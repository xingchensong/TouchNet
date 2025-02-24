import json
from typing import Any, Dict

import numpy
import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from touchnet.data import DataConfig, processor
from touchnet.data.dataset import TouchDataset
from touchnet.tokenizer import TokenizerConfig, build_tokenizer


# TODO(xcsong): is_build_on_rank
class TouchDatapipe(IterableDataset, Stateful):
    """The high-level interface dataset class

        We have two shuffle stage in the Dataset. The first is global
        shuffle at list level. The second is global shuffle
        at training samples level.
    """

    def __init__(self, config: DataConfig, dp_rank: int, dp_world_size: int):
        super().__init__()
        self.lists = []
        with open(config.datalist_path, "r") as f:
            lists = f.readlines()
            for l in lists:
                self.lists.append(l.strip())
        self.config = config
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size

        # Variables for checkpointing
        self.epoch = 0
        self.consumed_samples = 0

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.epoch = state_dict["epoch"]
        self.consumed_samples = state_dict["consumed_samples"]

    def state_dict(self):
        return {"epoch": self.epoch, "consumed_samples": self.consumed_samples}

    def __iter__(self):
        list_idxs = list(range(len(self.lists)))

        # 1st shuffle on lists
        if self.config.datalist_shuffling:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            list_idxs = torch.randperm(len(self.lists), generator=g).tolist()

        # 1st sharding on dp ranks
        if self.config.datalist_sharding:
            list_idxs = list_idxs[self.dp_rank::self.dp_world_size]

        # 2nd sharding on dataloader workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        list_idxs = list_idxs[worker_id::num_workers]

        for list_idx in list_idxs:
            _dataset = TouchDataset(self.lists[list_idx], self.config.dataset_mmap,
                                    self.config.dataset_datatypes)

            # 2nd shuffle on samples
            num_samples = len(_dataset)
            g = torch.Generator()
            g.manual_seed(self.epoch)
            if self.config.dataset_shuffling:
                sample_idxs = torch.randperm(num_samples, generator=g).tolist()
            else:
                sample_idxs = list(range(num_samples))

            for i, sample_idx in enumerate(sample_idxs):
                self.consumed_samples = i
                if self.config.dataset_datatypes == "pure_text":
                    # for text pre-training
                    text = _dataset.get(sample_idx, "pure_text")
                    text = text.tobytes().decode('utf-8')
                    text = json.loads(text.strip())
                    yield text
                elif self.config.dataset_datatypes == "pure_audio":
                    # for audio pre-training
                    if self.config.dataset_random_cut_audio:
                        audio_p, audio_l = _dataset.get_idx(sample_idx, "pure_audio")
                        # TODO(xcsong): slice audio
                        pass
                    else:
                        audio = _dataset.get(sample_idx, "pure_audio")
                    yield dict(audio=audio)
                elif self.config.dataset_datatypes == "pair_audio+pair_text":
                    # for audio-text alignment
                    text = _dataset.get(sample_idx, "pair_text")
                    text = text.tobytes().decode('utf-8')
                    text = json.loads(text.strip())
                    offset = 0
                    length = None
                    info = text.get("info", None)
                    if info is not None:
                        segments = info.get("segments", None)
                        if segments is not None:
                            assert "sample_rate" in info
                            sample_rate = info["sample_rate"]
                            g = torch.Generator()
                            g.manual_seed(self.epoch + i)
                            segment = segments[torch.randint(len(segments), (1,), generator=g).item()]
                            start = int(float(segment["start"]) * sample_rate)
                            end = int(float(segment["end"]) * sample_rate)
                            offset = start
                            length = end - start
                    audio = _dataset.get(sample_idx, "pair_audio", offset=offset, length=length)
                    audio = audio.astype(numpy.float32) / 32768.0  # normalize to [-1.0, 1.0]
                    text["waveform"] = torch.from_numpy(audio).unsqueeze(0)  # [1, T]
                    yield text
                else:
                    raise NotImplementedError(f"unsupported datatypes: {self.config.dataset_datatypes}")


class Processor(IterableDataset):
    """
    Processor class for data processing.
    """

    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


def pairaudio_pairtext_datapipe(
    data_config: DataConfig,
    tokenizer_config: TokenizerConfig,
    dp_rank: int, dp_world_size: int,
):
    """ Construct datapipe from configs
    """
    datapipe = TouchDatapipe(data_config, dp_rank, dp_world_size)
    # TODO(xcsong): load state for datapipe
    tokenizer = build_tokenizer(tokenizer_config)

    datapipe = Processor(datapipe, processor.text_tokenize, tokenizer)
    datapipe = Processor(datapipe, processor.filter, data_config)

    datapipe = Processor(datapipe, processor.audio_resample, data_config)

    # wav-level augment
    if data_config.audio_speed_perturb:
        datapipe = Processor(datapipe, processor.audio_speed_perturb, data_config)

    if data_config.audio_feat_type == 'fbank':
        datapipe = Processor(datapipe, processor.audio_compute_fbank, data_config)
    elif data_config.audio_feat_type == 'mfcc':
        datapipe = Processor(datapipe, processor.audio_compute_mfcc, data_config)
    elif data_config.audio_feat_type == 'log_mel_spectrogram':
        datapipe = Processor(datapipe, processor.audio_compute_log_mel_spectrogram,
                             data_config)

    # feat-level augment
    if data_config.audiofeat_spec_aug:
        datapipe = Processor(datapipe, processor.audiofeat_spec_aug, data_config)
    if data_config.audiofeat_spec_sub:
        datapipe = Processor(datapipe, processor.audiofeat_spec_sub, data_config)
    if data_config.audiofeat_spec_trim:
        datapipe = Processor(datapipe, processor.audiofeat_spec_trim, data_config)

    datapipe = Processor(datapipe, processor.audiofeat_stack, data_config)
    datapipe = Processor(datapipe, processor.batch_pairaudio_pairtext, data_config)
    return datapipe


def pureaudio_datapipe():
    pass

def puretext_datapipe():
    pass
