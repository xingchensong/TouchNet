from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Union


@dataclass
class DataConfig:
    """Configuration object for datas"""

    _argument_group_name = "data"

    datapipe_type: str = field(
        default="texttoken",
        metadata={
            "help": (
                "type of datapipe: "
                "- 'texttoken': extract text token offline and save to *.bin "
                "- 'audio+metainfo': save wav and text to *.bin, extract text token online during training"
            ),
            "choices": ["texttoken", "audio+metainfo"],
        },
    )
    datalist_path: str = field(
        default=None,
        metadata={
            "help": (
                "list of dataset, each line is a prefix path to a `TouchDataset`. "
                "e.g. `head -2 /mnt/data/data.list`\n"
                "```\n"
                "/mnt/data/aishell1\n"
                "/mnt/data/aishell2\n"
                "```\n"
            )
        },
    )
    datalist_dev_path: str = field(
        default=None,
        metadata={
            "help": (
                "list of dataset, each line is a prefix path to a `TouchDataset`. "
                "e.g. `head -2 /mnt/data/data.list`\n"
                "```\n"
                "/mnt/data/aishell1\n"
                "/mnt/data/aishell2\n"
                "```\n"
            )
        },
    )
    datalist_test_path: str = field(
        default=None,
        metadata={
            "help": (
                "list of dataset, each line is a prefix path to a `TouchDataset`. "
                "e.g. `head -2 /mnt/data/data.list`\n"
                "```\n"
                "/mnt/data/aishell1\n"
                "/mnt/data/aishell2\n"
                "```\n"
            )
        },
    )
    datalist_sharding: bool = field(
        default=True,
        metadata={
            "help": (
                "Shard datalist for dp."
            )
        },
    )
    datalist_epoch: int = field(
        default=1,
        metadata={
            "help": (
                "Set this value to a larger value to ensure that the total number of training steps "
                "determined by the epoch is greater than or equal to `lr_scheduler_steps`."
            ),
        },
    )
    datalist_shuffling: bool = field(
        default=True,
        metadata={
            "help": (
                "shuffle datalist."
            )
        },
    )
    dataset_shuffling: bool = field(
        default=True,
        metadata={
            "help": (
                "shuffle dataset."
            )
        },
    )
    dataset_mmap: bool = field(
        default=True,
        metadata={
            "help": (
                "Use mmap for reading .bin files in Dataset."
            )
        },
    )
    dataset_random_cut_audio: bool = field(
        default=True,
        metadata={
            "help": (
                "Randomly cut audio in audio pretraining mode."
            )
        },
    )
    dataset_batchsize: int = field(
        default=8,
        metadata={
            "help": (
                "batch size."
            )
        },
    )
    dataset_audio_seqlen: int = field(
        default=8192,
        metadata={
            "help": (
                "max audio sequence length (after stack&stride)."
            )
        },
    )
    dataset_text_seqlen: int = field(
        default=2048,
        metadata={
            "help": (
                "max text sequence length in tokens. "
                "default to dataset_audio_seqlen // 4."
            )
        },
    )
    audio_max_length_in_ms_for_filter: int = field(
        default=800000,
        metadata={
            "help": (
                "Max length of audio in ms. "
                "Drop utterance which is greater than this value."
            )
        },
    )
    audio_min_length_in_ms_for_filter: int = field(
        default=200,
        metadata={
            "help": (
                "Min length of audio in ms. "
                "Drop utterance which is less than this value."
            )
        },
    )
    text_max_length_in_tokens_for_filter: int = field(
        default=800000,
        metadata={
            "help": (
                "Max length of tokens in text. "
                "Drop utterance which is greater than this value."
            )
        },
    )
    text_min_length_in_tokens_for_filter: int = field(
        default=1,
        metadata={
            "help": (
                "Min length of tokens in text. "
                "Drop utterance which is less than this value."
            )
        },
    )
    max_text_audio_ratio: float = field(
        default=1.0,
        metadata={
            "help": (
                "Max ratio of len(text) / len(audio). "
                "Drop utterance which is greater than this value. "
                "only valid in `audio+metainfo` mode."
            )
        },
    )
    min_text_audio_ratio: float = field(
        default=0.0005,
        metadata={
            "help": (
                "Min ratio of len(text) / len(audio). "
                "Drop utterance which is less than this value."
            )
        },
    )
    audio_resample_rate: int = field(
        default=16000,
        metadata={
            "help": (
                "target sample rate"
            )
        },
    )
    audio_speed_perturb: bool = field(
        default=True,
        metadata={
            "help": (
                "Apply speed perturb to the data."
            )
        },
    )
    audio_speed_perturb_speeds: List[float] = field(
        default_factory=lambda: [0.9, 1.0, 1.1],
        metadata={
            "help": (
                "Optional speed."
            )
        },
    )
    audio_feat_type: str = field(
        default="fbank",
        metadata={
            "help": (
                "feature type."
            ),
            "choices": [
                "fbank", "mfcc", "log_mel_spectrogram"
            ],
        },
    )
    audiofeat_spec_aug: bool = field(
        default=True,
        metadata={
            "help": (
                "Apply spec aug to the data."
            )
        },
    )
    audiofeat_spec_aug_num_t_mask: int = field(
        default=2,
        metadata={
            "help": (
                "number of time mask."
            )
        },
    )
    audiofeat_spec_aug_num_f_mask: int = field(
        default=2,
        metadata={
            "help": (
                "number of frequence mask."
            )
        },
    )
    audiofeat_spec_aug_max_t: int = field(
        default=50,
        metadata={
            "help": (
                "max length of time mask."
            )
        },
    )
    audiofeat_spec_aug_max_f: int = field(
        default=10,
        metadata={
            "help": (
                "max length of frequence mask."
            )
        },
    )
    audiofeat_spec_sub: bool = field(
        default=True,
        metadata={
            "help": (
                "Apply spec sub to the data."
            )
        },
    )
    audiofeat_spec_sub_num_t_sub: int = field(
        default=3,
        metadata={
            "help": (
                "number of substitution."
            )
        },
    )
    audiofeat_spec_sub_max_t: int = field(
        default=20,
        metadata={
            "help": (
                "max length of sub chunk."
            )
        },
    )
    audiofeat_spec_trim: bool = field(
        default=False,
        metadata={
            "help": (
                "Apply spec trim to the data."
            )
        },
    )
    audiofeat_spec_trim_max_t: int = field(
        default=20,
        metadata={
            "help": (
                "max length for triming."
            )
        },
    )
    audiofeat_num_mel_bins: int = field(
        default=23,
        metadata={
            "help": (
                "Number of triangular mel-frequency bins. "
                "used in fbank/mfcc/log_mel_spectrogram."
            )
        },
    )
    audiofeat_frame_length: int = field(
        default=25,
        metadata={
            "help": (
                "Frame length in milliseconds."
                "used in fbank/mfcc."
            )
        },
    )
    audiofeat_frame_shift: int = field(
        default=10,
        metadata={
            "help": (
                "Frame shift in milliseconds."
                "used in fbank/mfcc."
            )
        },
    )
    audiofeat_dither: float = field(
        default=0.0,
        metadata={
            "help": (
                "Dithering constant (0.0 means no dither). If you turn this off, you should set "
                "the energy_floor option, e.g. to 1.0 or 0.1. "
                "used in fbank/mfcc."
            )
        },
    )
    audiofeat_num_ceps: int = field(
        default=40,
        metadata={
            "help": (
                "Number of cepstra in MFCC computation (including C0). "
                "used in mfcc."
            )
        },
    )
    audiofeat_high_freq: float = field(
        default=0.0,
        metadata={
            "help": (
                "High cutoff frequency for mel bins (if <= 0, offset from Nyquist). "
                "used in mfcc."
            )
        },
    )
    audiofeat_low_freq: float = field(
        default=20.0,
        metadata={
            "help": (
                "Low cutoff frequency for mel bins. "
                "used in mfcc."
            )
        },
    )
    audiofeat_padding: int = field(
        default=0,
        metadata={
            "help": (
                " "
                "used in log_mel_spectrogram."
            )
        },
    )
    audiofeat_n_fft: int = field(
        default=400,
        metadata={
            "help": (
                "size of Fourier transform. "
                "used in log_mel_spectrogram."
            )
        },
    )
    audiofeat_hop_length: int = field(
        default=160,
        metadata={
            "help": (
                "the distance between neighboring sliding window frames. "
                "used in log_mel_spectrogram."
            )
        },
    )
    audiofeat_stack_length: int = field(
        default=7,
        metadata={
            "help": (
                "number of frames to stack."
            )
        },
    )
    audiofeat_stride_length: int = field(
        default=6,
        metadata={
            "help": (
                "number of frames to stride."
            )
        },
    )
    audiofeat_normalize: bool = field(
        default=True,
        metadata={
            "help": (
                "normalize stacked feat to stablize training."
            )
        },
    )
    dataloader_drop_last_batch: bool = field(
        default=True,
        metadata={
            "help": (
                "drop last batch as it might contain too much padding values."
            )
        },
    )
    dataloader_num_workers: int = field(
        default=6,
        metadata={
            "help": (
                "num_workers for dataloader."
            )
        },
    )
    dataloader_prefetch_factor: int = field(
        default=6,
        metadata={
            "help": (
                "prefetch fector for dataloader."
            )
        },
    )
