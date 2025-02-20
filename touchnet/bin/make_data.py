import numpy
import torch
import os
import json
import multiprocessing

from typing import List, Type
from dataclasses import dataclass, field
from transformers.hf_argparser import HfArgumentParser
from subprocess import run, CalledProcessError

from touchnet.data.dataset import IndexWriter
from touchnet.utils.logging import init_logger, logger


@dataclass
class MakeDataConfig:
    """Configuration object for make_data"""

    _argument_group_name = "make_data"

    save_dir: str = field(
        metadata={
            "help": (
                "dir to save data."
            ),
        },
    )
    jsonl_path: str = field(
        metadata={
            "help": (
                "each line contains a json dict, "
                "e.g. `head -2 /mnt/data/data.jsonl`\n"
                "```\n"
                "{\"key\": 1, \"wav\": \"/mnt/data/audio/1.wav\", \"text\": \"hello world\"}\n"
                "{\"key\": 2, \"wav\": \"/mnt/data/audio/2.wav\", \"text\": \"wow cool\"}\n"
                "```\n"
            )
        },
    )
    num_utt_per_shard: int = field(
        default=1000,
        metadata={
            "help": (
                "number of utterances per shard."
            ),
        },
    )
    audio_resample: int = field(
        default=16000,
        metadata={
            "help": (
                "reample rate of audio."
            ),
        },
    )
    num_workers: int = field(
        default=10,
        metadata={
            "help": (
                "parallel workers."
            ),
        },
    )
    datatypes: str = field(
        default="pair_audio+pair_text",
        metadata={
            "help": (
                "types of multimodel Dataset."
            ),
            "choices": [
                "pair_audio+pair_text", "pure_audio", "pure_text",
            ],
        },
    )


class DataBuilder(object):
    """Builder class for the TouchDataset class

    Args:
        bin_path (str): The path to the data (.bin) file

        dtype (Type[numpy.number], optional): The dtype of the index file. Defaults to numpy.int32.

    Note:
        The format of the index file is as follows:
            [header]
            [version]
            [dtype]
            [seq_cnt==N] [doc_cnt==M]
            [seq1_len] [seq2_len] [...] [seqN_len]
            [seq1_idx] [seq2_idx] [...] [seqN_idx]
            [doc1_idx] [...] [docM_idx]  (M <= N)

        The format of the bin file is as follows:
            [bytes_of_seq1] [bytes_of_seq2] [...] [bytes_of_seqN]
    """

    def __init__(
        self, bin_path: str, dtype: Type[numpy.number] = numpy.int32
    ) -> None:
        self.data_file = open(bin_path, "wb")
        self.dtype = dtype

        self.sequence_lengths = []
        self.document_indices = [0]

    def add_item(self, tensor: torch.Tensor) -> None:
        """Add a single item to the dataset

        Args:
            tensor (torch.Tensor): The item to add to the data file
        """
        np_array = numpy.array(tensor.numpy(), dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.append(np_array.size)

    def add_document(
        self, tensor: torch.Tensor, lengths: List[int]
    ) -> None:
        """Add an entire document to the dataset

        Args:
            tensor (torch.Tensor): The document to add

            lengths (List[int]): The lengths of each item in the document
        """
        np_array = numpy.array(tensor, dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.extend(lengths)
        self.document_indices.append(len(self.sequence_lengths))

    def end_document(self) -> None:
        """Finalize the document, for use with DataBuilder.add_item"""
        self.document_indices.append(len(self.sequence_lengths))

    def finalize(self, idx_path: str) -> None:
        """Clean up and write the index (.idx) file

        Args:
            idx_path (str): The path to the index file
        """
        self.data_file.close()
        with IndexWriter(idx_path, self.dtype) as writer:
            writer.write(self.sequence_lengths, self.document_indices)


def load_audio(file: str,
               sr: int = 16000,
               start_time: float = 0.0,
               end_time: float = None):
    """Open an audio file and read as mono waveform,
       resampling as necessary, for a specific time segment

    Args:
        file: str
            The audio file to open

        sr: int
            The sample rate to resample the audio if necessary

        start_time: float
            The start time in seconds from which to extract the audio

        end_time: float
            The end time in seconds until which to extract the audio

    Returns
        A NumPy array containing the audio waveform, in int16 dtype.
    """

    # Base ffmpeg command
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-ss",
        str(start_time),  # Move -ss before -i for faster seeking
        "-i",
        file,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr)
    ]

    # Calculate and add duration
    if end_time is not None:
        duration = end_time - start_time
        cmd.extend(["-t", str(duration)])

    # Specify output to stdout
    cmd.append("-")

    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise Exception(f"Failed to load audio: {e.stderr.decode()}") from e

    # NOTE(xcsong): return int16 for memory efficiency, remember to normalize
    #               to [-1.0, 1.0] in float32 format before using it in the model
    #  e.g.
    #  >>>   waveform = load_audio(data["wav"])
    #  >>>   waveform = torch.from_numpy(waveform.astype(numpy.float32) / 32768.0)
    return numpy.frombuffer(out, numpy.int16).flatten()


def build_pure_text():
    pass


def build_pure_audio():
    pass


def build_pair_audio_pair_text(chunk: List[str], path_prefix: str,
                               cur_chunk: int, num_chunks: int,
                               conf: MakeDataConfig):
    builders = {
        "pair_audio": DataBuilder(f"{path_prefix}/pair_audio.bin", numpy.int16),
        "pair_text": DataBuilder(f"{path_prefix}/pair_text.bin", numpy.uint8),
    }
    logger.info('Processing {} {}/{}'.format(path_prefix, cur_chunk, num_chunks))

    for sample in chunk:
        try:
            data = json.loads(sample.strip())
            waveform = load_audio(data["wav"], conf.audio_resample)
            waveform = torch.from_numpy(waveform)
            sample_utf8 = sample.strip().encode('utf-8')
            sample_utf8 = numpy.frombuffer(sample_utf8, dtype=numpy.uint8)
            text = torch.from_numpy(numpy.copy(sample_utf8))
        except Exception as ex:
            logger.warning(f"Catch exception in reading {sample}: {ex}")
            continue
        builders["pair_audio"].add_item(waveform)
        builders["pair_text"].add_item(text)
        # documents contain only one sentence.
        builders["pair_audio"].end_document()
        builders["pair_text"].end_document()

    builders["pair_audio"].finalize(f"{path_prefix}/pair_audio.idx")
    builders["pair_text"].finalize(f"{path_prefix}/pair_text.idx")


if __name__ == "__main__":
    torch.set_num_threads(1)
    parser = HfArgumentParser(MakeDataConfig)
    conf = parser.parse_args_into_dataclasses()[0]
    init_logger()

    assert conf.jsonl_path is not None
    samples = []
    with open(conf.jsonl_path, "r") as f:
        for line in f:
            samples.append(line.strip())
    num = conf.num_utt_per_shard
    chunks = [samples[i:i + num] for i in range(0, len(samples), num)]
    os.makedirs(conf.save_dir, exist_ok=True)

    if conf.datatypes == "pair_audio+pair_text":
        processor = build_pair_audio_pair_text
    else:
        raise NotImplementedError()

    pool = multiprocessing.Pool(processes=conf.num_workers)
    shards_list = []
    num_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        path_prefix = '{}/{:09d}'.format(conf.save_dir, i)
        os.makedirs(path_prefix, exist_ok=True)
        shards_list.append(path_prefix)
        pool.apply_async(processor, (chunk, path_prefix, i,
                                     num_chunks, conf))

    pool.close()
    pool.join()

    with open(f"{conf.save_dir}/data.list", 'w', encoding='utf8') as fout:
        for name in shards_list:
            fout.write(name + '\n')
