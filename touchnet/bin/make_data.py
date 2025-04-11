import json
import multiprocessing
import os
from subprocess import CalledProcessError, run
from typing import List, Type

import numpy
import torch
from transformers.hf_argparser import HfArgumentParser

from touchnet.bin import MakeDataConfig
from touchnet.data.dataset import DType, IndexWriter
from touchnet.tokenizer import TokenizerConfig
from touchnet.tokenizer.tokenizer import build_tokenizer
from touchnet.utils.logging import init_logger, logger


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

    def __init__(self, bin_path: str, dtype: Type[numpy.number] = numpy.int32) -> None:
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

    def add_document(self, tensor: torch.Tensor, lengths: List[int]) -> None:
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


def load_audio(
    file: str, sr: int = 16000, start_time: float = 0.0, end_time: float = None
):
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
        str(sr),
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


def build_texttoken(
    chunk: List[str],
    path_prefix: str,
    cur_chunk: int,
    num_chunks: int,
    conf: MakeDataConfig,
    tok_conf: TokenizerConfig,
    *args,
    **kwargs,
):
    assert tok_conf.tokenizer_model is not None, "tok_conf.tokenizer_model cannot be None"
    tokenizer = build_tokenizer(tok_conf)
    builders = {
        "texttoken": DataBuilder(f"{path_prefix}/texttoken.bin",
                                 DType.optimal_dtype(tokenizer.vocab_size))
    }
    logger.info("Processing {} {}/{}".format(path_prefix, cur_chunk, num_chunks))

    for sample in chunk:
        try:
            data = json.loads(sample.strip())
            if len(data["text"]) == 0:
                continue
            # TODO(xcsong): split sentence ?
            # NOTE(xcsong): add bos/eos in batch_xxx()
            texttoken = tokenizer.tokenize(data["text"], add_special_tokens=False)
        except Exception as ex:
            logger.warning(f"Catch exception in reading {sample}: {ex}")
            continue
        builders["texttoken"].add_item(torch.IntTensor(texttoken))
        # documents contain only one sentence.
        builders["texttoken"].end_document()

    builders["texttoken"].finalize(f"{path_prefix}/texttoken.idx")


def build_audio_and_metainfo(
    chunk: List[str],
    path_prefix: str,
    cur_chunk: int,
    num_chunks: int,
    conf: MakeDataConfig,
    *args,
    **kwargs,
):
    builders = {
        "audio": DataBuilder(f"{path_prefix}/audio.bin", numpy.int16),
        "metainfo": DataBuilder(f"{path_prefix}/metainfo.bin", numpy.uint8),
    }
    logger.info("Processing {} {}/{}".format(path_prefix, cur_chunk, num_chunks))

    for sample in chunk:
        try:
            data = json.loads(sample.strip())
            waveform = load_audio(data["wav"], conf.audio_resample)
            waveform = torch.from_numpy(waveform)
            data["sample_rate"] = conf.audio_resample
            sample = json.dumps(data, ensure_ascii=False)
            sample_utf8 = sample.strip().encode("utf-8")
            sample_utf8 = numpy.frombuffer(sample_utf8, dtype=numpy.uint8)
            text = torch.from_numpy(numpy.copy(sample_utf8))
        except Exception as ex:
            logger.warning(f"Catch exception in reading {sample}: {ex}")
            continue
        builders["audio"].add_item(waveform)
        builders["metainfo"].add_item(text)
        # documents contain only one sentence.
        builders["audio"].end_document()
        builders["metainfo"].end_document()

    builders["audio"].finalize(f"{path_prefix}/audio.idx")
    builders["metainfo"].finalize(f"{path_prefix}/metainfo.idx")


def handle_error(e):
    logger.error(f"Catch error in subprocess: {e}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    os.environ["PYTHONUNBUFFERED"] = "1"
    parser = HfArgumentParser([MakeDataConfig, TokenizerConfig])
    (conf, tok_conf) = parser.parse_args_into_dataclasses()
    init_logger()

    assert conf.jsonl_path is not None, "conf.jsonl_path cannot be None"
    samples = []
    with open(conf.jsonl_path, "r") as f:
        for line in f:
            samples.append(line.strip())
    num = conf.num_utt_per_shard
    chunks = [samples[i : i + num] for i in range(0, len(samples), num)]
    os.makedirs(conf.save_dir, exist_ok=True)

    if conf.datatypes == "audio+metainfo":
        processor = build_audio_and_metainfo
    elif conf.datatypes == "texttoken":
        processor = build_texttoken
    else:
        raise NotImplementedError()

    pool = multiprocessing.Pool(processes=conf.num_workers)
    shards_list = []
    num_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        path_prefix = "{}/{:09d}".format(conf.save_dir, i)
        os.makedirs(path_prefix, exist_ok=True)
        shards_list.append(path_prefix)
        pool.apply_async(processor, (chunk, path_prefix, i, num_chunks, conf, tok_conf),
                         error_callback=handle_error)

    pool.close()
    pool.join()

    with open(f"{conf.save_dir}/data.list", "w", encoding="utf8") as fout:
        for name in shards_list:
            fout.write(f"{name} {conf.datatypes}\n")
