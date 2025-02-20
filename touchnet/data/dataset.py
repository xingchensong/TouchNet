
# Copyright (c) Facebook, Inc. and its affiliates.
#               Megatron-LM team.
#               2025 WeNet Community. Xingchen Song(sxc19@tsinghua.org.cn)

import os
import struct
import time
from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache
from types import TracebackType
from typing import List, Optional, Tuple, Type, Union, Literal

import numpy
import torch

from touchnet.utils.logging import logger

_INDEX_HEADER = b"MMIDIDX\x00\x00"


class DType(Enum):
    """The NumPy data type Enum for writing/reading the TouchDataset indices"""

    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, value: Type[numpy.number]) -> int:
        """Get the code from the dtype

        Args:
            value (Type[numpy.number]): The dtype

        Returns:
            int: The code
        """
        return cls[value.__name__].value

    @classmethod
    def dtype_from_code(cls, value: int) -> Type[numpy.number]:
        """Get the dtype from the code

        Args:
            value (int): The code

        Returns:
            Type[numpy.number]: The dtype
        """
        return getattr(numpy, cls(value).name)

    @staticmethod
    def size(key: Union[int, Type[numpy.number]]) -> int:
        """Get the size of the dtype/code in bytes

        Args:
            key (Union[int, Type[numpy.number]]): The dtype or code

        Raises:
            ValueError: If the key is neither dtype nor integer code

        Returns:
            int: The size of the dtype/code in in bytes
        """
        if isinstance(key, int):
            return DType.dtype_from_code(key)().itemsize
        elif numpy.number in key.__mro__:
            return key().itemsize
        else:
            raise ValueError

    @staticmethod
    def optimal_dtype(cardinality: Optional[int]) -> Type[numpy.number]:
        """Get the dtype to use for an index of a certain cardinality

        Args:
            cardinality (Optional[int]): The number of elements to be indexed

        Returns:
            Type[numpy.number]: The dtype to use for the index
        """
        if cardinality is not None and cardinality < 65500:
            return numpy.uint16
        else:
            return numpy.int32


class IndexWriter(object):
    """Object class to write the index (.idx) file

    Args:
        idx_path (str): The path to the index file

        dtype (Type[numpy.number]): The dtype of the .bin file

    Note:
        The format of the index file is as follows:
            [header]
            [version]
            [dtype]
            [seq_cnt==N] [doc_cnt==M]
            [seq1_len] [seq2_len] [...] [seqN_len]
            [seq1_idx] [seq2_idx] [...] [seqN_idx]
            [doc1_idx] [...] [docM_idx]  (M <= N)
    """

    def __init__(self, idx_path: str, dtype: Type[numpy.number]) -> None:
        self.idx_path = idx_path
        self.dtype = dtype

    def __enter__(self) -> "IndexWriter":
        """Enter the context introduced by the 'with' keyword

        Returns:
            IndexWriter: The instance
        """
        self.idx_writer = open(self.idx_path, "wb")
        # fixed, vestigial practice
        self.idx_writer.write(_INDEX_HEADER)
        # fixed, vestigial practice
        self.idx_writer.write(struct.pack("<Q", 1))
        # the numeric code for the dtype
        self.idx_writer.write(struct.pack("<B", DType.code_from_dtype(self.dtype)))
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        """Exit the context introduced by the 'with' keyword

        Args:
            exc_type (Optional[Type[BaseException]]): Exception type

            exc_val (Optional[BaseException]): Exception value

            exc_tb (Optional[TracebackType]): Exception traceback object

        Returns:
            Optional[bool]: Whether to silence the exception
        """
        self.idx_writer.close()

    def write(
        self,
        sequence_lengths: List[int],
        document_indices: List[int],
    ) -> None:
        """Write the index (.idx) file

        Args:
            sequence_lengths (List[int]): The length of each sequence

            document_indices (List[int]): The seqyebce indices demarcating the end of each document
        """
        sequence_pointers = self._sequence_pointers(sequence_lengths)

        # the number of sequences in the dataset
        sequence_count = len(sequence_lengths)
        self.idx_writer.write(struct.pack("<Q", sequence_count))

        # the number of documents in the dataset
        document_count = len(document_indices)
        self.idx_writer.write(struct.pack("<Q", document_count))

        # the number of tokens per sequence
        sequence_lengths = numpy.array(sequence_lengths, dtype=numpy.int32)
        self.idx_writer.write(sequence_lengths.tobytes(order="C"))
        del sequence_lengths

        # the byte offsets for all sequences
        sequence_pointers = numpy.array(sequence_pointers, dtype=numpy.int64)
        self.idx_writer.write(sequence_pointers.tobytes(order="C"))
        del sequence_pointers

        # the sequence indices marking the end of each document
        document_indices = numpy.array(document_indices, dtype=numpy.int64)
        self.idx_writer.write(document_indices.tobytes(order="C"))

    def _sequence_pointers(self, sequence_lengths: List[int]) -> List[int]:
        """Build the sequence pointers per the sequence lengths and dtype size

        Args:
            sequence_lengths (List[int]): The length of each sequence

        Returns:
            List[int]: The pointer to the beginning of each sequence
        """
        itemsize = DType.size(self.dtype)
        curr_ptr = 0
        list_ptr = []
        for length in sequence_lengths:
            list_ptr.append(curr_ptr)
            curr_ptr += length * itemsize
        return list_ptr


class IndexReader(object):
    """Object class to read the index (.idx) file

    Args:
        idx_path (str): The path to the index file
    """

    def __init__(self, idx_path: str) -> None:

        logger.info(f"Load the {type(self).__name__} from {idx_path}")

        with open(idx_path, "rb") as stream:
            header = stream.read(9)
            assert header == _INDEX_HEADER, f"bad header, cannot read: {idx_path}"

            version = struct.unpack("<Q", stream.read(8))[0]
            assert version == 1, f"bad version, cannot read: {idx_path}"

            code = struct.unpack("<B", stream.read(1))[0]
            self.dtype = DType.dtype_from_code(code)
            self.dtype_size = DType.size(self.dtype)

            self.sequence_count = struct.unpack("<Q", stream.read(8))[0]
            self.document_count = struct.unpack("<Q", stream.read(8))[0]

            offset = stream.tell()

        self.bin_buffer_mmap = numpy.memmap(idx_path, mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

        logger.info("\tExtract the sequence lengths")
        t_beg = time.time()
        self.sequence_lengths = numpy.frombuffer(
            self.bin_buffer, dtype=numpy.int32, count=self.sequence_count, offset=offset
        )
        t_end = time.time()
        logger.info("\t> time elapsed: {t_end - t_beg:4f} seconds")

        logger.info("\tExtract the sequence pointers")
        t_beg = time.time()
        self.sequence_pointers = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )
        t_end = time.time()
        logger.info(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        logger.info("\tExtract the document indices")
        t_beg = time.time()
        self.document_indices = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.document_count,
            offset=offset + self.sequence_lengths.nbytes + self.sequence_pointers.nbytes,
        )
        t_end = time.time()
        logger.info(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        assert self.sequence_lengths.shape[0] == len(self)
        assert self.sequence_lengths.shape[0] == self.sequence_count
        assert self.sequence_lengths.shape[0] == self.document_indices[-1]

        logger.info(f"> total number of sequences: {len(self)}")
        logger.info(
            f"> total number of documents: {self.document_indices.shape[0] - 1}",
        )

    def __del__(self) -> None:
        """Clean up the object"""
        if hasattr(self, "bin_buffer_mmap"):
            self.bin_buffer_mmap._mmap.close()
            del self.bin_buffer_mmap

    def __len__(self) -> int:
        """Return the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return self.sequence_count

    @lru_cache(maxsize=8)
    def __getitem__(self, idx: int) -> Tuple[numpy.int32, numpy.int64, Optional[numpy.int8]]:
        """Return the pointer, length, and mode at the index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.int32, numpy.int64, Optional[numpy.int8]]: The pointer, length and mode at the index
        """
        return (
            self.sequence_pointers[idx],
            self.sequence_lengths[idx],
        )


class BinReader(ABC):
    """Abstract class to read the data (.bin) file"""

    @abstractmethod
    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (Type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype` constructed from reading bytes from the data file starting at `offset`.
        """
        pass


class MMapBinReader(BinReader):
    """A BinReader that memory maps the data (.bin) file

    Args:
        bin_path (str): bin_path (str): The path to the data (.bin) file.
    """

    def __init__(self, bin_path: str) -> None:
        self._bin_buffer_mmap = numpy.memmap(bin_path, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (Type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype` constructed from reading bytes from the data file starting at `offset`.
        """
        return numpy.frombuffer(self._bin_buffer, dtype=dtype, count=count, offset=offset)

    def __del__(self) -> None:
        """Clean up the object."""
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap


class FileBinReader(BinReader):
    """A BinReader that reads from the data (.bin) file using a file pointer

    Args:
        bin_path (str): bin_path (str): The path to the data (.bin) file.
    """

    def __init__(self, bin_path: str) -> None:
        self._bin_path = bin_path

    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (Type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype` constructed from reading bytes from the data file starting at `offset`.
        """
        sequence = numpy.empty(count, dtype=dtype)
        with open(self._bin_path, mode='rb', buffering=0) as bin_buffer_file:
            bin_buffer_file.seek(offset)
            bin_buffer_file.readinto(sequence)
        return sequence


class TouchDataset(torch.utils.data.Dataset):
    """The low-level interface dataset class

    Args:
        path_prefix (str): The index (.idx) and data (.bin) prefix

        mmap (bool): Whether to mmap the .bin files. Defaults to True.
    """

    def __init__(
        self,
        path_prefix: str,
        mmap: bool = True,
        datatypes: Literal["pure_audio", "pure_text", "pair_audio+pair_text"] = "pair_audio+pair_text"
    ) -> None:
        super().__init__()
        self.path_prefix = None
        self.mmap = None
        self.datatypes = None

        self.index = {}
        self.bin_reader = {}

        self.initialize(path_prefix, mmap, datatypes)

    def initialize(
        self, path_prefix: str, mmap: bool, datatypes: str
    ) -> None:
        """Initialize the dataset

        This method is called by IndexedDataset.__init__ during object creation and by
        IndexedDataset.__setstate__ during un-pickling

        Args:
            path_prefix (str): The index (.idx) and data (.bin) prefix

            mmap (bool): Whether to mmap the .bin file
        """
        self.path_prefix = path_prefix
        self.mmap = mmap
        self.datatypes = datatypes
        for d in datatypes.split('+'):
            idx_path = f"{path_prefix}/{d}.idx"
            bin_path = f"{path_prefix}/{d}.bin"
            assert os.path.exists(idx_path) and os.path.exists(
                bin_path
            ), f"One or both of the .idx and .bin files cannot be found at the path prefix {path_prefix}"
            if mmap:
                self.bin_reader[d] = MMapBinReader(bin_path)
            else:
                self.bin_reader[d] = FileBinReader(bin_path)
            self.index[d] = IndexReader(idx_path)
        l = len(self.index[list(self.index.keys())[0]])
        for d in self.index.keys():
            assert l == len(self.index[d])

    def __getstate__(self) -> Tuple[str, bool, str]:
        """Get the state during pickling

        Returns:
            Tuple[str, bool, str]: The state tuple
        """
        return self.path_prefix, self.mmap, self.datatypes

    def __setstate__(self, state: Tuple[str, bool, str]) -> None:
        """Set the state during un-pickling

        Args:
            state (Tuple[str, bool, str]): The state tuple
        """
        path_prefix, mmap, datatypes = state
        self.initialize(path_prefix, mmap, datatypes)

    def __del__(self) -> None:
        """Clean up the object"""
        del self.bin_reader
        del self.index

    def __len__(self) -> int:
        """Return the length of the dataset i.e. the number of sequences in the index

        Returns:
            int: The length of the dataset
        """
        return len(self.index[list(self.index.keys())[0]])

    def get_idx(self, idx: int, datatype: str) -> Tuple[numpy.int32, numpy.int64]:
        sequence_pointer, sequence_length = self.index[datatype][idx]
        return sequence_pointer, sequence_length

    def get(self, idx: int, datatype: str, offset: int = 0, length: Optional[int] = None) -> numpy.ndarray:
        """Retrieve a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.

        Args:
            idx (Union[int, numpy.integer]): The index into the dataset

            offset (int): The integer token offset in the sequence

            length (int): The number of tokens to grab from the sequence

        Returns:
            Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]: The sequence tokens and modes at the index
        """
        sequence_pointer, sequence_length = self.get_idx(idx, datatype)
        if length is None:
            length = sequence_length - offset
        sequence_pointer += offset * DType.size(self.index[datatype].dtype)
        sequence = self.bin_reader[datatype].read(
            dtype=self.index[datatype].dtype, count=length, offset=sequence_pointer
        )
        return sequence


# TODO(xcsong): InterleavedTouchDataset
