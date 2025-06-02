import os

import numpy
import pytest
import torch

from touchnet.bin.make_data import DataBuilder
from touchnet.data import DataConfig
from touchnet.data.dataloader import ParallelAwareDataloader
from touchnet.data.datapipe import LowLevelTouchDatapipe


def build_fake_data(nnodes, nproc_per_node, max_epoch):
    total_number = nnodes * nproc_per_node * max_epoch
    shards_list = []
    for i in range(0, nnodes * nproc_per_node):
        path_prefix = f"tests/tmp/fake_data_{total_number}/shards_{i}"
        os.makedirs(path_prefix, exist_ok=True)
        builders = {
            "texttoken": DataBuilder(f"{path_prefix}/texttoken.bin",
                                     numpy.uint16)
        }

        for j in range(0, max_epoch):
            builders["texttoken"].add_item(torch.IntTensor([i * max_epoch + j]))
            # documents contain only one sentence.
            builders["texttoken"].end_document()

        builders["texttoken"].finalize(f"{path_prefix}/texttoken.idx")
        shards_list.append(path_prefix)
    with open(f"tests/tmp/fake_data_{total_number}/data.list", "w", encoding="utf8") as fout:
        for name in shards_list:
            fout.write(f"{name} texttoken\n")


# TODO(xcsong): support breal_point for num_workers > 1
@pytest.mark.parametrize("nnodes, nproc_per_node, max_epoch, num_workers, dp_rank, dp_worldsize, break_point", [
    (4, 8, 6, 1, 3, 8, 5),
    (4, 8, 6, 1, 3, 8, 12),
    (4, 8, 6, 1, 3, 8, 24),
    (4, 8, 6, 0, 3, 8, 12),
    (4, 8, 6, 0, 3, 8, 15),
    (4, 8, 6, 1, 3, 8, -1),
    (1, 8, 6, 4, 1, 2, -1),
    (1, 8, 6, 2, 1, 4, -1),
    (4, 8, 6, 4, 3, 8, -1),
    (2, 8, 6, 4, 0, 2, -1),
])
def test_dataloader(nnodes, nproc_per_node, max_epoch, num_workers, dp_rank, dp_worldsize, break_point):
    if num_workers > 0:
        assert (nnodes * nproc_per_node) % (dp_worldsize * num_workers) == 0
        assert nnodes * nproc_per_node * max_epoch // dp_worldsize >= break_point
    total_number = nnodes * nproc_per_node * max_epoch
    build_fake_data(nnodes, nproc_per_node, max_epoch)
    config = DataConfig(datalist_path=f"tests/tmp/fake_data_{total_number}/data.list",
                        datalist_sharding=True,
                        datalist_shuffling=False,
                        dataset_shuffling=False,
                        dataset_mmap=True)
    datapipe = LowLevelTouchDatapipe(config, dp_rank, dp_worldsize)
    dataloader = ParallelAwareDataloader(
        dataset=datapipe,
        dp_rank=dp_rank,
        dp_world_size=dp_worldsize,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    state_dict = {}
    loaded_data = []
    for i, data in enumerate(dataloader):
        if i == break_point:
            state_dict = dataloader.state_dict()
            break
        input_ids = data["input_ids"]
        assert len(input_ids) == 1
        loaded_data.append(input_ids[0])
    del dataloader, datapipe

    # resume from mid-checkpoint
    if len(state_dict.keys()) > 0:
        datapipe = LowLevelTouchDatapipe(config, dp_rank, dp_worldsize)
        dataloader = ParallelAwareDataloader(
            dataset=datapipe,
            dp_rank=dp_rank,
            dp_world_size=dp_worldsize,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
        )
        print(state_dict)
        for k in state_dict:
            if "dp_rank" in k:
                print(state_dict[k])
        dataloader.load_state_dict(state_dict)

        for i, data in enumerate(dataloader):
            input_ids = data["input_ids"]
            assert len(input_ids) == 1
            loaded_data.append(input_ids[0])

    loaded_data = numpy.array(loaded_data, dtype=numpy.int32)

    expected_data = numpy.array([i for i in range(0, total_number)],
                                dtype=numpy.int32).reshape(-1, max_epoch)
    expected_data = expected_data[dp_rank::dp_worldsize, :]
    if num_workers > 0:
        buffer = []
        for i in range(num_workers):
            tmp_data = expected_data[i::num_workers, :].reshape(1, -1)
            if tmp_data.shape[-1] > 0:
                buffer.append(tmp_data)
        expected_data = numpy.concatenate(buffer, axis=0).transpose()
    expected_data = expected_data.reshape(-1)
    assert numpy.allclose(loaded_data, expected_data)
