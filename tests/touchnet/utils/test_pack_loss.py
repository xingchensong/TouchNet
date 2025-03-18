from multiprocessing import Manager

import pytest
import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.nn.functional import all_gather


def calc_batch_dp_loss(batch_input_ids=None, batch_labels=None):
    """
    Calculate loss using data parallelism (batch splitting).

    Args:
        batch_input_ids: Tensor of shape [batch, length, vocab] containing logits
        batch_labels: Tensor of shape [batch, length] containing target indices

    Returns:
        float: The average loss across all processes
    """
    batch_input_ids = batch_input_ids.detach().clone()  # [batch, length, vocab]
    batch_labels = batch_labels.detach().clone()        # [batch, length]
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert len(batch_input_ids) % world_size == 0

    # Split data in batch-dim to simulate data parallel
    batch_input_ids = torch.split(
        batch_input_ids,
        len(batch_input_ids) // world_size, dim=0
    )[rank]
    batch_labels = torch.split(
        batch_labels,
        len(batch_labels) // world_size, dim=0
    )[rank]

    vocab = batch_input_ids.size(-1)
    loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    batch_size = batch_input_ids.size(0)
    loss = loss_fn(batch_input_ids.reshape(-1, vocab), batch_labels.reshape(-1))
    # 1. reduce loss over sentences
    loss = loss.reshape(batch_size, -1).sum(dim=1) / ((batch_labels != -100).sum(dim=1).float() + 1e-12)
    # 2. reduce loss over batches
    loss = loss.mean()
    # 3. reduce loss over dp
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    loss = loss / world_size
    print(f"rank {rank}: {loss.item()}")
    return loss.item()


def calc_pack_sp_loss(pack_input_ids=None, pack_labels=None, num_tokens=None):
    """
    Calculate loss using (packed) sequence parallelism (sequence splitting).

    Args:
        pack_input_ids: Tensor of shape [length, vocab] containing logits
        pack_labels: Tensor of shape [length] containing target indices
        num_tokens: number of tokens for each sentence

    Returns:
        float: The average loss across all processes
    """
    # NOTE(xcsong): In pack mode, we assume batch_size == 1 and sp == world_size
    pack_input_ids = pack_input_ids.detach().clone()  # [length, vocab]
    pack_labels = pack_labels.detach().clone()        # [length]
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert len(pack_input_ids) % world_size == 0
    assert len(pack_input_ids) == len(pack_labels)
    assert sum(num_tokens) == len(pack_labels)
    orig_pack_labels = pack_labels.detach().clone()

    # Split data in sequence-dim to simulate sequence parallel
    pack_input_ids = torch.split(
        pack_input_ids,
        len(pack_input_ids) // world_size, dim=0
    )[rank]
    pack_labels = torch.split(
        pack_labels,
        len(pack_labels) // world_size, dim=0
    )[rank]

    loss_fc = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    loss = loss_fc(pack_input_ids, pack_labels)

    all_loss = all_gather(loss)
    all_loss = torch.cat(all_loss)

    loss_list = all_loss.split(num_tokens)
    labels_list = orig_pack_labels.split(num_tokens)
    # 1. reduce loss over sentences
    loss_list = [
        loss.sum() / ((label != -100).sum().float() + 1e-12)
        for loss, label in zip(loss_list, labels_list)
    ]
    # 2. reduce loss over batches
    loss = torch.stack(loss_list).mean()
    # 3. since sp == world_size, we got dp == 1, no need for reducing over dp
    print(f"rank {rank}: {loss.item()}")
    return loss.item()


def run_distributed(func, world_size, *args):
    with Manager() as manager:
        results = manager.list([None] * world_size)
        torch.multiprocessing.spawn(
            _dist_worker,
            args=(func, world_size, args, results),
            nprocs=world_size
        )
        return list(results)


def _dist_worker(rank, func, world_size, args, results):
    torch.distributed.init_process_group(
        backend='gloo',
        init_method='tcp://127.0.0.1:29505',
        world_size=world_size,
        rank=rank
    )
    try:
        result = func(*args)
        results[rank] = result
    finally:
        dist.barrier()
        dist.destroy_process_group()


# NOTE(xcsong): The following references provide context for pack loss implementation:
# - Technical explanation of pack mode vs batch mode: https://zhuanlan.zhihu.com/p/721652210
# - Related implementation discussion: https://github.com/THUDM/LongAlign/issues/3
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_pack_loss(world_size):
    a1 = torch.randn(5, 9).float()
    b1 = torch.Tensor([-100, -100, 1, 2, 3]).long()
    a2 = torch.randn(8, 9).float()
    b2 = torch.Tensor([4, -100, 3, 4, 6, -100, -100, 7]).long()
    a3 = torch.randn(3, 9).float()
    b3 = torch.Tensor([-100, 6, 8]).long()
    a4 = torch.randn(4, 9).float()
    b4 = torch.Tensor([-100, 7, 8, -100]).long()
    a5 = torch.randn(6, 9).float()
    b5 = torch.Tensor([-100, -100, 7, 4, 2, 5]).long()
    a6 = torch.randn(3, 9).float()
    b6 = torch.Tensor([5, 8, -100]).long()

    max_item_length = 8
    batch_input_ids = torch.zeros(8, max_item_length, 9)
    batch_labels = torch.ones(8, max_item_length).long() * -100
    for i, (a, b) in enumerate(
        [(a1, b1), (a2, b2), (a3, b3), (a2, b2),
         (a6, b6), (a4, b4), (a5, b5), (a6, b6)]
    ):
        batch_input_ids[i, :a.size(0)] = a
        batch_labels[i, :b.size(0)] = b

    # NOTE(xcsong): In pack mode, we assume batch_size == 1
    pack_input_ids = torch.cat([a1, a2, a3, a2, a6, a4, a5, a6], dim=0)
    pack_labels = torch.cat([b1, b2, b3, b2, b6, b4, b5, b6], dim=0)
    num_tokens = [5, 8, 3, 8, 3, 4, 6, 3]

    data_batch = (batch_input_ids, batch_labels)
    data_pack = (pack_input_ids, pack_labels, num_tokens)

    results_batch = run_distributed(calc_batch_dp_loss, world_size, *data_batch)
    results_pack = run_distributed(calc_pack_sp_loss, world_size, *data_pack)

    assert len(set(results_batch)) == 1, f"The results of each child process are inconsistent: {results_batch}"
    assert len(set(results_pack)) == 1, f"The results of each child process are inconsistent: {results_pack}"
    assert results_batch[0] == pytest.approx(results_pack[0], abs=1e-6)
