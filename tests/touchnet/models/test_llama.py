import os
import subprocess
from multiprocessing import Manager

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch import distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM

from touchnet.bin import TrainConfig
from touchnet.utils.distributed import ParallelDims
from touchnet.utils.train_spec import get_train_spec


@pytest.fixture
def run_shell():
    def _run(cmd, check=True):
        return subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=check
        )
    return _run


def tiny_eval(parallel_dims: ParallelDims, folder: str, shard_folder: str):
    world_mesh = parallel_dims.build_mesh(device_type="cpu")
    train_spec = get_train_spec("llama")

    model_config = AutoConfig.from_pretrained("tests/assets/config/tiny_llama.json",
                                              attn_implementation="eager")
    model_config.return_dict = False  # NOTE: for compatibility with pipeline parallel
    with torch.device("meta"):
        shard_model = AutoModelForCausalLM.from_config(model_config)
        shard_model.apply(lambda m: setattr(m, "_is_hf_initialized", False))

    job_config = TrainConfig()
    job_config.training_compile = False
    train_spec.parallelize_fn(shard_model, world_mesh, parallel_dims, job_config)
    shard_model.to_empty(device="cpu")
    with torch.no_grad():
        shard_model.post_init()
        train_spec.additional_post_init_fn(shard_model, "cpu")
    shard_model.eval()

    # Load weights from un-shard ckpt
    dcp.load({"model": shard_model.state_dict()}, checkpoint_id=folder)

    # Save weights to shard ckpt
    dcp.save({"model": shard_model.state_dict()}, checkpoint_id=shard_folder)

    return True


def run_distributed(func, world_size, *args):
    with Manager() as manager:
        results = manager.list([None] * world_size)
        torch.multiprocessing.spawn(
            _dist_worker,
            args=(func, world_size, args, results),
            nprocs=world_size
        )
        return results


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


# TODO(xcsong): support PP?
@pytest.mark.parametrize("world_size, dp, pp, cp, tp", [
    (2, 1, 1, 1, 2),
    (8, 8, 1, 1, 1),
    (8, 2, 1, 4, 1),
    (8, 4, 1, 2, 1),
    (8, 2, 1, 2, 2),
])
def test_llama(run_shell, world_size, dp, pp, cp, tp):
    # NOTE(xcsong): cpu does not support sdpa or flexatt
    model_config = AutoConfig.from_pretrained("tests/assets/config/tiny_llama.json",
                                              attn_implementation="eager")
    model_config.return_dict = False  # NOTE: for compatibility with pipeline parallel
    model = AutoModelForCausalLM.from_config(model_config)
    with torch.no_grad():
        model.post_init()
    model.eval()

    folder = "tests/tmp/checkpoint/step-0"
    run_shell(f"rm -rf {folder}")
    os.makedirs(folder, exist_ok=True)
    dcp.save({"model": model.state_dict()}, checkpoint_id=folder)
    batch_size = 8
    max_len = 8
    assert max_len % cp == 0
    assert batch_size % dp == 0

    input_ids = torch.randint(low=0, high=model_config.vocab_size, size=(batch_size, max_len))
    position_ids = torch.arange(start=0, end=max_len, step=1, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)

    with torch.no_grad():
        results = model(
            input_ids=input_ids,
            position_ids=position_ids,
        )[0].float().cpu().numpy()

    parallel_dims = ParallelDims(
        dp_shard=dp, dp_replicate=1, cp=cp, tp=tp, pp=pp,
        world_size=world_size, enable_loss_parallel=True,
    )

    shard_folder = "tests/tmp/checkpoint/step-0-sharded"
    run_shell(f"rm -rf {shard_folder}")
    all_inputs = (parallel_dims, folder, shard_folder)
    run_distributed(tiny_eval, world_size, *all_inputs)

    train_spec = get_train_spec("llama")
    with torch.device("meta"):
        new_model = AutoModelForCausalLM.from_config(model_config)
        new_model.apply(lambda m: setattr(m, "_is_hf_initialized", False))
    new_model.to_empty(device="cpu")
    with torch.no_grad():
        new_model.post_init()
        train_spec.additional_post_init_fn(new_model, "cpu")
    # Load weights from shard ckpt
    dcp.load({"model": new_model.state_dict()}, checkpoint_id=shard_folder)
    new_model.eval()

    with torch.no_grad():
        new_results = new_model(
            input_ids=input_ids,
            position_ids=position_ids,
        )[0].float().cpu().numpy()

    assert results == pytest.approx(new_results, abs=1e-6)
    run_shell(f"rm -rf {folder}")
    run_shell(f"rm -rf {shard_folder}")
