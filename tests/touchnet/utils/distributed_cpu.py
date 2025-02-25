import os

import torch
from transformers.hf_argparser import HfArgumentParser

from touchnet.bin import TrainConfig
from touchnet.utils.distributed import ParallelDims
from touchnet.utils.logging import init_logger

init_logger()

parser = HfArgumentParser(TrainConfig)
job_config = parser.parse_args_into_dataclasses()[0]

# init distributed
world_size = int(os.environ["WORLD_SIZE"])
parallel_dims = ParallelDims(
    dp_shard=job_config.training_data_parallel_shard_degree,
    dp_replicate=job_config.training_data_parallel_replicate_degree,
    cp=job_config.training_context_parallel_degree,
    tp=job_config.training_tensor_parallel_degree,
    pp=1,
    world_size=world_size,
    enable_loss_parallel=job_config.training_enable_loss_parallel,
)

torch.distributed.init_process_group(backend="gloo")
world_mesh = parallel_dims.build_mesh(device_type="cpu")

if parallel_dims.dp_enabled:
    dp_mesh = world_mesh["dp"]
    dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
else:
    dp_degree, dp_rank = 1, 0

if parallel_dims.tp_enabled:
    tp_mesh = world_mesh["tp"]
    tp_degree, tp_rank = tp_mesh.size(), tp_mesh.get_local_rank()
else:
    tp_degree, tp_rank = 1, 0

if parallel_dims.cp_enabled:
    cp_mesh = world_mesh["cp"]
    cp_degree, cp_rank = cp_mesh.size(), cp_mesh.get_local_rank()
else:
    cp_degree, cp_rank = 1, 0

rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
print(f"""rank={rank}, world_size={world_size}, local_rank={local_rank},
          dp_degree={dp_degree}, dp_rank={dp_rank},
          tp_degree={tp_degree}, tp_rank={tp_rank},
          cp_degree={cp_degree}, cp_rank={cp_rank}""")

torch.distributed.destroy_process_group()
