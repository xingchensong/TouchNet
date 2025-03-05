import subprocess
import time

import pytest


def is_port_open(host, port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)
        return s.connect_ex((host, port)) == 0


# @pytest.mark.parametrize("master_port, nnodes, nproc_per_node, dp_shard, dp_replicate, cp, tp, lp", [
#     (29500, 4, 8, -1, 2, 2, 4, True),
#     (29501, 4, 8, -1, 1, 4, 2, False),
#     (29502, 4, 8, -1, 1, 2, 4, True),
# ])
@pytest.mark.skip(reason="太吃机器资源，离线测试就行")
def test_distributed_cpu(master_port, nnodes, nproc_per_node, dp_shard, dp_replicate, cp, tp, lp):
    master_addr = "127.0.0.1"
    processes = []

    master_cmd = [
        "torchrun",
        "--nnodes", str(nnodes),
        "--nproc_per_node", str(nproc_per_node),
        "--node_rank", "0",
        "--master_addr", master_addr,
        "--master_port", str(master_port),
        "--rdzv_endpoint", f"{master_addr}:{master_port}",
        "--rdzv_backend", "c10d",
        "tests/touchnet/utils/distributed_cpu.py",
        "--training_data_parallel_shard_degree", str(dp_shard),
        "--training_data_parallel_replicate_degree", str(dp_replicate),
        "--training_context_parallel_degree", str(cp),
        "--training_tensor_parallel_degree", str(tp),
        "--training_enable_loss_parallel", str(lp),
    ]
    processes.append(subprocess.Popen(master_cmd))

    while not is_port_open(master_addr, master_port):
        time.sleep(1)

    for node_rank in range(1, nnodes):
        cmd = [
            "torchrun",
            "--nnodes", str(nnodes),
            "--nproc_per_node", str(nproc_per_node),
            "--node_rank", str(node_rank),
            "--master_addr", master_addr,
            "--master_port", str(master_port),
            "--rdzv_endpoint", f"{master_addr}:{master_port}",
            "--rdzv_backend", "c10d",
            "tests/touchnet/utils/distributed_cpu.py",
            "--training_data_parallel_shard_degree", str(dp_shard),
            "--training_data_parallel_replicate_degree", str(dp_replicate),
            "--training_context_parallel_degree", str(cp),
            "--training_tensor_parallel_degree", str(tp),
            "--training_enable_loss_parallel", str(lp),
        ]
        processes.append(subprocess.Popen(cmd))

    for p in processes:
        p.wait()
