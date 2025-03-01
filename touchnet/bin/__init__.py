from dataclasses import asdict, dataclass, field, fields


@dataclass
class MakeDataConfig:
    """Configuration object for make_data"""

    _argument_group_name = "make_data"

    save_dir: str = field(
        default="./exp",
        metadata={
            "help": ("dir to save data."),
        },
    )
    jsonl_path: str = field(
        default=None,
        metadata={
            "help": (
                "each line contains a json dict, "
                "e.g. `head -2 /mnt/data/data.jsonl`\n"
                "```\n"
                '{"key": 1, "wav": "/mnt/data/audio/1.wav", "text": "hello world"}\n'
                '{"key": 2, "wav": "/mnt/data/audio/2.wav", "text": "wow cool"}\n'
                "```\n"
            )
        },
    )
    num_utt_per_shard: int = field(
        default=1000,
        metadata={
            "help": ("number of utterances per shard."),
        },
    )
    audio_resample: int = field(
        default=16000,
        metadata={
            "help": ("reample rate of audio."),
        },
    )
    num_workers: int = field(
        default=10,
        metadata={
            "help": ("parallel workers."),
        },
    )
    datatypes: str = field(
        default="audio+metainfo",
        metadata={
            "help": ("types of multimodel Dataset."),
            "choices": [
                "metainfo",
                "audio+metainfo",
                "audio",
                "audiotoken",
                "texttoken",
            ],
        },
    )


@dataclass
class TrainConfig:
    """Configuration object for training"""

    _argument_group_name = "training"

    training_log_req: int = field(
        default=100,
        metadata={
            "help": (
                "How often to log metrics to TensorBoard, in iterations."
            ),
        },
    )
    training_enable_wandb: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to log metrics to Weights & Biases."
            ),
        },
    )
    training_enable_tensorboard: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to log metrics to TensorBoard."
            ),
        },
    )
    training_save_tb_folder: str = field(
        default="tensorboard",
        metadata={
            "help": (
                "Folder to dump TensorBoard states."
            ),
        },
    )
    training_tb_rank_0_only: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to save TensorBoard metrics only for rank 0 or for all ranks. "
                "When pipeline_parallel_degree is > 1, this option uses the 0th rank of the last stage pipeline group, "
                "which is the only stage that computes loss metrics."
            ),
        },
    )
    training_enable_cpu_offload: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to apply CPU offloading of parameters, gradients, and optimizer states in FSDP."
            ),
        },
    )
    training_trace_buf_size: int = field(
        default=20000,
        metadata={
            "help": (
                "Flight recorder ring buffer size, >0 means recording by default, 0 means disabled."
            ),
        },
    )
    training_trace_dump_folder: str = field(
        default="./exp",
        metadata={
            "help": ("Folder to dump job outputs."),
        },
    )
    training_init_timeout_seconds: int = field(
        default=300,
        metadata={
            "help": (
                "Timeout for communication operations, during initialization and first train step."
            ),
        },
    )
    training_train_timeout_seconds: int = field(
        default=100,
        metadata={
            "help": (
                "Timeout for communication operations after the first train step -- "
                "usually a tighter bound than during initialization."
            ),
        },
    )
    training_mixed_precision_param: str = field(
        default="bfloat16",
        metadata={
            "help": (
                "torch dtype to use for parameters when applying mixed precision via FSDP. "
                "This feature only takes effect when data_parallel_shard_degree > 1."
            ),
            "choices": ["bfloat16", "float32"],
        },
    )
    training_mixed_precision_reduce: str = field(
        default="float32",
        metadata={
            "help": (
                "torch dtype to use for reductions when applying mixed precision via FSDP. "
                "This feature only takes effect when data_parallel_shard_degree > 1."
            ),
            "choices": ["float32"],
        },
    )
    training_compile: bool = field(
        default=False,
        metadata={
            "help": ("Whether to compile the model"),
        },
    )
    training_enable_compiled_autograd: bool = field(
        default=False,
        metadata={
            "help": ("Enable CompiledAutograd to compile the backward."),
        },
    )
    training_gc_freq: int = field(
        default=50,
        metadata={
            "help": ("Python garbage control scheduling interval, in steps"),
        },
    )
    training_seed: int = field(
        default=2025,
        metadata={
            "help": ("Choose the base RNG seed used for training."),
        },
    )
    training_deterministic: bool = field(
        default=False,
        metadata={
            "help": ("Use deterministic algorithms wherever possible, may be slower."),
        },
    )
    training_activation_checkpoint_mode: str = field(
        default="selective",
        metadata={
            "help": ("Type of activation checkpointing."),
            "choices": ["none", "full", "selective"],
        },
    )
    training_activation_checkpoint_selective_ac_option: str = field(
        default="2",  # 2 = checkpoint every other layer
        metadata={
            "help": (
                "Selective activation checkpointing options ['int', 'op']. "
                "'int' (e.g., 2) for every nth layer, or 'op' for op level ac."
            ),
        },
    )
    training_steps: int = field(
        default=10000,
        metadata={
            "help": ("How many train steps to run"),
        },
    )
    training_warmup_steps: int = field(
        default=200,
        metadata={
            "help": ("Steps for lr scheduler warmup, normally 1/5 of --training_steps"),
        },
    )
    training_max_norm: float = field(
        default=1.0,
        metadata={
            "help": ("Max norm for gradient clipping."),
        },
    )
    training_data_parallel_replicate_degree: int = field(
        default=1,
        metadata={
            "help": (
                "The `data_parallel_replicate_degree` argument specifies the degree of "
                "data parallelism for weight replication. When this value is greater "
                "than 1, weights will be replicated across `data_parallel_replicate_degree` "
                "ranks. If `data_parallel_shard_degree` is also greater than 1, the parallelism "
                "method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the "
                "parallelism method used is DDP (Distributed Data Parallelism). "
                "1 means disabled."
            ),
        },
    )
    training_data_parallel_shard_degree: int = field(
        default=-1,
        metadata={
            "help": (
                "The `data_parallel_shard_degree` argument specifies the degree of data "
                "parallelism for weight sharding. When this value is greater than 1, weights "
                "will be sharded across `data_parallel_shard_degree` ranks. If "
                "`data_parallel_replicate_degree` is also greater than 1, the parallelism "
                "method used is HSDP (Hybrid Sharded Data Parallelism).  Otherwise, the "
                "parallelism method used is FSDP (Fully Sharded Data Parallelism). "
                "-1 means leftover ranks will be used (After DP_REPLICATE/SP/PP). Note that "
                "only `data_parallel_shard_degree` can be negative. 1 means disabled."
            ),
        },
    )
    training_tensor_parallel_degree: int = field(
        default=1,
        metadata={
            "help": ("Tensor Parallelism degree. 1 means disabled."),
        },
    )
    training_context_parallel_degree: int = field(
        default=1,
        metadata={
            "help": ("Context parallelism degree. 1 means disabled."),
        },
    )
    training_context_parallel_rotate_method: str = field(
        default="allgather",
        metadata={
            "help": (
                "The collective to use in context parallel SDPA for kv shards exchange. "
                "'allgather' means to all-gather all kv shards on ranks after the first sub-SDPA computation, "
                "'alltoall' means to all-to-all shuffle the kv shards. "
                "The default value is 'allgather'."
            ),
            "choices": ["allgather", "alltoall"],
        },
    )
    training_enable_loss_parallel: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to apply loss parallel when sequence parallel is enabled"
            ),
        },
    )
    training_enable_async_tensor_parallel: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to apply async tensor parallel (currently only effective when compile is enabled)"
            ),
        },
    )
    training_fsdp_reshard_after_forward: str = field(
        default="default",
        metadata={
            "help": (
                "`reshard_after_forward` specifies the policy for applying `reshard_after_forward` "
                "within an FSDP setup. `reshard_after_forward` controls parameter behavior after forward, "
                "trading off memory and communication. See torch's `fully_shard` API for more documentation "
                "on `reshard_after_forward`. "
                "The supported policies include `default`, `always` and `never`: "
                "- `default` applies default resharding behavior, implementing `smart defaults` for known optimal "
                "  scenarios. "
                "- `always` will enable `reshard_after_forward` for all forward passes. "
                "- `never` will disable `reshard_after_forward` for all forward passes."
            ),
            "choices": ["default", "always", "never"],
        },
    )
