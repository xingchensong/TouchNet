from transformers.models.llama.configuration_llama import LlamaConfig


class LlamaForASRConfig(LlamaConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaForASR`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B with a linear projector.

    Configuration objects inherit from [`LlamaConfig`] and can be used to control the model outputs. Read the
    documentation from [`LlamaConfig`] for more information.


    Args:
        input_size (`int`, defaults to 4096):
            Dimension of the input representations (mel or fbank).
    """

    model_type = "llama4asr"

    def __init__(
        self,
        input_size: int = 4096,
        **kwargs,
    ):
        self.input_size = input_size
        super().__init__(
            **kwargs,
        )


__all__ = ["LlamaForASRConfig"]
