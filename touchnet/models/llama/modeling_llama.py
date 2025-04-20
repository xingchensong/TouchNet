# -*- coding: utf-8 -*-
# Copyright (c) 2025, Xingchen Song(sxc19@tsinghua.org.cn)

from typing import List, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (LLAMA_INPUTS_DOCSTRING,
                                                      KwargsForCausalLM,
                                                      LlamaForCausalLM,
                                                      LlamaPreTrainedModel)
from transformers.processing_utils import Unpack
from transformers.utils import (add_start_docstrings_to_model_forward,
                                replace_return_docstrings)

from touchnet.models.llama.configuration_llama import LlamaForASRConfig


class LlamaForASR(LlamaPreTrainedModel, GenerationMixin):
    config_class = LlamaForASRConfig
    base_model_prefix = "llm.model"

    def __init__(self, config: LlamaForASRConfig):
        super().__init__(config)
        self.llm = LlamaForCausalLM(config.llm_config)
        self.projector = torch.nn.Linear(config.input_size, config.llm_config.hidden_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.llm.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.llm.set_decoder(decoder)

    def get_decoder(self):
        return self.llm.get_decoder()

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class="LlamaForASRConfig")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            The arguments of this method are the same as the ones from [`LlamaForCausalLM`],
            Read the documentation from [`LlamaForCausalLM`] for more information.

            The only difference between [`LlamaForASR`] and [`LlamaForCausalLM`] lies in the `inputs_embeds`,
            which is projected to the hidden size in [`LlamaForASR`] before feeding it to the [`LlamaModel`].

            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from touchnet.models.llama.modeling_llama import LlamaForASR

        >>> model = LlamaForASR.from_pretrained("/absolute/path/to/llama_for_asr")
        >>> tokenizer = AutoTokenizer.from_pretrained("/absolute/path/to/llama_for_asr")

        >>> # Generate
        >>> generate_ids = model.generate(input_ids=input_ids, inputs_embeds=inputs_embeds, max_length=300)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert labels is None  # we calculate loss in train-loop
        assert inputs_embeds is not None  # (B, T, D)

        # NOTE(xcsong): This is the only difference between LlamaForASR and LlamaForCausalLM
        inputs_embeds_audio = self.projector(inputs_embeds)  # (B, T // sp // cp, D),  sp == tp

        if input_ids is not None:
            inputs_embeds_text = self.model.embed_tokens(input_ids)  # (B, T // sp // cp, D), sp == tp
            inputs_embeds = inputs_embeds_audio + inputs_embeds_text
        else:
            inputs_embeds = inputs_embeds_audio

        if torch.isnan(inputs_embeds).any():
            raise ValueError("NaN in data.")

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.llm(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None  # NOTE(xcsong): we calculate loss outside of the model

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        r"""

        Generates sequences of token ids for models with a language modeling head.

        See documents in GenerationMixin.generate() for more infos.

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.
        """

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()

__all__ = ["LlamaForASR"]
