# -*- coding: utf-8 -*-
# Copyright (c) 2025, Xingchen Song(sxc19@tsinghua.org.cn)

from typing import List, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (LLAMA_INPUTS_DOCSTRING,
                                                      KwargsForCausalLM,
                                                      LlamaForCausalLM)
from transformers.processing_utils import Unpack
from transformers.utils import (add_start_docstrings_to_model_forward,
                                replace_return_docstrings)

from touchnet.models.touch_audio.configuration_touch_audio import TouchAudioConfig


# TODO(xcsong): switch to Qwen
class TouchAudioForCausalLM(LlamaForCausalLM):
    config_class = TouchAudioConfig

    def __init__(self, config: TouchAudioConfig):
        super().__init__(config)
        self.projector = torch.nn.Linear(config.input_size, config.hidden_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class="TouchAudioConfig")
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

            The only difference between [`TouchAudioForCausalLM`] and [`LlamaForCausalLM`] lies in the `inputs_embeds`,
            which is projected to the hidden size in [`TouchAudioForCausalLM`] before feeding it to the [`LlamaModel`].

        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert labels is None  # we calculate loss in train-loop
        assert inputs_embeds is not None  # (B, T, D)

        # NOTE(xcsong): This is the only difference between TouchAudioForCausalLM and LlamaForCausalLM
        inputs_embeds_audio = self.projector(inputs_embeds)  # (B, T // sp // cp, D),  sp == tp

        if input_ids is not None:
            inputs_embeds_text = self.model.embed_tokens(input_ids)  # (B, T // sp // cp, D), sp == tp
            inputs_embeds = inputs_embeds_audio + inputs_embeds_text
        else:
            inputs_embeds = inputs_embeds_audio

        if torch.isnan(inputs_embeds).any():
            raise ValueError("NaN in data.")

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
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

__all__ = ["TouchAudioForCausalLM"]
