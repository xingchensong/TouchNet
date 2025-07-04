# -*- coding: utf-8 -*-
# Copyright (c) 2025, Xingchen Song(sxc19@tsinghua.org.cn)

from typing import Any, List, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModelForCausalLM
from transformers.utils import replace_return_docstrings

from touchnet.models.touch_audio.configuration_touch_audio import \
    TouchAudioConfig
from touchnet.utils.logging import logger


class TouchAudioForCausalLM(PreTrainedModel, GenerationMixin):
    """
    TouchAudio model for causal language modeling that extends AutoModelForCausalLM
    with audio input support through a linear projection layer.

    This model projects audio embeddings to the hidden size before feeding them
    to the base Qwen2/Llama/... model, enabling audio-based language generation tasks.
    """
    config_class = TouchAudioConfig
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def __init__(self, config: TouchAudioConfig):
        super().__init__(config)
        logger.info(f"config.audio_config:\n{config.audio_config}")
        self.projector = torch.nn.Linear(config.audio_config.input_size, config.text_config.hidden_size, bias=False)

        self.vocab_size = config.text_config.vocab_size
        logger.info(f"config.text_config:\n{config.text_config}")
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in self.language_model._tied_weights_keys]

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self._padding_side = "left"  # set it to left by default, user can use setter to change padding_sides

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def padding_side(self):
        return self._padding_side

    @padding_side.setter
    def padding_side(self, padding_side: str):
        if padding_side not in ["left", "right"]:
            raise ValueError(f"{padding_side} is not `left` or `right`.")
        self._padding_side = padding_side

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_input_embeddings
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_input_embeddings
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_output_embeddings
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_decoder
    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_decoder
    def get_decoder(self):
        return self.language_model.get_decoder()

    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class="TouchAudioConfig")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_features: Optional[torch.FloatTensor] = None,
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
        **kwargs: Any,
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

        if inputs_embeds is None:
            inputs_embeds_text = self.language_model.model.embed_tokens(input_ids)  # (B, T // sp // cp, D), sp == tp
            if input_features is not None and input_ids.shape[1] != 1:
                # NOTE(xcsong): This is the only difference between TouchAudioForCausalLM and LlamaForCausalLM
                inputs_embeds_audio = self.projector(input_features)  # (B, T // sp // cp, D),  sp == tp
            else:
                inputs_embeds_audio = self.projector(torch.zeros(input_ids.shape[0], input_ids.shape[1],
                                                                 self.config.audio_config.input_size, device=input_ids.device))
            inputs_embeds = inputs_embeds_audio + inputs_embeds_text

        if inputs_embeds is not None and torch.isnan(inputs_embeds).any():
            raise ValueError("NaN in data.")

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.language_model(
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

        outputs.attention_mask = attention_mask
        return outputs


__all__ = ["TouchAudioForCausalLM"]
