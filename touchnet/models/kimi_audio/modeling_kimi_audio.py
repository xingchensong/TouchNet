# -*- coding: utf-8 -*-
# Copyright (c) 2025, The Moonshot AI Team, Qwen Team, and HuggingFace Inc. team. All rights reserved.
#                     Xingchen Song(sxc19@tsinghua.org.cn)
#
# The code is based on Qwen2.5-7B, but modified for KimiAudio.
#
# Licensing Information:
# - Code derived from Qwen2.5-7B is licensed under the Apache License, Version 2.0.
# - Other parts of the code are licensed under the MIT License.
#
# Apache License, Version 2.0:
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""PyTorch KimiAudio model."""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from packaging import version
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

assert version.parse(transformers.__version__) >= version.parse("4.34.1")

from transformers import Qwen2Config
from transformers.cache_utils import (Cache, DynamicCache, SlidingWindowCache,
                                      StaticCache)
from transformers.generation import GenerationMixin
from transformers.integrations.flex_attention import \
    make_flex_block_causal_mask
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.models.qwen2.modeling_qwen2 import (QWEN2_INPUTS_DOCSTRING,
                                                      QWEN2_START_DOCSTRING,
                                                      KwargsForCausalLM,
                                                      Qwen2DecoderLayer,
                                                      Qwen2PreTrainedModel,
                                                      Qwen2RMSNorm,
                                                      Qwen2RotaryEmbedding)
from transformers.models.whisper.modeling_whisper import (
    WhisperEncoder, WhisperEncoderLayer, WhisperPreTrainedModel)
from transformers.processing_utils import Unpack
from transformers.utils import (add_start_docstrings,
                                add_start_docstrings_to_model_forward,
                                replace_return_docstrings)

from touchnet.models.kimi_audio.configuration_kimi_audio import (
    KimiAudioConfig, WhisperVQConfig)


def vector_quantize(inputs, codebook):
    embedding_size = codebook.size(1)
    inputs_flatten = inputs.reshape(-1, embedding_size)
    codebook_sqr = torch.sum(codebook ** 2, dim=1)
    inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
    # Compute the distances to the codebook
    distances = torch.addmm(codebook_sqr + inputs_sqr,
                            inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

    _, indices_flatten = torch.min(distances, dim=1)
    codes_flatten = torch.index_select(codebook, dim=0,
                                       index=indices_flatten)
    codes = codes_flatten.view_as(inputs)
    return codes, indices_flatten, distances


class CausalConv1d(nn.Conv1d):
    """
    NOTE(xcsong): This class is directly copied from
    https://github.com/THUDM/GLM-4-Voice/blob/eb00ce9142e8d98b0ed7c57cd47e0d6d5dce9a1a/speech_tokenizer/modeling_whisper.py
    with proper modifications
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        **kwargs
    ):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs
        )

        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, inp):
        x = torch.nn.functional.pad(inp.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        return super(CausalConv1d, self).forward(x)


class WhisperVQEncoder(WhisperPreTrainedModel):
    """
    NOTE(xcsong): This class is directly copied from
    https://github.com/THUDM/GLM-4-Voice/blob/eb00ce9142e8d98b0ed7c57cd47e0d6d5dce9a1a/speech_tokenizer/modeling_whisper.py
    with proper modifications, i.e., remove training-related code

    Transformer encoder consisting of *config.quantize_position* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: WhisperVQConfig
    """

    def __init__(self, config: WhisperVQConfig):
        super().__init__(config)
        self.config = config
        self.dropout = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        # NOTE(xcsong): For simplicity, we always assume encoder_causal_convolution == True
        assert config.encoder_causal_convolution
        self.conv1 = CausalConv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = CausalConv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)
        # NOTE(xcsong): For simplicity, we always assume quantize_encoder_only == True
        assert config.quantize_encoder_only
        self.layers = nn.ModuleList([WhisperEncoderLayer(config)
                                     for _ in range(config.quantize_position)])

        self.gradient_checkpointing = False
        # Parameters related to pooling layer
        self.pooling_layer = None
        # Parameters related to quantization layer
        self.codebook = None
        self.embed_positions2 = None
        # Initialize weights and apply final processing
        self.init_pooling_layer(config)
        self.init_quantize_layer(config)
        self.post_init()

    def init_pooling_layer(self, config: WhisperVQConfig):
        assert config.pooling_kernel_size is not None
        if config.pooling_type == "max":
            self.pooling_layer = nn.MaxPool1d(kernel_size=config.pooling_kernel_size)
        elif config.pooling_type == "avg":
            self.pooling_layer = nn.AvgPool1d(kernel_size=config.pooling_kernel_size)
        else:
            raise NotImplementedError(f"Pooling type {config.pooling_type} not implemented")

    def init_quantize_layer(self, config: WhisperVQConfig, quantize_load_codebook=None):
        # NOTE(xcsong): For simplicity, we always assume assertions == True
        assert config.quantize_vocab_size is not None
        assert config.pooling_position is not None
        assert config.quantize_position >= config.pooling_position
        assert config.pooling_kernel_size is not None
        self.codebook = nn.Embedding(config.quantize_vocab_size, self.config.d_model)
        if quantize_load_codebook is not None:
            init_codes = np.load(quantize_load_codebook)
            self.codebook.weight.data.copy_(torch.from_numpy(init_codes))
        max_source_positions = self.max_source_positions
        max_source_positions = math.ceil(max_source_positions / self.config.pooling_kernel_size)
        self.embed_positions2 = nn.Embedding(max_source_positions, self.config.d_model)
        self.embed_positions2.weight.data.copy_(self.embed_positions.weight.data[:max_source_positions])
        # NOTE(xcsong): we do not need ema for inference, keep it for disable the warning of missing weight
        if config.quantize_ema_decay is not None:
            self.codebook.weight.requires_grad = False
            self.register_buffer("ema_count", torch.ones(config.quantize_vocab_size, dtype=torch.float))
            self.register_buffer("ema_weight", self.codebook.weight.data.clone().float())

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def get_block_causal_attention_mask(self, attention_mask, block_size=50):
        dtype = self.dtype
        batch_size, seq_length = attention_mask.shape
        causal_mask = torch.tril(
            torch.ones(1, seq_length, seq_length, dtype=torch.bool, device=attention_mask.device))
        block_square_mask = []
        for start in range(0, seq_length, block_size):
            end = min(start + block_size, seq_length)
            length = end - start
            block_square_mask.append(causal_mask.new_ones((length, length)))
        block_square_mask = torch.block_diag(*block_square_mask)
        block_causal_mask = causal_mask | block_square_mask
        block_causal_mask = block_causal_mask & attention_mask[:, None, :]
        block_causal_mask = block_causal_mask.to(dtype=dtype)  # fp16 compatibility
        block_causal_mask = (1.0 - block_causal_mask) * torch.finfo(dtype).min
        block_causal_mask = block_causal_mask.unsqueeze(1)
        return block_causal_mask

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        quantized_token_ids=None,
        **kwargs,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
        """

        batch_size, feature_size, seq_length = input_features.shape
        seq_length = seq_length // (self.conv1.stride[0] * self.conv2.stride[0])

        attention_mask = attention_mask[:, :: self.conv1.stride[0] * self.conv2.stride[0]]
        # NOTE(xcsong): For simplicity, we always assume quantize_causal_block_size is not None, default 200
        extended_attention_mask = self.get_block_causal_attention_mask(attention_mask,
                                                                       block_size=self.config.quantize_causal_block_size)

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # (B, C, T) --> (B, T, C)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos[:seq_length]
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        assert attention_mask.shape[-1] == hidden_states.shape[1]
        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):

            layer_outputs = encoder_layer(
                hidden_states,
                extended_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
            )

            hidden_states = layer_outputs[0]

            if idx + 1 == self.config.pooling_position and self.config.pooling_kernel_size is not None:
                hidden_states = hidden_states.permute(0, 2, 1)
                if hidden_states.shape[-1] % self.config.pooling_kernel_size != 0:
                    hidden_states = torch.nn.functional.pad(hidden_states, (
                        0, self.config.pooling_kernel_size - hidden_states.shape[-1] % self.config.pooling_kernel_size))
                hidden_states = self.pooling_layer(hidden_states).permute(0, 2, 1)

            if idx + 1 == self.config.quantize_position and self.config.quantize_vocab_size is not None:
                if quantized_token_ids is not None:
                    hidden_states = self.codebook(quantized_token_ids)
                else:
                    hidden_quantized, indices_flat, distances = vector_quantize(hidden_states, self.codebook.weight)
                    quantized_token_ids = indices_flat.reshape(batch_size, hidden_quantized.shape[1])
                    hidden_states = hidden_quantized
                hidden_states = hidden_states + self.embed_positions2.weight[:hidden_states.shape[1]]

        return quantized_token_ids


class VQAdaptor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.kimia_adaptor_input_dim, config.hidden_size, bias=True),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, bias=True),
        )

    def forward(self, x):
        return self.layers(x)


class CustomWhisperEncoder(WhisperEncoder):
    def _init_weights(self, module: nn.Module):
        pass


@add_start_docstrings(
    "The bare MoonshotKimia Model outputting raw hidden-states without any specific head on top. ",
    "It is basically the same as `Qwen2Model`, except for mimo_layers/mimo_norm/vq_adapter.",
    QWEN2_START_DOCSTRING,
)
class MoonshotKimiaModel(Qwen2PreTrainedModel):
    """
    NOTE(xcsong): This class is directly copied from
    https://huggingface.co/moonshotai/Kimi-Audio-7B/blob/main/modeling_moonshot_kimia.py
    with proper modifications

    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`QwenDecoderLayer`]

    Args:
        config: KimiAudioConfig
    """

    config_class = KimiAudioConfig

    def __init__(self, config: KimiAudioConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.kimia_mimo_transformer_from_layer_index = (
            config.kimia_mimo_transformer_from_layer_index
        )

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        # NOTE(xcsong): Different from original impl from kimi-audio (see below), we use standard Qwen2DecoderLayer & Qwen2Attention
        #   https://huggingface.co/moonshotai/Kimi-Audio-7B/blob/main/modeling_moonshot_kimia.py#L232-L516
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # NOTE(xcsong): Different from original impl from kimi-audio (see below), we use standard qwen2-style rotary_emb
        #   https://huggingface.co/moonshotai/Kimi-Audio-7B/blob/main/modeling_moonshot_kimia.py#L190-L229
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        # NOTE(kimiaudio): Extra 1B audio transformers
        # NOTE(xcsong): Different from original impl from kimi-audio (see below), we use standard Qwen2DecoderLayer & Qwen2Attention
        #   https://huggingface.co/moonshotai/Kimi-Audio-7B/blob/main/modeling_moonshot_kimia.py#L232-L516
        self.mimo_layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx=layer_idx + config.num_hidden_layers) for layer_idx in range(config.kimia_mimo_layers)]  # noqa
        )
        self.mimo_norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_whisper_feature = config.use_whisper_feature
        if self.use_whisper_feature:
            self.vq_adaptor = VQAdaptor(config)

        self.kimia_media_begin = config.kimia_media_begin  # <|im_media_begin|>
        self.kimia_media_end = config.kimia_media_end      # <|im_media_end|>

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Cache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            print(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = [DynamicCache(), DynamicCache()]

        if cache_position is None:
            past_seen_tokens = past_key_values[0].get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if past_key_values is not None:
            past_key_value = past_key_values[0]
            past_key_value_mimo = past_key_values[1]
            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values[0], output_attentions
            )
        else:
            past_key_value = None
            past_key_value_mimo = None
            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, None, output_attentions
            )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if idx == self.kimia_mimo_transformer_from_layer_index:
                mimo_hidden_states = hidden_states.clone()

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # apply audio transformer layers
        for idx, decoder_layer in enumerate(self.mimo_layers):
            if output_hidden_states:
                all_hidden_states += (mimo_hidden_states,)

            layer_outputs = decoder_layer(
                mimo_hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_value_mimo,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            mimo_hidden_states = layer_outputs[0]

        mimo_hidden_states = self.mimo_norm(mimo_hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (mimo_hidden_states,)

        if not return_dict:
            output = (hidden_states, mimo_hidden_states, past_key_values,)
            if all_hidden_states is not None:
                output += all_hidden_states
            if all_self_attns is not None:
                output += all_self_attns
            return output

        return BaseModelOutputWithPast(
            last_hidden_state=(hidden_states, mimo_hidden_states),
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        """
        NOTE(xcsong): This function is directly copied from `modeling_qwen2.py::Qwen2Model::_update_causal_mask`
        """

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2Config,
        past_key_values: Cache,
    ):
        """
        NOTE(xcsong): This function is directly copied from `modeling_qwen2.py::Qwen2Model::_prepare_4d_causal_attention_mask_with_cache_position`  # noqa

        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.  # noqa
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.  # noqa
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            diagonal_attend_mask = torch.arange(target_length, device=cache_position.device) > cache_position.reshape(
                -1, 1
            )
            if config.get_text_config().sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=cache_position.device) <= (
                        cache_position.reshape(-1, 1) - config.get_text_config().sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class KimiASampler:
    def __init__(
        self,
        audio_top_k: int,
        audio_temperature: float,
        audio_repetition_penalty: float,
        audio_repetition_window_size: int,
        text_top_k: int,
        text_temperature: float,
        text_repetition_penalty: float,
        text_repetition_window_size: int,
        kimia_text_audiodelaytokens: int = 6,
    ):
        self.audio_top_k = audio_top_k
        self.audio_temperature = audio_temperature
        self.text_top_k = text_top_k
        self.text_temperature = text_temperature

        self.audio_repetition_penalty = audio_repetition_penalty
        self.audio_repetition_window_size = audio_repetition_window_size
        self.text_repetition_penalty = text_repetition_penalty
        self.text_repetition_window_size = text_repetition_window_size
        self.kimia_text_audiodelaytokens = kimia_text_audiodelaytokens

        # TODO(xcsong): Make those ids configurable
        self.kimia_text_blank = 151666   # <|im_kimia_text_blank|>
        self.kimia_text_eos = 151667     # <|im_kimia_text_eos|>
        self.eod_ids = [151645, 151663]  # <|im_msg_end|>, <|im_media_end|>

    def sample_audio_logits(
        self, logits: torch.Tensor, recent_tokens=None
    ) -> torch.Tensor:
        """Sample from audio logits with top-k, temperature and repetition penalty.

        Args:
            logits: Logits tensor of shape [batch_size, seq_len, vocab_size] or [batch_size, vocab_size]
            recent_tokens: Optional tensor of recent tokens for repetition penalty

        Returns:
            Sampled token ids
        """
        # Take the last token's logits if we have a sequence dimension
        if len(logits.shape) == 3:
            logits = logits[:, -1]
        batch_size = logits.size(0)
        logits = logits.clone()
        # Apply repetition penalty if needed
        if (
            self.audio_repetition_penalty > 1.0
            and recent_tokens is not None
            and recent_tokens.size(1) > self.audio_repetition_window_size
        ):
            for b in range(batch_size):
                recent_window = recent_tokens[b, -self.audio_repetition_window_size :].long()
                scores = torch.gather(logits[b], dim=0, index=recent_window)
                scores = torch.where(
                    scores < 0,
                    scores * self.audio_repetition_penalty,
                    scores / self.audio_repetition_penalty,
                )
                logits[b].scatter_(dim=0, index=recent_window, src=scores)
        # Convert to probabilities with softmax
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        # Apply temperature scaling if not greedy
        if self.audio_temperature > 1e-6:
            logprobs = logprobs / self.audio_temperature
            # Apply top-k sampling
            if self.audio_top_k > 0:
                probs = torch.exp(logprobs)
                top_k_probs, top_k_indices = torch.topk(probs, self.audio_top_k, dim=-1)
                sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(1)
                next_token = top_k_indices.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)
            else:
                next_token = torch.multinomial(torch.exp(logprobs), num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(logprobs, dim=-1)
        return next_token

    def sample_text_logits(
        self, logits: torch.Tensor, recent_tokens=None
    ) -> torch.Tensor:
        """Sample from text logits with top-k, temperature and repetition penalty.

        Args:
            logits: Logits tensor of shape [batch_size, seq_len, vocab_size] or [batch_size, vocab_size]
            recent_tokens: Optional tensor of recent tokens for repetition penalty

        Returns:
            Sampled token ids
        """
        # Take the last token's logits if we have a sequence dimension
        if len(logits.shape) == 3:
            logits = logits[:, -1]
        batch_size = logits.size(0)
        logits = logits.clone()
        # Apply repetition penalty if needed
        if (
            self.text_repetition_penalty > 1.0
            and recent_tokens is not None
            and recent_tokens.size(1) > self.text_repetition_window_size
        ):
            for b in range(batch_size):
                recent_window = recent_tokens[b, -self.text_repetition_window_size :].long()
                scores = torch.gather(logits[b], dim=0, index=recent_window)
                scores = torch.where(
                    scores < 0,
                    scores * self.text_repetition_penalty,
                    scores / self.text_repetition_penalty,
                )
                logits[b].scatter_(dim=0, index=recent_window, src=scores)
        # Convert to probabilities with softmax
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        # Apply temperature scaling if not greedy
        if self.text_temperature > 1e-6:
            logprobs = logprobs / self.text_temperature
            # Apply top-k sampling
            if self.text_top_k > 0:
                probs = torch.exp(logprobs)
                top_k_probs, top_k_indices = torch.topk(probs, self.text_top_k, dim=-1)
                sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(1)
                next_token = top_k_indices.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)
            else:
                next_token = torch.multinomial(torch.exp(logprobs), num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(logprobs, dim=-1)
        return next_token


class MoonshotKimiaForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight", "mimo_output.weight"]
    config_class = KimiAudioConfig
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.model = MoonshotKimiaModel(config)
        self.speech_encoder = CustomWhisperEncoder(config.speech_encoder_config)
        self.speech_tokenizer = WhisperVQEncoder(config.speech_tokenizer_config).eval()
        self.speech_tokenizer._freeze_parameters()
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mimo_output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.use_whisper_feature = config.use_whisper_feature

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def create_mask_between_markers(
        self,
        sequences_batch: torch.Tensor,
        marker1_val: int = 151661, marker2_val: int = 151663
    ) -> torch.Tensor:
        """
        Creates a boolean mask for a batch of sequences.
        For each sequence, the region strictly between the positions of marker1_val
        and marker2_val (exclusive of the markers themselves) will be set to True.
        The positions of the markers themselves will be False.

        Args:
            sequences_batch (torch.Tensor): A 2D integer tensor of shape (batch_size, seq_len).
                                            It's guaranteed that marker1_val and marker2_val
                                            appear exactly once in each row.
            marker1_val (int): The value of the first marker, defaults to 1.
            marker2_val (int): The value of the second marker, defaults to 2.

        Returns:
            torch.Tensor: A boolean tensor of the same shape as sequences_batch.
        """
        batch_size, seq_len = sequences_batch.shape
        device = sequences_batch.device

        # 1. Find the column indices of marker1 and marker2 in each row.
        #    Since each marker is guaranteed to appear once per row,
        #    .nonzero(as_tuple=True)[1] will return a tensor of batch_size column indices.
        col_indices_marker1 = (sequences_batch == marker1_val).nonzero(as_tuple=True)[1]
        col_indices_marker2 = (sequences_batch == marker2_val).nonzero(as_tuple=True)[1]

        # 2. Determine the minimum and maximum column indices where the markers actually reside.
        #    These still represent the positions of marker1 and marker2 themselves.
        start_cols = torch.minimum(col_indices_marker1, col_indices_marker2)  # (batch_size,)
        end_cols = torch.maximum(col_indices_marker1, col_indices_marker2)    # (batch_size,)

        # 3. Expand the shapes of the marker position indices to (batch_size, 1) for broadcasting.
        start_cols_exp = start_cols.unsqueeze(1)  # Shape: (batch_size, 1)
        end_cols_exp = end_cols.unsqueeze(1)      # Shape: (batch_size, 1)

        # 4. Create a tensor representing all column indices for each row.
        #    Shape: (batch_size, seq_len)
        arange_cols = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # 5. Generate the mask.
        #    The conditions are now: column_index > min_marker_column_index AND column_index < max_marker_column_index
        mask_ge_start = (arange_cols > start_cols_exp)
        mask_le_end = (arange_cols < end_cols_exp)
        final_mask = mask_ge_start & mask_le_end
        return final_mask

    def prepare_audio_input_embs(
        self,
        audio_input_ids: torch.LongTensor = None,
        audio_input_embs: torch.FloatTensor = None,
        whisper_input_features: torch.FloatTensor = None,
        whisper_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # NOTE(xcsong): extract `speech_encoder_embs` on-the-fly
        #   ref: https://github.com/MoonshotAI/Kimi-Audio/blob/master/kimia_infer/api/prompt_manager.py#L62-L67
        speech_encoder_embs = self.speech_encoder(
            input_features=whisper_input_features,
            attention_mask=whisper_attention_mask,
            return_dict=True,
        ).last_hidden_state
        speech_encoder_embs = speech_encoder_embs.reshape(
            speech_encoder_embs.shape[0],
            int(speech_encoder_embs.shape[1] // 4),
            speech_encoder_embs.shape[2] * 4,
        )
        speech_encoder_embs = self.model.vq_adaptor(speech_encoder_embs)

        # NOTE(xcsong): extract `speech_tokenizer_ids` on-the-fly
        #   ref: https://github.com/MoonshotAI/Kimi-Audio/blob/master/kimia_infer/api/prompt_manager.py#L46-L47
        with torch.no_grad():
            speech_tokenizer_ids = self.speech_tokenizer(
                input_features=whisper_input_features,
                attention_mask=whisper_attention_mask,
                return_dict=True,
            )
            speech_tokenizer_ids += self.config.kimia_token_offset
        speech_tokenizer_embs = self.get_input_embeddings()(speech_tokenizer_ids.clone())

        # NOTE(xcsong): merge continuous audio and discrete audio
        speech_embs = (speech_encoder_embs + speech_tokenizer_embs) * math.sqrt(2.0)
        speech_embs = speech_embs.to(audio_input_embs.dtype)
        special_mask = self.create_mask_between_markers(
            sequences_batch=audio_input_ids,
            marker1_val=self.model.kimia_media_begin,
            marker2_val=self.model.kimia_media_end
        ).to(speech_embs.device)
        special_mask = special_mask.unsqueeze(-1).expand_as(audio_input_embs)
        result = torch.stack([
            audio_input_embs[i].masked_scatter(special_mask[i], speech_embs[i])
            for i in range(audio_input_embs.shape[0])
        ], dim=0)
        return result

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class="KimiAudioConfig")
    def forward(
        self,
        text_input_ids: torch.LongTensor = None,
        audio_input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        whisper_input_features: Optional[torch.FloatTensor] = None,
        whisper_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Cache]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:
        """

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        audio_input_embs = self.get_input_embeddings()(audio_input_ids)
        if self.use_whisper_feature and whisper_input_features is not None:
            audio_input_embs = self.prepare_audio_input_embs(
                audio_input_ids=audio_input_ids,
                audio_input_embs=audio_input_embs,
                whisper_input_features=whisper_input_features,
                whisper_attention_mask=whisper_attention_mask,
            )

        # NOTE(xcsong): merge `text` with `audio`, get final `input_embeds`
        if text_input_ids is not None and text_input_ids.sum() != 0:
            text_input_embs = self.get_input_embeddings()(text_input_ids)
            inputs_embeds = audio_input_embs + text_input_embs
        else:
            inputs_embeds = audio_input_embs

        outputs = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        if not return_dict:
            hidden_states, mimo_hidden_states = (
                outputs[0],
                outputs[1],
            )
        else:
            hidden_states, mimo_hidden_states = (
                outputs.last_hidden_state[0],
                outputs.last_hidden_state[1],
            )

        text_logits = self.lm_head(hidden_states)
        audio_logits = self.mimo_output(mimo_hidden_states)

        if not return_dict:
            output = (text_logits, audio_logits,) + outputs[2:]
            return output

        return CausalLMOutputWithPast(
            loss=None,
            logits=(text_logits, audio_logits),
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.inference_mode()
    def generate(
        self,
        text_input_ids: torch.LongTensor = None,
        audio_input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        whisper_input_features: Optional[torch.FloatTensor] = None,
        whisper_attention_mask: Optional[torch.Tensor] = None,
        audio_temperature=0.8,
        audio_top_k=10,
        text_temperature=0.0,
        text_top_k=5,
        audio_repetition_penalty=1.0,
        audio_repetition_window_size=64,
        text_repetition_penalty=1.1,
        text_repetition_window_size=16,
        max_new_tokens=-1,
        kimia_text_audiodelaytokens=6,
    ):
        batch_size = audio_input_ids.shape[0]
        audio_input_embs = self.get_input_embeddings()(audio_input_ids)
        if self.use_whisper_feature and whisper_input_features is not None:
            audio_input_embs = self.prepare_audio_input_embs(
                audio_input_ids=audio_input_ids,
                audio_input_embs=audio_input_embs,
                whisper_input_features=whisper_input_features,
                whisper_attention_mask=whisper_attention_mask,
            )

        # NOTE(xcsong): merge `text` with `audio`, get final `input_embeds`
        if text_input_ids is not None and text_input_ids.sum() != 0:
            text_input_embs = self.get_input_embeddings()(text_input_ids)
            inputs_embeds = audio_input_embs + text_input_embs
        else:
            inputs_embeds = audio_input_embs

        if max_new_tokens == -1:
            max_new_tokens = 7500 - audio_input_ids.shape[1]

        sampler = KimiASampler(
            audio_top_k=audio_top_k,
            audio_temperature=audio_temperature,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
            kimia_text_audiodelaytokens=kimia_text_audiodelaytokens,
        )

        generated_wav_tokens, generated_text_tokens = self._generate_loop(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            sampler=sampler,
            batch_size=batch_size,
        )

        generated_wav_tokens = [
            [t - self.config.kimia_token_offset for t in seq if t >= self.config.kimia_token_offset]
            for seq in generated_wav_tokens
        ]

        generated_text_tokens = [
            [t for t in seq if t < self.config.kimia_token_offset]
            for seq in generated_text_tokens
        ]
        return generated_wav_tokens, generated_text_tokens

    @torch.inference_mode()
    def _generate_loop(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int = 50,
        sampler: KimiASampler = None,
        batch_size: int = 1,
        output_type: str = "text",
    ):
        device = inputs_embeds.device
        previous_audio_tokens = torch.zeros((batch_size, max_new_tokens), dtype=torch.int, device=device)
        text_previous_tokens = torch.zeros((batch_size, max_new_tokens), dtype=torch.int, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        past_key_values = None
        generated_wav_tokens = [[] for _ in range(batch_size)]
        generated_text_tokens = [[] for _ in range(batch_size)]

        for i in range(max_new_tokens):
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            text_logits = self.lm_head(outputs.last_hidden_state[0])
            audio_logits = self.mimo_output(outputs.last_hidden_state[1])

            next_token_text = sampler.sample_text_logits(
                text_logits, recent_tokens=text_previous_tokens[:, :i] if i > 0 else None
            )
            next_audio_token = sampler.sample_audio_logits(
                audio_logits, recent_tokens=previous_audio_tokens[:, :i] if i > 0 else None
            )

            for b in range(batch_size):
                if finished[b]:
                    next_token_text[b] = sampler.kimia_text_blank
                elif next_token_text[b].item() == sampler.kimia_text_eos:
                    finished[b] = True
                text_previous_tokens[b, i] = next_token_text[b]

                if i < sampler.kimia_text_audiodelaytokens:
                    next_audio_token[b] = sampler.kimia_text_blank
                else:
                    if output_type == "text":
                        next_audio_token[b] = sampler.kimia_text_blank
                previous_audio_tokens[b, i] = next_audio_token[b]

                if next_token_text[b].item() != sampler.kimia_text_blank and not finished[b]:
                    generated_text_tokens[b].append(next_token_text[b].item())
                if i >= sampler.kimia_text_audiodelaytokens:
                    if next_audio_token[b].item() != sampler.kimia_text_blank:
                        generated_wav_tokens[b].append(next_audio_token[b].item())

            if finished.all():
                break

            next_audio_emb = self.get_input_embeddings()(next_audio_token.unsqueeze(1))
            next_text_emb = self.get_input_embeddings()(next_token_text.unsqueeze(1))
            inputs_embeds = next_audio_emb + next_text_emb

        return generated_wav_tokens, generated_text_tokens
