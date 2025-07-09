# -*- coding: utf-8 -*-
# Copyright (c) 2025, Xingchen Song(sxc19@tsinghua.org.cn)

from typing import Optional, Tuple, Union

import torch
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.qwen2_audio import (
    Qwen2AudioConfig, Qwen2AudioForConditionalGeneration)
from transformers.models.qwen2_audio.modeling_qwen2_audio import (
    Qwen2AudioCausalLMOutputWithPast, Qwen2AudioEncoder)

from touchnet.bin import TrainConfig


def forward_audio_tower(
    self,
    input_features,
    attention_mask=None,
    head_mask=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    **kwargs,
):
    r"""
    NOTE(xcsong): This function is a modified version of the `forward` function in `Qwen2AudioEncoder.forward`.
                  The only difference is that the `forward` function in `Qwen2AudioEncoder.forward`
                  assumes the input features are padded to the fixed sequence length (30s), but in our case, the input features might be much longer than 30s (e.g. 1 hour).
                  So we add a slice&repeat operation to the position embedding to make it compatible with variable length input features.

    For input/output arguments, please refer to the `forward` function in `Qwen2AudioEncoder.forward`.
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Ignore copy
    input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

    inputs_embeds = torch.nn.functional.gelu(self.conv1(input_features))
    inputs_embeds = torch.nn.functional.gelu(self.conv2(inputs_embeds))

    inputs_embeds = inputs_embeds.permute(0, 2, 1)
    embed_pos = self.embed_positions.weight

    # NOTE(xcsong): Slice the position embedding to make it compatible with variable length input features
    #   This is the only difference from the original forward function
    seq_len = inputs_embeds.shape[1]
    embed_pos_len = embed_pos.shape[0]
    if embed_pos_len >= seq_len:
        # embed_pos is longer or equal, slice it
        pos_embeds = embed_pos[None, :seq_len, :]
    else:
        # embed_pos is shorter, repeat it to match seq_len
        repeat_times = seq_len // embed_pos_len
        remainder = seq_len % embed_pos_len
        if remainder == 0:
            # Perfect division, just repeat
            pos_embeds = embed_pos.repeat(repeat_times, 1)[None, :, :]
        else:
            # Need to handle remainder
            repeated_pos = embed_pos.repeat(repeat_times, 1)
            # Add the remaining part
            remaining_pos = embed_pos[:remainder, :]
            pos_embeds = torch.cat([repeated_pos, remaining_pos], dim=0)[None, :, :]

    hidden_states = inputs_embeds + pos_embeds
    hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    # check if head_mask has a correct number of layers specified if desired
    if head_mask is not None:
        assert head_mask.size()[0] == (len(self.layers)), (
            f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        )

    for idx, encoder_layer in enumerate(self.layers):
        if output_hidden_states and encoder_states is not None:
            encoder_states = encoder_states + (hidden_states,)
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        to_drop = False
        if self.training:
            dropout_probability = torch.rand([])
            if dropout_probability < self.layerdrop:  # skip the layer
                to_drop = True

        # Ignore copy
        if to_drop:
            layer_outputs = (None, None)
        else:
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    (head_mask[idx] if head_mask is not None else None),
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

        if output_attentions and all_attentions is not None:
            all_attentions = all_attentions + (layer_outputs[1],)

    # Ignore copy
    hidden_states = hidden_states.permute(0, 2, 1)
    hidden_states = self.avg_pooler(hidden_states)
    hidden_states = hidden_states.permute(0, 2, 1)

    hidden_states = self.layer_norm(hidden_states)
    if output_hidden_states and encoder_states is not None:
        encoder_states = encoder_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
    return BaseModelOutput(
        last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
    )


def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    input_features: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    feature_attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, Qwen2AudioCausalLMOutputWithPast]:
    r"""
    NOTE (xcsong): The liger kernel requires `shift_labels` to calculate loss,
                   and Qwen2AudioForConditionalGeneration forward function has no `shift_labels` argument
                   So add **kwargs to `forward` to pass it, which is the first difference from
                   the standard Qwen2AudioForConditionalGeneration forward function.

                   The second difference is that we use a modified version of the `forward` function for `Qwen2AudioEncoder`
                   to support variable length input features (see `forward_audio_tower`).

                   The third difference is that we force the `is_causal` attribute of the self-attention layer in `audio_tower` and `language_model` to be True,
                   and force the `attention_implementation` attribute of `audio_tower` and `language_model` to be "sdpa". This is to make both
                   `audio_tower` and `language_model` to be streamable and memory efficient.

    For input/output arguments, please refer to original `Qwen2AudioForConditionalGeneration.forward`.
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    target_device = self.audio_tower.device

    if input_features is not None:
        input_features = input_features.to(target_device)
        feature_attention_mask = feature_attention_mask.to(target_device)

    if inputs_embeds is None:
        # 1. Extract the input embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text and audios
        if input_features is not None and input_ids is not None and input_ids.shape[1] != 1:
            audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )

            assert self.config._attn_implementation == "sdpa", "sdpa is required"
            for layer in self.audio_tower.layers:
                layer.self_attn.is_causal = True
            audio_outputs = self.audio_tower(input_features)
            selected_audio_feature = audio_outputs.last_hidden_state
            audio_features = self.multi_modal_projector(selected_audio_feature)

            # if we have consecutive audio tokens, then it means we expanded input_ids in processing
            audio_tokens = input_ids == self.config.audio_token_index
            legacy_processing = (audio_tokens[:, :-1] & audio_tokens[:, 1:]).sum() == 0
            assert not legacy_processing, "legacy_processing should be False"

            num_audios, max_audio_tokens, embed_dim = audio_features.shape
            audio_features_mask = torch.arange(max_audio_tokens, device=audio_output_lengths.device)[None, :]
            audio_features_mask = audio_features_mask < audio_output_lengths[:, None]
            audio_features = audio_features[audio_features_mask]  # [total_valid_audio_tokens, embed_dim], one-dimensional arrangement

            n_audio_tokens = (input_ids == self.config.audio_token_index).sum().item()
            n_audio_features = audio_features.shape[0]

            if n_audio_tokens != n_audio_features:
                print(f"Audio features and audio tokens mismatch detected:")
                print(f"  n_audio_tokens: {n_audio_tokens}")
                print(f"  n_audio_features: {n_audio_features}")
                print(f"  difference: {abs(n_audio_tokens - n_audio_features)}")

                if n_audio_tokens > n_audio_features:
                    missing_features = n_audio_tokens - n_audio_features
                    print(f"  Applying padding: adding {missing_features} audio features")
                    last_feature = audio_features[-1:].expand(missing_features, -1)
                    audio_features = torch.cat([audio_features, last_feature], dim=0)
                else:
                    raise ValueError(
                        f"Audio features exceed audio tokens: tokens: {n_audio_tokens}, features {n_audio_features}. "
                        f"This would result in information loss. Please check the audio processing pipeline."
                    )
            special_audio_mask = (input_ids == self.config.audio_token_index).to(inputs_embeds.device)
            special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds)  # [batch_size, seq_len, embed_dim]
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)

    if self.training or inputs_embeds.shape[0] == 1:
        # NOTE(xcsong): Training use right padding while inference use left padding, attention mask can be ignored in training but not in inference.
        assert self.config._attn_implementation == "sdpa", "sdpa is required"
        for layer in self.language_model.model.layers:
            layer.self_attn.is_causal = True
        attention_mask = None
    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        **kwargs,
    )

    return outputs


def pre_init(args: TrainConfig):
    """Pre-initialization function for Qwen2AudioForConditionalGeneration."""
    if args.training_enable_liger_kernel:
        # 1. monkey patch the forward function to Qwen2Model
        apply_liger_kernel_to_qwen2()
        # 2. monkey patch the forward function of Qwen2AudioEncoder
        Qwen2AudioEncoder.forward = forward_audio_tower
        # 3. monkey patch the forward function to Qwen2AudioForConditionalGeneration
        Qwen2AudioForConditionalGeneration.forward = forward


def post_init(model: Qwen2AudioForConditionalGeneration, init_device: torch.device):
    """Post-initialization function for Qwen2AudioForConditionalGeneration."""

    # NOTE(xcsong): Init rope and norm.weight

    # 1. whisper encoder
    model.audio_tower.layer_norm.reset_parameters()
    for layer in model.audio_tower.layers:
        layer.self_attn_layer_norm.reset_parameters()
        layer.final_layer_norm.reset_parameters()

    # 2. qwen backbone
    inv_freq, attention_scaling = model.language_model.model.rotary_emb.rope_init_fn(
        model.language_model.model.rotary_emb.config, device=init_device)
    model.language_model.model.rotary_emb.inv_freq = inv_freq
    model.language_model.model.rotary_emb.attention_scaling = attention_scaling
    model.language_model.model.rotary_emb.original_inv_freq = inv_freq
    torch.nn.init.ones_(model.language_model.model.norm.weight)
    for layer in model.language_model.model.layers:
        torch.nn.init.ones_(layer.input_layernorm.weight)
        torch.nn.init.ones_(layer.post_attention_layernorm.weight)

    # NOTE(xcsong): Do some NaN check
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            raise ValueError(f"NaN/inf in model parameters `{name}`.")


def get_num_flop_per_token(num_params: int, model_config: Qwen2AudioConfig, seq_len: int) -> int:
    # NOTE(xcsong): We do not include flops from speech_encoder
    l, h, q, t = (
        model_config.text_config.num_hidden_layers,
        model_config.text_config.num_attention_heads,
        model_config.text_config.hidden_size // model_config.text_config.num_attention_heads,
        seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * num_params + 12 * l * h * q * t

    return flop_per_token


def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        base_model_prefix = getattr(model.language_model, "base_model_prefix", "model")
        submodel = getattr(model.language_model, f"{base_model_prefix}")
        num_params -= sum(
            sum(p.numel() for p in m.parameters())
            for m in submodel.children()
            if isinstance(m, torch.nn.Embedding)
        )
    return num_params
