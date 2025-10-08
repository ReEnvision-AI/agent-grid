#
# Copyright (c) 2025 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind import get_logger

from agentgrid.models.nemotron.block_config import AttentionConfig
from agentgrid.models.nemotron.configuration_decilm import DeciLMConfig
from agentgrid.models.nemotron.modeling_decilm import (
    DeciLMAttention,
    DeciLMDecoderLayer,
    DeciLMFlashAttention2,
    DeciLMLinearMLP,
    DeciLMMLP,
    DeciLMRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)
from agentgrid.models.nemotron.transformers_4_44_2__modeling_flash_attention_utils_backward_compat import (
    _flash_attention_forward,
)
from agentgrid.utils.cuda_graphs import make_inference_graphed_callable

try:
    from agentgrid.models.nemotron.kernels import apply_rotary_pos_emb_fused
    HAS_FUSED_ROPE_KERNEL = True
except ImportError:
    HAS_FUSED_ROPE_KERNEL = False

logger = get_logger(__name__)

class BaseNemotronAttention:
    ...


class OptimizedNemotronAttention(DeciLMAttention, BaseNemotronAttention):
    def __init__(self, config: DeciLMConfig, attention_config: AttentionConfig, layer_idx: Optional[int] = None):
        super().__init__(config, attention_config, layer_idx)
        self._rotary_graph = None
        self.num_attention_heads = attention_config.n_heads_in_group
        #self.rotary_emb = DeciLMRotaryEmbedding(config=self.config)
    
    def _optimized_apply_rotary(self, query_states, key_states, cos, sin):
        if self._rotary_graph is None:
            self._rotary_graph = make_inference_graphed_callable(
                apply_rotary_pos_emb, sample_args=(query_states, key_states, cos, sin)
            )
        return self._rotary_graph(query_states, key_states, cos, sin)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool= False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        if position_ids is None:
            past_seen_tokens = past_key_value[0].shape[2] if past_key_value is not None else 0
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            cos, sin = self.rotary_emb(value_states, position_ids)
            #cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)

        if q_len == 1 and torch.is_inference_mode_enabled() and hidden_states.device.type == "cuda":
            if HAS_FUSED_ROPE_KERNEL:
                query_states, key_states = apply_rotary_pos_emb_fused(query_states, key_states, cos, sin)
            else:
                query_states, key_states = self._optimized_apply_rotary(query_states, key_states, cos, sin)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
class OptimizedNemotronFlashAttention2(DeciLMFlashAttention2, BaseNemotronAttention):
    def __init__(self, config: DeciLMConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._rotary_graph = None
        self.num_attention_heads = config.num_attention_heads
    
    def _optimized_apply_rotary(self, query_states, key_states, cos, sin):
        if self._rotary_graph is None:
            self._rotary_graph = make_inference_graphed_callable(
                apply_rotary_pos_emb, sample_args=(query_states, key_states, cos, sin)
            )
        return self._rotary_graph(query_states, key_states, cos, sin)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  
    ):
        output_attentions = False

        if position_ids is None:
            past_seen_tokens = past_key_value[0].shape[2] if past_key_value is not None else 0
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            cos, sin = self.rotary_emb(value_states, position_ids)
            #cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)

        if q_len == 1 and torch.is_inference_mode_enabled() and hidden_states.device.type == "cuda":
            if HAS_FUSED_ROPE_KERNEL:
                query_states, key_states = apply_rotary_pos_emb_fused(query_states, key_states, cos, sin)
            else:
                query_states, key_states = self._optimized_apply_rotary(query_states, key_states, cos, sin)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # reuse k, v, self_attention
            #logger.error(f"{past_key_value[0].size()=}, {past_key_value[1].size()=}")
            #logger.error(f"{key_states.size()=}, {value_states.size()=}")
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


DECILM_ATTENTION_CLASSES = {
    "eager": OptimizedNemotronAttention,
    "flash_attention_2": OptimizedNemotronFlashAttention2,
}

class OptimizedNemotronDecoderLayer(DeciLMDecoderLayer):
    def __init__(self, config: DeciLMConfig, layer_idx):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.block_config = config.block_configs[layer_idx]
        self.attention_config = self.block_config.attention
        self.ffn_config = self.block_config.ffn

        if not self.attention_config.no_op:
            self.input_layernorm = DeciLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if not self.attention_config.replace_with_linear:
                self.self_attn = DECILM_ATTENTION_CLASSES[config._attn_implementation or 'eager'](
                    config=config, attention_config=self.attention_config, layer_idx=layer_idx)
            else:
                self.self_attn = OptimizedNemotronAttention(config)

        if not self.ffn_config.no_op:
            self.post_attention_layernorm = DeciLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if not self.ffn_config.replace_with_linear:
                self.mlp = DeciLMMLP(config, self.ffn_config)
            else:
                self.mlp = DeciLMLinearMLP(config)

        self.is_sliding = self.attention_config.is_sliding
        self.sliding_window = self.attention_config.prefill_sliding_window

        self.pre_attn_graph = None
        self.post_attn_graph = None

    def _optimized_input_layernorm(self, hidden_states):
        if self.pre_attn_graph is None:
            self.pre_attn_graph = make_inference_graphed_callable(
                self.input_layernorm.forward, sample_args=(hidden_states,)
            )
        return self.pre_attn_graph(hidden_states)

    def _optimized_output_layernorm(self, hidden_states):
        if self.post_attn_graph is None:
            self.post_attn_graph = make_inference_graphed_callable(
                self.post_attention_layernorm.forward, sample_args=(hidden_states,)
            )
        return self.post_attn_graph(hidden_states)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        if self.attention_config.unshifted_sink and self.attention_config.is_sink:
            attention_mask = self._unshifted_sink_mask(
                attention_mask, hidden_states,
                self.attention_config.window_length, self.attention_config.num_sink_tokens)
        else:
            attention_mask = self._gemma2_window_mask(attention_mask, hidden_states, past_key_value)

        self_attn_weights = None
        present_key_value = past_key_value
        if self.attention_config.no_op:
            pass
        elif self.attention_config.replace_with_linear:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(hidden_states)
            hidden_states = residual + hidden_states
        else:
            residual = hidden_states

            if hidden_states.size(1) == 1 and torch.is_inference_mode_enabled() and hidden_states.device.type == "cuda":
                hidden_states = self._optimized_input_layernorm(hidden_states)
            else:
                hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + hidden_states

        if not self.ffn_config.no_op:
            residual = hidden_states
            if hidden_states.size(1) == 1 and torch.is_inference_mode_enabled() and hidden_states.device.type == "cuda":
                hidden_states = self._optimized_output_layernorm(hidden_states)
            else:
                hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
class WrappedNemotronBlock(OptimizedNemotronDecoderLayer):
    def __init__(self, config: DeciLMConfig, layer_idx):
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        past_key_value = layer_past

        outputs = super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        return outputs
