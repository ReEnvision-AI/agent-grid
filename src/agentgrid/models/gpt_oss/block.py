#
# Copyright (c) 2025 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#

from typing import Optional, Tuple

import torch
from tensor_parallel.per_device_tensors import PerDeviceTensors
from transformers.cache_utils import Cache, DynamicCache
from hivemind import get_logger

logger = get_logger(__name__)
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer, GptOssRotaryEmbedding


class _NormalizedDynamicCache(DynamicCache):
    def __init__(self, normalizer):
        super().__init__()
        self._normalizer = normalizer

    @classmethod
    def from_cache(cls, cache: DynamicCache, normalizer) -> "_NormalizedDynamicCache":
        if isinstance(cache, cls):
            return cache
        normalized_cache = cls(normalizer)
        for layer_idx, layer in enumerate(cache.layers):
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is None or values is None:
                continue
            keys, values = normalizer(keys, values)
            normalized_cache.update(keys, values, layer_idx)
        return normalized_cache

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        logger.debug(
            "Cache update input: layer=%s key=%s value=%s",
            layer_idx,
            tuple(key_states.shape) if isinstance(key_states, torch.Tensor) else type(key_states),
            tuple(value_states.shape) if isinstance(value_states, torch.Tensor) else type(value_states),
        )
        key_states, value_states = self._normalizer(key_states, value_states)
        if layer_idx < len(self.layers):
            layer = self.layers[layer_idx]
            existing_keys = getattr(layer, "keys", None)
            if existing_keys is not None and existing_keys.shape[1] != key_states.shape[1]:
                logger.warning(
                    "GPT-OSS cache head mismatch before update: existing=%s new=%s",
                    tuple(existing_keys.shape),
                    tuple(key_states.shape),
                )
        try:
            result_keys, result_values = super().update(key_states, value_states, layer_idx, cache_kwargs)
        except RuntimeError as exc:
            logger.error(
                "Failed to append cache: layer_idx=%s existing=%s new=%s",
                layer_idx,
                tuple(getattr(self.layers[layer_idx], "keys", torch.empty(0)).shape)
                if layer_idx < len(self.layers) else None,
                tuple(key_states.shape),
            )
            raise
        normalized_keys, normalized_values = self._normalizer(result_keys, result_values)
        layer = self.layers[layer_idx]
        layer.keys = normalized_keys
        layer.values = normalized_values
        return normalized_keys, normalized_values


class WrappedGptOssBlock(GptOssDecoderLayer):
    def __init__(self, config: GptOssConfig, layer_idx: int):
        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = "eager"
        self._config = config
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self._rotary_emb: GptOssRotaryEmbedding | None = None

    def _prepare_attention_mask(self, attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if isinstance(attention_mask, dict):
            if self.attention_type not in attention_mask:
                raise KeyError(f"Missing attention mask for layer type '{self.attention_type}'")
            return attention_mask[self.attention_type]
        return attention_mask

    def _prepare_cache(self, layer_past: Optional[object]) -> Optional[Cache]:
        if layer_past is None:
            return None
        if isinstance(layer_past, Cache):
            if isinstance(layer_past, DynamicCache):
                return _NormalizedDynamicCache.from_cache(layer_past, self._normalize_cache_tensors)
            return layer_past
        if isinstance(layer_past, PerDeviceTensors):
            layer_past = layer_past.tensors
        if isinstance(layer_past, (tuple, list)):
            if len(layer_past) == 2:
                key_states, value_states = layer_past
            else:
                if len(layer_past) % 2 != 0:
                    raise ValueError(
                        f"Expected layer_past to contain key/value tensors, got {len(layer_past)} elements"
                    )
                key_states = torch.cat(layer_past[0::2], dim=1)
                value_states = torch.cat(layer_past[1::2], dim=1)
            cache = _NormalizedDynamicCache(self._normalize_cache_tensors)
            key_states, value_states = self._normalize_cache_tensors(key_states, value_states)
            cache.update(key_states, value_states, self.layer_idx)
            return cache
        raise TypeError(f"Unsupported cache type: {type(layer_past)}")

    def _ensure_rotary_emb(self) -> GptOssRotaryEmbedding:
        if self._rotary_emb is None:
            self._rotary_emb = GptOssRotaryEmbedding(config=self._config)
        return self._rotary_emb

    def _cache_length(self, cache: Optional[Cache]) -> int:
        if cache is None or len(cache.layers) <= self.layer_idx:
            return 0
        layer_cache = cache.layers[self.layer_idx]
        if layer_cache is None or getattr(layer_cache, "keys", None) is None:
            return 0
        return layer_cache.keys.shape[-2]

    def _normalize_cache_tensors(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(key_states, torch.Tensor) or not isinstance(value_states, torch.Tensor):
            raise TypeError("Expected tensors for cache normalization")

        head_dim = getattr(self._config, "head_dim", self._config.hidden_size // self._config.num_attention_heads)
        if key_states.shape[-2] == head_dim and key_states.shape[-1] != head_dim:
            key_states = key_states.transpose(-1, -2).contiguous()
        if value_states.shape[-2] == head_dim and value_states.shape[-1] != head_dim:
            value_states = value_states.transpose(-1, -2).contiguous()

        expected_kv = getattr(self._config, "num_key_value_heads", self._config.num_attention_heads)
        if key_states.shape[1] == expected_kv:
            return key_states, value_states

        attn_heads = self._config.num_attention_heads
        if key_states.shape[1] == attn_heads:
            if attn_heads % expected_kv != 0:
                raise ValueError(
                    f"Cannot reshape cache heads: attention heads {attn_heads} not divisible by kv heads {expected_kv}"
                )
            factor = attn_heads // expected_kv
            key_states = key_states.view(
                key_states.shape[0], expected_kv, factor, key_states.shape[2], key_states.shape[3]
            )[:, :, 0, :, :]
            value_states = value_states.view(
                value_states.shape[0], expected_kv, factor, value_states.shape[2], value_states.shape[3]
            )[:, :, 0, :, :]
            return key_states.contiguous(), value_states.contiguous()

        raise ValueError(
            "Unsupported cache head shape:"
            f" got {key_states.shape[1]}, expected {expected_kv} or {attn_heads}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attention_mask = self._prepare_attention_mask(attention_mask)
        cache = self._prepare_cache(layer_past)
        if use_cache and cache is None:
            cache = _NormalizedDynamicCache(self._normalize_cache_tensors)

        past_length = self._cache_length(cache)

        device = hidden_states.device
        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + hidden_states.shape[1], device=device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if position_embeddings is None:
            rotary_emb = self._ensure_rotary_emb()
            position_embeddings = rotary_emb(hidden_states, position_ids)

        if attention_mask is not None and attention_mask.device != device:
            attention_mask = attention_mask.to(device)
        if position_ids.device != device:
            position_ids = position_ids.to(device)
        if isinstance(position_embeddings, tuple):
            position_embeddings = tuple(t.to(device) for t in position_embeddings)

        hidden_states = super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        outputs: Tuple[torch.Tensor, ...]
        outputs = (hidden_states,)
        if use_cache:
            assert cache is not None
            if len(cache.layers) <= self.layer_idx:
                raise RuntimeError("Cache did not record states for the current layer")
            layer_cache = cache.layers[self.layer_idx]
            if layer_cache is None or layer_cache.keys is None or layer_cache.values is None:
                present = None
            else:
                key_states, value_states = self._normalize_cache_tensors(
                    layer_cache.keys, layer_cache.values
                )
                present = (key_states, value_states)
            outputs += (present,)
        return outputs
