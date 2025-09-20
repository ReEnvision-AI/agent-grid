from __future__ import annotations

from collections import Counter, deque
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple, TypeAlias, Union

import os
import torch
from hivemind import BatchTensorDescriptor, TensorDescriptor
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.utils import get_logger
from tensor_parallel import TensorParallel
from tensor_parallel.per_device_tensors import PerDeviceTensors
from transformers import PretrainedConfig
from transformers.cache_utils import Cache, CacheLayerMixin

import time

from agentgrid.data_structures import InferenceMetadata
from agentgrid.server.memory_cache import MemoryCache
from agentgrid.server.task_pool import PrioritizedTaskPool
from agentgrid.utils.misc import get_size_in_bytes, is_dummy

logger = get_logger(__name__)

ExpertUID: TypeAlias = str

# Check for the fused Triton kernel once at module load time for efficiency.
# This avoids repeated import attempts in the hot path of inference.
try:
    from agentgrid.server.kernels import update_cache_fused, reorder_cache_fused

    HAS_FUSED_UPDATE_KERNEL = True
    HAS_FUSED_REORDER_KERNEL = True
    logger.info("Triton fused kernels for cache update and reorder are available and will be used.")
except ImportError:
    HAS_FUSED_UPDATE_KERNEL = False
    HAS_FUSED_REORDER_KERNEL = False
    logger.warning(
        "Triton fused kernels not available. "
        "Falling back to slower python-level implementation. "
        "For performance, install Triton (`pip install triton`)."
    )

class AgentGridLayer(CacheLayerMixin):
    def __init__(self, key_shards: Sequence[torch.Tensor] = None, value_shards: Sequence[torch.Tensor] = None):
        self.key_shards = key_shards
        self.value_shards = value_shards

    @property
    def keys(self) -> PerDeviceTensors:
        return PerDeviceTensors(*self.key_shards)

    @property
    def values(self) -> PerDeviceTensors:
        return PerDeviceTensors(*self.value_shards)

    def update(self, key_states: PerDeviceTensors, value_states: PerDeviceTensors, cache_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[PerDeviceTensors, PerDeviceTensors]:
        prefix_length = (cache_kwargs or {}).get("prefix_length", 0)
        new_length = key_states.shape[2]

        key_state_shards = getattr(key_states, "tensor_shards", [key_states])
        value_state_shards = getattr(value_states, "tensor_shards", [value_states])

        for i in range(len(self.key_shards)):
            key_cache_shard = self.key_shards[i]
            value_cache_shard = self.value_shards[i]
            key_state_shard = key_state_shards[i]
            value_state_shard = value_state_shards[i]

            if prefix_length < new_length:
                update_slice = slice(prefix_length, new_length)
                key_cache_shard[:, :, update_slice, :] = key_state_shard[:, :, update_slice, :]
                value_cache_shard[:, :, update_slice, :] = value_state_shard[:, :, update_slice, :]
        return self.keys, self.values

    def get_seq_length(self, cache_position=None) -> int:
        return self.keys.shape[2]

    def get_max_cache_shape(self) -> int:
        return self.keys.shape[2]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        if HAS_FUSED_REORDER_KERNEL:
            shards_by_device = {}
            for key_shard, value_shard in zip(self.key_shards, self.value_shards):
                device = key_shard.device
                if device not in shards_by_device:
                    shards_by_device[device] = []
                shards_by_device[device].append((key_shard, value_shard))

            for device, shards in shards_by_device.items():
                device_beam_idx = beam_idx.to(device)
                for key_cache_shard, value_cache_shard in shards:
                    reorder_cache_fused(key_cache_shard, value_cache_shard, device_beam_idx)
        else:
            for key_shard, value_shard in zip(self.key_shards, self.value_shards):
                key_shard[...] = key_shard[beam_idx.to(key_shard.device)]
                value_shard[...] = value_shard[beam_idx.to(value_shard.device)]

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return self.get_seq_length(), 0

class AgentGridCache(Cache):
    def __init__(self, tensors: Sequence[torch.Tensor], max_length: int, num_shards: int):
        super().__init__(layer_classes=AgentGridLayer)
        key_shards = list(tensors[0::2])
        value_shards = list(tensors[1::2])
        self.layers = [AgentGridLayer(key_shards, value_shards)]
        self.max_length = max_length
        self.num_shards = num_shards

    def select_layer_past(self, prefix_length: int) -> Sequence[torch.Tensor]:
        layer = self.layers[0]
        key_shards = [k[:, :, :prefix_length, :] for k in layer.key_shards]
        value_shards = [v[:, :, :prefix_length, :] for v in layer.value_shards]
        layer_past = tuple(chain(*zip(key_shards, value_shards)))
        return PerDeviceTensors(*layer_past) if self.num_shards > 1 else layer_past

    def update(self, key_states: PerDeviceTensors, value_states: PerDeviceTensors, layer_idx: int, cache_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[PerDeviceTensors, PerDeviceTensors]:
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)



class TransformerBackend(ModuleBackend):
    """A wrapper for a transformer block that can process requests for forward, backward and inference"""

    _peft_module = None

    def __init__(
        self,
        *args,
        config: PretrainedConfig,
        memory_cache: MemoryCache,
        backend_dtype: torch.dtype,
        max_chunk_size_bytes: int,
        **kwargs,
    ):
        import agentgrid.utils.peft as _peft_module

        self._peft_module = _peft_module

        super().__init__(*args, **kwargs)
        assert isinstance(self.module, TensorParallel)
        self.config = config
        self.memory_cache = memory_cache
        self.max_chunk_size_bytes = max_chunk_size_bytes

        for name, param in self.module.named_parameters():
            assert not param.requires_grad, f"Block parameters must not accumulate gradients, but {name} does"
        for name, buf in self.module.named_buffers():
            assert not buf.requires_grad, f"Block parameters must not accumulate gradients, but {name} does"

        max_batch_size = self.forward_pool.max_batch_size
        device = self.module.devices[self.module.output_device_index]
        self.inference_pool = PrioritizedTaskPool(
            self.inference_step, max_batch_size=max_batch_size, device=device, name=f"{self.name}_inference"
        )  # note: inference_pools may be merged later, see merge_inference_pools_inplace
        self.forward_pool = PrioritizedTaskPool(
            self.forward, max_batch_size=max_batch_size, device=device, name=f"{self.name}_forward"
        )
        

        self.dtype = backend_dtype
        self.dtype_bytes = get_size_in_bytes(self.dtype)
        self.shard_num_heads = []
        self.shard_num_kv_heads = []
        self.shard_head_dims = []

        profile_env = os.getenv("AGENTGRID_PROFILE_INFERENCE", "0").lower()
        self.enable_profiling = profile_env not in {"0", "false", "no", ""}
        self._profile_samples: deque[Tuple[float, int]] = deque(maxlen=int(os.getenv("AGENTGRID_PROFILE_WINDOW", "100")))
        self._profile_last_log = time.perf_counter()
        self._profile_log_interval = float(os.getenv("AGENTGRID_PROFILE_LOG_INTERVAL", "30"))
        for shard in self.module.module_shards:
            num_heads_on_shard = 0
            num_kv_heads_on_shard = 0
            head_dims_on_shard = []
            for submodule in shard.modules():
                if isinstance(submodule, config.attn_class):
                    attn_heads = getattr(submodule, "num_attention_heads", config.num_attention_heads)
                    num_heads_on_shard += attn_heads
                    kv_heads = getattr(submodule, "num_key_value_heads", None)
                    if kv_heads is None:
                        kv_heads = getattr(config, "num_key_value_heads", attn_heads)
                    num_kv_heads_on_shard += kv_heads
                    if hasattr(submodule, "head_dim"):
                        head_dims_on_shard.append(submodule.head_dim)

            self.shard_num_heads.append(num_heads_on_shard)
            self.shard_num_kv_heads.append(num_kv_heads_on_shard)
            if head_dims_on_shard:
                assert all(d == head_dims_on_shard[0] for d in head_dims_on_shard)
                self.shard_head_dims.append(head_dims_on_shard[0])
            else:
                self.shard_head_dims.append(0)

        assert len(self.shard_num_heads) == len(self.module.devices)
        assert len(self.shard_head_dims) == len(self.module.devices)

        self.overall_head_dim = next((dim for dim in self.shard_head_dims if dim > 0), 0)
        self._warmup_inference()

        self.inference_schema = (
            (
                *self.args_schema,
                BatchTensorDescriptor((), dtype=self.dtype),  # Inputs
                BatchTensorDescriptor((), dtype=self.dtype),  # Prompts
                BatchTensorDescriptor((), dtype=self.dtype),  # Attention Mask
                BatchTensorDescriptor((), dtype=self.dtype),  # Position IDs
                (
                    BatchTensorDescriptor((), dtype=self.dtype),
                    BatchTensorDescriptor((), dtype=self.dtype),
                ),  # Position Embeddings
            ),
            self.kwargs_schema,
        )

        self.cache_bytes_per_token: Dict[torch.device, int] = Counter()
        for descr in self.get_inference_cache_descriptors(batch_size=1, max_length=1):
            self.cache_bytes_per_token[descr.device] += descr.numel() * get_size_in_bytes(descr.dtype)

    def get_inference_cache_descriptors(self, batch_size: int, max_length: int) -> Sequence[TensorDescriptor]:
        """Create tensor descriptors for attention cache tensors used during inference_step"""
        head_dim = self.overall_head_dim

        cache_tensors = []
        for device_idx, device in enumerate(self.module.devices):
            num_heads = self.shard_num_kv_heads[device_idx]
            keys = TensorDescriptor((batch_size, num_heads, max_length, head_dim), dtype=self.dtype, device=device)
            values = TensorDescriptor((batch_size, num_heads, max_length, head_dim), dtype=self.dtype, device=device)
            cache_tensors.extend((keys, values))
        return cache_tensors

    def forward(self, *inputs: Union[torch.Tensor, str]) -> Tuple[torch.Tensor, ...]:
        *inputs, active_adapter = inputs
        with self._peft_module.using_adapter(active_adapter):
            return super().forward(*inputs)

    

    @torch.inference_mode()
    def inference_step(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        inference_info: InferenceMetadata,
    ) -> Tuple[torch.Tensor, ...]:
        assert hidden_states.ndim == 3, "expected hidden states to be 3-dimensional: [batch_size, seq_len, hid_size]"

        assert isinstance(position_embeddings, tuple), "expected position_embeddings to be a tuple"
        seq_len = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]

        profile_start = time.perf_counter() if self.enable_profiling else 0.0

        with self.memory_cache.use_cache(
            *inference_info.cache_handles
        ) as cache_tensors, self._peft_module.using_adapter(inference_info.active_adapter):
            max_length = cache_tensors[0].shape[2]  # Get max_length from the cache tensor shape
            cache = AgentGridCache(cache_tensors, max_length, len(self.module.module_shards))

            max_chunk_length = self._estimate_max_chunk_length(hidden_states, inference_info)
            layer_past = cache.select_layer_past(inference_info.prefix_length)
            new_kvs = None

            if seq_len <= max_chunk_length:
                output_hidden_states, new_kvs = self.module.forward(
                    hidden_states,
                    layer_past=layer_past,
                    use_cache=True,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
            else:
                output_hidden_states = torch.empty_like(hidden_states)
                for offset in range(0, seq_len, max_chunk_length):
                    hidden_states_chunk = hidden_states[:, offset : offset + max_chunk_length, :]
                    output_hidden_states_chunk, new_kvs = self.module.forward(
                        hidden_states_chunk,
                        layer_past=layer_past,
                        use_cache=True,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )
                    output_hidden_states[:, offset : offset + max_chunk_length] = output_hidden_states_chunk
                    layer_past = new_kvs

            if new_kvs is not None:
                key_states, value_states = new_kvs
                cache.update(key_states, value_states, 0, cache_kwargs={"prefix_length": inference_info.prefix_length})

            if self.enable_profiling:
                duration = time.perf_counter() - profile_start
                processed_tokens = batch_size * seq_len
                self._profile_samples.append((duration, processed_tokens))
                now = time.perf_counter()
                if now - self._profile_last_log >= self._profile_log_interval and self._profile_samples:
                    total_time = sum(sample[0] for sample in self._profile_samples)
                    total_tokens = sum(sample[1] for sample in self._profile_samples)
                    avg_latency = total_time / len(self._profile_samples)
                    avg_batch_tokens = total_tokens / len(self._profile_samples)
                    throughput = total_tokens / total_time if total_time else float("inf")
                    logger.info(
                        "[profiling] backend=%s samples=%d avg_latency=%.3fs avg_tokens=%.1f tokens_per_sec=%.1f",
                        self.name,
                        len(self._profile_samples),
                        avg_latency,
                        avg_batch_tokens,
                        throughput,
                    )
                    self._profile_last_log = now

            return (output_hidden_states,)

    def _estimate_max_chunk_length(self, hidden_states: torch.Tensor, inference_info: InferenceMetadata) -> int:
        # We assume that attention logit matrices are the main thing that consumes memory
        batch_size, seq_length, hidden_size = hidden_states.shape
        worst_case_length = inference_info.prefix_length + seq_length
        attn_bytes_per_token = max(self.shard_num_heads) * batch_size * self.dtype_bytes * worst_case_length
        if attn_bytes_per_token == 0:
            return 1
        return max(1, self.max_chunk_size_bytes // attn_bytes_per_token)

    def _warmup_inference(self) -> None:
        warmup_tokens = int(os.getenv("AGENTGRID_WARMUP_TOKENS", "16"))
        if warmup_tokens <= 0:
            return
        try:
            device = self.module.devices[self.module.output_device_index]
            dtype = self.dtype
            hidden_size = self.config.hidden_size
            seq_len = min(warmup_tokens, getattr(self.config, "max_position_embeddings", warmup_tokens))
            hidden_states = torch.zeros((1, seq_len, hidden_size), dtype=dtype, device=device)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            for block in self.module.module_shards:
                kwargs = {
                    "attention_mask": None,
                    "position_ids": position_ids,
                    "layer_past": None,
                    "use_cache": False,
                }
                if hasattr(block, "rotary_emb"):
                    position_embeddings = block.rotary_emb(hidden_states, position_ids)
                    kwargs["position_embeddings"] = position_embeddings
                try:
                    outputs = block(hidden_states, **kwargs)
                except TypeError:
                    outputs = block(hidden_states)
                if isinstance(outputs, tuple):
                    hidden_states = outputs[0]
                else:
                    hidden_states = outputs

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            logger.info("Warmup inference completed for %s with %d tokens", self.name, seq_len)
        except Exception as exc:  # pragma: no cover
            logger.debug("Warmup inference skipped for %s: %s", self.name, exc)

    def _reorder_cache_inplace(self, cache_tensors: torch.Tensor, hypo_ids: torch.Tensor):
        """If hypo_ids is specified, reorder elements of each cache tensor in-place by taking indices from hypo_ids"""
        if not is_dummy(hypo_ids):
            for cache_tensor in cache_tensors:
                cache_tensor[...] = cache_tensor[hypo_ids.to(cache_tensor.device)]  # in-place reorder cache by hypo ids

    def get_pools(self) -> Sequence[PrioritizedTaskPool]:
        return self.forward_pool, self.inference_pool

    def get_info(self) -> Dict[str, Any]:
        """Get module parameters and stats. Used by RemoteExpert to check shapes and for DMoE orchestration."""
        return dict(super().get_info(), inference_schema=self.inference_schema)

    def shutdown(self):
        # Break the cyclic references, otherwise TransformerBackend may be not garbage-collected
        self.forward_pool = self.inference_pool = None

        # Explicitly free the GPU memory. This is not necessary at the time this code is written,
        # but may help to avoid future issues when the module is not garbage-collected for some reasons
        dummy = torch.tensor([])
        for p in self.module.parameters():
            p.data = dummy


def merge_inference_pools_inplace(backends: Dict[ExpertUID, TransformerBackend]):
    """Replace each backend's rpc_inference pools with a combined pool runs multiple blocks in one call"""
    assert len(backends) != 0 and all(isinstance(b, TransformerBackend) for b in backends.values())
    first_pool = next(iter(backends.values())).inference_pool
    merged_pool = PrioritizedTaskPool(
        _MergedInferenceStep(backends),
        max_batch_size=first_pool.max_batch_size,
        device=first_pool.device,
        name=f"merged_inference",
    )
    for backend in backends.values():
        assert not backend.inference_pool.is_alive()
        backend.inference_pool = merged_pool


class _MergedInferenceStep:
    def __init__(self, backends: Dict[ExpertUID, TransformerBackend]):
        self.backends = backends

    @torch.inference_mode()
    def __call__(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        inference_infos: Sequence[InferenceMetadata],
        *optional_prompts: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        assert len(inference_infos) == len(
            optional_prompts
        ), f"found {len(inference_infos)} blocks but {len(optional_prompts)} prompts"

        assert isinstance(position_embeddings, tuple), "expected position_embeddings to be a tuple"

        for inference_info, optional_prompt in zip(inference_infos, optional_prompts):
            if isinstance(optional_prompt, tuple) and optional_prompt:
                # TODO: this is a hack, find out why prompts are wrapped in a tuple
                optional_prompt = optional_prompt[0]

            if optional_prompt is not None and optional_prompt.numel() > 0:
                prompt_to_add = optional_prompt
                while prompt_to_add.ndim < hidden_states.ndim:
                    prompt_to_add = prompt_to_add.unsqueeze(0)

                prompt_len = prompt_to_add.shape[1]
                hidden_states[:, :prompt_len] += prompt_to_add

            outputs = self.backends[inference_info.uid].inference_step(
                hidden_states, attention_mask, position_ids, position_embeddings, inference_info
            )
            hidden_states = outputs[0]
        return (hidden_states,)
