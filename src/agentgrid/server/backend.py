from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple, TypeAlias, Union

import torch
from hivemind import BatchTensorDescriptor, TensorDescriptor
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.utils import get_logger
from tensor_parallel import TensorParallel
from tensor_parallel.per_device_tensors import PerDeviceTensors
from transformers import PretrainedConfig
from transformers.cache_utils import Cache

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

class AgentGridCache(Cache):
    """
    A cache that uses agentgrid's MemoryCache for storing KV cache tensors.
    """

    def __init__(
        self,
        tensors: Sequence[torch.Tensor],
        max_length: int,
        num_shards: int,
    ):
        super().__init__()
        self.tensors = tensors
        self.max_length = max_length
        self.num_shards = num_shards
        self._seq_length = 0

    def select_layer_past(self, prefix_length: int) -> Sequence[torch.Tensor]:
        """Extract first {prefix_length} tokens and reshape them such that they can be used as layer_past"""
        self._seq_length = prefix_length
        key_cache, value_cache = list(self.tensors[0::2]), list(self.tensors[1::2])
        for i in range(len(key_cache)):
            key_cache[i] = key_cache[i][:, :, :prefix_length, :]
            value_cache[i] = value_cache[i][:, :, :prefix_length, :]
        layer_past = tuple(chain(*zip(key_cache, value_cache)))
        return PerDeviceTensors(*layer_past) if self.num_shards > 1 else layer_past

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states."""
        return self._seq_length

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_length

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        if HAS_FUSED_REORDER_KERNEL:
            # Fuse reordering for all shards on the same device
            shards_by_device = {}
            for i in range(self.num_shards):
                key_cache_shard = self.tensors[i * 2]
                value_cache_shard = self.tensors[i * 2 + 1]
                device = key_cache_shard.device
                if device not in shards_by_device:
                    shards_by_device[device] = []
                shards_by_device[device].append((key_cache_shard, value_cache_shard))

            for device, shards in shards_by_device.items():
                device_beam_idx = beam_idx.to(device)
                for key_cache_shard, value_cache_shard in shards:
                    reorder_cache_fused(key_cache_shard, value_cache_shard, device_beam_idx)
        else:
            # Fallback to the original implementation
            for cache_tensor in self.tensors:
                cache_tensor[...] = cache_tensor[beam_idx.to(cache_tensor.device)]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        This method is kept for API compatibility with transformers.Cache but the main update logic
        for this backend is in `update_sharded`.
        """
        # This method is not expected to be called in the current agentgrid flow,
        # but is implemented for correctness and compatibility.
        prefix_length = (cache_kwargs or {}).get("prefix_length", self._seq_length)
        self.update_sharded(
            PerDeviceTensors(key_states), PerDeviceTensors(value_states), prefix_length=prefix_length
        )
        return key_states, value_states

    def update_sharded(
        self,
        key_states: Union[PerDeviceTensors, torch.Tensor],
        value_states: Union[PerDeviceTensors, torch.Tensor],
        prefix_length: int,
    ):
        """
        Updates the sharded cache tensors in-place using the new key-value states.
        It attempts to use a fused Triton kernel for performance and falls back to
        a standard PyTorch implementation if the kernel is unavailable.
        """
        if isinstance(key_states, PerDeviceTensors):
            key_state_shards = key_states.tensor_shards
            value_state_shards = value_states.tensor_shards
            new_length = key_states.shape[2]
        else:
            # Handle the case of a single, non-sharded tensor
            key_state_shards = [key_states]
            value_state_shards = [value_states]
            new_length = key_states.shape[2]

        self._seq_length = new_length

        num_shards = len(key_state_shards)
        assert num_shards == len(value_state_shards)
        assert num_shards * 2 == len(self.tensors)

        for i in range(num_shards):
            key_cache_shard = self.tensors[i * 2]
            value_cache_shard = self.tensors[i * 2 + 1]

            key_state_shard = key_state_shards[i]
            value_state_shard = value_state_shards[i]

            if HAS_FUSED_UPDATE_KERNEL:
                update_cache_fused(
                    key_cache_shard, value_cache_shard, key_state_shard, value_state_shard, prefix_length
                )
            else:
                # Original python-level copy as a fallback
                if prefix_length < new_length:
                    update_slice = slice(prefix_length, new_length)
                    key_cache_shard[:, :, update_slice, :] = key_state_shard[:, :, update_slice, :]
                    value_cache_shard[:, :, update_slice, :] = value_state_shard[:, :, update_slice, :]


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
        self.backward_pool = PrioritizedTaskPool(
            self.backward, max_batch_size=max_batch_size, device=device, name=f"{self.name}_backward"
        )

        self.dtype = backend_dtype
        self.dtype_bytes = get_size_in_bytes(self.dtype)
        self.shard_num_heads = []
        self.shard_num_kv_heads = []
        self.shard_head_dims = []
        for shard in self.module.module_shards:
            num_heads_on_shard = 0
            num_kv_heads_on_shard = 0
            head_dims_on_shard = []
            for submodule in shard.modules():
                if isinstance(submodule, config.attn_class):
                    num_heads_on_shard += submodule.num_attention_heads
                    if hasattr(submodule, "num_key_value_heads"):
                        num_kv_heads_on_shard += submodule.num_key_value_heads
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

    def backward(self, *inputs: Union[torch.Tensor, str]) -> Tuple[torch.Tensor, ...]:
        *inputs, active_adapter = inputs
        with self._peft_module.using_adapter(active_adapter):
            return super().backward(*inputs)

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

            if new_kvs is not None and len(new_kvs) == 2:
                # new_kvs is a tuple of (PerDeviceTensors(keys), PerDeviceTensors(values))
                key_states, value_states = new_kvs
                cache.update_sharded(key_states, value_states, prefix_length=inference_info.prefix_length)

            return (output_hidden_states,)

    def _estimate_max_chunk_length(self, hidden_states: torch.Tensor, inference_info: InferenceMetadata) -> int:
        # We assume that attention logit matrices are the main thing that consumes memory
        batch_size, seq_length, hidden_size = hidden_states.shape
        worst_case_length = inference_info.prefix_length + seq_length
        attn_bytes_per_token = max(self.shard_num_heads) * batch_size * self.dtype_bytes * worst_case_length
        if attn_bytes_per_token == 0:
            return 1
        return max(1, self.max_chunk_size_bytes // attn_bytes_per_token)

    def _reorder_cache_inplace(self, cache_tensors: torch.Tensor, hypo_ids: torch.Tensor):
        """If hypo_ids is specified, reorder elements of each cache tensor in-place by taking indices from hypo_ids"""
        if not is_dummy(hypo_ids):
            for cache_tensor in cache_tensors:
                cache_tensor[...] = cache_tensor[hypo_ids.to(cache_tensor.device)]  # in-place reorder cache by hypo ids

    def get_pools(self) -> Sequence[PrioritizedTaskPool]:
        return self.forward_pool, self.backward_pool, self.inference_pool

    def get_info(self) -> Dict[str, Any]:
        """Get module parameters and stats. Used by RemoteExpert to check shapes and for DMoE orchestration."""
        return dict(super().get_info(), inference_schema=self.inference_schema)

    def shutdown(self):
        # Break the cyclic references, otherwise TransformerBackend may be not garbage-collected
        self.forward_pool = self.backward_pool = self.inference_pool = None

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
            if optional_prompt is not None:
                hidden_states[:, : optional_prompt.shape[1]] += optional_prompt
            outputs = self.backends[inference_info.uid].inference_step(
                hidden_states, attention_mask, position_ids, position_embeddings, inference_info
            )
            hidden_states = outputs[0]
        return (hidden_states,)
