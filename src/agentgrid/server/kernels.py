import torch
import triton
import triton.language as tl


@triton.jit
def _copy_kv_kernel(
    key_cache_ptr,
    value_cache_ptr,
    key_states_ptr,
    value_states_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    A fused Triton kernel to copy key and value tensors.
    It copies `n_elements` from `key_states_ptr` to `key_cache_ptr`
    and from `value_states_ptr` to `value_cache_ptr`. This is more efficient
    than two separate copy operations as it reduces kernel launch overhead.
    """
    # Each program instance computes a block of the copy
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to guard memory operations against out-of-bounds access
    mask = offsets < n_elements

    # Load keys and values from source tensors (key_states, value_states)
    keys = tl.load(key_states_ptr + offsets, mask=mask)
    values = tl.load(value_states_ptr + offsets, mask=mask)

    # Store keys and values into destination cache tensors (key_cache, value_cache)
    tl.store(key_cache_ptr + offsets, keys, mask=mask)
    tl.store(value_cache_ptr + offsets, values, mask=mask)


def update_cache_fused(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    prefix_length: int,
):
    """
    Updates slices of key_cache and value_cache in-place with values from key_states and value_states
    using a fused Triton kernel.

    :param key_cache: The destination tensor for keys, shape [bs, num_heads, max_len, head_dim]
    :param value_cache: The destination tensor for values, shape [bs, num_heads, max_len, head_dim]
    :param key_states: The source tensor for keys, shape [bs, num_heads, new_len, head_dim]
    :param value_states: The source tensor for values, shape [bs, num_heads, new_len, head_dim]
    :param prefix_length: The starting sequence offset from which to update the cache.
    """
    # Basic shape and device checks for tensor validity
    assert key_cache.ndim == 4 and value_cache.ndim == 4
    assert key_states.ndim == 4 and value_states.ndim == 4
    assert key_states.shape == value_states.shape
    assert key_cache.device == value_cache.device == key_states.device == value_states.device
    assert key_cache.dtype == value_cache.dtype == key_states.dtype == value_states.dtype

    new_length = key_states.shape[2]

    # If there are no new tokens, there's nothing to update
    if prefix_length >= new_length:
        return

    # Create views into the tensors for the slices we need to update
    key_cache_view = key_cache[:, :, prefix_length:new_length, :]
    value_cache_view = value_cache[:, :, prefix_length:new_length, :]
    key_states_view = key_states[:, :, prefix_length:new_length, :]
    value_states_view = value_states[:, :, prefix_length:new_length, :]

    # Ensure the views have the same shape
    assert key_cache_view.shape == key_states_view.shape
    assert value_cache_view.shape == value_states_view.shape

    # Triton works best with contiguous tensors. Slicing can create non-contiguous views.
    # The .contiguous() call is necessary if the view is not already contiguous.
    key_cache_c = key_cache_view.contiguous()
    value_cache_c = value_cache_view.contiguous()
    key_states_c = key_states_view.contiguous()
    value_states_c = value_states_view.contiguous()

    n_elements = key_cache_c.numel()
    if n_elements == 0:
        return

    # Define the grid for launching the kernel.
    # The grid is 1D, with a size equal to the number of blocks needed.
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch the kernel.
    _copy_kv_kernel[grid](
        key_cache_c,
        value_cache_c,
        key_states_c,
        value_states_c,
        n_elements,
        BLOCK_SIZE=1024,  # A common block size, can be tuned for specific hardware
    )

    # If .contiguous() created a copy, we must copy the result back to the original view.
    # This ensures the original cache tensor is actually modified.
    if key_cache_view.data_ptr() != key_cache_c.data_ptr():
        key_cache_view.copy_(key_cache_c)
    if value_cache_view.data_ptr() != value_cache_c.data_ptr():
        value_cache_view.copy_(value_cache_c)
