#
# Copyright (c) 2025 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#

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


@triton.jit
def _reorder_kv_cache_kernel(
    key_out_ptr,
    value_out_ptr,
    key_in_ptr,
    value_in_ptr,
    beam_idx_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_COLS: tl.constexpr,
):
    """
    A fused Triton kernel to reorder cache tensors based on beam indices.
    It performs a gather operation for both key and value caches in a single kernel launch.
    `output = input[beam_idx]`
    """
    row_idx = tl.program_id(axis=0)

    # The index of the row to gather from in the input tensor
    gather_idx = tl.load(beam_idx_ptr + row_idx)

    # Pointers to the start of the rows in the input and output tensors
    key_in_row_ptr = key_in_ptr + gather_idx * n_cols
    value_in_row_ptr = value_in_ptr + gather_idx * n_cols
    key_out_row_ptr = key_out_ptr + row_idx * n_cols
    value_out_row_ptr = value_out_ptr + row_idx * n_cols

    # Iterate over the columns of the row in blocks
    for col_block_start in range(0, n_cols, BLOCK_SIZE_COLS):
        col_offsets = col_block_start + tl.arange(0, BLOCK_SIZE_COLS)
        mask = col_offsets < n_cols

        # Reorder keys
        keys = tl.load(key_in_row_ptr + col_offsets, mask=mask)
        tl.store(key_out_row_ptr + col_offsets, keys, mask=mask)

        # Reorder values
        values = tl.load(value_in_row_ptr + col_offsets, mask=mask)
        tl.store(value_out_row_ptr + col_offsets, values, mask=mask)


def reorder_cache_fused(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    beam_idx: torch.LongTensor,
):
    """
    Reorders key_cache and value_cache in-place according to beam_idx using a fused Triton kernel.
    This is equivalent to:
    key_cache[...] = key_cache[beam_idx]
    value_cache[...] = value_cache[beam_idx]

    :param key_cache: The key cache tensor to reorder, shape [bs, num_heads, max_len, head_dim]
    :param value_cache: The value cache tensor to reorder, shape [bs, num_heads, max_len, head_dim]
    :param beam_idx: The beam indices for reordering, shape [bs]
    """
    # Basic shape and device checks
    assert key_cache.ndim == 4 and value_cache.ndim == 4
    assert key_cache.shape == value_cache.shape
    assert beam_idx.ndim == 1
    assert key_cache.shape[0] == beam_idx.shape[0]
    assert key_cache.device == value_cache.device == beam_idx.device

    batch_size, num_heads, seq_len, head_dim = key_cache.shape

    # The kernel writes to temporary output tensors to avoid in-place issues
    key_out = torch.empty_like(key_cache)
    value_out = torch.empty_like(value_cache)

    n_rows = batch_size
    n_cols = num_heads * seq_len * head_dim

    if n_rows == 0 or n_cols == 0:
        return

    # Ensure tensors are contiguous for the kernel
    key_in_c = key_cache.contiguous()
    value_in_c = value_cache.contiguous()
    beam_idx_c = beam_idx.contiguous()

    # Define the grid for launching the kernel
    grid = (n_rows,)

    # A reasonable block size for the column dimension
    BLOCK_SIZE_COLS = 1024

    _reorder_kv_cache_kernel[grid](
        key_out,
        value_out,
        key_in_c,
        value_in_c,
        beam_idx_c,
        n_rows,
        n_cols,
        BLOCK_SIZE_COLS=BLOCK_SIZE_COLS,
    )

    # Copy the results back to the original tensors for in-place semantics
    key_cache.copy_(key_out)
    value_cache.copy_(value_out)


@triton.jit
def _reorder_kv_cache_kernel(
    key_out_ptr,
    value_out_ptr,
    key_in_ptr,
    value_in_ptr,
    beam_idx_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_COLS: tl.constexpr,
):
    """
    A fused Triton kernel to reorder cache tensors based on beam indices.
    It performs a gather operation for both key and value caches in a single kernel launch.
    `output = input[beam_idx]`
    """
    row_idx = tl.program_id(axis=0)

    # The index of the row to gather from in the input tensor
    gather_idx = tl.load(beam_idx_ptr + row_idx)

    # Pointers to the start of the rows in the input and output tensors
    key_in_row_ptr = key_in_ptr + gather_idx * n_cols
    value_in_row_ptr = value_in_ptr + gather_idx * n_cols
    key_out_row_ptr = key_out_ptr + row_idx * n_cols
    value_out_row_ptr = value_out_ptr + row_idx * n_cols

    # Iterate over the columns of the row in blocks
    for col_block_start in range(0, n_cols, BLOCK_SIZE_COLS):
        col_offsets = col_block_start + tl.arange(0, BLOCK_SIZE_COLS)
        mask = col_offsets < n_cols

        # Reorder keys
        keys = tl.load(key_in_row_ptr + col_offsets, mask=mask)
        tl.store(key_out_row_ptr + col_offsets, keys, mask=mask)

        # Reorder values
        values = tl.load(value_in_row_ptr + col_offsets, mask=mask)
        tl.store(value_out_row_ptr + col_offsets, values, mask=mask)

