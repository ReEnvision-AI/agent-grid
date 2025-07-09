
import torch
import triton
import triton.language as tl


@triton.jit
def _apply_rope_kernel(
    t_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_t_b, stride_t_h, stride_t_s,
    stride_cos_b, stride_cos_s,
    stride_sin_b, stride_sin_s,
    head_dim: tl.constexpr,
    half_head_dim: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    t_offset = pid_b * stride_t_b + pid_h * stride_t_h + pid_s * stride_t_s
    cos_offset = pid_b * stride_cos_b + pid_s * stride_cos_s
    sin_offset = pid_b * stride_sin_b + pid_s * stride_sin_s

    t_ptr += t_offset
    out_ptr += t_offset
    cos_ptr += cos_offset
    sin_ptr += sin_offset

    # Offsets for the first and second halves of the head dimension
    h1_offsets = tl.arange(0, half_head_dim)
    h2_offsets = tl.arange(half_head_dim, head_dim)

    # Load the two halves of the input tensor
    t_h1 = tl.load(t_ptr + h1_offsets)
    t_h2 = tl.load(t_ptr + h2_offsets)

    # Load the corresponding cos and sin values (only the first half is needed)
    cos_h1 = tl.load(cos_ptr + h1_offsets)
    sin_h1 = tl.load(sin_ptr + h1_offsets)

    # Apply the rotary embedding transformation
    out_h1 = t_h1 * cos_h1 - t_h2 * sin_h1
    out_h2 = t_h2 * cos_h1 + t_h1 * sin_h1

    # Store the results back to the output tensor
    tl.store(out_ptr + h1_offsets, out_h1)
    tl.store(out_ptr + h2_offsets, out_h2)


def apply_rotary_pos_emb_fused(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary position embedding to q and k tensors using a fused Triton kernel.
    """
    bsz, num_q_heads, seq_len, head_dim = q.shape
    _, num_k_heads, _, _ = k.shape
    
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    grid_q = (bsz, num_q_heads, seq_len)
    grid_k = (bsz, num_k_heads, seq_len)

    half_head_dim = head_dim // 2

    _apply_rope_kernel[grid_q](
        q, cos, sin, q_out,
        q.stride(0), q.stride(1), q.stride(2),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        head_dim=head_dim,
        half_head_dim=half_head_dim,
    )
    _apply_rope_kernel[grid_k](
        k, cos, sin, k_out,
        k.stride(0), k.stride(1), k.stride(2),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        head_dim=head_dim,
        half_head_dim=half_head_dim,
    )
    return q_out, k_out
