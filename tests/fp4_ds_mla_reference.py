"""Independent fp4_ds_mla cache encoder/decoder used by CUDA tests."""

from __future__ import annotations

import torch

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
FP4_GROUP_SIZE = 32
FP4_SCALE_GROUPS = NOPE_DIM // FP4_GROUP_SIZE
FP4_DATA_BYTES = NOPE_DIM // 2
ROPE_DATA_BYTES = ROPE_DIM * 2
TOKEN_DATA_BYTES = FP4_DATA_BYTES + ROPE_DATA_BYTES
SCALE_BYTES = 16
ROW_BYTES = TOKEN_DATA_BYTES + SCALE_BYTES

_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _quantize_e2m1_codes(scaled: torch.Tensor) -> torch.Tensor:
    """Round to E2M1 with ties-to-even, returning one nibble per value."""
    magnitude = scaled.abs()
    code = torch.zeros_like(magnitude, dtype=torch.int32)
    code = torch.where(magnitude > 0.25, 1, code)
    code = torch.where(magnitude >= 0.75, 2, code)
    code = torch.where(magnitude > 1.25, 3, code)
    code = torch.where(magnitude >= 1.75, 4, code)
    code = torch.where(magnitude > 2.5, 5, code)
    code = torch.where(magnitude >= 3.5, 6, code)
    code = torch.where(magnitude > 5.0, 7, code)
    sign = torch.signbit(scaled).to(torch.uint8)
    return code.to(torch.uint8) | (sign << 3)


def quantize_fp4_ds_mla_rows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode semantic BF16 rows into token-data and per-token scale sections.

    Returns:
        token_data: ``[rows, 352]`` uint8 (224 packed NoPE + 128 BF16 RoPE).
        scale_data: ``[rows, 16]`` uint8 (14 UE8M0 scales + 2 zero padding).
    """
    if rows.ndim != 2 or rows.shape[1] != HEAD_DIM:
        raise ValueError(f"rows must have shape [N, {HEAD_DIM}], got {tuple(rows.shape)}")
    if rows.dtype != torch.bfloat16:
        raise ValueError(f"rows must be bfloat16, got {rows.dtype}")

    nope = rows[:, :NOPE_DIM].float().reshape(-1, FP4_SCALE_GROUPS, FP4_GROUP_SIZE)
    amax = nope.abs().amax(dim=-1, keepdim=True).clamp_min(6.0 * (2.0**-126))
    exponent = torch.ceil(torch.log2(amax * (1.0 / 6.0))).clamp(-127.0, 127.0)
    scale = torch.exp2(exponent)
    scaled = (nope / scale).clamp(-6.0, 6.0)
    codes = _quantize_e2m1_codes(scaled).reshape(-1, NOPE_DIM)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)

    rope_bytes = rows[:, NOPE_DIM:].contiguous().view(torch.uint8)
    token_data = torch.cat((packed, rope_bytes), dim=-1).contiguous()
    scale_data = torch.zeros(rows.shape[0], SCALE_BYTES, dtype=torch.uint8, device=rows.device)
    scale_data[:, :FP4_SCALE_GROUPS] = (exponent.squeeze(-1) + 127.0).to(torch.uint8)
    return token_data, scale_data


def decode_fp4_ds_mla_rows(
    token_data: torch.Tensor, scale_data: torch.Tensor
) -> torch.Tensor:
    """Decode encoded sections to the BF16 values consumed by attention."""
    if token_data.ndim != 2 or token_data.shape[1] != TOKEN_DATA_BYTES:
        raise ValueError(
            f"token_data must have shape [N, {TOKEN_DATA_BYTES}], "
            f"got {tuple(token_data.shape)}"
        )
    if scale_data.ndim != 2 or scale_data.shape != (token_data.shape[0], SCALE_BYTES):
        raise ValueError(
            f"scale_data must have shape [N, {SCALE_BYTES}], "
            f"got {tuple(scale_data.shape)}"
        )

    packed = token_data[:, :FP4_DATA_BYTES]
    codes = torch.empty(
        token_data.shape[0], NOPE_DIM, dtype=torch.uint8, device=token_data.device
    )
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    magnitude_lut = torch.tensor(
        _E2M1_MAGNITUDES, dtype=torch.float32, device=token_data.device
    )
    magnitude = magnitude_lut[(codes & 0x07).long()]
    values = torch.where((codes & 0x08) != 0, -magnitude, magnitude)
    group_scale = torch.exp2(
        scale_data[:, :FP4_SCALE_GROUPS].to(torch.float32) - 127.0
    ).repeat_interleave(FP4_GROUP_SIZE, dim=-1)
    nope = (values * group_scale).to(torch.bfloat16)
    rope = (
        token_data[:, FP4_DATA_BYTES:]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(-1, ROPE_DIM)
    )
    return torch.cat((nope, rope), dim=-1).contiguous()


def build_fp4_ds_mla_cache(
    num_rows: int, block_size: int, device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build paged raw cache plus its independently decoded semantic rows."""
    num_blocks = (num_rows + block_size - 1) // block_size
    capacity = num_blocks * block_size
    source = torch.randn(capacity, HEAD_DIM, dtype=torch.bfloat16, device=device)
    token_data, scale_data = quantize_fp4_ds_mla_rows(source)

    raw = torch.zeros(
        num_blocks, block_size * ROW_BYTES, dtype=torch.uint8, device=device
    )
    raw[:, : block_size * TOKEN_DATA_BYTES].view(
        num_blocks, block_size, TOKEN_DATA_BYTES
    ).copy_(token_data.view(num_blocks, block_size, TOKEN_DATA_BYTES))
    raw[:, block_size * TOKEN_DATA_BYTES :].view(
        num_blocks, block_size, SCALE_BYTES
    ).copy_(scale_data.view(num_blocks, block_size, SCALE_BYTES))

    decoded = decode_fp4_ds_mla_rows(token_data, scale_data)
    return raw.view(num_blocks, block_size, ROW_BYTES), decoded
