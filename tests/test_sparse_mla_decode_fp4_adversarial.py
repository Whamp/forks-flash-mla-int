"""Adversarial parity for native fp4_ds_mla sparse-MLA decode on SM86."""

import math

import pytest
import torch

from fp4_ds_mla_reference import (
    FP4_DATA_BYTES,
    FP4_SCALE_GROUPS,
    HEAD_DIM,
    ROW_BYTES,
    SCALE_BYTES,
    TOKEN_DATA_BYTES,
    build_fp4_ds_mla_cache,
    decode_fp4_ds_mla_rows,
    quantize_fp4_ds_mla_rows,
)
from test_sparse_mla_prefill_adversarial import _ref
from test_sparse_mla_prefill_int8_adversarial import cos_diff


def _requires_ampere() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 6):
        pytest.skip("requires an SM86 GPU")


def test_fp4_ds_mla_physical_layout_and_round_trip() -> None:
    torch.manual_seed(101)
    rows = torch.randn(5, HEAD_DIM, dtype=torch.bfloat16)

    token_data, scales = quantize_fp4_ds_mla_rows(rows)
    decoded = decode_fp4_ds_mla_rows(token_data, scales)

    assert token_data.shape == (5, TOKEN_DATA_BYTES)
    assert scales.shape == (5, SCALE_BYTES)
    assert TOKEN_DATA_BYTES == 352
    assert FP4_DATA_BYTES == 224
    assert FP4_SCALE_GROUPS == 14
    assert ROW_BYTES == 368
    assert torch.count_nonzero(scales[:, FP4_SCALE_GROUPS:]) == 0
    torch.testing.assert_close(decoded[:, 448:], rows[:, 448:], rtol=0, atol=0)
    assert torch.isfinite(decoded).all()


@pytest.mark.parametrize("swa_topk,extra_topk", [(128, 384), (128, 896)])
@pytest.mark.parametrize("num_tokens", [1, 4])
def test_fp4_decode_adversarial_parity(swa_topk, extra_topk, num_tokens) -> None:
    _requires_ampere()
    from flash_mla import sparse_mla_decode_fp4

    torch.manual_seed(103)
    device = "cuda"
    num_heads = 64
    scale = 1.0 / math.sqrt(HEAD_DIM)
    swa_cache, swa_rows = build_fp4_ds_mla_cache(swa_topk + 40, 64, device)
    extra_cache, extra_rows = build_fp4_ds_mla_cache(extra_topk + 40, 16, device)

    q = torch.randn(
        num_tokens, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    swa_lens = torch.randint(
        0, swa_topk + 1, (num_tokens,), dtype=torch.int32, device=device
    )
    extra_lens = torch.randint(
        0, extra_topk + 1, (num_tokens,), dtype=torch.int32, device=device
    )
    swa_lens[0] = swa_topk
    extra_lens[0] = extra_topk
    if num_tokens > 1:
        swa_lens[1] = 0
        extra_lens[1] = 1
    swa_indices = torch.randint(
        0,
        swa_rows.shape[0],
        (num_tokens, swa_topk),
        dtype=torch.int32,
        device=device,
    )
    extra_indices = torch.randint(
        0,
        extra_rows.shape[0],
        (num_tokens, extra_topk),
        dtype=torch.int32,
        device=device,
    )
    swa_indices[0, 2] = -1
    swa_indices[0, 5] = swa_rows.shape[0] + 5
    extra_indices[0, 1] = -1
    sink = torch.randn(num_heads, dtype=torch.float32, device=device) * 0.1

    expected = _ref(
        q,
        swa_rows,
        swa_indices,
        swa_lens,
        extra_rows,
        extra_indices,
        extra_lens,
        scale,
        sink,
    )
    actual = sparse_mla_decode_fp4(
        q,
        swa_cache,
        swa_indices,
        swa_lens,
        scale=scale,
        attn_sink=sink,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
    )

    difference = cos_diff(actual.float(), expected)
    assert difference < 8e-5, (
        f"native fp4 decode cos_diff={difference:.2e} "
        f"(topk={swa_topk}+{extra_topk}, T={num_tokens})"
    )
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)


def test_fp4_decode_accepts_vllm_singleton_index_head() -> None:
    _requires_ampere()
    from flash_mla import sparse_mla_decode_fp4

    torch.manual_seed(107)
    device = "cuda"
    num_tokens, num_heads, topk = 3, 64, 256
    cache, _ = build_fp4_ds_mla_cache(topk + 16, 64, device)
    q = torch.randn(
        num_tokens, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    lens = torch.full((num_tokens,), topk, dtype=torch.int32, device=device)
    indices = torch.randint(
        0, cache.shape[0] * cache.shape[1], (num_tokens, topk),
        dtype=torch.int32, device=device
    )

    expected = sparse_mla_decode_fp4(q, cache, indices, lens)
    actual = sparse_mla_decode_fp4(q, cache, indices.unsqueeze(1), lens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fp4_prefill_adversarial_parity() -> None:
    _requires_ampere()
    from flash_mla import sparse_mla_prefill_fp4

    torch.manual_seed(109)
    device = "cuda"
    num_tokens, num_heads = 33, 64
    swa_topk, extra_topk = 128, 384
    scale = 1.0 / math.sqrt(HEAD_DIM)
    swa_cache, swa_rows = build_fp4_ds_mla_cache(swa_topk + 32, 64, device)
    extra_cache, extra_rows = build_fp4_ds_mla_cache(extra_topk + 32, 16, device)
    q = torch.randn(
        num_tokens, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    swa_lens = torch.randint(
        0, swa_topk + 1, (num_tokens,), dtype=torch.int32, device=device
    )
    extra_lens = torch.randint(
        0, extra_topk + 1, (num_tokens,), dtype=torch.int32, device=device
    )
    swa_lens[:6] = torch.tensor([0, 1, 15, 16, 17, swa_topk], device=device)
    extra_lens[:6] = torch.tensor([1, 0, 31, 32, 33, extra_topk], device=device)
    swa_indices = torch.randint(
        0, swa_rows.shape[0], (num_tokens, swa_topk),
        dtype=torch.int32, device=device
    )
    extra_indices = torch.randint(
        0, extra_rows.shape[0], (num_tokens, extra_topk),
        dtype=torch.int32, device=device
    )
    swa_indices[3, 2] = -1
    extra_indices[4, 0] = extra_rows.shape[0] + 1
    sink = torch.randn(num_heads, dtype=torch.float32, device=device) * 0.1

    expected = _ref(
        q, swa_rows, swa_indices, swa_lens,
        extra_rows, extra_indices, extra_lens, scale, sink
    )
    actual = sparse_mla_prefill_fp4(
        q, swa_cache, swa_indices, swa_lens,
        scale=scale, attn_sink=sink,
        extra_cache=extra_cache, extra_indices=extra_indices,
        extra_lens=extra_lens,
    )

    difference = cos_diff(actual.float(), expected)
    assert difference < 8e-5, f"native fp4 prefill cos_diff={difference:.2e}"
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)


def test_fp4_decode_cuda_graph_replay_is_deterministic() -> None:
    _requires_ampere()
    from flash_mla import sparse_mla_decode_fp4

    torch.manual_seed(113)
    device = "cuda"
    num_tokens, num_heads, topk = 2, 64, 256
    cache, _ = build_fp4_ds_mla_cache(topk + 16, 64, device)
    q = torch.randn(
        num_tokens, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    indices = torch.randint(
        0, cache.shape[0] * cache.shape[1], (num_tokens, topk),
        dtype=torch.int32, device=device
    )
    lens = torch.full((num_tokens,), topk, dtype=torch.int32, device=device)

    eager = None
    for _ in range(3):
        eager = sparse_mla_decode_fp4(q, cache, indices, lens)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = sparse_mla_decode_fp4(q, cache, indices, lens)
    graph.replay()
    torch.cuda.synchronize()
    expected = captured.clone()
    assert eager is not None
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)
    for _ in range(10):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)


def test_fp4_decode_rejects_wrong_physical_row_width() -> None:
    _requires_ampere()
    from flash_mla import sparse_mla_decode_fp4

    q = torch.zeros(1, 64, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    cache = torch.zeros(1, 64, 367, dtype=torch.uint8, device="cuda")
    indices = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
    lens = torch.ones(1, dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="368 bytes/token"):
        sparse_mla_decode_fp4(q, cache, indices, lens)
