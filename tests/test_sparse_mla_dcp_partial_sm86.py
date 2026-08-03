"""P9 fork patches: (a) the PARTIAL (context-parallel) sparse-MLA decode op and
(b) the BLOCK_M=16 fused sparse-prefill instantiation.

Both are checked against the SAME fp32 oracle the existing suites use, plus an
algebraic identity that is the whole point of the partial op:

    merging the per-shard (out, lse) partials of an arbitrary PARTITION of the
    selection, then folding attn_sink once at the global max, must reproduce the
    combined op's single-shard answer.

That identity is exactly what vLLM's `dcp_merge_flashmla_output` computes across
DCP ranks, so it is the test that actually protects the production path -- an
un-normalized or sink-folded partial passes an oracle-vs-out check on one shard
and still corrupts the merge.

Run (Ampere, single GPU):
  CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_sparse_mla_dcp_partial_sm86.py -x -q
"""
import math

import pytest
import torch

# tests/ has no __init__.py: pytest puts tests/ itself on sys.path, but a plain
# `python -m pytest` from the repo root puts the root there. Accept both.
try:
    from test_sparse_mla_decode_sm86 import (  # noqa: E402
        _HEAD_DIM,
        _SCALE_DIM,
        _TOKEN_DATA_SIZE,
        _write_fp8_ds_mla_token,
        cos_diff,
    )
except ImportError:  # pragma: no cover
    from tests.test_sparse_mla_decode_sm86 import (  # noqa: E402
        _HEAD_DIM,
        _SCALE_DIM,
        _TOKEN_DATA_SIZE,
        _write_fp8_ds_mla_token,
        cos_diff,
    )

# must match PARTIAL_LSE_SENTINEL in csrc/flash_sparse_mla_decode_sm80.cu and
# DCP_LSE_SENTINEL in vLLM's models/deepseek_v4/common/ops/dcp.py
LSE_SENTINEL = -1.0e30


def _build_cache(num_slots, block_size, dev):
    nb = (num_slots + block_size - 1) // block_size
    cache = torch.zeros(
        nb, block_size, _TOKEN_DATA_SIZE + _SCALE_DIM, dtype=torch.uint8, device=dev
    )
    K = torch.zeros(nb * block_size, _HEAD_DIM, dtype=torch.bfloat16, device=dev)
    for slot in range(num_slots):
        K[slot] = _write_fp8_ds_mla_token(cache, slot, block_size)
    return cache, K


def _ref_partial(q, K_by_slot, indices, lens, scale):
    """fp32 oracle for the PARTIAL contract: normalized pre-sink out + natural lse."""
    T, H, D = q.shape
    out = torch.zeros(T, H, D, device=q.device, dtype=torch.float32)
    lse = torch.full((T, H), LSE_SENTINEL, device=q.device, dtype=torch.float32)
    for t in range(T):
        n = int(lens[t].item())
        if n == 0:
            continue
        sl = indices[t, :n].long()
        K = K_by_slot[sl].float()
        scores = (q[t].float() @ K.t()) * scale          # [H, n]
        m = scores.max(dim=-1, keepdim=True).values
        ex = torch.exp(scores - m)
        l = ex.sum(-1, keepdim=True)
        out[t] = (ex / l) @ K
        lse[t] = (m + torch.log(l)).squeeze(-1)
    return out, lse


def _merge(outs, lses, sink):
    """The dcp.py merge, in float64: LSE-weighted combine then sink once."""
    o = torch.stack([x.double() for x in outs])          # [N, T, H, D]
    ls = torch.stack([x.double() for x in lses])         # [N, T, H]
    mx = ls.max(dim=0).values
    w = torch.exp(ls - mx)
    ws = w.sum(0)
    merged = (o * (w / ws).unsqueeze(-1)).sum(0)
    glse = torch.log(ws) + mx
    if sink is None:
        return merged
    out_lse = torch.logaddexp(glse, sink.double().unsqueeze(0))
    return merged * torch.exp(glse - out_lse).unsqueeze(-1)


@pytest.mark.parametrize("H", [64, 16])
@pytest.mark.parametrize("topk", [512])
def test_decode_partial_matches_oracle(H, topk):
    """out is NORMALIZED and PRE-SINK; lse is natural-log; both match fp32."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from flash_mla import sparse_mla_decode_fp8_partial

    torch.manual_seed(0)
    dev, T, block_size = "cuda", 4, 32
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    num_slots = topk + 64
    cache, K = _build_cache(num_slots, block_size, dev)

    q = torch.randn(T, H, _HEAD_DIM, device=dev, dtype=torch.bfloat16)
    lens = torch.tensor([topk, topk - 7, 1, 0], dtype=torch.int32, device=dev)[:T]
    idx = torch.stack(
        [torch.randperm(num_slots, device=dev)[:topk].to(torch.int32) for _ in range(T)]
    )

    o_ref, lse_ref = _ref_partial(q, K, idx, lens, scale)
    o, lse = sparse_mla_decode_fp8_partial(
        q=q, swa_cache=cache, swa_indices=idx, swa_lens=lens, scale=scale
    )
    assert lse.dtype == torch.float32 and tuple(lse.shape) == (T, H)
    assert torch.isfinite(o.float()).all(), "partial out must never be NaN/inf"
    assert torch.isfinite(lse).all(), "partial lse must never be NaN/inf"

    valid = lens > 0
    cd = cos_diff(o[valid].float(), o_ref[valid])
    assert cd < 8e-5, f"partial out cos_diff={cd:.2e}"
    assert torch.allclose(lse[valid], lse_ref[valid], atol=2e-2, rtol=2e-3)

    # rule 9: the empty shard is a finite sentinel and an exactly-zero row.
    empty = ~valid
    if empty.any():
        assert (lse[empty] <= LSE_SENTINEL / 2).all(), "empty shard needs the sentinel"
        assert (o[empty].float() == 0).all(), "empty shard must be exactly zero"


@pytest.mark.parametrize("H", [64])
@pytest.mark.parametrize("shards", [2, 4])
def test_decode_partial_merge_equals_combined(H, shards):
    """Partition the selection across `shards`; merged+sinked == combined op."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from flash_mla import sparse_mla_decode_fp8, sparse_mla_decode_fp8_partial

    torch.manual_seed(7)
    dev, T, block_size, topk = "cuda", 3, 32, 512
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    num_slots = topk + 64
    cache, _K = _build_cache(num_slots, block_size, dev)

    q = torch.randn(T, H, _HEAD_DIM, device=dev, dtype=torch.bfloat16)
    lens = torch.tensor([topk, topk - 13, topk // 2], dtype=torch.int32, device=dev)[:T]
    idx = torch.stack(
        [torch.randperm(num_slots, device=dev)[:topk].to(torch.int32) for _ in range(T)]
    )
    sink = torch.randn(H, device=dev, dtype=torch.float32) * 0.1

    combined = sparse_mla_decode_fp8(
        q=q, swa_cache=cache, swa_indices=idx, swa_lens=lens, scale=scale, attn_sink=sink
    )

    # round-robin partition -- every shard keeps a prefix-compact row, one shard
    # is deliberately given zero entries for token 0 (exercises the sentinel).
    outs, lses = [], []
    for r in range(shards):
        sidx = torch.zeros_like(idx)
        slen = torch.zeros_like(lens)
        for t in range(T):
            n = int(lens[t].item())
            take = idx[t, r:n:shards]
            if t == 0 and r == shards - 1:
                take = take[:0]
            sidx[t, : take.numel()] = take
            slen[t] = take.numel()
        o, l = sparse_mla_decode_fp8_partial(
            q=q, swa_cache=cache, swa_indices=sidx, swa_lens=slen, scale=scale
        )
        assert torch.isfinite(o.float()).all() and torch.isfinite(l).all()
        outs.append(o.float())
        lses.append(l)

    merged = _merge(outs, lses, sink)
    cd = cos_diff(merged.float(), combined.float())
    assert cd < 5e-5, f"merged partials vs combined cos_diff={cd:.2e} (shards={shards})"


@pytest.mark.parametrize("H", [8, 16, 32, 64])
def test_prefill_block_m_dispatch_matches_oracle(H):
    """BLOCK_M=16 (H<=16) and BLOCK_M=32 (H>16) both match the fp32 oracle.

    H=16 and H=8 exercise the new narrow instantiation (4-way QK k-split, 8 PV
    d64 slices); H=32/64 re-run the untouched default so this doubles as the
    BLOCK_M=32 regression.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from flash_mla import sparse_mla_prefill

    torch.manual_seed(3)
    dev, T, block_size, topk = "cuda", 5, 1, 96
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    num_slots = topk + 32
    cache, K = _build_cache(num_slots, block_size, dev)

    q = torch.randn(T, H, _HEAD_DIM, device=dev, dtype=torch.bfloat16)
    lens = torch.randint(1, topk + 1, (T,), dtype=torch.int32, device=dev)
    idx = torch.stack(
        [torch.randperm(num_slots, device=dev)[:topk].to(torch.int32) for _ in range(T)]
    )
    sink = torch.randn(H, device=dev, dtype=torch.float32) * 0.1

    ref = torch.zeros(T, H, _HEAD_DIM, device=dev, dtype=torch.float32)
    for t in range(T):
        n = int(lens[t].item())
        Kt = K[idx[t, :n].long()].float()
        scores = (q[t].float() @ Kt.t()) * scale
        s = sink[:, None].float()
        m = torch.maximum(scores.max(dim=-1, keepdim=True).values, s)
        ex = torch.exp(scores - m)
        ref[t] = (ex / (ex.sum(-1, keepdim=True) + torch.exp(s - m))) @ Kt

    out = sparse_mla_prefill(
        q=q, swa_cache=cache, swa_indices=idx, swa_lens=lens, scale=scale, attn_sink=sink
    )
    cd = cos_diff(out.float(), ref)
    assert cd < 8e-5, f"prefill H={H} cos_diff={cd:.2e}"


def test_prefill_block_m_16_and_32_agree():
    """At H=16 the two instantiations must agree (same math, different tiling)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import os
    import subprocess
    import sys

    # the dispatch is a process-level static (env read once), so A/B in
    # subprocesses rather than flipping it mid-process.
    script = (
        "import math,torch;from flash_mla import sparse_mla_prefill;"
        "torch.manual_seed(11);d='cuda';T,H,topk=4,16,96;n=topk+32;"
        # deterministic non-degenerate cache bytes; the kernel dequantizes them
        # identically at either BLOCK_M, so only the tiling differs.
        # sane fp8 payload -- three NaN sources in raw random bytes, all
        # neutralized: E4M3 data bytes & 0x7E (0x7F/0xFF are E4M3 NaN),
        # RoPE bf16 region zeroed (random bytes decode to inf/NaN), scale
        # exponents clamped to a +/-2^4 window (raw scales reach 2^128).
        "g=torch.Generator(device='cpu').manual_seed(5);"
        "c=torch.randint(0,255,(n,1,584),generator=g,dtype=torch.uint8);"
        "c[:,:,:448]&=126;"
        "c[:,:,448:576]=0;"
        "c[:,:,576:584]=torch.randint(123,132,(n,1,8),generator=g,dtype=torch.uint8);"
        "c=c.to(d);"
        "q=torch.randn(T,H,512,device=d,dtype=torch.bfloat16);"
        "l=torch.full((T,),topk,dtype=torch.int32,device=d);"
        "i=torch.stack([torch.arange(topk,device=d,dtype=torch.int32) for _ in range(T)]);"
        "o=sparse_mla_prefill(q=q,swa_cache=c,swa_indices=i,swa_lens=l,"
        "scale=512**-0.5,attn_sink=None);"
        "print(float(o.float().sum()), float((o.float()**2).sum()))"
    )
    res = {}
    for bm in ("16", "32"):
        env = dict(os.environ, FLASH_MLA_PREFILL_BLOCK_M=bm)
        p = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True
        )
        assert p.returncode == 0, p.stderr
        res[bm] = [float(x) for x in p.stdout.split()]
    for a, b in zip(res["16"], res["32"]):
        assert abs(a - b) <= 1e-3 * max(1.0, abs(b)), f"BLOCK_M 16 vs 32 differ: {res}"
