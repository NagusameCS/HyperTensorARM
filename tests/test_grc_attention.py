"""Tests for GRC shared-basis attention compression (GGUF path).

Ports of the original HyperTensor method (build_shared_basis / project /
sink_indices) into the unified GGUF compression pipeline. These tests pin
the math-level properties: full-rank identity, sink-column restoration,
per-layer grouping, and the manifest contract.
"""

import numpy as np

from hyperretro.models._compress import (
    _GGUF_ATTN_RE,
    _grc_attention,
    _grc_compress_layer,
    _group_gguf_attn,
)


def _rand_mat(m, n, seed):
    return np.random.default_rng(seed).standard_normal((m, n)).astype(np.float32)


def test_full_rank_projection_is_identity():
    rng = np.random.default_rng(0)
    Wq, Wk, Wv = (_rand_mat(64, 64, s) for s in (1, 2, 3))
    q2, k2, v2 = _grc_compress_layer(Wq, Wk, Wv, k=64, sink_T=0)
    assert np.abs(q2 - Wq).max() < 1e-4
    assert np.abs(k2 - Wk).max() < 1e-4
    assert np.abs(v2 - Wv).max() < 1e-4


def test_sink_columns_restored_verbatim():
    rng = np.random.default_rng(7)
    Wq = rng.standard_normal((32, 96)).astype(np.float32)
    Wk = rng.standard_normal((16, 96)).astype(np.float32)
    Wv = rng.standard_normal((16, 96)).astype(np.float32)
    q2, k2, v2 = _grc_compress_layer(Wq, Wk, Wv, k=32, sink_T=8)
    from hyperretro.hf.compress import sink_indices
    sink = sink_indices(Wq, Wk, Wv, 8)
    assert len(sink) == 8
    assert np.array_equal(q2[:, sink], Wq[:, sink])
    assert np.array_equal(k2[:, sink], Wk[:, sink])
    assert np.array_equal(v2[:, sink], Wv[:, sink])
    # Rank-limited compression actually changes the non-sink columns.
    rest = np.ones(Wq.shape[1], dtype=bool)
    rest[sink] = False
    assert np.abs(q2[:, rest] - Wq[:, rest]).max() > 1e-6


def test_group_gguf_attn_names():
    sd = {
        "blk.0.attn_k.weight": None,
        "blk.0.attn_q.weight": None,
        "blk.0.attn_v.weight": None,
        "blk.0.attn_output.weight": None,
        "blk.1.attn_k.weight": None,
        "blk.1.attn_q.weight": None,
        "token_embd.weight": None,
    }
    layers = _group_gguf_attn(sd)
    assert set(layers) == {0, 1}
    assert layers[0] == {
        "k": "blk.0.attn_k.weight",
        "q": "blk.0.attn_q.weight",
        "v": "blk.0.attn_v.weight",
    }
    assert _GGUF_ATTN_RE.match("blk.12.attn_v.weight").group(1) == "12"


def test_grc_attention_skips_incomplete_layers():
    sd = {
        "blk.0.attn_k.weight": _rand_mat(16, 64, 1),
        "blk.0.attn_q.weight": _rand_mat(64, 64, 2),
        # blk.0.attn_v.weight missing: layer must be skipped
        "blk.1.attn_k.weight": _rand_mat(16, 64, 3),
        "blk.1.attn_q.weight": _rand_mat(64, 64, 4),
        "blk.1.attn_v.weight": _rand_mat(16, 64, 5),
    }
    import torch
    sd = {k: torch.from_numpy(v) for k, v in sd.items()}
    out = _grc_attention(sd, attn_rank=32, sink_T=0)
    assert "blk.0.attn_k.weight" not in out
    assert "blk.1.attn_q.weight" in out
    assert "blk.1.attn_k.weight" in out
    assert "blk.1.attn_v.weight" in out


def test_manifest_grc_flags():
    import torch
    sd = {}
    rng = np.random.default_rng(11)
    n_layers = 2
    for li in range(n_layers):
        for slot, m in (("q", 64), ("k", 16), ("v", 16)):
            sd[f"blk.{li}.attn_{slot}.weight"] = torch.from_numpy(
                rng.standard_normal((m, 64)).astype(np.float32))
        sd[f"blk.{li}.ffn_gate.weight"] = torch.from_numpy(
            rng.standard_normal((256, 64)).astype(np.float32))

    from hyperretro.models._compress import CompressedModel
    model = type("M", (), {
        "state_dict": sd,
        "backend": "gguf",
        "config": {},
    })()
    cm = _compress_generic_wrapper(model, attn_rank=32, sink_T=4)
    grc_entries = [e for e in cm.manifest["layers"] if e.get("grc")]
    assert len(grc_entries) == n_layers * 3
    assert all(e["rank"] == 32 for e in grc_entries)


def _compress_generic_wrapper(model, *, attn_rank, sink_T):
    from hyperretro.models._compress import _compress_generic
    return _compress_generic(
        model, ffn_rank=0, attn_rank=attn_rank, int4=False,
        block_size=128, awq=False, sink_T=sink_T,
    )
