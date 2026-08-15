"""Streaming GGUF-to-GGUF compression.

Unlike the in-memory pipeline (load_model -> compress_model -> export_model),
this processes one tensor at a time with peak memory bounded by a single
matrix. It is intended for models too large to materialize as a float32
state dict (e.g. 70B-class GGUFs on a laptop):

- FFN / attention matrices: dequantized one at a time, SVD-factored
  (B @ A), optionally int4-quantized, reconstructed and written as fp16.
- Everything else (embeddings, norms, biases, tied heads): byte-exact
  copy preserving the source quantized type.

The output is a standard GGUF loadable by geodessical and llama.cpp.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

FFN_RE = re.compile(r"^blk\.\d+\.ffn_(gate|up|down)\.weight$")
ATTN_RE = re.compile(r"^blk\.\d+\.attn_(q|k|v)\.weight$")

# Above this many elements we switch from exact truncated SVD to randomized
# svd_lowrank so a large FFN matrix still fits comfortably in memory.
RANDOMIZED_SVD_THRESHOLD = 64_000_000


def _svd_factor(W: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (A, B) with B @ A ~= W, rank <= k. A: (k, n), B: (m, k)."""
    m, n = W.shape
    k_eff = min(k, m, n)
    if m * n <= RANDOMIZED_SVD_THRESHOLD:
        from hyperretro.hf.factored import _svd_factor as _exact
        return _exact(W, k_eff)
    import torch
    T = torch.from_numpy(W.astype(np.float32))
    U, S, V = torch.svd_lowrank(T, q=k_eff, niter=2)
    A = (S[:, None] * V.t()).numpy()          # (k, n)
    B = U.numpy()                              # (m, k)
    return A.astype(np.float32), B.astype(np.float32)


def stream_compress_gguf(
    src_path: str | Path,
    dst_path: str | Path,
    *,
    ffn_rank: int = 1024,
    attn_rank: int = 0,
    int4: bool = True,
    int4_block_size: int = 128,
    int4_awq: bool = False,
    sink_T: int = 0,
    quiet: bool = False,
) -> dict:
    """Compress a GGUF in a single streaming pass.

    Args:
        src_path: source GGUF (any llama.cpp compatible quantized file).
        dst_path: output GGUF (fp16 for reconstructed matrices, source
            quantized type preserved elsewhere).
        ffn_rank: SVD rank for FFN matrices (0 = skip factoring).
        attn_rank: GRC shared-basis rank for attention Q/K/V (0 = skip).
        int4: quantize the factors to block-wise int4 before reconstruction.
        int4_block_size: int4 quantization block size.
        int4_awq: reserved for AWQ-aware calibration (no-op without a corpus).
        sink_T: attention-sink columns restored verbatim (0 = vanilla GRC).
        quiet: suppress progress output.

    Returns:
        dict with stats: n_tensors, n_factored, n_copied, out_size_mb,
        elapsed_s.
    """
    import gc
    import gguf
    from gguf import GGUFWriter

    from hyperretro.models.gguf import GGUFAdapter
    from hyperretro.models._compress import _grc_compress_layer
    from hyperretro.models._export import (
        _copy_arch_metadata_from_gguf,
        _copy_tokenizer_from_gguf,
        _gguf_arch,
    )

    src_path = Path(src_path)
    dst_path = Path(dst_path)

    adapter = GGUFAdapter(str(src_path))
    config = dict(adapter._config)
    config["_gguf_path"] = str(src_path)
    arch = _gguf_arch("gguf", config)

    writer = GGUFWriter(str(dst_path), arch)
    writer.add_name(f"HyperRetro-stream-{arch}")
    writer.add_description(
        f"HyperRetro streaming compression: ffn_rank={ffn_rank}, "
        f"attn_rank={attn_rank}, int4={int4} (block {int4_block_size})"
    )
    writer.add_block_count(config.get("num_hidden_layers", config.get("n_layers", 0)))
    writer.add_embedding_length(config.get("hidden_size", config.get("dim", 0)))
    writer.add_feed_forward_length(config.get("intermediate_size", config.get("expert_dim", 0)))
    writer.add_head_count(config.get("num_attention_heads", config.get("n_heads", 0)))
    writer.add_head_count_kv(config.get("num_key_value_heads", config.get("n_kv_heads", 0)))
    writer.add_context_length(config.get("max_position_embeddings", config.get("max_seq_len", 4096)))
    writer.add_vocab_size(config.get("vocab_size", 0))
    writer.add_file_type(0)
    _copy_tokenizer_from_gguf(writer, str(src_path))
    _copy_arch_metadata_from_gguf(writer, str(src_path), arch)

    if int4:
        from hyperretro.hf.factor_int4 import pack_int4_rows, unpack_int4_rows
        from hyperretro.hf.factor_quantize import (
            dequantize_blockwise_int4,
            quantize_blockwise_int4,
        )

    reader = adapter._reader
    n_tensors = len(reader.tensors)
    n_factored = 0
    n_copied = 0
    t0 = time.time()

    # Pass 1: register tensor infos (cheap — no data is loaded). Factored
    # matrices become fp16 dense tensors; everything else keeps its source
    # quantized type with a byte-exact raw copy.
    def _factor_rank(name: str) -> int:
        if FFN_RE.match(name):
            return ffn_rank
        if ATTN_RE.match(name):
            return attn_rank
        return 0

    # GRC attention Q/K/V of a layer are emitted only when all three are
    # buffered (at attn_v's turn). Register their infos right before the
    # first FFN/next-layer tensor so the FIFO pop order in write_tensor_data
    # matches emission order: k, q, v after the layer's bias/norm/output.
    _GRC_RE = re.compile(r"^blk\.(\d+)\.attn_(q|k|v)\.weight$")
    deferred: list[tuple[str, tuple, type, int]] = []
    deferred_blk = -1

    def _flush_deferred() -> None:
        nonlocal deferred_blk
        for nm, shp, dt, nb in deferred:
            writer.add_tensor_info(nm, shp, dt, nb)
        deferred.clear()
        deferred_blk = -1

    def _register(t) -> None:
        name = t.name
        qt = t.tensor_type
        nd = len(t.shape)
        if qt == gguf.GGMLQuantizationType.F32:
            writer.add_tensor_info(
                name, tuple(int(s) for s in reversed(t.shape)),
                np.float32, t.n_bytes,
            )
        elif qt == gguf.GGMLQuantizationType.F16:
            writer.add_tensor_info(
                name, tuple(int(s) for s in reversed(t.shape)),
                np.float16, t.n_bytes,
            )
        elif nd == 1:
            writer.add_tensor_info(
                name, (t.n_bytes,), np.uint8, t.n_bytes, raw_dtype=qt,
            )
        else:
            # Byte shape = numpy shape of the raw array: (rows, bytes/row).
            rows = int(t.shape[1])
            byte_shape = (rows, t.n_bytes // rows)
            writer.add_tensor_info(
                name, byte_shape, np.uint8, t.n_bytes, raw_dtype=qt,
            )

    for t in reader.tensors:
        name = t.name
        rank = _factor_rank(name)
        mm = _GRC_RE.match(name) if attn_rank > 0 else None
        if mm and rank > 0 and t.n_elements > 10000:
            # Defer registration until the layer's non-attention tensors
            # are registered (see emission order in pass 2).
            m = int(t.shape[1])  # rows
            n = int(t.shape[0])  # columns
            deferred.append((name, (m, n), np.float16, m * n * 2))
            deferred_blk = int(mm.group(1))
            continue
        cur_blk = int(mm2.group(1)) if (mm2 := re.match(
            r"^blk\.(\d+)\.", name)) else -1
        if deferred and (cur_blk != deferred_blk
                         or name.startswith(f"blk.{deferred_blk}.ffn_")):
            # Deferred Q/K/V belong to an earlier layer (or the FFN section
            # of their own layer): flush before registering this tensor.
            _flush_deferred()
        if rank > 0 and t.n_elements > 10000:
            m = int(t.shape[1])  # rows
            n = int(t.shape[0])  # columns
            writer.add_tensor_info(
                name,
                (m, n),  # np shape (rows, cols); writer reverses to GGUF dims
                np.float16,
                m * n * 2,
            )
        else:
            _register(t)
    _flush_deferred()

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    def _dequant(t) -> np.ndarray:
        qt = t.tensor_type
        raw = t.data
        if qt == gguf.GGMLQuantizationType.F32:
            return np.ascontiguousarray(raw, dtype=np.float32)
        if qt == gguf.GGMLQuantizationType.F16:
            return np.ascontiguousarray(raw, dtype=np.float16).astype(np.float32)
        return gguf.dequantize(raw, qt).astype(np.float32)

    def _write_dense(W: np.ndarray) -> None:
        """Reconstruct an int4-quantized dense matrix (or plain) and write."""
        if int4:
            Wq, Ws = quantize_blockwise_int4(W, block_size=int4_block_size)
            pW = pack_int4_rows(Wq)
            Wd = dequantize_blockwise_int4(unpack_int4_rows(pW, pW.shape[1] * 2), Ws)
            W2 = Wd.astype(np.float16)
        else:
            W2 = W.astype(np.float16)
        writer.write_tensor_data(np.ascontiguousarray(W2))

    # Pass 2: write tensor data in order. Attention Q/K/V of a layer are
    # buffered until all three arrive (sorted GGUF order puts k before q
    # before v within a layer block), then GRC-compressed and emitted in
    # registration order.
    pending_attn: dict[int, dict[str, np.ndarray]] = {}

    def _emit_grc_layer(li: int) -> None:
        nonlocal n_factored
        parts = pending_attn.pop(li)
        Wqn, Wkn, Wvn = _grc_compress_layer(
            parts["q"], parts["k"], parts["v"],
            min(attn_rank, parts["q"].shape[1]), sink_T)
        # registration order within the layer block: k, q, v
        _write_dense(Wkn)
        _write_dense(Wqn)
        _write_dense(Wvn)
        n_factored += 3

    for i, t in enumerate(reader.tensors):
        name = t.name
        rank = _factor_rank(name)
        mm = _GRC_RE.match(name) if attn_rank > 0 else None
        if mm and rank > 0 and t.n_elements > 10000:
            li = int(mm.group(1))
            slot = mm.group(2)
            buf = pending_attn.setdefault(li, {})
            buf[slot] = _dequant(t)
            if all(s in buf for s in ("q", "k", "v")):
                _emit_grc_layer(li)
            continue
        raw = t.data
        if rank > 0 and t.n_elements > 10000:
            W = _dequant(t)
            A, B = _svd_factor(W, rank)
            del W
            gc.collect()
            if int4:
                Aq, As = quantize_blockwise_int4(A, block_size=int4_block_size)
                Bq, Bs = quantize_blockwise_int4(B, block_size=int4_block_size)
                pA = pack_int4_rows(Aq)
                pB = pack_int4_rows(Bq)
                Ad = dequantize_blockwise_int4(unpack_int4_rows(pA, pA.shape[1] * 2), As)
                Bd = dequantize_blockwise_int4(unpack_int4_rows(pB, pB.shape[1] * 2), Bs)
                W2 = (Bd.astype(np.float64) @ Ad.astype(np.float64)).astype(np.float16)
            else:
                W2 = (B.astype(np.float64) @ A.astype(np.float64)).astype(np.float16)
            writer.write_tensor_data(np.ascontiguousarray(W2))
            n_factored += 1
            del A, B, W2
        else:
            arr = np.ascontiguousarray(raw)
            writer.write_tensor_data(arr)
            n_copied += 1
        del raw
        gc.collect()
        if not quiet and (i + 1) % 50 == 0:
            print(f"[stream] {i + 1}/{n_tensors} tensors "
                  f"({n_factored} factored, {n_copied} copied, {time.time() - t0:.0f}s)")

    if pending_attn:
        # Incomplete Q/K/V groups: emit the untouched originals would break
        # ordering; this cannot happen for well-formed GGUFs.
        raise RuntimeError(f"stream: incomplete attention groups: {sorted(pending_attn)}")

    # All tensor data was written via write_tensor_data (which handles
    # per-tensor alignment padding). No final write_tensors_to_file needed;
    # it would re-enter write_ti_data_to_file and fail in WEIGHTS state.
    writer.close()

    stats = {
        "n_tensors": n_tensors,
        "n_factored": n_factored,
        "n_copied": n_copied,
        "out_size_mb": round(dst_path.stat().st_size / 1e6, 1),
        "elapsed_s": round(time.time() - t0, 1),
    }
    if not quiet:
        print(f"[stream] done: {stats}")
    return stats
