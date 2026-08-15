"""Unified model export — GGUF, safetensors, HF formats.

Exports any AbstractModel or CompressedModel to industry-standard formats.
The export adapts to the model architecture:

- GGUF: llama.cpp / Ollama compatible
- safetensors: standard weight format
- HF: HuggingFace-compatible directory (config.json + model.safetensors)
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from hyperretro.models import AbstractModel, CompressedModel


def _export_model(
    model: AbstractModel | CompressedModel,
    path: str | Path,
    *,
    format: str = "auto",
    **kwargs,
) -> Path:
    """Export model to an industry-standard format.

    Args:
        model: AbstractModel or CompressedModel.
        path: output path.
        format: 'gguf', 'safetensors', 'hf', or 'auto'.
        **kwargs: format-specific options.

    Returns:
        Output path.
    """
    path = Path(path)

    if format == "auto":
        format = _detect_format(path)

    if format == "gguf":
        return _export_gguf(model, path, **kwargs)
    elif format in ("safetensors", "hf"):
        return _export_safetensors(model, path, **kwargs)
    else:
        raise ValueError(f"Unknown format: {format}. Supported: gguf, safetensors, hf")


def _detect_format(path: Path) -> str:
    """Detect format from file extension."""
    if path.suffix == ".gguf":
        return "gguf"
    return "safetensors"


# ---------------------------------------------------------------------------
# GGUF export
# ---------------------------------------------------------------------------

def _copy_tokenizer_from_gguf(writer, src_path: str) -> None:
    """Copy tokenizer KV (tokens/merges/types/scores/special ids) from a source
    GGUF into the writer, so the exported file is self-contained."""
    import numpy as np
    from gguf import GGUFReader

    r = GGUFReader(src_path)
    f = r.fields

    def _string_field(key):
        v = f.get(key)
        if v is None:
            return None
        parts = getattr(v, "parts", None)
        if not parts or len(parts) < 3:
            return None
        # Scalar string layout: [key_type, key_name, ...]; value = parts[-1].
        p = parts[-1]
        if hasattr(p, "shape"):
            return bytes(np.asarray(p, dtype=np.uint8).tolist())
        return None

    def _array_of_strings(key):
        """ARRAY of STRING: gguf-python stores per-element part offsets in
        field.data; each element's bytes live at parts[offset]. (String
        lengths are varints and take a variable number of parts, so a fixed
        stride over parts is NOT valid.)"""
        v = f.get(key)
        if v is None:
            return None
        parts = getattr(v, "parts", None)
        data = getattr(v, "data", None)
        if not parts or data is None:
            return None
        out = []
        for idx in data:
            p = parts[int(idx)]
            if hasattr(p, "shape"):
                out.append(bytes(np.asarray(p, dtype=np.uint8).tolist()))
        return out

    def _array_of_ints(key):
        v = f.get(key)
        if v is None:
            return None
        parts = getattr(v, "parts", None)
        data = getattr(v, "data", None)
        if not parts or data is None:
            return None
        out = []
        for idx in data:
            try:
                p = parts[int(idx)]
                out.append(int(p.item() if hasattr(p, "item") else p))
            except Exception:
                out.append(0)
        return out

    def _array_of_floats(key):
        v = f.get(key)
        if v is None:
            return None
        parts = getattr(v, "parts", None)
        data = getattr(v, "data", None)
        if not parts or data is None:
            return None
        out = []
        for idx in data:
            try:
                p = parts[int(idx)]
                out.append(float(p.item() if hasattr(p, "item") else p))
            except Exception:
                out.append(0.0)
        return out

    def _int_field(key):
        v = f.get(key)
        if v is None:
            return None
        parts = getattr(v, "parts", None)
        if not parts:
            return None
        p = parts[-1]
        try:
            return int(p.item() if hasattr(p, "item") else p)
        except Exception:
            return None

    model = _string_field("tokenizer.ggml.model")
    if model is not None:
        try:
            writer.add_tokenizer_model(model.decode("utf-8", "replace"))
        except Exception:
            pass

    tokens = _array_of_strings("tokenizer.ggml.tokens")
    if tokens:
        writer.add_token_list(tokens)
    types = _array_of_ints("tokenizer.ggml.token_type")
    if types:
        try:
            writer.add_token_types([int(t) for t in types])
        except Exception:
            pass
    merges = _array_of_strings("tokenizer.ggml.merges")
    if merges:
        try:
            writer.add_token_merges([m.decode("utf-8", "replace") for m in merges])
        except Exception:
            pass
    scores = _array_of_floats("tokenizer.ggml.scores")
    if scores:
        try:
            writer.add_token_scores(scores)
        except Exception:
            pass

    for kv, fn in (
        ("tokenizer.ggml.bos_token_id", "add_bos_token_id"),
        ("tokenizer.ggml.eos_token_id", "add_eos_token_id"),
        ("tokenizer.ggml.unknown_token_id", "add_unk_token_id"),
        ("tokenizer.ggml.separator_token_id", "add_sep_token_id"),
        ("tokenizer.ggml.padding_token_id", "add_pad_token_id"),
        ("tokenizer.ggml.add_bos_token", "add_add_bos_token"),
        ("tokenizer.ggml.add_eos_token", "add_add_eos_token"),
        ("tokenizer.ggml.add_space_prefix", "add_add_space_prefix"),
    ):
        val = _int_field(kv)
        if val is not None:
            try:
                getattr(writer, fn)(val)
            except Exception:
                pass

    pre = _string_field("tokenizer.ggml.pre")
    if pre is not None:
        try:
            writer.add_tokenizer_pre(pre.decode("utf-8", "replace"))
        except Exception:
            pass
    chat_tpl = _string_field("tokenizer.chat_template")
    if chat_tpl is not None:
        try:
            writer.add_chat_template(chat_tpl.decode("utf-8", "replace"))
        except Exception:
            pass


def _copy_arch_metadata_from_gguf(writer, src_path: str, arch: str) -> None:
    """Copy architecture metadata KV (rope.freq_base, rms_epsilon, ...) from
    the source GGUF so the exported file keeps correct hyperparameters."""
    from gguf import GGUFReader, GGUFValueType

    r = GGUFReader(src_path)
    skip = {
        f"{arch}.block_count", f"{arch}.embedding_length",
        f"{arch}.feed_forward_length", f"{arch}.head_count",
        f"{arch}.head_count_kv", f"{arch}.context_length",
        f"{arch}.vocab_size", "general.name", "general.description",
        "general.file_type", "general.architecture",
    }
    for key, field in r.fields.items():
        if key in skip or key.startswith("GGUF.") or key.startswith("tokenizer."):
            continue
        if not (key.startswith(f"{arch}.") or key.startswith("general.")):
            continue
        try:
            types = getattr(field, "types", None) or []
            vtype = types[-1] if types else None
            parts = getattr(field, "parts", None)
            if not parts or vtype is None:
                continue
            if key.startswith("general."):
                # GGUFWriter already emits general.* defaults; skip the copy.
                continue
            if vtype == GGUFValueType.ARRAY:
                sub = types[-2] if len(types) >= 2 else None
                vals = [
                    (p.item() if hasattr(p, "item") else p) for p in parts[4:]
                ]
                writer.add_key_value(key, vals, vtype, sub)
            else:
                p = parts[-1]
                if hasattr(p, "shape") and len(p.shape) > 0:
                    continue
                writer.add_key_value(key, p.item() if hasattr(p, "item") else p, vtype)
        except Exception as e:
            print(f"[gguf] metadata copy failed for {key}: {e}")


def _export_gguf(
    model: AbstractModel | CompressedModel,
    path: Path,
    **kwargs,
) -> Path:
    """Export to GGUF format. Set quantize='Q4_K_M' to auto-quantize."""
    import torch, numpy as np, subprocess
    from gguf import GGUFWriter

    quantize_type = kwargs.get("quantize", None)

    if isinstance(model, CompressedModel):
        backend, config = model.source_backend, model.source_config
    else:
        backend, config = model.backend, model.config

    arch = _gguf_arch(backend, config)
    state_dict = _reconstruct_for_export(model) if isinstance(model, CompressedModel) else model.state_dict

    writer = GGUFWriter(str(path), arch)
    writer.add_name(kwargs.get("name", f"HyperRetro-{arch}"))
    writer.add_description(
        f"HyperRetro compressed ({backend}). "
        f"Quantize: llama-quantize {path.name} out.gguf Q4_K_M"
    )
    n_layers = config.get("num_hidden_layers", config.get("n_layers",
                         config.get("prelude_layers", 0) + 1 + config.get("coda_layers", 0)))
    writer.add_block_count(n_layers)
    writer.add_embedding_length(config.get("hidden_size", config.get("dim", 0)))
    writer.add_feed_forward_length(config.get("intermediate_size", config.get("expert_dim", 0)))
    writer.add_head_count(config.get("num_attention_heads", config.get("n_heads", 0)))
    writer.add_head_count_kv(config.get("num_key_value_heads", config.get("n_kv_heads", 0)))
    writer.add_context_length(config.get("max_position_embeddings", config.get("max_seq_len", 4096)))
    writer.add_vocab_size(config.get("vocab_size", 0))
    # 0 = ALL_F32: prevents loaders (llama.cpp) from auto-converting our
    # mixed fp16/fp32 tensor layout (RMS norms must stay fp32).
    writer.add_file_type(0)
    if backend == "gguf":
        src = config.get("_gguf_path")
        if src and Path(src).exists():
            try:
                _copy_tokenizer_from_gguf(writer, src)
                _copy_arch_metadata_from_gguf(writer, src, arch)
            except Exception as e:
                print(f"[gguf] tokenizer copy failed: {e}")

    tensor_count = 0
    fp32_all = bool(kwargs.get("fp32", False))
    for key, tensor in sorted(state_dict.items()):
        if any(key.endswith(s) for s in (".q", ".scales", ".awq_scales", ".factored_A", ".factored_B")):
            continue
        if not hasattr(tensor, "cpu"):
            continue
        name = _hf_to_gguf_name(key, config)
        # RMS norm weights and biases must stay F32 (llama.cpp does
        # ggml_mul(f32, f32); our runtime assumes F32 biases).
        is_f32 = name.endswith("_norm.weight") or name.endswith(".bias")
        W = tensor.float().cpu().numpy()
        if fp32_all or is_f32:
            W = W.astype(np.float32)
        else:
            W = W.astype(np.float16)
        W = np.ascontiguousarray(W)
        writer.add_tensor(name, W)
        tensor_count += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    fp16_mb = path.stat().st_size / 1e6
    print(f"[gguf] {path} ({fp16_mb:.1f} MB fp16, {tensor_count} tensors)")

    if quantize_type:
        qpath = path.with_suffix(f".{quantize_type}.gguf")
        print(f"[gguf] quantizing to {quantize_type} (requires llama-quantize on PATH) ...")
        try:
            subprocess.run(["llama-quantize", str(path), str(qpath), quantize_type],
                         check=True, capture_output=True, timeout=600)
            q_mb = qpath.stat().st_size / 1e6
            print(f"[gguf] → {qpath} ({q_mb:.1f} MB {quantize_type}, {fp16_mb/q_mb:.1f}x shrink)")
            path.unlink(); qpath.rename(path)
        except FileNotFoundError:
            print(f"[gguf] ⚠️  llama-quantize not found. Install llama.cpp.")
            print(f"[gguf]    Then run: llama-quantize {path} {qpath} {quantize_type}")
        except Exception as e:
            print(f"[gguf] ⚠️  quantize failed: {e}")
    return path


def _export_safetensors(
    model: AbstractModel | CompressedModel,
    path: Path,
    **kwargs,
) -> Path:
    """Export to safetensors / HuggingFace format."""
    import torch
    import numpy as np
    from safetensors.torch import save_file
    import json

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Get state dict
    if isinstance(model, CompressedModel):
        state_dict = model.state_dict
        config = model.source_config
    else:
        state_dict = model.state_dict
        config = model.config

    # Write safetensors (handle int4 and factored keys properly)
    out_sd = {}
    for k, v in state_dict.items():
        if hasattr(v, "cpu"):
            v = v.contiguous().cpu()
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(np.ascontiguousarray(v))
        if k.endswith(".q"):
            v = v.to(torch.uint8)
        elif k.endswith(".scales") or k.endswith(".awq_scales"):
            v = v.to(torch.float16)
        elif isinstance(v, torch.Tensor) and v.dtype in (torch.float32, torch.float64):
            v = v.to(torch.float16)
        out_sd[k] = v

    save_file(out_sd, str(path / "model.safetensors"))

    # Write config
    (path / "config.json").write_text(json.dumps(config, indent=2))

    # Write manifest if compressed
    if isinstance(model, CompressedModel) and model.manifest:
        (path / "hyperretro_factored.json").write_text(
            json.dumps(model.manifest, indent=2)
        )

    size_mb = (path / "model.safetensors").stat().st_size / 1e6
    print(f"[safetensors] {path} ({size_mb:.1f} MB, {len(out_sd)} tensors)")
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gguf_arch(backend: str, config: dict) -> str:
    """Map backend + config to GGUF architecture string."""
    if backend == "openmythos":
        return "mythos"
    # HuggingFace arch detection
    arch_list = config.get("architectures", [])
    arch_map = {
        "Qwen2ForCausalLM": "qwen2",
        "LlamaForCausalLM": "llama",
        "MistralForCausalLM": "llama",
        "GPT2LMHeadModel": "gpt2",
        "Phi3ForCausalLM": "phi3",
    }
    for arch_name in arch_list:
        if arch_name in arch_map:
            return arch_map[arch_name]
    return "llama"


def _hf_to_gguf_name(hf_name: str, config: dict) -> str:
    """Map HuggingFace/OM state_dict key to GGUF tensor name."""
    name = hf_name
    if name.startswith("model."):
        name = name[len("model."):]

    # Embedding
    if name == "embed_tokens.weight" or name == "embed.weight":
        return "token_embd.weight"
    if name == "norm.weight":
        return "output_norm.weight"
    if name == "lm_head.weight" or name == "head.weight":
        return "output.weight"

    # Layer patterns
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "layers" and parts[1].isdigit():
        li = int(parts[1])
        rest = ".".join(parts[2:])
        layer_map = {
            "input_layernorm.weight": f"blk.{li}.attn_norm.weight",
            "post_attention_layernorm.weight": f"blk.{li}.ffn_norm.weight",
            "self_attn.q_proj.weight": f"blk.{li}.attn_q.weight",
            "self_attn.k_proj.weight": f"blk.{li}.attn_k.weight",
            "self_attn.v_proj.weight": f"blk.{li}.attn_v.weight",
            "self_attn.o_proj.weight": f"blk.{li}.attn_output.weight",
            "mlp.gate_proj.weight": f"blk.{li}.ffn_gate.weight",
            "mlp.up_proj.weight": f"blk.{li}.ffn_up.weight",
            "mlp.down_proj.weight": f"blk.{li}.ffn_down.weight",
        }
        if rest in layer_map:
            return layer_map[rest]

    # OpenMythos patterns
    if "prelude" in name or "coda" in name or "recurrent" in name:
        # Use a simple fallback mapping
        return name.replace(".", "_")

    return name


def _reconstruct_for_export(compressed: CompressedModel) -> dict:
    """Reconstruct dense weights from a CompressedModel for GGUF export.

    GGUF needs dense weights — factored and int4 matrices must be
    materialized back to fp16.
    """
    import torch
    import numpy as np
    from hyperretro.hf.factor_int4 import unpack_int4_rows
    from hyperretro.hf.factor_quantize import dequantize_blockwise_int4

    dense_sd = {}
    factored_map = {}
    int4_map = {}

    for key, tensor in compressed.state_dict.items():
        if key.endswith(".q"):
            base = key[:-2]
            scales_key = base + ".scales"
            awq_key = base + ".awq_scales"
            if scales_key in compressed.state_dict:
                packed = compressed.state_dict[key].cpu().numpy() if hasattr(compressed.state_dict[key], "cpu") else np.asarray(compressed.state_dict[key])
                scales = compressed.state_dict[scales_key].cpu().numpy() if hasattr(compressed.state_dict[scales_key], "cpu") else np.asarray(compressed.state_dict[scales_key])
                awq = None
                if awq_key in compressed.state_dict:
                    awq = compressed.state_dict[awq_key].cpu().numpy() if hasattr(compressed.state_dict[awq_key], "cpu") else np.asarray(compressed.state_dict[awq_key])
                int4_map[base] = (packed, scales, awq)
        elif key.endswith(".scales") or key.endswith(".awq_scales"):
            continue
        elif key.endswith(".factored_A"):
            base = key[:-len(".factored_A")]
            if base + ".factored_B" in compressed.state_dict or base + ".factored_B.q" in compressed.state_dict:
                factored_map[base] = [compressed.state_dict[key], None]
        elif key.endswith(".factored_B"):
            base = key[:-len(".factored_B")]
            if base in factored_map:
                factored_map[base][1] = compressed.state_dict[key]

    # Detect int4 factored pairs
    for base in list(int4_map.keys()):
        if base.endswith(".factored_A"):
            prefix = base[:-len(".factored_A")]
            b_base = prefix + ".factored_B"
            if b_base in int4_map:
                packed_a, scales_a, awq_a = int4_map[base]
                packed_b, scales_b, awq_b = int4_map[b_base]
                a_ncols = packed_a.shape[1] * 2
                b_ncols = packed_b.shape[1] * 2
                W_qa = unpack_int4_rows(packed_a, a_ncols)
                W_qb = unpack_int4_rows(packed_b, b_ncols)
                A = dequantize_blockwise_int4(W_qa, scales_a)
                B = dequantize_blockwise_int4(W_qb, scales_b)
                if awq_a is not None:
                    A = A / awq_a[None, :]
                if awq_b is not None:
                    B = B / awq_b[None, :]
                factored_map[prefix] = [torch.from_numpy(A), torch.from_numpy(B)]

    # Materialize factored
    for base, (A_t, B_t) in factored_map.items():
        A = A_t.float().cpu().numpy() if hasattr(A_t, "cpu") else np.asarray(A_t)
        B = B_t.float().cpu().numpy() if hasattr(B_t, "cpu") else np.asarray(B_t)
        W = (B.astype(np.float64) @ A.astype(np.float64)).astype(np.float16)
        dense_sd[base + ".weight"] = torch.from_numpy(W)

    # Dequantize non-factored int4
    for base, (packed, scales, awq) in int4_map.items():
        if any(base.startswith(fb) for fb in factored_map):
            continue
        n_cols = packed.shape[1] * 2
        W_q = unpack_int4_rows(packed, n_cols)
        W = dequantize_blockwise_int4(W_q, scales)
        if awq is not None:
            W = W / awq[None, :]
        # restore the ".weight" suffix the compression step strips
        dense_sd[base + ".weight"] = torch.from_numpy(W.astype(np.float16))

    # Copy remaining
    for key, tensor in compressed.state_dict.items():
        if key in dense_sd:
            continue
        if any(key.endswith(s) for s in (".q", ".scales", ".awq_scales",
                                          ".factored_A", ".factored_B")):
            continue
        dense_sd[key] = tensor

    return dense_sd
