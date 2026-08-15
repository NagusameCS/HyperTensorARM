# End-to-End Pipeline: GGUF -> HyperRetro -> GGUF -> geodessical

This document describes the complete compression pipeline on Apple Silicon,
its two operating modes, the verified configurations, and how to reproduce
every number.

## Pipeline

```
llama.cpp GGUF (any quantized type)
        |
        |  load / stream
        v
HyperRetro compression
  - FFN matrices (blk.N.ffn_gate/up/down.weight): truncated SVD to rank k,
    factors optionally block-quantized to int4, reconstructed as fp16
  - attention Q/K/V (blk.N.attn_q/k/v.weight): GRC shared-basis compression
    (attn_rank > 0) — the original HyperTensor method: build a shared basis
    from the sum Gram matrix of Q/K/V, project all three onto the top-k
    subspace, optionally restore the top-T magnitude columns verbatim
    (sink-aware, --sink T). o_proj is never SVD-factored (oracle runs show
    it hurts; it stays int4 / byte-copied).
  - everything else: unchanged (or byte-copied, streaming mode)
        |
        |  export / stream-write
        v
Output GGUF (loadable by geodessical AND llama.cpp)
        |
        |  optional: llama-quantize -> Q8_0 / Q4_K_M / ...
        v
geodessical --ppl-eval / generation  (ARM NEON runtime)
```

## Modes

### In-memory (`compress`)

`load_model` -> `compress_model` -> `export_model`. Simple, fine for models up
to a few billion parameters. Peak memory is the float32 state dict plus SVD
scratch.

```bash
python scripts/e2e.py compress in.gguf out.gguf --ffn-rank 1024 --int4
```

### Streaming (`stream`)

Single pass over the source GGUF; peak memory is bounded by one tensor.
Intended for models too large to materialize in float32 (e.g. 70B-class).

- FFN/attention matrices: dequantized one at a time, SVD-factored
  (exact truncated SVD below 64M elements, randomized `torch.svd_lowrank`
  above), int4-quantized, reconstructed, written as fp16.
- All other tensors: byte-exact copy preserving the source quantized type.
- Two-pass write: tensor infos are registered first, then data streams out.

```bash
python scripts/e2e.py stream in.gguf out.gguf --ffn-rank 1024 --int4
```

Python API: `hyperretro.models.stream_compress_gguf(...)`.

## Verified configurations (Qwen2.5-0.5B-Instruct, WikiText-2 slice, 512 tokens)

PPL is measured twice: on geodessical (`--ppl-eval`) and on a fresh llama.cpp
build (`third_party/llama.cpp`, per-token NLL harness). Tokenization is
bit-identical between the two runtimes (560/560 tokens).

| Config | geodessical PPL | llama.cpp PPL | Note |
|---|---|---|---|
| original Q4_K_M | 7.94 | 7.58 | baseline |
| int4-only (all 2D matrices, block 128) | 10.98 | 10.66 | no SVD |
| ffn_rank=1024 + int4, in-memory export | 14.89 | 14.20 | all weights pass through int4+fp16 |
| ffn_rank=1024 + int4, streaming export | 11.66 | 11.09 | non-FFN tensors byte-copied; 861 MB |
| ffn_rank=256 + int4 | ~2M | ~1.8M | rank too low; pipeline stays oracle-consistent |

Decode throughput (geodessical, 0.5B compressed model):

| Format | tok/s | PPL |
|---|---|---|
| fp16 (exported) | 37.7 | 10.98 |
| Q8_0 (re-quantized) | 108.7 | 10.98 |
| Q4_K_M (re-quantized) | 109.2 | 11.24 |

1.5B: int4-only PPL 6.94 (llama 6.73). ffn_rank=2048 + int4 streaming gives
PPL 7.73 (llama 7.35), 2729 MB, 15.9 tok/s decode. Plain truncated SVD at rank
1024 is too aggressive for the 8960-dim FFN (PPL ~3.5K); the GRC method
(shared basis, sink-aware) is required at lower ranks — see below.

## GRC attention compression (ported from the original, verified)

Q/K/V are compressed jointly per layer with a shared basis (rank k) instead
of per-tensor SVD. In-memory exports int4-quantize every 2D matrix; streaming
exports byte-copy everything except the GRC-compressed tensors.

| Config | geodessical PPL | llama.cpp PPL | Note |
|---|---|---|---|
| 1.5B attn_rank=1024 + int4, in-memory | 7.46 | 7.20 | vanilla GRC, sink_T=0 |
| 1.5B attn_rank=1024 + int4, in-memory | 7.68 | 7.45 | sink-aware GRC, sink_T=4 |
| 1.5B attn_rank=1024 + int4, streaming | 6.40 | 6.14 | 1242.5 MB, 52 s compress |
| 0.5B attn_rank=600 + int4, streaming | 9.48 | 9.01 | sink_T=4, 523.4 MB |
| 0.5B attn_rank=256 + int4, in-memory | 68.5 / 58.7 | — | sink_T=0 / sink_T=4 (rank too low for this model) |
| 0.5B attn_rank=1024 + int4 (full rank) | 10.96 | 10.59 | identity sanity check = int4-only |

1.5B baselines for reference: original 5.99 (llama 5.84), int4-only 6.94
(llama 6.73). GRC at k=1024 (2/3 of d=1536) costs only +0.4 PPL vs int4-only
in streaming mode — the original's claim-level method (their numbers: GRC
k=1024 PPL 14.58 vs 12.94 baseline on their own eval set, no int4). Sink-aware
helps at very low ranks (0.5B k=256: 68.5 -> 58.7); at k=1024 on 1.5B vanilla
is marginally better. Report per-config.

## Reproducing

```bash
./build_host_arm.sh                     # C runtime
./build_tests_arm.sh                    # C test suite
.venv311/bin/python -m pytest tests/    # 114 tests incl. emoji-free guard

.venv311/bin/python scripts/e2e.py compress \
    models/qwen2.5-0.5b-instruct-q4_k_m.gguf out.gguf --ffn-rank 1024 --int4
.venv311/bin/python scripts/e2e.py stream \
    in.gguf out.gguf --ffn-rank 0 --attn-rank 1024 --sink 4 --int4
.venv311/bin/python scripts/e2e.py verify out.gguf
.venv311/bin/python scripts/e2e.py quantize out.gguf out-q4km.gguf Q4_K_M
```

## Oracle protocol

- llama.cpp oracle: `third_party/llama.cpp` (gitignored) built with
  `cmake -B build -DCMAKE_BUILD_TYPE=Release`; `llama-quantize`,
  `llama-perplexity`, plus the per-token NLL harness pattern used in
  `benchmarks/arm/`.
- PyTorch oracle: load the dense reconstruction into a vanilla
  `Qwen2ForCausalLM` and compare top-k logits.
- Both runtimes must agree on PPL within quantization noise (~0.05 NLL).
