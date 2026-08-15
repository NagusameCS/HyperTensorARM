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
  - attention matrices (blk.N.attn_q/k/v/output.weight): optional SVD
    (attn_rank > 0)
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

1.5B: int4-only PPL 6.94 (llama 6.73). Plain truncated SVD at rank 1024 is too
aggressive for its 8960-dim FFN (PPL ~3.5K); the civilized GRC method
(sink-aware) is required there.

## Reproducing

```bash
./build_host_arm.sh                     # C runtime
./build_tests_arm.sh                    # C test suite
.venv311/bin/python -m pytest tests/    # 109 tests incl. emoji-free guard

.venv311/bin/python scripts/e2e.py compress \
    models/qwen2.5-0.5b-instruct-q4_k_m.gguf out.gguf --ffn-rank 1024 --int4
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
