# HyperTensor — ARM (Apple Silicon)

The full HyperTensor project, ported to and verified on **ARM64
(Apple Silicon, macOS)**. This repository mirrors the original HyperTensor
and includes an ARM-native C runtime, ARM NEON kernels, and ARM-verified
claim evidence. See [`ARM_CLAIMS.md`](ARM_CLAIMS.md) for the claim-parity
matrix.

## What runs on ARM today

| Layer | Status |
|---|---|
| C runtime `geodessical` (GGUF inference, NEON backend, arm64 JIT, Accelerate) | OK Builds & runs — numerics match llama.cpp oracle |
| Apple ARM NEON dotprod (SDOT) GEMV kernels (Q5_0/Q8_0/Q4_K/Q6_K, SMP split) | OK **4.9 → 88.5 tok/s decode** (~18×), parity kept |
| C tests (kernels, model_meta, chat, tokenizer) | OK All pass |
| Python suites (pytest: 108/108, audit 33/33, external verification 28/28) | OK Pass |
| ht-repro REST + SQLite | OK Pass — `/health`, `/gpu`, `/sort`, `/jobs`, and real `/infer` (gpt2 generation on ARM) |
| AGT 50K primes (Riemann) | OK 100%, k90=k95=1, 676× separation |
| Jury scaling | OK 547× @128 jurors (claim: 53×) |
| HyperRetro fused dual-Q8 GEMV (NEON SDOT) | OK 8.87× vs two Q8 GEMVs (claim: 2.3×) |
| CECI grafting (ht-graft) | OK 5 grafts built, "GRAFTING WORKS" |
| External verification on real 1.5B model (claim: 14/14) | OK **14/14 PASS** (UGT overlap 0.970 vs 0.968 claim) |
| HyperRetro GRC compression (rank 1024, all 28 layers) | OK Compressed HF checkpoint loads + runs: PPL 14.58 vs 12.94 baseline |
| E2E: GGUF → compress (FFN SVD + int4) → GGUF → ARM runtime | OK **Verified vs oracle**: exports run on geodessical AND llama.cpp; tokenizer bit-identical to llama.cpp. 0.5B (PPL ours / llama.cpp): int4-only 10.98 / 10.66; ffn_rank=1024+int4 14.89 / 14.20 (in-memory) and 11.66 / 11.09 (streaming, 861 MB, 46.4 tok/s). 1.5B: int4-only 6.94 / 6.73; ffn_rank=2048+int4 streaming 7.73 / 7.35 (2729 MB, 15.9 tok/s). Decode: fp16 37.7 tok/s → Q8_0 108.7 / Q4_K_M 109.2 tok/s |
| civilized-HyperTensor models module | OK Imported, fixed for ARM, tests pass |
| Unified memory: Qwen2.5-1.5B Q4_K_M (1.07 GB) | OK 32.8 tok/s decode, PPL 16.68, oracle parity |
| i8mm (SMMLA) + GCD parallel hyperretro kernel | OK 12.78× vs two Q8 GEMVs (claim 2.3×) |

## Build the C runtime

```bash
./build_host_arm.sh                 # -> build_host_arm/geodessical
./build_tests_arm.sh                # C test suite
```

Run inference (example model: Qwen2.5 0.5B Instruct Q4_K_M GGUF):

```bash
./build_host_arm/geodessical models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    -p "Hello" -n 64 --temp 0.5
./build_host_arm/geodessical models/qwen2.5-0.5b-instruct-q4_k_m.gguf --ppl-eval
```

## Python (ARM)

```bash
# venv with py3.11 + torch + pytest (see scripts for the full set)
.venv311/bin/pip install -e. --no-build-isolation --no-deps
.venv311/bin/python -m pytest tests/
.venv311/bin/python scripts/verify_external.py
.venv311/bin/python scripts/jury_scaling.py
.venv311/bin/python hyperretro/bench/run.py kernel --rows 4096 --in-dim 4096
.venv311/bin/python -m ht_repro.cli serve --port 8765 --no-browser
```

## Compression / end-to-end pipeline

Compress any llama.cpp GGUF and run the result on the ARM runtime. Two modes:

```bash
# 1) In-memory (fine up to ~few-B params): load, SVD+int4, export
.venv311/bin/python scripts/e2e.py compress \
    models/qwen2.5-0.5b-instruct-q4_k_m.gguf out.gguf \
    --ffn-rank 1024 --int4

# 2) Streaming (peak memory = one tensor; for very large models):
#    FFN/attention matrices are SVD-factored and written as fp16;
#    all other tensors are byte-copied in their source quantized type.
.venv311/bin/python scripts/e2e.py stream \
    models/qwen2.5-0.5b-instruct-q4_k_m.gguf out.gguf \
    --ffn-rank 1024 --int4

# Verify (built-in WikiText perplexity on geodessical)
.venv311/bin/python scripts/e2e.py verify out.gguf

# Optional re-quantize (llama-quantize vendored in third_party/llama.cpp)
.venv311/bin/python scripts/e2e.py quantize out.gguf out-q4km.gguf Q4_K_M
```

Same pipeline as a Python API:

```python
from hyperretro.models import load_model, compress_model, export_model, stream_compress_gguf

# in-memory
m = load_model("in.gguf")
cm = compress_model(m, ffn_rank=1024, attn_rank=0, int4=True, int4_block_size=128)
export_model(cm, "out.gguf", format="gguf")

# streaming
stream_compress_gguf("in.gguf", "out.gguf", ffn_rank=1024, int4=True)
```

Verified numbers are in [`benchmarks/arm/`](benchmarks/arm/) and the
claim-parity matrix in [`ARM_CLAIMS.md`](ARM_CLAIMS.md).

## NEON kernel rebuild

```bash
clang -O3 -shared -fPIC -march=armv8.4-a+dotprod \
  -o hyperretro/kernels/csrc/cpu/hyperretro_cpu_neon.dylib \
  hyperretro/kernels/csrc/cpu/hyperretro_cpu_neon.c
```

## Notes

- GGUF Qwen2-family models use NeoX (half-rotation) RoPE — handled
  automatically by the runtime.
- `HT_BACKEND=cpu` forces the scalar/NEON CPU path; default on arm64
  auto-selects the ARM NEON backend.
- Model files are not committed (see `models/` in `.gitignore`).
