# HyperTensor — ARM (Apple Silicon)

The full HyperTensor project, ported to and verified on **ARM64
(Apple Silicon, macOS)**. This repository mirrors the original HyperTensor
and includes an ARM-native C runtime, ARM NEON kernels, and ARM-verified
claim evidence. See [`ARM_CLAIMS.md`](ARM_CLAIMS.md) for the claim-parity
matrix.

## What runs on ARM today

| Layer | Status |
|---|---|
| C runtime `geodessical` (GGUF inference, NEON backend, arm64 JIT, Accelerate) | ✅ Builds & runs — numerics match llama.cpp oracle |
| Apple ARM NEON dotprod (SDOT) GEMV kernels (Q5_0/Q8_0/Q4_K/Q6_K, SMP split) | ✅ **4.9 → 88.5 tok/s decode** (~18×), parity kept |
| C tests (kernels, model_meta, chat, tokenizer) | ✅ All pass |
| Python suites (pytest: 108/108, audit 33/33, external verification 28/28) | ✅ Pass |
| ht-repro REST + SQLite | ✅ Pass — `/health`, `/gpu`, `/sort`, `/jobs`, and real `/infer` (gpt2 generation on ARM) |
| AGT 50K primes (Riemann) | ✅ 100%, k90=k95=1, 676× separation |
| Jury scaling | ✅ 547× @128 jurors (claim: 53×) |
| HyperRetro fused dual-Q8 GEMV (NEON SDOT) | ✅ 8.87× vs two Q8 GEMVs (claim: 2.3×) |
| CECI grafting (ht-graft) | ✅ 5 grafts built, "GRAFTING WORKS" |
| External verification on real 1.5B model (claim: 14/14) | ✅ **14/14 PASS** (UGT overlap 0.970 vs 0.968 claim) |
| HyperRetro GRC compression (rank 1024, all 28 layers) | ✅ Compressed HF checkpoint loads + runs: PPL 14.58 vs 12.94 baseline |
| E2E: GGUF → compress (FFN+int4) → GGUF → ARM runtime | ✅ Compressed GGUF runs on geodessical |
| civilized-HyperTensor models module | ✅ Imported, fixed for ARM, tests pass |
| Unified memory: Qwen2.5-1.5B Q4_K_M (1.07 GB) | ✅ 32.8 tok/s decode, PPL 16.68, oracle parity |
| i8mm (SMMLA) + GCD parallel hyperretro kernel | ✅ 12.78× vs two Q8 GEMVs (claim 2.3×) |

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
.venv311/bin/pip install -e . --no-build-isolation --no-deps
.venv311/bin/python -m pytest tests/
.venv311/bin/python scripts/verify_external.py
.venv311/bin/python scripts/jury_scaling.py
.venv311/bin/python hyperretro/bench/run.py kernel --rows 4096 --in-dim 4096
.venv311/bin/python -m ht_repro.cli serve --port 8765 --no-browser
```

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
