# AGENTS.md — HyperTensor ARM

This file is for AI coding agents (Copilot, Claude Code, Codex, Cursor) and for
humans who want the machine-readable summary of this repository. Read this
before making changes.

## What this repository is

HyperTensor ported to Apple Silicon (arm64). It contains:

1. A C inference runtime (`geodessical`) with NEON kernels, verified against
   llama.cpp as a numeric oracle.
2. A Python stack (`hyperretro`) for loading, compressing (SVD + int4) and
   exporting GGUF models, including a streaming compressor with peak memory
   bounded by one tensor.
3. Claim parity evidence: every headline claim of the original HyperTensor is
   re-verified on ARM in `ARM_CLAIMS.md`, with machine-readable results in
   `benchmarks/arm/*.json`.

Reference clones `civilized-HyperTensor/`, `HyperTensor-original/` and
`third_party/llama.cpp/` are gitignored references — do not edit them.

## Commands that must keep working

```bash
./build_host_arm.sh                 # builds build_host_arm/geodessical
./build_tests_arm.sh                # builds libht_arm.a + C tests (39/39)
.venv311/bin/python -m pytest tests/  # 114 tests, includes the emoji guard

# Inference + eval (verified models in models/)
./build_host_arm/geodessical models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    -p "Hello" -n 32
./build_host_arm/geodessical models/qwen2.5-0.5b-instruct-q4_k_m.gguf --ppl-eval

# End-to-end compression (installed as `hyperarm`, or scripts/e2e.py)
.venv311/bin/python scripts/e2e.py compress in.gguf out.gguf --ffn-rank 1024 --int4
.venv311/bin/python scripts/e2e.py stream   in.gguf out.gguf --ffn-rank 1024 --int4
.venv311/bin/python scripts/e2e.py stream   in.gguf out.gguf --ffn-rank 0 --attn-rank 1024 --sink 4 --int4  # GRC attention
.venv311/bin/python scripts/e2e.py verify   out.gguf
.venv311/bin/python scripts/e2e.py quantize out.gguf out-q4km.gguf Q4_K_M

# Packaging (prod install of the hyperarm CLI with the runtime shipped in-wheel)
./scripts/package.sh --venv .venv311/bin/python   # build C + ship binary + pip install .
make package
hyperarm doctor        # env status; verify/infer/ppl work with only numpy+gguf
hyperarm install-runtime --repo .   # build + copy geodessical to ~/.hyperarm
```

## Repo map

| Path | What it is |
|---|---|
| `runtime/nn/llm.c` | Model forward, GEMV kernels, tokenizer (large file). NEON fast path near `llm_gemv`; BPE tokenizer near `llm_tokenize_segment`; vocab + merge-rank tables near `llm_build_vocab` |
| `runtime/nn/flash_attn.c` | Flash attention decode (SMP head split) |
| `runtime/nn/gguf.c` | GGUF parser |
| `host/main.c` | `geodessical` CLI (`--ppl-eval`, generation) |
| `host/hal.c` | SMP (8 CPUs), memory, timers |
| `hyperretro/models/` | Python model API: `load_model`, `compress_model`, `export_model`, `stream_compress_gguf`, `e2e_cli` (the `hyperarm` command), `runtime.py` (geodessical discovery) |
| `hyperretro/bin/geodessical` | compiled ARM runtime shipped in the wheel — re-copy after C changes |
| `hyperretro/hf/` | SVD factoring (`factored.py`), int4 quant (`factor_int4.py`, `factor_quantize.py`) |
| `ht_repro/` | REST + SQLite service |
| `scripts/` | Experiments, verification harnesses, `e2e.py` shim |
| `tests/` | pytest suite incl. `test_no_emoji.py` |
| `benchmarks/arm/` | Evidence JSONs — gitignored, use `git add -f` |
| `docs/E2E_PIPELINE.md` | Pipeline modes, verified configs, oracle protocol |
| `ARM_CLAIMS.md` | Claim parity matrix (plain-text status markers only) |

## Verified pipeline numbers (do not regress)

PPL measured twice: `geodessical --ppl-eval` and a fresh llama.cpp build
(`third_party/llama.cpp`). Both must agree within ~0.05 NLL.

| Config (Qwen2.5-0.5B) | PPL ours | PPL llama.cpp |
|---|---|---|
| original Q4_K_M | 7.94 | 7.58 |
| int4-only | 10.98 | 10.66 |
| ffn_rank=1024 + int4, streaming | 11.66 | 11.09 |
| ffn_rank=256 + int4 | ~2M (rank too low) | ~1.8M |
| attn_rank=600 (GRC) + int4, streaming, sink 4 | 9.48 | 9.01 |

1.5B: original 5.99/5.84; int4-only 6.94/6.73; ffn_rank=1024 breaks; ffn_rank=2048
streaming 7.73/7.35; **GRC attn_rank=1024+int4 streaming 6.40/6.14** (in-memory
7.46/7.20 vanilla, 7.68/7.45 sink 4).

Decode throughput: fp16 37.7 tok/s, Q8_0 108.7, Q4_K_M 109.2 (0.5B).

## Rules

1. **Oracle parity first.** Any change to kernels, the tokenizer, or the export
   path must be re-verified against llama.cpp (and for exports, against a
   PyTorch dense reconstruction). The per-token NLL harness pattern lives in
   `docs/E2E_PIPELINE.md`.
2. **No emoji** in tracked docs or code. `tests/test_no_emoji.py` enforces this;
   use the plain words `OK`, `PENDING`, `WARN` for status markers.
3. **C code**: run `./build_tests_arm.sh` (it deletes `build_host_arm/libht_arm.a`
   first — always rebuild the archive after touching runtime sources).
4. **Python code**: run `.venv311/bin/python -m pytest tests/`.
5. `benchmarks/` is gitignored — force-add evidence with `git add -f`.
6. Keep the claims matrix honest: mark machine-dependent results as `WARN` and
   GPU-only results as `PENDING T3`.

## Pitfalls that have already burned us

- `ffn_rank` silently does nothing unless tensor names match the factor
  patterns. GGUF names are `blk.N.ffn_gate/up/down.weight`; HF names are
  `gate_proj.` etc. Always assert `manifest["ffn"]` count after compressing.
- gguf-python `Field.parts`: ARRAY fields store per-element offsets in
  `field.data` — values are at `parts[data[i]]`. Scalar values are `parts[-1]`.
- `tokenizer.ggml.merges` entries are space-separated pairs (`"t h"`), not
  concatenations. Rank = list index.
- Qwen2.5-Instruct has real trained QKV biases even though its config omits
  `bias`. Exports must keep biases and write them as F32; the runtime adds
  biases whenever the tensor exists (llama.cpp semantics).
- Exported GGUFs: `file_type=0`, norm weights and biases F32, everything else
  fp16 — otherwise llama.cpp converts F32 norms to F16 and crashes.
- The BPE tokenizer must be bit-identical to llama.cpp. It starts from
  byte-level symbols (GPT-2 `bytes_to_unicode`, Latin-1 identity as 2-byte
  UTF-8) and merges pairs in merge-rank order.
- SMP splits: BSP takes the first chunk; worker ranges start after it. A wrong
  split silently corrupts results only when SMP is on — validate NLL with and
  without `hal_init()`.
- gguf-python writer: when streaming with `write_tensor_data` (interleaved),
  do NOT call `write_tensors_to_file` afterwards — it re-enters the TI state
  and raises. `write_tensor_data` already handles alignment padding.
- llama-quantize refuses to requantize legacy types (Q4_0/Q5_0/Q8_0) even with
  `--allow-requantize`; it can only requantize k-quants/i-quants.
- Sink-aware GRC (`--sink`) currently applies to the GGUF/generic path
  (dense projection + sink restore). The HF factored path (`factor_attn_state_dict`)
  uses shared-basis factoring without sink restore — extending it needs the
  rank-k+T trick (append sink columns to A/B).

## Definition of done for a change

- [ ] C changes: 39/39 C tests pass.
- [ ] Python changes: 114 pytest tests pass (incl. emoji guard).
- [ ] Numeric changes: oracle parity shown (llama.cpp, and torch for exports).
- [ ] Docs and `ARM_CLAIMS.md` updated with measured numbers only.
- [ ] Committed with a message stating what changed and the evidence.
