# Copilot instructions for the HyperTensor ARM repository

Before making any change, read `AGENTS.md` at the repository root and follow
its rules. The essentials:

- Build the C runtime with `./build_host_arm.sh`, then run the C suite with
  `./build_tests_arm.sh` (expect 39/39).
- Run Python tests with `.venv311/bin/python -m pytest tests/` (expect 114/114).
- The end-to-end CLI is `.venv311/bin/python scripts/e2e.py`
  (`compress`, `stream`, `verify`, `quantize`); installed as `hyperarm`.
- Every change to kernels, the tokenizer, or GGUF export must be re-verified
  against the llama.cpp oracle (see `docs/E2E_PIPELINE.md`). Report measured
  numbers, never estimates.
- Do not add emoji to tracked files (`tests/test_no_emoji.py` enforces this).
- `benchmarks/` and `third_party/` are gitignored; force-add evidence with
  `git add -f`.
- Reference clones `civilized-HyperTensor/` and `HyperTensor-original/` are
  read-only.
- When unsure about a numeric behavior, compare against both loaders
  (geodessical and llama.cpp) before and after the change.
