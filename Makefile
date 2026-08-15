# HyperTensor ARM — standard targets for humans and AI agents.
# See AGENTS.md for context.

PY     ?= .venv311/bin/python
MODEL  ?= models/qwen2.5-0.5b-instruct-q4_k_m.gguf
IN     ?= $(MODEL)
OUT    ?= outputs/compressed.gguf
RANK   ?= 1024
ATTN   ?= 0
SINK   ?= 0

.PHONY: help build test pytest all infer ppl compress stream verify quantize clean package

help:
	@echo "Targets:"
	@echo "  build     - build C runtime + libht_arm.a (./build_host_arm.sh)"
	@echo "  test      - C test suite (39/39)"
	@echo "  pytest    - Python test suite (114/114)"
	@echo "  all       - build + test + pytest"
	@echo "  infer     - geodessical generation on MODEL"
	@echo "  ppl       - geodessical --ppl-eval on MODEL"
	@echo "  compress  - in-memory compress (IN -> OUT, --ffn-rank RANK --int4)"
	@echo "  stream    - streaming compress (IN -> OUT, --ffn-rank RANK --int4)"
	@echo "  verify    - verify OUT with geodessical --ppl-eval"
	@echo "  quantize  - llama-quantize OUT to OUT.q4km.gguf (Q4_K_M)"
	@echo "  package   - ship runtime in package + pip install (hyperarm CLI)"
	@echo "  clean     - remove build directories"
	@echo ""
	@echo "Variables: PY, MODEL, IN, OUT, RANK, ATTN, SINK"

build:
	./build_host_arm.sh

test:
	./build_tests_arm.sh

pytest:
	$(PY) -m pytest tests/ -q

all: build test pytest

infer:
	./build_host_arm/geodessical $(MODEL) -p "The capital of France is" -n 32

ppl:
	./build_host_arm/geodessical $(MODEL) --ppl-eval

compress:
	$(PY) scripts/e2e.py compress $(IN) $(OUT) --ffn-rank $(RANK) --attn-rank $(ATTN) --sink $(SINK) --int4

stream:
	$(PY) scripts/e2e.py stream $(IN) $(OUT) --ffn-rank $(RANK) --attn-rank $(ATTN) --sink $(SINK) --int4

verify:
	$(PY) scripts/e2e.py verify $(OUT)

quantize:
	third_party/llama.cpp/build/bin/llama-quantize $(OUT) $(OUT).q4km.gguf Q4_K_M 8

package:
	./scripts/package.sh --venv $(PY)

clean:
	rm -rf build_host_arm build_asan build_tests_arm
