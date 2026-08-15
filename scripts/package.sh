#!/bin/bash
# =============================================================================
# HyperTensor ARM — production packaging
#
# Builds the geodessical ARM runtime, ships it inside the Python package,
# and installs the `hyperarm` CLI. Result is a pip-installable tool that
# works from anywhere (no checkout needed after install).
#
# Usage:
#   ./scripts/package.sh [--venv PYTHON] [--no-build] [--wheel]
#     PYTHON   python interpreter to install into (default: .venv311/bin/python)
#     --no-build   skip the C rebuild (use the existing binary)
#     --wheel      additionally build a distributable .whl into dist/
# =============================================================================
set -e
cd "$(dirname "$0")/.."

PY=.venv311/bin/python
NO_BUILD=0
WHEEL=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv) PY="$2"; shift 2 ;;
        --no-build) NO_BUILD=1; shift ;;
        --wheel) WHEEL=1; shift ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

if [[ $NO_BUILD -eq 0 ]]; then
    echo "[package] building geodessical (clang, arm64)"
    ./build_host_arm.sh
fi

echo "[package] shipping runtime into hyperretro/bin/"
mkdir -p hyperretro/bin
cp build_host_arm/geodessical hyperretro/bin/geodessical
chmod +x hyperretro/bin/geodessical

echo "[package] installing into $PY"
"$PY" -m pip install --upgrade pip >/dev/null
"$PY" -m pip install . >/dev/null

if [[ $WHEEL -eq 1 ]]; then
    echo "[package] building wheel into dist/"
    "$PY" -m pip wheel . --no-deps -w dist/ 2>/dev/null || \
        "$PY" -m pip wheel . --no-deps --no-build-isolation -w dist/
    ls dist/*.whl | tail -3
fi

echo "[package] done. Try:"
echo "  hyperarm doctor"
echo "  hyperarm verify models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
echo "  hyperarm infer   models/qwen2.5-0.5b-instruct-q4_k_m.gguf -p \"Hello\""
echo "  hyperarm stream  in.gguf out.gguf --ffn-rank 0 --attn-rank 1024 --sink 4 --int4"
