#!/bin/bash
# =============================================================================
# Geodessical — Host-Mode Build Script for Apple Silicon (arm64) / macOS
#
# ARM-native equivalent of build_host.ps1: builds the geodessical inference
# runtime with clang targeting arm64-apple-macos, linking Apple's Accelerate
# framework for BLAS, plus the ARM64 NEON JIT and ARM backend kernels.
#
# Usage: ./build_host_arm.sh [clean]
# Output: build_host_arm/geodessical
# =============================================================================
set -e
cd "$(dirname "$0")"

BUILD="build_host_arm"
OUT="$BUILD/geodessical"

# macOS SDK vecLib headers (cblas.h) live inside the Accelerate framework.
SDKROOT="$(xcrun --show-sdk-path 2>/dev/null || echo '')"
VECLIB_INC="${SDKROOT}/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers"

CFLAGS=(
    -O2
    -DGEODESSICAL_HOSTED=1
    -DHT_ARM64=1
    -DHT_DEBUG_FWD=1
    -Ihost/shims
    -I.
    -Ihost
    ${VECLIB_INC:+-I"$VECLIB_INC"}
    -Wno-unused-function -Wno-unused-variable -Wno-format
    -Wno-incompatible-pointer-types -Wno-int-conversion
    -Wno-sign-compare -Wno-missing-field-initializers
    -Wno-comment
    -Wno-unused-parameter
    -Wno-attributes
)

SOURCES=(
    "host/hal.c"
    "host/main.c"
    "host/api_server.c"
    "host/gd_daemon.c"
    "host/mcp_server.c"
    "runtime/nn/llm.c"
    "runtime/nn/gguf.c"
    "runtime/nn/backend.c"
    "runtime/nn/backend_arm.c"
    "runtime/nn/model_meta.c"
    "runtime/nn/tensor_bridge.c"
    "runtime/nn/mod_package.c"
    "runtime/nn/token_comm.c"
    "runtime/nn/hf_download.c"
    "runtime/nn/flash_attn.c"
    "runtime/nn/axiom_linalg.c"
    "runtime/nn/axiom_geo.c"
    "runtime/nn/axiom_beta.c"
    "runtime/nn/axiom_exploit.c"
    "runtime/nn/axiom_gauge.c"
    "runtime/nn/axiom_vis.c"
    "runtime/nn/online_basis.c"
    "runtime/nn/geo_research.c"
    "runtime/nn/mcr_compress.c"
    "runtime/nn/thermal_rank.c"
    "runtime/nn/qspec_basis.c"
    "runtime/nn/jit_kernel.c"
    "runtime/jit/arm64_jit.c"
    "runtime/jit/llm_jit.c"
    "runtime/arm/arm_shims.c"
)

LDFLAGS=(
    "-framework" "Accelerate"
    "-framework" "CoreFoundation"
    "-lpthread"
    "-lm"
)

if [ "$1" = "clean" ]; then
    echo "Cleaning $BUILD..."
    rm -rf "$BUILD"
fi

mkdir -p "$BUILD"

echo "  ARM64 host build (clang, Accelerate, NEON) ..."
"${CC:-clang}" "${CFLAGS[@]}" -o "$OUT" "${SOURCES[@]}" "${LDFLAGS[@]}"

echo ""
echo "  Build OK: $OUT"
ls -la "$OUT"
