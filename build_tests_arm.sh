#!/bin/bash
# =============================================================================
# Build & run the C test suite on Apple Silicon (arm64).
# Produces build_host_arm/libht_arm.a + test binaries, then runs them.
# Usage: ./build_tests_arm.sh [clean]
# =============================================================================
set -e
cd "$(dirname "$0")"

BUILD="build_host_arm"
SDKROOT="$(xcrun --show-sdk-path 2>/dev/null || echo '')"
VECLIB_INC="${SDKROOT}/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers"

CFLAGS=(
    -O2
    -DGEODESSICAL_HOSTED=1
    -DHT_ARM64=1
    -DHT_DEBUG_FWD
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

LDFLAGS=(
    "-framework" "Accelerate"
    "-framework" "CoreFoundation"
    "-lpthread"
    "-lm"
)

LIB_SOURCES=(
    "host/hal.c"
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

TESTS=(
    "test_kernels"
    "test_model_meta"
    "test_tokenizer"
    "test_chat"
)

mkdir -p "$BUILD"

echo "  Building libht_arm.a ..."
rm -f "$BUILD/libht_arm.a"
OBJS=()
for src in "${LIB_SOURCES[@]}"; do
    obj="$BUILD/$(basename "$src" .c).o"
    "${CC:-clang}" "${CFLAGS[@]}" -c "$src" -o "$obj"
    OBJS+=("$obj")
done
"${AR:-ar}" rcs "$BUILD/libht_arm.a" "${OBJS[@]}"

for t in "${TESTS[@]}"; do
    echo "  Building $t ..."
    "${CC:-clang}" "${CFLAGS[@]}" "tests/runtime/$t.c" -o "$BUILD/$t" \
        "$BUILD/libht_arm.a" "${LDFLAGS[@]}"
done

echo ""
echo "  Running C test suite ..."
FAILED=0
for t in "${TESTS[@]}"; do
    echo "  --- $t ---"
    if "$BUILD/$t"; then
        echo "  PASS: $t"
    else
        echo "  FAIL: $t"
        FAILED=1
    fi
done

exit $FAILED
