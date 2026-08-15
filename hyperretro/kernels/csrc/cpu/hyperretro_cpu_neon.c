/*
 * HyperRetro CPU backend — ARM64 NEON fused dual Q8_0 GEMV.
 *
 * ARM-native twin of hyperretro_cpu_avx2.c: computes
 *     out_a[i] = dequant(W_a[i]) · x
 *     out_b[i] = dequant(W_b[i]) · x
 * simultaneously, sharing the input load (reduces DRAM traffic ~16%).
 *
 * Q8_0 layout: 32-element blocks. Per block: float32 scale, int8 codes[32].
 *
 * Compile (macOS / Apple Silicon):
 *     clang -O3 -shared -fPIC -o hyperretro_cpu_neon.dylib hyperretro_cpu_neon.c
 *
 * Exports (same ABI as the AVX2 build so cpu_opt.py works unchanged):
 *     void gemv_dual_q8_0_avx2(...)   — compatibility alias
 *     void gemv_dual_q8_0_neon(...)
 *     void gemv_dual_q8_0_scalar(...)
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#define BLOCK_SIZE 32

/* --------------------------------------------------------------------------
   NEON kernel: one row of dual Q8_0 GEMV (16 int8 lanes widened per step).
   -------------------------------------------------------------------------- */
#if defined(__aarch64__)
static inline void
gemv_dual_q8_0_row_neon(const float *x, int n_blocks,
                        const float *scales_a, const int8_t *codes_a,
                        const float *scales_b, const int8_t *codes_b,
                        float *out_a, float *out_b)
{
    float32x4_t sum_a0 = vdupq_n_f32(0.0f), sum_a1 = vdupq_n_f32(0.0f);
    float32x4_t sum_b0 = vdupq_n_f32(0.0f), sum_b1 = vdupq_n_f32(0.0f);

    for (int b = 0; b < n_blocks; ++b) {
        float sa = scales_a[b];
        float sb = scales_b[b];
        const float32x4_t vsa = vdupq_n_f32(sa);
        const float32x4_t vsb = vdupq_n_f32(sb);

        for (int i = 0; i < BLOCK_SIZE; i += 16) {
            /* 16 int8 codes -> 4x int32 (two 8x8 half loads) */
            const int8x8_t ca_l = vld1_s8(codes_a + i);
            const int8x8_t ca_h = vld1_s8(codes_a + i + 8);
            const int8x8_t cb_l = vld1_s8(codes_b + i);
            const int8x8_t cb_h = vld1_s8(codes_b + i + 8);

            /* int8 -> int16 -> int32 -> float */
            const int16x8_t ca16_l = vmovl_s8(ca_l);
            const int16x8_t ca16_h = vmovl_s8(ca_h);
            const int16x8_t cb16_l = vmovl_s8(cb_l);
            const int16x8_t cb16_h = vmovl_s8(cb_h);

            const float32x4_t cfa0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(ca16_l)));
            const float32x4_t cfa1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(ca16_l)));
            const float32x4_t cfa2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(ca16_h)));
            const float32x4_t cfa3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(ca16_h)));
            const float32x4_t cfb0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(cb16_l)));
            const float32x4_t cfb1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(cb16_l)));
            const float32x4_t cfb2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(cb16_h)));
            const float32x4_t cfb3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(cb16_h)));

            /* x loads (shared) — advance across blocks and within block
             * (fixes the original AVX2 kernel's missing block offset). */
            const float *xb = x + b * BLOCK_SIZE + i;
            const float32x4_t vx0 = vld1q_f32(xb);
            const float32x4_t vx1 = vld1q_f32(xb + 4);
            const float32x4_t vx2 = vld1q_f32(xb + 8);
            const float32x4_t vx3 = vld1q_f32(xb + 12);

            /* FMA: sum += code * x * scale */
            sum_a0 = vmlaq_f32(sum_a0, vmulq_f32(cfa0, vx0), vsa);
            sum_a0 = vmlaq_f32(sum_a0, vmulq_f32(cfa1, vx1), vsa);
            sum_a1 = vmlaq_f32(sum_a1, vmulq_f32(cfa2, vx2), vsa);
            sum_a1 = vmlaq_f32(sum_a1, vmulq_f32(cfa3, vx3), vsa);
            sum_b0 = vmlaq_f32(sum_b0, vmulq_f32(cfb0, vx0), vsb);
            sum_b0 = vmlaq_f32(sum_b0, vmulq_f32(cfb1, vx1), vsb);
            sum_b1 = vmlaq_f32(sum_b1, vmulq_f32(cfb2, vx2), vsb);
            sum_b1 = vmlaq_f32(sum_b1, vmulq_f32(cfb3, vx3), vsb);
        }
        codes_a += BLOCK_SIZE;
        codes_b += BLOCK_SIZE;
    }

    *out_a = vaddvq_f32(vaddq_f32(sum_a0, sum_a1));
    *out_b = vaddvq_f32(vaddq_f32(sum_b0, sum_b1));
}
#endif /* __aarch64__ */

/* --------------------------------------------------------------------------
   NEON SDOT fast path: quantize x to Q8_0 blocks once per call (shared by
   both matrices), then compute both dot products with 4-way int8
   dot-product instructions (vdotq_s32).  This is the ARM twin of the
   fused CUDA kernel: same output contract (dual Q8_0 GEMV), different
   implementation — used for the bench "fused" path.
   -------------------------------------------------------------------------- */
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
static inline void
gemv_dual_q8_0_row_neon_sdot(const int8_t *xq, const float *xs, int n_blocks,
                             const float *scales_a, const int8_t *codes_a,
                             const float *scales_b, const int8_t *codes_b,
                             float *out_a, float *out_b)
{
    float32x4_t acc_a = vdupq_n_f32(0.0f);
    float32x4_t acc_b = vdupq_n_f32(0.0f);
    const int32x4_t zero = vdupq_n_s32(0);

    for (int b = 0; b < n_blocks; ++b) {
        const int8x16_t xq0 = vld1q_s8(xq + b * 32);
        const int8x16_t xq1 = vld1q_s8(xq + b * 32 + 16);

        const int8x16_t ca0 = vld1q_s8(codes_a);
        const int8x16_t ca1 = vld1q_s8(codes_a + 16);
        const int8x16_t cb0 = vld1q_s8(codes_b);
        const int8x16_t cb1 = vld1q_s8(codes_b + 16);
        codes_a += 32;
        codes_b += 32;

        int32x4_t da = vdotq_s32(zero, ca0, xq0);
        da = vdotq_s32(da, ca1, xq1);
        int32x4_t db = vdotq_s32(zero, cb0, xq0);
        db = vdotq_s32(db, cb1, xq1);

        /* Block scale: W scale * x block scale. */
        const float32x4_t sa_sx = vdupq_n_f32(scales_a[b] * xs[b]);
        const float32x4_t sb_sx = vdupq_n_f32(scales_b[b] * xs[b]);

        acc_a = vmlaq_f32(acc_a, vcvtq_f32_s32(da), sa_sx);
        acc_b = vmlaq_f32(acc_b, vcvtq_f32_s32(db), sb_sx);
    }

    *out_a = vaddvq_f32(acc_a);
    *out_b = vaddvq_f32(acc_b);
}
#endif /* __aarch64__ && __ARM_FEATURE_DOTPROD */

/* --------------------------------------------------------------------------
   Public entry point — processes all rows.
   -------------------------------------------------------------------------- */
EXPORT void
gemv_dual_q8_0_neon(
    const float *x, int in_dim,
    const float *scales_a, const int8_t *codes_a, int rows,
    const float *scales_b, const int8_t *codes_b,
    float *restrict out_a, float *restrict out_b)
{
    int n_blocks = in_dim / BLOCK_SIZE;
    int stride = n_blocks * BLOCK_SIZE;  /* elements per row in codes */

#if defined(__aarch64__)
    for (int r = 0; r < rows; ++r) {
        gemv_dual_q8_0_row_neon(
            x, n_blocks,
            scales_a, codes_a,
            scales_b, codes_b,
            out_a + r, out_b + r);
        scales_a += n_blocks;
        codes_a += stride;
        scales_b += n_blocks;
        codes_b += stride;
    }
#else
    (void)x; (void)scales_a; (void)codes_a; (void)rows;
    (void)scales_b; (void)codes_b; (void)out_a; (void)out_b;
    (void)n_blocks; (void)stride;
#endif
}

/* Compatibility alias: same symbol name the AVX2 build exports, so
 * cpu_opt.py works unchanged on either architecture. */
EXPORT void
gemv_dual_q8_0_avx2(
    const float *x, int in_dim,
    const float *scales_a, const int8_t *codes_a, int rows,
    const float *scales_b, const int8_t *codes_b,
    float *restrict out_a, float *restrict out_b)
{
    gemv_dual_q8_0_neon(x, in_dim, scales_a, codes_a, rows,
                        scales_b, codes_b, out_a, out_b);
}

/* --------------------------------------------------------------------------
   Fast fused dual Q8_0 GEMV (SDOT).  Quantizes x per 32-element block and
   uses integer dot products; same output contract as gemv_dual_q8_0_neon
   with ~1e-3 relative accuracy (still Q8-class).
   -------------------------------------------------------------------------- */
EXPORT void
gemv_dual_q8_0_neon_fast(
    const float *x, int in_dim,
    const float *scales_a, const int8_t *codes_a, int rows,
    const float *scales_b, const int8_t *codes_b,
    float *restrict out_a, float *restrict out_b)
{
    int n_blocks = in_dim / BLOCK_SIZE;
    int stride = n_blocks * BLOCK_SIZE;

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
    /* Quantize x once per call — shared between the two matrices. */
    int8_t *xq = (int8_t *)malloc((size_t)in_dim);
    float *xs = (float *)malloc((size_t)n_blocks * sizeof(float));
    if (xq == NULL || xs == NULL) {
        free(xq); free(xs);
        gemv_dual_q8_0_neon(x, in_dim, scales_a, codes_a, rows,
                            scales_b, codes_b, out_a, out_b);
        return;
    }

    for (int b = 0; b < n_blocks; ++b) {
        const float *xb = x + b * BLOCK_SIZE;
        float mx = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            float a = fabsf(xb[i]);
            if (a > mx) mx = a;
        }
        float d = mx / 127.0f;
        if (d <= 0.0f) d = 1.0f;
        xs[b] = d;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            int iv = (int)lrintf(xb[i] / d);
            if (iv > 127) iv = 127;
            else if (iv < -127) iv = -127;
            xq[b * BLOCK_SIZE + i] = (int8_t)iv;
        }
    }

    for (int r = 0; r < rows; ++r) {
        gemv_dual_q8_0_row_neon_sdot(
            xq, xs, n_blocks,
            scales_a, codes_a,
            scales_b, codes_b,
            out_a + r, out_b + r);
        scales_a += n_blocks;
        codes_a += stride;
        scales_b += n_blocks;
        codes_b += stride;
    }

    free(xq);
    free(xs);
#else
    (void)x; (void)in_dim; (void)rows; (void)out_a; (void)out_b;
    (void)scales_a; (void)codes_a; (void)scales_b; (void)codes_b;
    (void)n_blocks; (void)stride;
#endif
}

/* Compatibility alias: same name the fast AVX2 build would export. */
EXPORT void
gemv_dual_q8_0_avx2_fast(
    const float *x, int in_dim,
    const float *scales_a, const int8_t *codes_a, int rows,
    const float *scales_b, const int8_t *codes_b,
    float *restrict out_a, float *restrict out_b)
{
    gemv_dual_q8_0_neon_fast(x, in_dim, scales_a, codes_a, rows,
                             scales_b, codes_b, out_a, out_b);
}

/* --------------------------------------------------------------------------
   Fallback scalar implementation — identical math, any architecture.
   -------------------------------------------------------------------------- */
EXPORT void
gemv_dual_q8_0_scalar(
    const float *x, int in_dim,
    const float *scales_a, const int8_t *codes_a, int rows,
    const float *scales_b, const int8_t *codes_b,
    float *restrict out_a, float *restrict out_b)
{
    int n_blocks = in_dim / BLOCK_SIZE;
    int stride = n_blocks * BLOCK_SIZE;

    for (int r = 0; r < rows; ++r) {
        double acc_a = 0.0, acc_b = 0.0;
        for (int b = 0; b < n_blocks; ++b) {
            float sa = scales_a[b];
            float sb = scales_b[b];
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                float xv = x[b * BLOCK_SIZE + i];
                acc_a += (double)codes_a[b * BLOCK_SIZE + i] * (double)xv * (double)sa;
                acc_b += (double)codes_b[b * BLOCK_SIZE + i] * (double)xv * (double)sb;
            }
        }
        out_a[r] = (float)acc_a;
        out_b[r] = (float)acc_b;
        scales_a += n_blocks;
        codes_a += stride;
        scales_b += n_blocks;
        codes_b += stride;
    }
}
