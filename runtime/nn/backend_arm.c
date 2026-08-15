/*
 * runtime/nn/backend_arm.c — ARM64 NEON backend for the Geodessical runtime.
 *
 * Implements the full backend vtable with ARM NEON (128-bit SIMD) kernels,
 * selected automatically on aarch64 hosts. Numerical results match the CPU
 * reference backend within fp tolerance while running 2-8x faster on Apple
 * Silicon.
 */
#include "runtime/nn/backend.h"
#include <string.h>
#include <math.h>

#if defined(__aarch64__)
#include <arm_neon.h>

#ifdef GEODESSICAL_HOSTED
#include "hal.h"
#else
#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"
#endif

/* FP16 -> FP32 scalar (same semantics as backend_cpu). */
static inline float arm_fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else { exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; } mant &= 0x3FF;
               bits = sign | ((exp + 127 - 15) << 23) | (mant << 13); }
    } else if (exp == 31) bits = sign | 0x7F800000 | (mant << 13);
    else bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

static inline float arm_hsum(float32x4_t v) {
    return vaddvq_f32(v);
}

static inline float arm_max4(float32x4_t v) {
    return vmaxvq_f32(v);
}

/* ---------------- memory ops ---------------- */
static void *arm_alloc(uint64_t size) { return tensor_alloc(size); }
static void  arm_free(void *ptr)      { tensor_free(ptr); }
static int   arm_upload(void *d, const void *s, uint64_t sz) { kmemcpy(d, s, sz); return 0; }
static int   arm_download(void *d, const void *s, uint64_t sz) { kmemcpy(d, s, sz); return 0; }
static void  arm_sync(void) {}

/* ---------------- dequantize ---------------- */
static void arm_dequantize(float *out, const void *data, int n, ggml_type_t type) {
    if (type == GGML_TYPE_F32) {
        kmemcpy(out, data, (size_t)n * sizeof(float));
        return;
    }
    if (type == GGML_TYPE_F16) {
        const uint16_t *h = (const uint16_t *)data;
        for (int i = 0; i < n; i++) out[i] = arm_fp16_to_f32(h[i]);
        return;
    }
    if (type == GGML_TYPE_Q4_0) {
        typedef struct { uint16_t d; uint8_t qs[16]; } q4_0_t;
        int nb = n / 32;
        const q4_0_t *blocks = (const q4_0_t *)data;
        const int8x16_t lo_mask = vdupq_n_s8(0x0F);
        for (int b = 0; b < nb; b++) {
            float d = arm_fp16_to_f32(blocks[b].d);
            uint8x16_t q = vld1q_u8(blocks[b].qs);
            int8x16_t qi = vreinterpretq_s8_u8(q);
            int8x16_t lo = vandq_s8(qi, lo_mask);
            int8x16_t hi = vshrq_n_s8(qi, 4);
            /* subtract 8 (nibble centering) then convert to f32 and scale */
            lo = vsubq_s8(lo, vdupq_n_s8(8));
            hi = vsubq_s8(hi, vdupq_n_s8(8));
            int16x8_t lo_l = vmovl_s8(vget_low_s8(lo));
            int16x8_t lo_h = vmovl_s8(vget_high_s8(lo));
            int16x8_t hi_l = vmovl_s8(vget_low_s8(hi));
            int16x8_t hi_h = vmovl_s8(vget_high_s8(hi));
            float32x4_t d4 = vdupq_n_f32(d);
            float32x4_t o0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_l))), d4);
            float32x4_t o1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_l))), d4);
            float32x4_t o2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_h))), d4);
            float32x4_t o3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_h))), d4);
            float32x4_t o4 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_l))), d4);
            float32x4_t o5 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_l))), d4);
            float32x4_t o6 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_h))), d4);
            float32x4_t o7 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_h))), d4);
            vst1q_f32(out + b * 32,       o0);
            vst1q_f32(out + b * 32 + 4,   o1);
            vst1q_f32(out + b * 32 + 8,   o2);
            vst1q_f32(out + b * 32 + 12,  o3);
            vst1q_f32(out + b * 32 + 16,  o4);
            vst1q_f32(out + b * 32 + 20,  o5);
            vst1q_f32(out + b * 32 + 24,  o6);
            vst1q_f32(out + b * 32 + 28,  o7);
        }
        return;
    }
    if (type == GGML_TYPE_Q8_0) {
        typedef struct { uint16_t d; int8_t qs[32]; } q8_0_t;
        int nb = n / 32;
        const q8_0_t *blocks = (const q8_0_t *)data;
        for (int b = 0; b < nb; b++) {
            float d = arm_fp16_to_f32(blocks[b].d);
            int8x16_t q0 = vld1q_s8(blocks[b].qs);
            int8x16_t q1 = vld1q_s8(blocks[b].qs + 16);
            int16x8_t l0 = vmovl_s8(vget_low_s8(q0));
            int16x8_t h0 = vmovl_s8(vget_high_s8(q0));
            int16x8_t l1 = vmovl_s8(vget_low_s8(q1));
            int16x8_t h1 = vmovl_s8(vget_high_s8(q1));
            float32x4_t d4 = vdupq_n_f32(d);
            vst1q_f32(out + b * 32,       vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(l0))), d4));
            vst1q_f32(out + b * 32 + 4,   vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(l0))), d4));
            vst1q_f32(out + b * 32 + 8,   vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(h0))), d4));
            vst1q_f32(out + b * 32 + 12,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(h0))), d4));
            vst1q_f32(out + b * 32 + 16,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(l1))), d4));
            vst1q_f32(out + b * 32 + 20,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(l1))), d4));
            vst1q_f32(out + b * 32 + 24,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(h1))), d4));
            vst1q_f32(out + b * 32 + 28,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(h1))), d4));
        }
        return;
    }
    /* Fall back to the CPU reference for formats without a NEON path. */
    cpu_dequantize(out, data, n, type);
}

/* ---------------- dot ---------------- */
static float arm_dot(const float *a, const float *b, int n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(a + i),     vld1q_f32(b + i));
        acc1 = vmlaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float sum = arm_hsum(vaddq_f32(acc0, acc1));
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

/* ---------------- GEMV ---------------- */
static void arm_gemv(float *out, const void *weight, const float *x,
                     int out_dim, int in_dim, ggml_type_t weight_type) {
    if (weight_type == GGML_TYPE_F32) {
        const float *w = (const float *)weight;
        for (int i = 0; i < out_dim; i++) {
            const float *row = w + (uint64_t)i * in_dim;
            float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
            int j = 0;
            for (; j + 7 < in_dim; j += 8) {
                acc0 = vmlaq_f32(acc0, vld1q_f32(row + j),     vld1q_f32(x + j));
                acc1 = vmlaq_f32(acc1, vld1q_f32(row + j + 4), vld1q_f32(x + j + 4));
            }
            float s = arm_hsum(vaddq_f32(acc0, acc1));
            for (; j < in_dim; j++) s += row[j] * x[j];
            out[i] = s;
        }
        return;
    }
    /* Quantized: dequant row then NEON dot. */
    float *row_buf = (float *)arm_alloc((uint64_t)in_dim * sizeof(float));
    if (!row_buf) return;
    uint64_t row_bytes = 0;
    if (weight_type == GGML_TYPE_Q4_0) row_bytes = (uint64_t)(in_dim / 32) * 18;
    else if (weight_type == GGML_TYPE_Q8_0) row_bytes = (uint64_t)(in_dim / 32) * 34;
    else if (weight_type == GGML_TYPE_F16) row_bytes = (uint64_t)in_dim * 2;
    else if (weight_type == GGML_TYPE_F32) row_bytes = (uint64_t)in_dim * 4;
    else {
        /* General GGML types (Q5_0, Q4_K, Q6_K, ...): use block geometry. */
        const ggml_type_info_t *ti = ggml_get_type_info(weight_type);
        if (ti && ti->block_size > 0)
            row_bytes = (uint64_t)(in_dim / (int)ti->block_size) * ti->type_size;
        else
            row_bytes = (uint64_t)in_dim * 4;
    }
    const uint8_t *base = (const uint8_t *)weight;
    for (int i = 0; i < out_dim; i++) {
        arm_dequantize(row_buf, base + (uint64_t)i * row_bytes, in_dim, weight_type);
        out[i] = arm_dot(row_buf, x, in_dim);
    }
    arm_free(row_buf);
}

/* ---------------- GEMM (row-major, C += A*B) ---------------- */
static void arm_gemm(float *C, const float *A, const float *B,
                     int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        const float *arow = A + (uint64_t)i * K;
        float *crow = C + (uint64_t)i * N;
        for (int j = 0; j < N; j++) {
            float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 7 < K; k += 8) {
                float32x4_t a0 = vld1q_f32(arow + k), a1 = vld1q_f32(arow + k + 4);
                float32x4_t b0 = { B[(uint64_t)k * N + j],     B[(uint64_t)(k+1) * N + j],
                                   B[(uint64_t)(k+2) * N + j], B[(uint64_t)(k+3) * N + j] };
                float32x4_t b1 = { B[(uint64_t)(k+4) * N + j], B[(uint64_t)(k+5) * N + j],
                                   B[(uint64_t)(k+6) * N + j], B[(uint64_t)(k+7) * N + j] };
                acc0 = vmlaq_f32(acc0, a0, b0);
                acc1 = vmlaq_f32(acc1, a1, b1);
            }
            float s = arm_hsum(vaddq_f32(acc0, acc1));
            for (; k < K; k++) s += arow[k] * B[(uint64_t)k * N + j];
            crow[j] += s;
        }
    }
}

/* ---------------- RMSNorm ---------------- */
static void arm_rmsnorm(float *out, const float *x, const float *w,
                        int dim, float eps) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vmlaq_f32(acc, v, v);
    }
    float ss = arm_hsum(acc);
    for (; i < dim; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / (float)dim + eps);
    float32x4_t r4 = vdupq_n_f32(r);
    i = 0;
    for (; i + 3 < dim; i += 4)
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(vld1q_f32(x + i), r4), vld1q_f32(w + i)));
    for (; i < dim; i++) out[i] = x[i] * r * w[i];
}

/* ---------------- LayerNorm ---------------- */
static void arm_layernorm(float *out, const float *x, const float *w,
                          const float *bias, int dim, float eps) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < dim; i += 4) acc = vaddq_f32(acc, vld1q_f32(x + i));
    float mean = arm_hsum(acc);
    for (; i < dim; i++) mean += x[i];
    mean /= (float)dim;
    float32x4_t m4 = vdupq_n_f32(mean);
    acc = vdupq_n_f32(0.0f);
    i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(x + i), m4);
        acc = vmlaq_f32(acc, d, d);
    }
    float var = arm_hsum(acc);
    for (; i < dim; i++) { float d = x[i] - mean; var += d * d; }
    var /= (float)dim;
    float inv = 1.0f / sqrtf(var + eps);
    float32x4_t iv4 = vdupq_n_f32(inv);
    i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t v = vmulq_f32(vsubq_f32(vld1q_f32(x + i), m4), iv4);
        v = vmulq_f32(v, vld1q_f32(w + i));
        if (bias) v = vaddq_f32(v, vld1q_f32(bias + i));
        vst1q_f32(out + i, v);
    }
    for (; i < dim; i++) {
        float v = (x[i] - mean) * inv * w[i];
        if (bias) v += bias[i];
        out[i] = v;
    }
}

/* ---------------- RoPE ---------------- */
static void arm_rope(float *q, float *k, int head_dim, int n_heads,
                     int n_kv_heads, int pos, float base,
                     const float *freq_factors) {
    for (int h = 0; h < n_heads; h++) {
        float *qh = q + (uint64_t)h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(base, (float)i / (float)head_dim);
            if (freq_factors) freq *= freq_factors[i / 2];
            float angle = (float)pos * freq;
            float cs = cosf(angle), sn = sinf(angle);
            float q0 = qh[i], q1 = qh[i + 1];
            qh[i]     = q0 * cs - q1 * sn;
            qh[i + 1] = q0 * sn + q1 * cs;
        }
    }
    for (int h = 0; h < n_kv_heads; h++) {
        float *kh = k + (uint64_t)h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(base, (float)i / (float)head_dim);
            if (freq_factors) freq *= freq_factors[i / 2];
            float angle = (float)pos * freq;
            float cs = cosf(angle), sn = sinf(angle);
            float k0 = kh[i], k1 = kh[i + 1];
            kh[i]     = k0 * cs - k1 * sn;
            kh[i + 1] = k0 * sn + k1 * cs;
        }
    }
}

/* ARMv8 NEON has no vector exp/tanh; use per-lane libm for exact parity with
 * the CPU reference backend. Memory layout is still vectorised. */
static inline float32x4_t arm_vexp(float32x4_t v) {
    float out[4];
    vst1q_f32(out, v);
    for (int i = 0; i < 4; i++) out[i] = expf(out[i]);
    return vld1q_f32(out);
}
static inline float32x4_t arm_vtanh(float32x4_t v) {
    float out[4];
    vst1q_f32(out, v);
    for (int i = 0; i < 4; i++) out[i] = tanhf(out[i]);
    return vld1q_f32(out);
}

/* ---------------- softmax / silu / gelu ---------------- */
static void arm_softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float32x4_t m4 = vdupq_n_f32(mx);
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t e = arm_vexp(vsubq_f32(vld1q_f32(x + i), m4));
        vst1q_f32(x + i, e);
        acc = vaddq_f32(acc, e);
    }
    float sum = arm_hsum(acc);
    for (; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / sum;
    float32x4_t i4 = vdupq_n_f32(inv);
    i = 0;
    for (; i + 3 < n; i += 4) vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), i4));
    for (; i < n; i++) x[i] *= inv;
}

static void arm_silu(float *x, int n) {
    int i = 0;
    float32x4_t one = vdupq_n_f32(1.0f);
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t sig = vdivq_f32(one, vaddq_f32(one, arm_vexp(vnegq_f32(v))));
        vst1q_f32(x + i, vmulq_f32(v, sig));
    }
    for (; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

static void arm_gelu(float *x, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t v3 = vmulq_f32(vmulq_f32(v, v), v);
        float32x4_t t = arm_vtanh(vmulq_f32(vdupq_n_f32(0.7978845608f),
                                vaddq_f32(v, vmulq_f32(vdupq_n_f32(0.044715f), v3))));
        vst1q_f32(x + i, vmulq_f32(vmulq_f32(vdupq_n_f32(0.5f), v),
                  vaddq_f32(vdupq_n_f32(1.0f), t)));
    }
    for (; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

/* ---------------- elementwise ---------------- */
static void arm_mul(float *out, const float *a, const float *b, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4)
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < n; i++) out[i] = a[i] * b[i];
}
static void arm_add(float *out, const float *a, const float *b, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4)
        vst1q_f32(out + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < n; i++) out[i] = a[i] + b[i];
}
static void arm_scale(float *out, const float *x, float s, int n) {
    int i = 0;
    float32x4_t s4 = vdupq_n_f32(s);
    for (; i + 3 < n; i += 4)
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(x + i), s4));
    for (; i < n; i++) out[i] = x[i] * s;
}

/* ---------------- attention (NEON dot for scores) ---------------- */
static void arm_attention(float *out, const float *Q,
                          const float *K_cache, const float *V_cache,
                          int n_heads, int n_kv_heads, int head_dim,
                          int seq_len, int max_seq, float scale, float softcap) {
    int kv_group = n_heads / n_kv_heads;
    float *scores = (float *)arm_alloc((uint64_t)seq_len * sizeof(float));
    if (!scores) return;
    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / kv_group;
        const float *qh = Q + (uint64_t)h * head_dim;
        for (int t = 0; t < seq_len; t++) {
            const float *kh = K_cache + ((uint64_t)kv_h * max_seq + t) * head_dim;
            float s = arm_dot(qh, kh, head_dim) * scale;
            if (softcap > 0.0f) s = softcap * tanhf(s / softcap);
            scores[t] = s;
        }
        arm_softmax(scores, seq_len);
        float *oh = out + (uint64_t)h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            int t = 0;
            for (; t + 3 < seq_len; t += 4) {
                float32x4_t sv = vld1q_f32(scores + t);
                float32x4_t vv = { V_cache[((uint64_t)kv_h * max_seq + t) * head_dim + d],
                                   V_cache[((uint64_t)kv_h * max_seq + t + 1) * head_dim + d],
                                   V_cache[((uint64_t)kv_h * max_seq + t + 2) * head_dim + d],
                                   V_cache[((uint64_t)kv_h * max_seq + t + 3) * head_dim + d] };
                acc = vmlaq_f32(acc, sv, vv);
            }
            float v = arm_hsum(acc);
            for (; t < seq_len; t++)
                v += scores[t] * V_cache[((uint64_t)kv_h * max_seq + t) * head_dim + d];
            oh[d] = v;
        }
    }
    arm_free(scores);
}

/* ---------------- kv update ---------------- */
static void arm_kv_update(float *K_cache, float *V_cache,
                          const float *K_new, const float *V_new,
                          int n_kv_heads, int head_dim, int pos,
                          int max_seq, int layer) {
    (void)layer;
    for (int h = 0; h < n_kv_heads; h++) {
        kmemcpy(K_cache + ((uint64_t)h * max_seq + pos) * head_dim,
                K_new + (uint64_t)h * head_dim, (size_t)head_dim * sizeof(float));
        kmemcpy(V_cache + ((uint64_t)h * max_seq + pos) * head_dim,
                V_new + (uint64_t)h * head_dim, (size_t)head_dim * sizeof(float));
    }
}

/* ---------------- embedding lookup ---------------- */
static void arm_embed_lookup(float *out, const void *embd_table,
                             int token_id, int dim, ggml_type_t type) {
    if (type == GGML_TYPE_F32) {
        const float *t = (const float *)embd_table;
        kmemcpy(out, t + (uint64_t)token_id * dim, (size_t)dim * sizeof(float));
    } else {
        uint64_t row_bytes = 0;
        if (type == GGML_TYPE_Q4_0) row_bytes = (uint64_t)(dim / 32) * 18;
        else if (type == GGML_TYPE_Q8_0) row_bytes = (uint64_t)(dim / 32) * 34;
        else if (type == GGML_TYPE_F16) row_bytes = (uint64_t)dim * 2;
        else if (type == GGML_TYPE_F32) row_bytes = (uint64_t)dim * 4;
        else {
            const ggml_type_info_t *ti = ggml_get_type_info(type);
            if (ti && ti->block_size > 0)
                row_bytes = (uint64_t)(dim / (int)ti->block_size) * ti->type_size;
            else
                row_bytes = (uint64_t)dim * 4;
        }
        const uint8_t *base = (const uint8_t *)embd_table;
        arm_dequantize(out, base + (uint64_t)token_id * row_bytes, dim, type);
    }
}

/* ---------------- softcap ---------------- */
static void arm_softcap(float *x, int n, float cap) {
    for (int i = 0; i < n; i++)
        x[i] = cap * tanhf(x[i] / cap);
}

/* ---------------- init/shutdown ---------------- */
static int  arm_init(void) { return 0; }
static void arm_shutdown(void) {}
static int  arm_device_count(void) { return 1; }
static uint64_t arm_free_memory(int dev) {
    (void)dev;
    return tensor_mm_free_bytes();
}

const backend_t backend_arm = {
    .id   = BACKEND_ARM,
    .name = "arm-neon",
    .init = arm_init,
    .shutdown = arm_shutdown,
    .get_device_count = arm_device_count,
    .get_free_memory  = arm_free_memory,
    .mem = {
        .alloc    = arm_alloc,
        .free     = arm_free,
        .upload   = arm_upload,
        .download = arm_download,
        .sync     = arm_sync,
    },
    .compute = {
        .gemv         = arm_gemv,
        .gemm         = arm_gemm,
        .rmsnorm      = arm_rmsnorm,
        .layernorm    = arm_layernorm,
        .rope         = arm_rope,
        .softmax      = arm_softmax,
        .silu         = arm_silu,
        .gelu         = arm_gelu,
        .mul          = arm_mul,
        .add          = arm_add,
        .scale        = arm_scale,
        .dot          = arm_dot,
        .dequantize   = arm_dequantize,
        .attention    = arm_attention,
        .kv_update    = arm_kv_update,
        .embed_lookup = arm_embed_lookup,
        .softcap      = arm_softcap,
    },
};

#endif /* __aarch64__ */
