/**
 * Fused single-kernel decode for Qwen3.5-0.8B (hybrid DeltaNet + Full Attention).
 * Supports Pascal (sm_6x), Volta/Turing (sm_70-75), and Ampere+ (sm_80+).
 * BF16 for sm_80+, F16 for sm_6x-sm_75. Accumulation always f32.
 * DeltaNet state: f32 (recurrence needs precision).
 *
 * Pascal-specific adaptations (via TARGET_SM guards):
 *   - __shfl / __shfl_down instead of __shfl_sync / __shfl_down_sync
 *   - membar.gl instead of fence.acq_rel.gpu
 *   - __ldg() instead of ld.global.cg / ld.global.L1::no_allocate
 *   - No tensor cores, no WMMA, no cp.async
 */

#include "half_type.h"
#include "rope.h"

// ── Pascal (sm_6x) compatibility shims ──
// Pascal lacks the _sync suffixed warp shuffle and the fence.acq_rel.gpu
// PTX instruction.  Provide thin wrappers so the same source compiles for
// both Pascal and Volta+.
#if TARGET_SM >= 70
// Volta+: native _sync shuffles and fence.acq_rel.gpu
#define SHFL_SYNC(mask, val, lane)    __shfl_sync((mask), (val), (lane))
#define SHFL_DOWN_SYNC(mask, val, d)  __shfl_down_sync((mask), (val), (d))
#define GPU_MEMORY_FENCE()            asm volatile("fence.acq_rel.gpu;" ::: "memory")
#else
// Pascal: no mask argument, no _sync suffix; membar.gl for global fence
#define SHFL_SYNC(mask, val, lane)    __shfl((val), (lane))
#define SHFL_DOWN_SYNC(mask, val, d)  __shfl_down((val), (d))
#define GPU_MEMORY_FENCE()            asm volatile("membar.gl;" ::: "memory")
#endif
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef MEGAKERNEL_DECODE_GQA_REUSE
#define MEGAKERNEL_DECODE_GQA_REUSE 0
#endif
#ifndef MEGAKERNEL_DECODE_ATTENTION_VEC2
#define MEGAKERNEL_DECODE_ATTENTION_VEC2 1
#endif
#ifndef MEGAKERNEL_ATTENTION_EXP_APPROX
#define MEGAKERNEL_ATTENTION_EXP_APPROX 0
#endif
#ifndef MEGAKERNEL_APPROX_ATTN_WINDOW
#define MEGAKERNEL_APPROX_ATTN_WINDOW 0
#endif
#ifndef MEGAKERNEL_APPROX_MLP_WIDTH
#define MEGAKERNEL_APPROX_MLP_WIDTH 0
#endif
#ifndef MEGAKERNEL_APPROX_SKIP_MLP
#define MEGAKERNEL_APPROX_SKIP_MLP 0
#endif
#ifndef MEGAKERNEL_APPROX_SKIP_MLP_EVERY
#define MEGAKERNEL_APPROX_SKIP_MLP_EVERY 0
#endif
#ifndef MEGAKERNEL_DECODE_SPLITKV
#define MEGAKERNEL_DECODE_SPLITKV 0
#endif
#ifndef MEGAKERNEL_DECODE_SPLITKV_MIN_CONTEXT
#define MEGAKERNEL_DECODE_SPLITKV_MIN_CONTEXT 512
#endif
#ifndef MEGAKERNEL_LAYER_DP4A
#define MEGAKERNEL_LAYER_DP4A 0
#endif
#ifndef MEGAKERNEL_DN_HEAD_SPLIT
#define MEGAKERNEL_DN_HEAD_SPLIT 1
#endif

// =============================================================================
// Model constants
// =============================================================================

constexpr int WARP_SIZE = 32;
constexpr int HIDDEN_SIZE = 1024;
constexpr int INTERMEDIATE_SIZE = 3584;
constexpr int NUM_LAYERS = 24;
constexpr float RMS_EPS = 1e-6f;
constexpr int VOCAB_SIZE = 248320;

// Full Attention
constexpr int FA_NUM_Q_HEADS = 8;
constexpr int FA_NUM_KV_HEADS = 2;
constexpr int FA_HEAD_DIM = 256;
constexpr int FA_GQA_RATIO = FA_NUM_Q_HEADS / FA_NUM_KV_HEADS;
constexpr int FA_Q_SIZE = FA_NUM_Q_HEADS * FA_HEAD_DIM;
constexpr int FA_GATE_SIZE = FA_Q_SIZE;
constexpr int FA_QPROJ_SIZE = FA_Q_SIZE + FA_GATE_SIZE;
constexpr int FA_KV_SIZE = FA_NUM_KV_HEADS * FA_HEAD_DIM;
constexpr int FA_ROTARY_DIM = 64;
constexpr float FA_ROPE_THETA = 10000000.0f;

// DeltaNet
constexpr int DN_NUM_HEADS = 16;
constexpr int DN_KEY_DIM = 128;
constexpr int DN_VALUE_DIM = 128;
constexpr int DN_CONV_KERNEL = 4;
constexpr int DN_QK_SIZE = DN_NUM_HEADS * DN_KEY_DIM;
constexpr int DN_V_SIZE = DN_NUM_HEADS * DN_VALUE_DIM;
constexpr int DN_CONV_CHANNELS = DN_QK_SIZE + DN_QK_SIZE + DN_V_SIZE;

constexpr int MAX_ACT_DIM = (HIDDEN_SIZE > INTERMEDIATE_SIZE) ? HIDDEN_SIZE : INTERMEDIATE_SIZE;

// A reduced-width MLP is an intentionally lossy approximation used for
// experiments.  Keeping the full-width default makes the production path
// bit-for-bit unchanged; positive values cap only the decode MLP channels.
#if MEGAKERNEL_APPROX_MLP_WIDTH > 0 && MEGAKERNEL_APPROX_MLP_WIDTH < INTERMEDIATE_SIZE
constexpr int DECODE_MLP_WIDTH = MEGAKERNEL_APPROX_MLP_WIDTH;
#else
constexpr int DECODE_MLP_WIDTH = INTERMEDIATE_SIZE;
#endif

#ifndef NUM_BLOCKS
#define NUM_BLOCKS 82
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
// Decode activations reserve a tail for one-pass split-KV partials. The
// buffer is reused across layers, so this does not multiply by layer count.
constexpr int ATTN_PARTIAL_SLOT = FA_HEAD_DIM + 2;
constexpr int ACTIVATIONS_STRIDE = HIDDEN_SIZE + NUM_BLOCKS * FA_NUM_Q_HEADS * ATTN_PARTIAL_SLOT;

#if MEGAKERNEL_DECODE_GQA_REUSE
// The GQA decode path stages a small K/V tile before the four query-head
// groups consume it.  Keep the existing attention partials after the tile.
constexpr int GQA_TILE_POS = 32;
constexpr int GQA_TILE_FLOATS = GQA_TILE_POS * FA_HEAD_DIM;
#endif
constexpr int SHMEM_FLOATS_BASE = (MAX_ACT_DIM > NUM_WARPS * FA_HEAD_DIM)
    ? MAX_ACT_DIM : NUM_WARPS * FA_HEAD_DIM;
constexpr int SHMEM_FLOATS_GQA =
#if MEGAKERNEL_DECODE_GQA_REUSE
    (SHMEM_FLOATS_BASE > GQA_TILE_FLOATS + NUM_WARPS * FA_HEAD_DIM)
        ? SHMEM_FLOATS_BASE : GQA_TILE_FLOATS + NUM_WARPS * FA_HEAD_DIM;
#else
    SHMEM_FLOATS_BASE;
#endif
// Optional layer DP4A uses a per-group int8 activation copy.  Keep it in a
// separate shared-memory tail so the original BF16/F32 staging remains
// available while the projection reads the quantized values.
#if MEGAKERNEL_LAYER_DP4A
constexpr int LAYER_Q_GROUPS = (MAX_ACT_DIM + 31) / 32;
constexpr int LAYER_Q_BYTES = MAX_ACT_DIM + LAYER_Q_GROUPS * sizeof(float);
constexpr int LAYER_Q_FLOATS = (LAYER_Q_BYTES + sizeof(float) - 1) / sizeof(float);
#else
constexpr int LAYER_Q_FLOATS = 0;
#endif
constexpr int SHMEM_FLOATS = SHMEM_FLOATS_GQA + LAYER_Q_FLOATS;

#ifndef LM_NUM_BLOCKS
#define LM_NUM_BLOCKS 512
#endif
#ifndef LM_BLOCK_SIZE
#define LM_BLOCK_SIZE 256
#endif
#ifndef MEGAKERNEL_LM_INT8
#define MEGAKERNEL_LM_INT8 0
#endif
#ifndef MEGAKERNEL_LM_DP4A
#define MEGAKERNEL_LM_DP4A 0
#endif
#ifndef MEGAKERNEL_LAYER_INT8
#define MEGAKERNEL_LAYER_INT8 0
#endif
static int g_decode_blocks_override = 0;

__device__ __constant__ int LAYER_TYPE[NUM_LAYERS] = {
    0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1
};

// =============================================================================
// Weight structs — ALL BF16
// =============================================================================

struct FullAttnWeights {
    const half_t *input_layernorm_weight;   // [1024]
    const half_t *q_proj_weight;             // [4096, 1024]
    const half_t *k_proj_weight;             // [512, 1024]
    const half_t *v_proj_weight;             // [512, 1024]
    const half_t *q_norm_weight;              // [256]
    const half_t *k_norm_weight;              // [256]
    const half_t *o_proj_weight;             // [1024, 2048]
    const half_t *post_attn_layernorm_weight;
    const half_t *gate_proj_weight;          // [3584, 1024]
    const half_t *up_proj_weight;            // [3584, 1024]
    const half_t *down_proj_weight;          // [1024, 3584]
};

struct DeltaNetWeights {
    const half_t *input_layernorm_weight;
    const half_t *qkv_proj_weight;           // [6144, 1024]
    const half_t *z_proj_weight;             // [2048, 1024]
    const half_t *beta_proj_weight;          // [16, 1024]
    const half_t *alpha_proj_weight;         // [16, 1024]
    const half_t *conv1d_weight;             // [6144, 1, 4]
    const half_t *a_log;                     // [16]
    const half_t *dt_bias;                   // [16]
    const half_t *norm_weight;               // [128]
    const half_t *out_proj_weight;           // [1024, 2048]
    const half_t *post_attn_layernorm_weight;
    const half_t *gate_proj_weight;
    const half_t *up_proj_weight;
    const half_t *down_proj_weight;
};

struct LayerWeights {
    int layer_type;
    int _pad[3];
    union {
        DeltaNetWeights dn;
        FullAttnWeights fa;
    };
};

// =============================================================================
// Atomic barrier
// =============================================================================

struct AtomicGridSync {
    unsigned int *counter;
    unsigned int *generation;
    unsigned int nblocks;
    unsigned int local_gen;

    __device__ void sync() {
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int my_gen = local_gen;
            GPU_MEMORY_FENCE();
            unsigned int arrived = atomicAdd(counter, 1);
            if (arrived == nblocks - 1) {
                *counter = 0;
                GPU_MEMORY_FENCE();
                atomicAdd(generation, 1);
            } else {
                volatile unsigned int *vgen = (volatile unsigned int *)generation;
                while (*vgen <= my_gen) {}
            }
            local_gen = my_gen + 1;
        }
        __syncthreads();
    }
};

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += SHFL_DOWN_SYNC(0xffffffff, val, offset);
    return val;
}

#if MEGAKERNEL_LAYER_DP4A
// Quantize one activation vector into 32-element groups.  A group scale keeps
// the DP4A path close to the existing per-row weight-only int8 result while
// avoiding the large error from a single fixed activation scale.
__device__ void layer_quantize_bf16(
    const half_t *__restrict__ input, int8_t *__restrict__ q,
    float *__restrict__ scales, int n) {
    const int groups = (n + 31) / 32;
    for (int g = threadIdx.x; g < groups; g += BLOCK_SIZE) {
        const int start = g * 32;
        const int end = min(start + 32, n);
        float amax = 0.0f;
        for (int i = start; i < end; ++i) amax = fmaxf(amax, fabsf(H2F(input[i])));
        scales[g] = amax / 127.0f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += BLOCK_SIZE) {
        const float s = scales[i / 32];
        q[i] = s > 0.0f ? (int8_t)max(-127, min(127, __float2int_rn(H2F(input[i]) / s))) : 0;
    }
    __syncthreads();
}

__device__ void layer_quantize_f32(
    const float *__restrict__ input, int8_t *__restrict__ q,
    float *__restrict__ scales, int n) {
    const int groups = (n + 31) / 32;
    for (int g = threadIdx.x; g < groups; g += BLOCK_SIZE) {
        const int start = g * 32;
        const int end = min(start + 32, n);
        float amax = 0.0f;
        for (int i = start; i < end; ++i) amax = fmaxf(amax, fabsf(input[i]));
        scales[g] = amax / 127.0f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += BLOCK_SIZE) {
        const float s = scales[i / 32];
        q[i] = s > 0.0f ? (int8_t)max(-127, min(127, __float2int_rn(input[i] / s))) : 0;
    }
    __syncthreads();
}
#endif

__device__ __forceinline__ float fast_exp(float x) {
    // Keep the fused path numerically aligned with Transformers' reference
    // implementation.  The previous ex2.approx approximation was fast, but
    // its accumulated error could change greedy token selection after a
    // prefill/decode handoff.
    return expf(x);
}

__device__ __forceinline__ float attention_exp(float x) {
#if MEGAKERNEL_ATTENTION_EXP_APPROX
    return __expf(x);
#else
    return expf(x);
#endif
}

// Deliberately lossy long-context approximation.  Qwen3.5 has only six
// full-attention layers; bounding their KV scan leaves the recurrent DeltaNet
// layers untouched while making decode cost nearly context-independent.  This
// is compile-time so the inner loops retain the same address arithmetic as the
// exact path.  A value of zero is the exact implementation.
__device__ __forceinline__ int attention_scan_start(int cache_len) {
#if MEGAKERNEL_APPROX_ATTN_WINDOW > 0
    return cache_len > MEGAKERNEL_APPROX_ATTN_WINDOW
        ? cache_len - MEGAKERNEL_APPROX_ATTN_WINDOW : 0;
#else
    (void)cache_len;
    return 0;
#endif
}

#if MEGAKERNEL_DECODE_SPLITKV
// Float atomic max implemented with CAS.  Attention scores may be negative,
// so integer atomicMax cannot be used directly on their bit representation.
__device__ __forceinline__ float atomic_max_float(float *address, float value) {
    int *as_int = reinterpret_cast<int *>(address);
    int old = *as_int;
    while (value > __int_as_float(old)) {
        int assumed = old;
        old = atomicCAS(as_int, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
    return __int_as_float(old);
}
#endif

#if MEGAKERNEL_DECODE_SPLITKV
template <int EPL>
__device__ __forceinline__ float splitkv_dot(const float *q, const half_t *p, int lane_id) {
    float score = 0.0f;
#if MEGAKERNEL_DECODE_ATTENTION_VEC2
    for (int e = 0; e < EPL; e += 2) {
        int idx = lane_id * EPL + e;
#if TARGET_SM >= 80
        __nv_bfloat162 kv = __ldg(reinterpret_cast<const __nv_bfloat162 *>(p + idx));
        float2 kf = __bfloat1622float2(kv);
        score += q[e] * kf.x + q[e + 1] * kf.y;
#else
        __half2 kv = __ldg(reinterpret_cast<const __half2 *>(p + idx));
        float2 kf = __half22float2(kv);
        score += q[e] * kf.x + q[e + 1] * kf.y;
#endif
    }
#else
    for (int e = 0; e < EPL; ++e)
        score += q[e] * H2F(__ldg(p + lane_id * EPL + e));
#endif
    return score;
}

__device__ void split_kv_attention_onepass(
    AtomicGridSync &grid, const float *g_q, half_t *k_cache,
    half_t *v_cache, float *g_partial, float *g_attn_out,
    int position, int max_seq_len, float attn_scale, half_t *shmem)
{
    constexpr int EPL = FA_HEAD_DIM / WARP_SIZE;
    constexpr int WARPS_PER_Q = NUM_WARPS / FA_NUM_Q_HEADS;
    constexpr int SLOT = FA_HEAD_DIM + 2;
    const int block_id = blockIdx.x;
    const int num_blocks = gridDim.x;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int cache_len = position + 1;
    const int scan_start = attention_scan_start(cache_len);
    __shared__ float s_max[NUM_WARPS];
    __shared__ float s_sum[NUM_WARPS];
    float *s_out = reinterpret_cast<float *>(shmem);

    // At short context there is not enough work to amortize a grid-wide
    // partial reduction. Reuse the original warp scheduling and reduction
    // order so this fast path remains numerically compatible with the
    // non-split implementation.
    if (cache_len < MEGAKERNEL_DECODE_SPLITKV_MIN_CONTEXT) {
        int hpb = (FA_NUM_Q_HEADS + num_blocks - 1) / num_blocks;
        int hs = block_id * hpb, he = min(hs + hpb, FA_NUM_Q_HEADS);
        constexpr int SHORT_EPL = FA_HEAD_DIM / WARP_SIZE;
        for (int qh = hs; qh < he; ++qh) {
            int kvh = qh / FA_GQA_RATIO;
            const float *q_head = g_q + qh * FA_HEAD_DIM * 2;
            float *out_head = g_attn_out + qh * FA_HEAD_DIM;
            float max_score = -INFINITY, sum_exp = 0.0f;
            float out_acc[SHORT_EPL], q_local[SHORT_EPL];
            for (int e = 0; e < SHORT_EPL; ++e) {
                out_acc[e] = 0.0f;
                q_local[e] = q_head[lane_id * SHORT_EPL + e];
            }
#pragma unroll 4
            for (int pos = scan_start + warp_id; pos < cache_len; pos += NUM_WARPS) {
                const half_t *kp = k_cache + kvh * max_seq_len * FA_HEAD_DIM + pos * FA_HEAD_DIM;
                const half_t *vp = v_cache + kvh * max_seq_len * FA_HEAD_DIM + pos * FA_HEAD_DIM;
                float score = splitkv_dot<SHORT_EPL>(q_local, kp, lane_id);
                score = warp_reduce_sum(score) * attn_scale;
                score = SHFL_SYNC(0xffffffff, score, 0);
                float old_max = max_score;
                max_score = fmaxf(max_score, score);
                float scale = attention_exp(old_max - max_score);
                float wt = attention_exp(score - max_score);
                sum_exp = sum_exp * scale + wt;
#if MEGAKERNEL_DECODE_ATTENTION_VEC2
                for (int e = 0; e < SHORT_EPL; e += 2) {
                    int idx = lane_id * SHORT_EPL + e;
#if TARGET_SM >= 80
                    __nv_bfloat162 vv = __ldg(reinterpret_cast<const __nv_bfloat162 *>(vp + idx));
                    float2 vf = __bfloat1622float2(vv);
#else
                    __half2 vv = __ldg(reinterpret_cast<const __half2 *>(vp + idx));
                    float2 vf = __half22float2(vv);
#endif
                    out_acc[e] = out_acc[e] * scale + wt * vf.x;
                    out_acc[e + 1] = out_acc[e + 1] * scale + wt * vf.y;
                }
#else
                for (int e = 0; e < SHORT_EPL; ++e)
                    out_acc[e] = out_acc[e] * scale + wt * H2F(__ldg(vp + lane_id * SHORT_EPL + e));
#endif
            }
            if (lane_id == 0) {
                s_max[warp_id] = max_score;
                s_sum[warp_id] = sum_exp;
            }
            for (int e = 0; e < SHORT_EPL; ++e)
                s_out[warp_id * FA_HEAD_DIM + lane_id * SHORT_EPL + e] = out_acc[e];
            __syncthreads();
            if (warp_id == 0) {
                float global_max = -INFINITY;
                for (int ww = 0; ww < NUM_WARPS; ++ww)
                    if (s_max[ww] > -INFINITY) global_max = fmaxf(global_max, s_max[ww]);
                float total = 0.0f, final_out[SHORT_EPL];
                for (int e = 0; e < SHORT_EPL; ++e) final_out[e] = 0.0f;
                for (int ww = 0; ww < NUM_WARPS; ++ww) {
                    if (s_max[ww] > -INFINITY) {
                        float scale = attention_exp(s_max[ww] - global_max);
                        total += s_sum[ww] * scale;
                        for (int e = 0; e < SHORT_EPL; ++e)
                            final_out[e] += s_out[ww * FA_HEAD_DIM + lane_id * SHORT_EPL + e] * scale;
                    }
                }
                float rcp = 1.0f / total;
                for (int e = 0; e < SHORT_EPL; ++e) {
                    int idx = lane_id * SHORT_EPL + e;
                    out_head[idx] = final_out[e] * rcp *
                        (1.0f / (1.0f + expf(-q_head[FA_HEAD_DIM + idx])));
                }
            }
            __syncthreads();
        }
        grid.sync();
        return;
    }

    const int qh = warp_id / WARPS_PER_Q;
    const int q_warp = warp_id % WARPS_PER_Q;
    if (qh < FA_NUM_Q_HEADS) {
        const float *q_head = g_q + qh * FA_HEAD_DIM * 2;
        float q_local[EPL], out_acc[EPL];
        for (int e = 0; e < EPL; ++e) {
            q_local[e] = q_head[lane_id * EPL + e];
            out_acc[e] = 0.0f;
        }
        float max_score = -INFINITY, sum_exp = 0.0f;
#pragma unroll 4
        for (int pos = scan_start + block_id + q_warp * num_blocks;
             pos < cache_len; pos += num_blocks * WARPS_PER_Q) {
            const int kvh = qh / FA_GQA_RATIO;
            const half_t *kp = k_cache + kvh * max_seq_len * FA_HEAD_DIM + pos * FA_HEAD_DIM;
            const half_t *vp = v_cache + kvh * max_seq_len * FA_HEAD_DIM + pos * FA_HEAD_DIM;
            float score = SHFL_SYNC(0xffffffff,
                warp_reduce_sum(splitkv_dot<EPL>(q_local, kp, lane_id)), 0) * attn_scale;
            float old_max = max_score;
            max_score = fmaxf(max_score, score);
            float scale = attention_exp(old_max - max_score);
            float wt = attention_exp(score - max_score);
            sum_exp = sum_exp * scale + wt;
#if MEGAKERNEL_DECODE_ATTENTION_VEC2
            for (int e = 0; e < EPL; e += 2) {
                int idx = lane_id * EPL + e;
#if TARGET_SM >= 80
                __nv_bfloat162 vv = __ldg(reinterpret_cast<const __nv_bfloat162 *>(vp + idx));
                float2 vf = __bfloat1622float2(vv);
                out_acc[e] = out_acc[e] * scale + wt * vf.x;
                out_acc[e + 1] = out_acc[e + 1] * scale + wt * vf.y;
#else
                __half2 vv = __ldg(reinterpret_cast<const __half2 *>(vp + idx));
                float2 vf = __half22float2(vv);
                out_acc[e] = out_acc[e] * scale + wt * vf.x;
                out_acc[e + 1] = out_acc[e + 1] * scale + wt * vf.y;
#endif
            }
#else
            for (int e = 0; e < EPL; ++e)
                out_acc[e] = out_acc[e] * scale + wt * H2F(__ldg(vp + lane_id * EPL + e));
#endif
        }
        s_max[warp_id] = max_score;
        s_sum[warp_id] = sum_exp;
        for (int e = 0; e < EPL; ++e)
            s_out[warp_id * FA_HEAD_DIM + lane_id * EPL + e] = out_acc[e];
        __syncthreads();

        // One warp group per query head merges the local online-softmax
        // states into one partial record for this block.
        if (q_warp == 0) {
            float block_max = -INFINITY, block_sum = 0.0f;
            for (int ww = 0; ww < WARPS_PER_Q; ++ww)
                block_max = fmaxf(block_max, s_max[qh * WARPS_PER_Q + ww]);
            float block_out[EPL];
            for (int e = 0; e < EPL; ++e) block_out[e] = 0.0f;
            if (block_max > -INFINITY) {
                for (int ww = 0; ww < WARPS_PER_Q; ++ww) {
                    int slot = qh * WARPS_PER_Q + ww;
                    float scale = attention_exp(s_max[slot] - block_max);
                    block_sum += s_sum[slot] * scale;
                    for (int e = 0; e < EPL; ++e)
                        block_out[e] += s_out[slot * FA_HEAD_DIM + lane_id * EPL + e] * scale;
                }
            }
            float *dst = g_partial + block_id * FA_NUM_Q_HEADS * SLOT + qh * SLOT;
            if (lane_id == 0) { dst[0] = block_max; dst[1] = block_sum; }
            for (int e = 0; e < EPL; ++e)
                dst[2 + lane_id * EPL + e] = block_out[e];
        }
        __syncthreads();
    }
    grid.sync();

    // Block zero performs the final stable merge and gate.  Each warp owns a
    // query head and reads all block partials, so no second KV scan is needed.
    if (block_id == 0 && warp_id < FA_NUM_Q_HEADS) {
        const int qh0 = warp_id;
        float global_max = -INFINITY;
        for (int b = 0; b < num_blocks; ++b)
            global_max = fmaxf(global_max,
                g_partial[b * FA_NUM_Q_HEADS * SLOT + qh0 * SLOT]);
        global_max = SHFL_SYNC(0xffffffff, global_max, 0);
        float denom = 0.0f;
        float *out_head = g_attn_out + qh0 * FA_HEAD_DIM;
        for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) {
            float value = 0.0f;
            for (int b = 0; b < num_blocks; ++b) {
                const float *src = g_partial + b * FA_NUM_Q_HEADS * SLOT + qh0 * SLOT;
                float scale = attention_exp(src[0] - global_max);
                if (i == lane_id) denom += src[1] * scale;
                value += src[2 + i] * scale;
            }
            out_head[i] = value;
        }
        denom = SHFL_SYNC(0xffffffff, denom, 0);
        const float *q_head = g_q + qh0 * FA_HEAD_DIM * 2;
        for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE)
            out_head[i] = out_head[i] / denom * (1.0f / (1.0f + expf(-q_head[FA_HEAD_DIM + i])));
    }
    grid.sync();
}
#endif

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

#if MEGAKERNEL_DECODE_SPLITKV
__device__ void split_kv_attention(
    AtomicGridSync &grid, const float *g_q, half_t *k_cache,
    half_t *v_cache, float *g_kv, float *g_attn_out,
    int position, int max_seq_len, float attn_scale)
{
    constexpr int EPL = FA_HEAD_DIM / WARP_SIZE;
    constexpr int WARPS_PER_Q = NUM_WARPS / FA_NUM_Q_HEADS;
    const int block_id = blockIdx.x;
    const int num_blocks = gridDim.x;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int cache_len = position + 1;
    const int scan_start = attention_scan_start(cache_len);
    __shared__ float s_max_score[NUM_WARPS];

    // g_kv is no longer needed for projections after the cache write.  Use
    // its first 16 floats for per-head max and denominator scratch, and use
    // g_attn_out for the value numerator.  All blocks initialize identically
    // before the first grid barrier.
    if (threadIdx.x < FA_NUM_Q_HEADS * 2)
        g_kv[threadIdx.x] = threadIdx.x < FA_NUM_Q_HEADS ? -INFINITY : 0.0f;
    for (int i = threadIdx.x; i < FA_Q_SIZE; i += BLOCK_SIZE)
        g_attn_out[i] = 0.0f;
    __syncthreads();
    grid.sync();

    const int qh = warp_id / WARPS_PER_Q;
    const int q_warp = warp_id % WARPS_PER_Q;
    if (qh < FA_NUM_Q_HEADS) {
        const float *q_head = g_q + qh * FA_HEAD_DIM * 2;
        float q_local[EPL];
        for (int e = 0; e < EPL; ++e)
            q_local[e] = q_head[lane_id * EPL + e];
        float local_max = -INFINITY;
#pragma unroll 4
        for (int pos = scan_start + block_id + q_warp * num_blocks;
             pos < cache_len; pos += num_blocks * WARPS_PER_Q) {
            const int kvh = qh / FA_GQA_RATIO;
            const half_t *k_pos = k_cache + kvh * max_seq_len * FA_HEAD_DIM + pos * FA_HEAD_DIM;
            float score = 0.0f;
#if MEGAKERNEL_DECODE_ATTENTION_VEC2
            for (int e = 0; e < EPL; e += 2) {
                int idx = lane_id * EPL + e;
#if TARGET_SM >= 80
                __nv_bfloat162 kv = __ldg(reinterpret_cast<const __nv_bfloat162 *>(k_pos + idx));
                float2 kf = __bfloat1622float2(kv);
                score += q_local[e] * kf.x + q_local[e + 1] * kf.y;
#else
                __half2 kv = __ldg(reinterpret_cast<const __half2 *>(k_pos + idx));
                float2 kf = __half22float2(kv);
                score += q_local[e] * kf.x + q_local[e + 1] * kf.y;
#endif
            }
#else
            for (int e = 0; e < EPL; ++e)
                score += q_local[e] * H2F(__ldg(k_pos + lane_id * EPL + e));
#endif
            local_max = fmaxf(local_max,
                              SHFL_SYNC(0xffffffff, warp_reduce_sum(score), 0) * attn_scale);
        }
        if (lane_id == 0) s_max_score[warp_id] = local_max;
        __syncthreads();
        if (q_warp == 0 && lane_id == 0) {
            float qmax = -INFINITY;
            for (int ww = 0; ww < WARPS_PER_Q; ++ww)
                qmax = fmaxf(qmax, s_max_score[qh * WARPS_PER_Q + ww]);
            atomic_max_float(g_kv + qh, qmax);
        }
    }
    grid.sync();

    if (qh < FA_NUM_Q_HEADS) {
        const float *q_head = g_q + qh * FA_HEAD_DIM * 2;
        float q_local[EPL], out_acc[EPL];
        for (int e = 0; e < EPL; ++e) {
            q_local[e] = q_head[lane_id * EPL + e];
            out_acc[e] = 0.0f;
        }
        const float global_max = g_kv[qh];
        float local_sum = 0.0f;
#pragma unroll 4
        for (int pos = scan_start + block_id + q_warp * num_blocks;
             pos < cache_len; pos += num_blocks * WARPS_PER_Q) {
            const int kvh = qh / FA_GQA_RATIO;
            const half_t *k_pos = k_cache + kvh * max_seq_len * FA_HEAD_DIM + pos * FA_HEAD_DIM;
            const half_t *v_pos = v_cache + kvh * max_seq_len * FA_HEAD_DIM + pos * FA_HEAD_DIM;
            float score = 0.0f;
#if MEGAKERNEL_DECODE_ATTENTION_VEC2
            for (int e = 0; e < EPL; e += 2) {
                int idx = lane_id * EPL + e;
#if TARGET_SM >= 80
                __nv_bfloat162 kv = __ldg(reinterpret_cast<const __nv_bfloat162 *>(k_pos + idx));
                float2 kf = __bfloat1622float2(kv);
                score += q_local[e] * kf.x + q_local[e + 1] * kf.y;
#else
                __half2 kv = __ldg(reinterpret_cast<const __half2 *>(k_pos + idx));
                float2 kf = __half22float2(kv);
                score += q_local[e] * kf.x + q_local[e + 1] * kf.y;
#endif
            }
#else
            for (int e = 0; e < EPL; ++e)
                score += q_local[e] * H2F(__ldg(k_pos + lane_id * EPL + e));
#endif
            score = SHFL_SYNC(0xffffffff, warp_reduce_sum(score), 0) * attn_scale;
            const float wt = attention_exp(score - global_max);
            local_sum += wt;
#if MEGAKERNEL_DECODE_ATTENTION_VEC2
            for (int e = 0; e < EPL; e += 2) {
                int idx = lane_id * EPL + e;
#if TARGET_SM >= 80
                __nv_bfloat162 vv = __ldg(reinterpret_cast<const __nv_bfloat162 *>(v_pos + idx));
                float2 vf = __bfloat1622float2(vv);
                out_acc[e] += wt * vf.x;
                out_acc[e + 1] += wt * vf.y;
#else
                __half2 vv = __ldg(reinterpret_cast<const __half2 *>(v_pos + idx));
                float2 vf = __half22float2(vv);
                out_acc[e] += wt * vf.x;
                out_acc[e + 1] += wt * vf.y;
#endif
            }
#else
            for (int e = 0; e < EPL; ++e)
                out_acc[e] += wt * H2F(__ldg(v_pos + lane_id * EPL + e));
#endif
        }
        if (lane_id == 0) atomicAdd(g_kv + FA_NUM_Q_HEADS + qh, local_sum);
        float *dst = g_attn_out + qh * FA_HEAD_DIM + lane_id * EPL;
        for (int e = 0; e < EPL; ++e) atomicAdd(dst + e, out_acc[e]);
    }
    grid.sync();

    if (block_id == 0 && warp_id < FA_NUM_Q_HEADS) {
        const int qh0 = warp_id;
        const float *q_head = g_q + qh0 * FA_HEAD_DIM * 2;
        float *out_head = g_attn_out + qh0 * FA_HEAD_DIM;
        const float rcp = 1.0f / g_kv[FA_NUM_Q_HEADS + qh0];
        for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE)
            out_head[i] = out_head[i] * rcp * fast_sigmoid(q_head[FA_HEAD_DIM + i]);
    }
    grid.sync();
}
#endif

__device__ __forceinline__ float fast_silu(float x) { return x * fast_sigmoid(x); }

__device__ __forceinline__ uint4 load_128bit(const uint4 *ptr) {
    uint4 out;
#if TARGET_SM >= 80
    // Ampere+: bypass L1 to avoid read-for-ownership pollution
    asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w) : "l"(ptr));
#elif TARGET_SM >= 70
    // Volta/Turing: .cg (cache global) for read-only coalesced access
    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w) : "l"(ptr));
#else
    // Pascal: no .cg cache modifier — use __ldg (cached read-only)
    out = __ldg(ptr);
#endif
    return out;
}

// BF16 dot product: 8 bf16 weights × 8 bf16 activations → f32
__device__ __forceinline__ float dot8_bf16(const uint4 &w_u4, const half_t *act) {
    const half_t *w = reinterpret_cast<const half_t *>(&w_u4);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; i++)
        sum += H2F(w[i]) * H2F(act[i]);
    return sum;
}


// =============================================================================
// RMSNorm — reads bf16 input, writes bf16 output
// =============================================================================

__device__ void rmsnorm_redundant(
    const half_t *__restrict__ input,
    const half_t *__restrict__ weight,
    half_t *__restrict__ s_out,        // shared memory bf16
    half_t *__restrict__ g_residual)   // global bf16
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ float smem_reduce[NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float v = H2F(__ldg(input + i));
        s_out[i] = F2H(v);
        local_sum_sq += v * v;
    }

    if (block_id == 0) {
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE)
            g_residual[i] = s_out[i];
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
            smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
    }
    __syncthreads();

    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float w = H2F(__ldg(weight + i));
        float v = H2F(s_out[i]);
        s_out[i] = F2H(v * rstd * (1.0f + w));
    }
    __syncthreads();
}

// RMSNorm from bf16 buffer (for post-attn norm)
__device__ void rmsnorm_from_bf16(
    const half_t *__restrict__ input,
    const half_t *__restrict__ weight,
    half_t *__restrict__ s_out,
    half_t *__restrict__ g_residual)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ float smem_reduce[NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float v = H2F(input[i]);
        s_out[i] = F2H(v);
        local_sum_sq += v * v;
    }

    if (block_id == 0) {
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE)
            g_residual[i] = s_out[i];
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
            smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
    }
    __syncthreads();

    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float w = H2F(__ldg(weight + i));
        float v = H2F(s_out[i]);
        s_out[i] = F2H(v * rstd * (1.0f + w));
    }
    __syncthreads();
}

// =============================================================================
// BF16 Matvec: warp-per-row, activations in shared memory (bf16)
// =============================================================================

__device__ void matvec_bf16(
    const half_t *__restrict__ s_input,  // shared memory bf16 [in_dim]
    const half_t *__restrict__ weight,   // [out_dim, in_dim] bf16
    float *__restrict__ output,                  // [out_dim] f32 (accumulate in f32)
    int in_dim, int out_dim, int num_blocks,
    const int8_t *__restrict__ q_input = nullptr,
    const float *__restrict__ q_scales = nullptr)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
#if MEGAKERNEL_LAYER_INT8
            const int8_t *w_row = reinterpret_cast<const int8_t *>(weight) + m * (in_dim + 16);
            const float scale = *reinterpret_cast<const float *>(w_row + in_dim);
            float sum = 0.0f;
#if MEGAKERNEL_LAYER_DP4A
            for (int k = lane_id * 4; k < in_dim; k += WARP_SIZE * 4) {
                const int qw = *reinterpret_cast<const int *>(w_row + k);
                const int qx = *reinterpret_cast<const int *>(q_input + k);
                sum += float(__dp4a(qw, qx, 0)) * q_scales[k / 32];
            }
#else
#pragma unroll 2
            for (int k = lane_id * 16; k < in_dim; k += WARP_SIZE * 16) {
                uint4 q_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                const int8_t *q = reinterpret_cast<const int8_t *>(&q_u4);
#if TARGET_SM >= 80
#pragma unroll
                for (int i = 0; i < 16; i += 2) {
                    const __nv_bfloat162 av = *reinterpret_cast<const __nv_bfloat162 *>(s_input + k + i);
                    const float2 af = __bfloat1622float2(av);
                    sum += float(q[i]) * af.x + float(q[i + 1]) * af.y;
                }
#else
#pragma unroll
                for (int i = 0; i < 16; ++i) sum += float(q[i]) * H2F(s_input[k+i]);
#endif
            }
#endif
            sum = warp_reduce_sum(sum) * scale;
#else
            const half_t *w_row = weight + m * in_dim;
            float sum = 0.0f;
#pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                sum += dot8_bf16(w_u4, s_input + k);
            }
            sum = warp_reduce_sum(sum);
#endif
            if (lane_id == 0) output[m] = sum;
        }
    }
}

// Fused gate+up+SiLU matvec (bf16 weights, bf16 activations)
__device__ void matvec_gate_up_silu_bf16(
    const half_t *__restrict__ s_input,
    const half_t *__restrict__ gate_weight,
    const half_t *__restrict__ up_weight,
    float *__restrict__ output,
    int in_dim, int out_dim, int num_blocks,
    const int8_t *__restrict__ q_input = nullptr,
    const float *__restrict__ q_scales = nullptr)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
#if MEGAKERNEL_LAYER_INT8
            const int8_t *g_row = reinterpret_cast<const int8_t *>(gate_weight) + m * (in_dim + 16);
            const int8_t *u_row = reinterpret_cast<const int8_t *>(up_weight) + m * (in_dim + 16);
            const float gs = *reinterpret_cast<const float *>(g_row + in_dim);
            const float us = *reinterpret_cast<const float *>(u_row + in_dim);
            float gate_sum = 0.0f, up_sum = 0.0f;
#if MEGAKERNEL_LAYER_DP4A
            for (int k = lane_id * 4; k < in_dim; k += WARP_SIZE * 4) {
                const int qx = *reinterpret_cast<const int *>(q_input + k);
                const float qs = q_scales[k / 32];
                gate_sum += float(__dp4a(*reinterpret_cast<const int *>(g_row + k), qx, 0)) * qs;
                up_sum += float(__dp4a(*reinterpret_cast<const int *>(u_row + k), qx, 0)) * qs;
            }
#else
#pragma unroll 2
            for (int k = lane_id * 16; k < in_dim; k += WARP_SIZE * 16) {
                uint4 g_u4 = load_128bit(reinterpret_cast<const uint4 *>(g_row + k));
                uint4 u_u4 = load_128bit(reinterpret_cast<const uint4 *>(u_row + k));
                const int8_t *g = reinterpret_cast<const int8_t *>(&g_u4);
                const int8_t *u = reinterpret_cast<const int8_t *>(&u_u4);
#if TARGET_SM >= 80
#pragma unroll
                for (int i = 0; i < 16; i += 2) {
                    const __nv_bfloat162 av = *reinterpret_cast<const __nv_bfloat162 *>(s_input + k + i);
                    const float2 af = __bfloat1622float2(av);
                    gate_sum += float(g[i]) * af.x + float(g[i + 1]) * af.y;
                    up_sum += float(u[i]) * af.x + float(u[i + 1]) * af.y;
                }
#else
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    float x = H2F(s_input[k+i]);
                    gate_sum += float(g[i]) * x;
                    up_sum += float(u[i]) * x;
                }
#endif
            }
#endif
            gate_sum = warp_reduce_sum(gate_sum) * gs;
            up_sum = warp_reduce_sum(up_sum) * us;
#else
            const half_t *g_row = gate_weight + m * in_dim;
            const half_t *u_row = up_weight + m * in_dim;
            float gate_sum = 0.0f, up_sum = 0.0f;
#pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 g_u4 = load_128bit(reinterpret_cast<const uint4 *>(g_row + k));
                uint4 u_u4 = load_128bit(reinterpret_cast<const uint4 *>(u_row + k));
                gate_sum += dot8_bf16(g_u4, s_input + k);
                up_sum += dot8_bf16(u_u4, s_input + k);
            }
            gate_sum = warp_reduce_sum(gate_sum);
            up_sum = warp_reduce_sum(up_sum);
#endif
            if (lane_id == 0)
                output[m] = fast_silu(gate_sum) * up_sum;
        }
    }
}

// Down projection + residual → bf16 hidden
__device__ void matvec_down_residual_bf16(
    const float *__restrict__ s_input,           // shared [INTER] f32
    const half_t *__restrict__ weight,    // [HIDDEN, INTER] bf16
    const half_t *__restrict__ residual,  // [HIDDEN] bf16
    half_t *__restrict__ hidden_out,      // [HIDDEN] bf16
    int in_dim, int out_dim, int num_blocks, int weight_in_dim = -1,
    const int8_t *__restrict__ q_input = nullptr,
    const float *__restrict__ q_scales = nullptr)
{
    // This needs f32 input (MLP intermediate is f32). Convert on the fly.
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    const int packed_in_dim = weight_in_dim > 0 ? weight_in_dim : in_dim;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
#if MEGAKERNEL_LAYER_INT8
            const int8_t *w_row = reinterpret_cast<const int8_t *>(weight) + m * (packed_in_dim + 16);
            const float scale = *reinterpret_cast<const float *>(w_row + in_dim);
            float sum = 0.0f;
#if MEGAKERNEL_LAYER_DP4A
            for (int k = lane_id * 4; k < in_dim; k += WARP_SIZE * 4) {
                sum += float(__dp4a(*reinterpret_cast<const int *>(w_row + k),
                                    *reinterpret_cast<const int *>(q_input + k), 0))
                       * q_scales[k / 32];
            }
#else
#pragma unroll 2
            for (int k = lane_id * 16; k < in_dim; k += WARP_SIZE * 16) {
                uint4 q_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                const int8_t *q = reinterpret_cast<const int8_t *>(&q_u4);
#pragma unroll
                for (int i = 0; i < 16; ++i) sum += float(q[i]) * s_input[k+i];
            }
#endif
            sum = warp_reduce_sum(sum) * scale;
#else
            const half_t *w_row = weight + m * packed_in_dim;
            float sum = 0.0f;
            // Weight is bf16, input is f32 — convert input to bf16 on the fly
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                const half_t *w = reinterpret_cast<const half_t *>(&w_u4);
#pragma unroll
                for (int i = 0; i < 8; i++)
                    sum += H2F(w[i]) * s_input[k + i];
            }
            sum = warp_reduce_sum(sum);
#endif
            if (lane_id == 0)
                hidden_out[m] = F2H(sum + H2F(residual[m]));
        }
    }
}

// O projection + residual → bf16
__device__ void matvec_o_residual_bf16(
    const float *__restrict__ s_input,           // shared [Q_SIZE] f32
    const half_t *__restrict__ weight,
    const half_t *__restrict__ residual,
    half_t *__restrict__ hidden_out,
    int in_dim, int out_dim, int num_blocks,
    const int8_t *__restrict__ q_input = nullptr,
    const float *__restrict__ q_scales = nullptr)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
#if MEGAKERNEL_LAYER_INT8
            const int8_t *w_row = reinterpret_cast<const int8_t *>(weight) + m * (in_dim + 16);
            const float scale = *reinterpret_cast<const float *>(w_row + in_dim);
            float sum = 0.0f;
#if MEGAKERNEL_LAYER_DP4A
            for (int k = lane_id * 4; k < in_dim; k += WARP_SIZE * 4) {
                sum += float(__dp4a(*reinterpret_cast<const int *>(w_row + k),
                                    *reinterpret_cast<const int *>(q_input + k), 0))
                       * q_scales[k / 32];
            }
#else
#pragma unroll 2
            for (int k = lane_id * 16; k < in_dim; k += WARP_SIZE * 16) {
                uint4 q_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                const int8_t *q = reinterpret_cast<const int8_t *>(&q_u4);
#pragma unroll
                for (int i = 0; i < 16; ++i) sum += float(q[i]) * s_input[k+i];
            }
#endif
            sum = warp_reduce_sum(sum) * scale;
#else
            const half_t *w_row = weight + m * in_dim;
            float sum = 0.0f;
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                const half_t *w = reinterpret_cast<const half_t *>(&w_u4);
#pragma unroll
                for (int i = 0; i < 8; i++)
                    sum += H2F(w[i]) * s_input[k + i];
            }
            sum = warp_reduce_sum(sum);
#endif
            if (lane_id == 0)
                hidden_out[m] = F2H(sum + H2F(residual[m]));
        }
    }
}

// =============================================================================
// Full Attention layer (bf16)
// =============================================================================

__device__ void full_attention_layer(
    AtomicGridSync &grid,
    const FullAttnWeights &w,
    const half_t *__restrict__ input,
    half_t *__restrict__ k_cache,
    half_t *__restrict__ v_cache,
    half_t *__restrict__ g_residual,  // [HIDDEN] bf16
    float *__restrict__ g_activations,        // scratch f32
    float *__restrict__ g_q,                  // [FA_QPROJ_SIZE] f32
    float *__restrict__ g_kv,                 // [FA_KV_SIZE*2] f32
    float *__restrict__ g_attn_out,           // [FA_Q_SIZE] f32
    float *__restrict__ g_mlp_inter,          // [INTER] f32
    half_t *__restrict__ hidden_out,   // [HIDDEN] bf16
    int layer_idx, int position, int max_seq_len,
    half_t *__restrict__ shmem)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Phase 1: RMSNorm + QKV projection
    half_t *s_norm = shmem;
    rmsnorm_redundant(input, w.input_layernorm_weight, s_norm, g_residual);
#if MEGAKERNEL_LAYER_DP4A
    int8_t *q_input = reinterpret_cast<int8_t *>(reinterpret_cast<char *>(shmem) + SHMEM_FLOATS_GQA * sizeof(float));
    float *q_scales = reinterpret_cast<float *>(q_input + MAX_ACT_DIM);
    layer_quantize_bf16(s_norm, q_input, q_scales, HIDDEN_SIZE);
#else
    const int8_t *q_input = nullptr;
    const float *q_scales = nullptr;
#endif
    matvec_bf16(s_norm, w.q_proj_weight, g_q, HIDDEN_SIZE, FA_QPROJ_SIZE, num_blocks, q_input, q_scales);
    matvec_bf16(s_norm, w.k_proj_weight, g_kv, HIDDEN_SIZE, FA_KV_SIZE, num_blocks, q_input, q_scales);
    matvec_bf16(s_norm, w.v_proj_weight, g_kv + FA_KV_SIZE, HIDDEN_SIZE, FA_KV_SIZE, num_blocks, q_input, q_scales);
    grid.sync();

    // Phase 2: QK norm + partial RoPE + KV cache write
    if (block_id == 0) {
        float *k_buf = g_kv, *v_buf = g_kv + FA_KV_SIZE;
        for (int h = warp_id; h < FA_NUM_KV_HEADS; h += NUM_WARPS) {
            float *kh = k_buf + h * FA_HEAD_DIM, *vh = v_buf + h * FA_HEAD_DIM;
            half_t *kc = k_cache + h * max_seq_len * FA_HEAD_DIM + position * FA_HEAD_DIM;
            half_t *vc = v_cache + h * max_seq_len * FA_HEAD_DIM + position * FA_HEAD_DIM;
            float ss = 0; for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) ss += kh[i]*kh[i];
            ss = warp_reduce_sum(ss); float sc = rsqrtf(ss / float(FA_HEAD_DIM) + RMS_EPS);
            sc = SHFL_SYNC(0xffffffff, sc, 0);
            for (int i = lane_id; i < FA_ROTARY_DIM/2; i += WARP_SIZE) {
                int p = i + FA_ROTARY_DIM/2;
                float fe = float(2*i) / float(FA_ROTARY_DIM);
                float freq = float(position) / powf(FA_ROPE_THETA, fe);
                float cv = cosf(freq), sv = sinf(freq);
                apply_rope_pair(kh, kc, w.k_norm_weight, i, p, sc, cv, sv);
                vc[i] = F2H(vh[i]); vc[p] = F2H(vh[p]);
            }
            for (int i = FA_ROTARY_DIM + lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) {
                kc[i] = F2H(kh[i]*sc*(1.0f+H2F(__ldg(w.k_norm_weight+i))));
                vc[i] = F2H(vh[i]);
            }
        }
    }
    // Q norm + RoPE (all blocks)
    {
        int hpb = (FA_NUM_Q_HEADS + num_blocks - 1) / num_blocks;
        int hs = block_id * hpb, he = min(hs + hpb, FA_NUM_Q_HEADS);
        for (int qh = hs; qh < he; qh++) {
            float *qh_ptr = g_q + qh * FA_HEAD_DIM * 2;
            if (warp_id == 0) {
                float ss = 0; for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) ss += qh_ptr[i]*qh_ptr[i];
                ss = warp_reduce_sum(ss); float sc = rsqrtf(ss / float(FA_HEAD_DIM) + RMS_EPS);
                sc = SHFL_SYNC(0xffffffff, sc, 0);
                for (int i = lane_id; i < FA_ROTARY_DIM/2; i += WARP_SIZE) {
                    int p = i + FA_ROTARY_DIM/2;
                    float fe = float(2*i) / float(FA_ROTARY_DIM);
                    float freq = float(position) / powf(FA_ROPE_THETA, fe);
                    float cv = cosf(freq), sv = sinf(freq);
                    apply_rope_pair(qh_ptr, qh_ptr, w.q_norm_weight, i, p, sc, cv, sv);
                }
                for (int i = FA_ROTARY_DIM + lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) {
                    qh_ptr[i] = qh_ptr[i]*sc*(1.0f+H2F(__ldg(w.q_norm_weight+i)));
                }
            }
        }
    }
    grid.sync();

    // Phase 3: Attention decode (online softmax + sigmoid gate)
    {
        int cache_len = position + 1;
        const int scan_start = attention_scan_start(cache_len);
        float attn_scale = 1.0f / sqrtf(float(FA_HEAD_DIM));
        __shared__ float s_max_score[NUM_WARPS];
        __shared__ float s_sum_exp[NUM_WARPS];
        float *s_attn_scratch = reinterpret_cast<float *>(shmem);
#if MEGAKERNEL_DECODE_SPLITKV
        split_kv_attention_onepass(
            grid, g_q, k_cache, v_cache,
            g_activations + HIDDEN_SIZE, g_attn_out,
            position, max_seq_len, attn_scale, shmem);
#elif MEGAKERNEL_DECODE_GQA_REUSE
        // Four Q heads share each KV head. Stage K/V tiles once, then let four
        // warp groups (one per Q head) consume the same tile. This preserves
        // the legacy parallelism while removing the four redundant cache reads.
        if (block_id < FA_NUM_KV_HEADS) {
            constexpr int Q_PER_KV = FA_GQA_RATIO;
            // Split the block evenly across the four query heads sharing this
            // KV head.  The old fixed value (4) was tuned for 512 threads;
            // the default 1024-thread block has 32 warps and needs 8 per Q.
            constexpr int WARPS_PER_Q = NUM_WARPS / Q_PER_KV;
            static_assert(NUM_WARPS % Q_PER_KV == 0, "GQA warp partition");
            constexpr int EPL = FA_HEAD_DIM / WARP_SIZE;
            float *gqa_scratch = s_attn_scratch + GQA_TILE_FLOATS;
            half_t *tile_k = reinterpret_cast<half_t *>(shmem);
            half_t *tile_v = tile_k + GQA_TILE_POS * FA_HEAD_DIM;
            int q_group = warp_id / WARPS_PER_Q;
            int q_warp = warp_id % WARPS_PER_Q;
            float q_local[EPL], out_acc[EPL];
            float max_score = -INFINITY, sum_exp = 0.0f;
            int qh = block_id * Q_PER_KV + q_group;
            float *q_head = g_q + qh * FA_HEAD_DIM * 2;
            for (int e = 0; e < EPL; ++e) {
                q_local[e] = q_head[lane_id * EPL + e];
                out_acc[e] = 0.0f;
            }
            for (int tile_start = scan_start; tile_start < cache_len; tile_start += GQA_TILE_POS) {
                int tile_count = min(GQA_TILE_POS, cache_len - tile_start);
                int tile_elems = tile_count * FA_HEAD_DIM;
                for (int i = threadIdx.x; i < tile_elems; i += BLOCK_SIZE) {
                    int pos = i / FA_HEAD_DIM;
                    int dim = i - pos * FA_HEAD_DIM;
                    const half_t *kp = k_cache + block_id * max_seq_len * FA_HEAD_DIM
                                     + (tile_start + pos) * FA_HEAD_DIM;
                    const half_t *vp = v_cache + block_id * max_seq_len * FA_HEAD_DIM
                                     + (tile_start + pos) * FA_HEAD_DIM;
                    tile_k[pos * FA_HEAD_DIM + dim] = __ldg(kp + dim);
                    tile_v[pos * FA_HEAD_DIM + dim] = __ldg(vp + dim);
                }
                __syncthreads();
                for (int ti = q_warp; ti < tile_count; ti += WARPS_PER_Q) {
                    const half_t *kp = tile_k + ti * FA_HEAD_DIM;
                    const half_t *vp = tile_v + ti * FA_HEAD_DIM;
                    float score = 0.0f;
                    for (int e = 0; e < EPL; ++e)
                        score += q_local[e] * H2F(kp[lane_id * EPL + e]);
                    score = warp_reduce_sum(score) * attn_scale;
                    score = SHFL_SYNC(0xffffffff, score, 0);
                    float old_max = max_score;
                    max_score = fmaxf(max_score, score);
                    float exp_diff = attention_exp(old_max - max_score);
                    sum_exp = sum_exp * exp_diff + attention_exp(score - max_score);
                    float wt = attention_exp(score - max_score);
                    for (int e = 0; e < EPL; ++e)
                        out_acc[e] = out_acc[e] * exp_diff + wt * H2F(vp[lane_id * EPL + e]);
                }
                __syncthreads();
            }
            int slot = q_group * WARPS_PER_Q + q_warp;
            if (lane_id == 0) {
                s_max_score[slot] = max_score;
                s_sum_exp[slot] = sum_exp;
            }
            float *dst = gqa_scratch + slot * FA_HEAD_DIM;
            for (int e = 0; e < EPL; ++e)
                dst[lane_id * EPL + e] = out_acc[e];
            __syncthreads();
            if (warp_id == 0) {
                for (int qi = 0; qi < Q_PER_KV; ++qi) {
                    int out_qh = block_id * Q_PER_KV + qi;
                    float *out_q_head = g_q + out_qh * FA_HEAD_DIM * 2;
                    float gm = -INFINITY;
                    for (int ww = 0; ww < WARPS_PER_Q; ++ww)
                        gm = fmaxf(gm, s_max_score[qi * WARPS_PER_Q + ww]);
                    float ts = 0.0f, fo[EPL];
                    for (int e = 0; e < EPL; ++e) fo[e] = 0.0f;
                    for (int ww = 0; ww < WARPS_PER_Q; ++ww) {
                        int slot = qi * WARPS_PER_Q + ww;
                        float scale = attention_exp(s_max_score[slot] - gm);
                        ts += s_sum_exp[slot] * scale;
                        const float *src = gqa_scratch + slot * FA_HEAD_DIM;
                        for (int e = 0; e < EPL; ++e)
                            fo[e] += src[lane_id * EPL + e] * scale;
                    }
                    float *gate_ptr = out_q_head + FA_HEAD_DIM;
                    float rcp = 1.0f / ts;
                    float *out_head = g_attn_out + out_qh * FA_HEAD_DIM;
                    for (int e = 0; e < EPL; ++e) {
                        int idx = lane_id * EPL + e;
                        out_head[idx] = fo[e] * rcp * fast_sigmoid(gate_ptr[idx]);
                    }
                }
            }
            __syncthreads();
        } else {
            // Other cooperative-grid blocks are idle during this phase.
        }
        grid.sync();
#else
        int hpb = (FA_NUM_Q_HEADS + num_blocks - 1) / num_blocks;
        int hs = block_id * hpb, he = min(hs + hpb, FA_NUM_Q_HEADS);
        constexpr int EPL = FA_HEAD_DIM / WARP_SIZE;

        for (int qh = hs; qh < he; qh++) {
            int kvh = qh / FA_GQA_RATIO;
            float *q_head = g_q + qh * FA_HEAD_DIM * 2;
            float *out_head = g_attn_out + qh * FA_HEAD_DIM;
            float max_score = -INFINITY, sum_exp = 0;
            float out_acc[EPL], q_local[EPL];
            for (int e = 0; e < EPL; e++) {
                out_acc[e] = 0;
                q_local[e] = q_head[lane_id * EPL + e];
            }
            // The long-context case executes thousands of iterations per
            // query head. Unroll a small factor to reduce loop/index overhead
            // without changing the online-softmax order.
#pragma unroll 4
            for (int pos = scan_start + warp_id; pos < cache_len; pos += NUM_WARPS) {
                const half_t *k_pos = k_cache + kvh*max_seq_len*FA_HEAD_DIM + pos*FA_HEAD_DIM;
                const half_t *v_pos = v_cache + kvh*max_seq_len*FA_HEAD_DIM + pos*FA_HEAD_DIM;
                float score = 0;
#if MEGAKERNEL_DECODE_ATTENTION_VEC2
                for (int e = 0; e < EPL; e += 2) {
                    int idx = lane_id * EPL + e;
#if TARGET_SM >= 80
                    __nv_bfloat162 kv = __ldg(reinterpret_cast<const __nv_bfloat162 *>(k_pos + idx));
                    float2 kf = __bfloat1622float2(kv);
                    score += q_local[e] * kf.x + q_local[e + 1] * kf.y;
#else
                    __half2 kv = __ldg(reinterpret_cast<const __half2 *>(k_pos + idx));
                    float2 kf = __half22float2(kv);
                    score += q_local[e] * kf.x + q_local[e + 1] * kf.y;
#endif
                }
#else
                for (int e = 0; e < EPL; e++) score += q_local[e] * H2F(__ldg(k_pos + lane_id*EPL+e));
#endif
                score = warp_reduce_sum(score) * attn_scale;
                score = SHFL_SYNC(0xffffffff, score, 0);
                float old_max = max_score; max_score = fmaxf(max_score, score);
                float exp_diff = attention_exp(old_max - max_score);
                sum_exp = sum_exp * exp_diff + attention_exp(score - max_score);
                float wt = attention_exp(score - max_score);
#if MEGAKERNEL_DECODE_ATTENTION_VEC2
                for (int e = 0; e < EPL; e += 2) {
                    int idx = lane_id * EPL + e;
#if TARGET_SM >= 80
                    __nv_bfloat162 vv = __ldg(reinterpret_cast<const __nv_bfloat162 *>(v_pos + idx));
                    float2 vf = __bfloat1622float2(vv);
                    out_acc[e] = out_acc[e] * exp_diff + wt * vf.x;
                    out_acc[e + 1] = out_acc[e + 1] * exp_diff + wt * vf.y;
#else
                    __half2 vv = __ldg(reinterpret_cast<const __half2 *>(v_pos + idx));
                    float2 vf = __half22float2(vv);
                    out_acc[e] = out_acc[e] * exp_diff + wt * vf.x;
                    out_acc[e + 1] = out_acc[e + 1] * exp_diff + wt * vf.y;
#endif
                }
#else
                for (int e = 0; e < EPL; e++)
                    out_acc[e] = out_acc[e]*exp_diff + wt*H2F(__ldg(v_pos + lane_id*EPL+e));
#endif
            }
            if (lane_id == 0) { s_max_score[warp_id] = max_score; s_sum_exp[warp_id] = sum_exp; }
            for (int e = 0; e < EPL; e++) s_attn_scratch[warp_id*FA_HEAD_DIM + lane_id*EPL+e] = out_acc[e];
            __syncthreads();

            if (warp_id == 0) {
                float gm = -INFINITY; for (int ww = 0; ww < NUM_WARPS; ww++) if (s_max_score[ww] > -INFINITY) gm = fmaxf(gm, s_max_score[ww]);
                float ts = 0; float fo[EPL]; for (int e = 0; e < EPL; e++) fo[e] = 0;
                for (int ww = 0; ww < NUM_WARPS; ww++) {
                    if (s_max_score[ww] > -INFINITY) {
                        float s = attention_exp(s_max_score[ww]-gm); ts += s_sum_exp[ww]*s;
                        for (int e = 0; e < EPL; e++) fo[e] += s_attn_scratch[ww*FA_HEAD_DIM+lane_id*EPL+e]*s;
                    }
                }
                float *gate_ptr = q_head + FA_HEAD_DIM;
                float rcp = 1.0f / ts;
                for (int e = 0; e < EPL; e++) {
                    int idx = lane_id*EPL+e;
                    out_head[idx] = fo[e]*rcp * fast_sigmoid(gate_ptr[idx]);
                }
            }
            __syncthreads();
        }
#endif
    }
    grid.sync();

    // Phase 4: O projection + residual → bf16
    {
        float *s_attn = reinterpret_cast<float *>(shmem);
        for (int i = threadIdx.x; i < FA_Q_SIZE; i += BLOCK_SIZE) s_attn[i] = g_attn_out[i];
        __syncthreads();
#if MEGAKERNEL_LAYER_DP4A
        layer_quantize_f32(s_attn, q_input, q_scales, FA_Q_SIZE);
#endif
        matvec_o_residual_bf16(s_attn, w.o_proj_weight, g_residual, hidden_out, FA_Q_SIZE, HIDDEN_SIZE, num_blocks, q_input, q_scales);
    }
    grid.sync();

#if MEGAKERNEL_APPROX_SKIP_MLP || MEGAKERNEL_APPROX_SKIP_MLP_EVERY > 0
    // Extreme approximation for speed/ablation experiments: retain the
    // attention/DeltaNet residual output and omit this layer's MLP entirely.
    // The phase-4 barrier above keeps all blocks aligned for the next layer.
    // Default zero is the exact path.
    if (MEGAKERNEL_APPROX_SKIP_MLP ||
        (MEGAKERNEL_APPROX_SKIP_MLP_EVERY > 0 &&
         (layer_idx % MEGAKERNEL_APPROX_SKIP_MLP_EVERY) == 0)) return;
#endif

    // Phase 5: Post-attn norm + MLP
    half_t *s_act = shmem;
    rmsnorm_from_bf16(hidden_out, w.post_attn_layernorm_weight, s_act, g_residual);
#if MEGAKERNEL_LAYER_DP4A
    layer_quantize_bf16(s_act, q_input, q_scales, HIDDEN_SIZE);
#endif
    matvec_gate_up_silu_bf16(s_act, w.gate_proj_weight, w.up_proj_weight,
                              g_mlp_inter, HIDDEN_SIZE, DECODE_MLP_WIDTH, num_blocks, q_input, q_scales);
    grid.sync();

    // Load MLP intermediate to shared (f32)
    float *s_mlp = reinterpret_cast<float *>(shmem);
    for (int i = threadIdx.x; i < DECODE_MLP_WIDTH; i += BLOCK_SIZE) s_mlp[i] = g_mlp_inter[i];
    __syncthreads();
#if MEGAKERNEL_LAYER_DP4A
    layer_quantize_f32(s_mlp, q_input, q_scales, INTERMEDIATE_SIZE);
#endif

    matvec_down_residual_bf16(s_mlp, w.down_proj_weight, g_residual, hidden_out,
                               DECODE_MLP_WIDTH, HIDDEN_SIZE, num_blocks, INTERMEDIATE_SIZE,
                               q_input, q_scales);
    grid.sync();
}

// =============================================================================
// DeltaNet layer (bf16) — warp-cooperative state-in-registers recurrence
// =============================================================================

#if MEGAKERNEL_DN_HEAD_SPLIT > 1
// Split each recurrent head's 128 value rows across multiple cooperative
// blocks. The projection grid already has 82 resident blocks, but the old
// recurrence used only one block per head (16 blocks total), leaving most of
// the GPU idle during the state update. One owner block performs conv/SiLU and
// writes normalized Q/K/V back to the scratch vector; split blocks then update
// disjoint state rows. The two grid barriers preserve the original ordering.
__device__ void deltanet_phase_split(
    AtomicGridSync &grid,
    const DeltaNetWeights &w,
    float *__restrict__ g_qkv,
    float *__restrict__ g_z,
    float *__restrict__ g_beta,
    float *__restrict__ g_alpha,
    float *__restrict__ g_dn_out,
    float *__restrict__ dn_state,
    float *__restrict__ conv_buf,
    int dn_layer_idx)
{
    constexpr int SPLIT = MEGAKERNEL_DN_HEAD_SPLIT;
    constexpr int J_PER_BLOCK = DN_VALUE_DIM / SPLIT;
    constexpr int J_PER_WARP = J_PER_BLOCK / NUM_WARPS;
    static_assert(DN_VALUE_DIM % SPLIT == 0, "DeltaNet head split must divide value rows");
    static_assert(J_PER_BLOCK >= NUM_WARPS && J_PER_BLOCK % NUM_WARPS == 0,
                  "DeltaNet head split must leave at least one row per warp");

    const int block_id = blockIdx.x;
    const int head_group = block_id / SPLIT;
    const int split_id = block_id % SPLIT;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const bool active = head_group < DN_NUM_HEADS;

    __shared__ float s_q[DN_KEY_DIM], s_k[DN_KEY_DIM], s_v[DN_VALUE_DIM];
    __shared__ float s_kq;

    if (active && split_id == 0) {
        const int h = head_group;
        float *layer_conv = conv_buf + dn_layer_idx * DN_CONV_CHANNELS * DN_CONV_KERNEL;
        const int head_ch[3] = {h * DN_KEY_DIM, DN_QK_SIZE + h * DN_KEY_DIM,
                                2 * DN_QK_SIZE + h * DN_VALUE_DIM};
        for (int region = 0; region < 3; ++region) {
            const int ch_base = head_ch[region];
            const int ch_count = (region < 2) ? DN_KEY_DIM : DN_VALUE_DIM;
            float *dst = (region == 0) ? s_q : (region == 1) ? s_k : s_v;
            for (int c = threadIdx.x; c < ch_count; c += BLOCK_SIZE) {
                const int ch = ch_base + c;
                const float h0 = layer_conv[ch * DN_CONV_KERNEL + 1];
                const float h1 = layer_conv[ch * DN_CONV_KERNEL + 2];
                const float h2 = layer_conv[ch * DN_CONV_KERNEL + 3];
                layer_conv[ch * DN_CONV_KERNEL] = h0;
                layer_conv[ch * DN_CONV_KERNEL + 1] = h1;
                layer_conv[ch * DN_CONV_KERNEL + 2] = h2;
                layer_conv[ch * DN_CONV_KERNEL + 3] = g_qkv[ch];
                float co = 0.0f;
#pragma unroll
                for (int t = 0; t < DN_CONV_KERNEL; ++t)
                    co += layer_conv[ch * DN_CONV_KERNEL + t] * H2F(__ldg(w.conv1d_weight + ch * DN_CONV_KERNEL + t));
                dst[c] = fast_silu(co);
            }
        }
        __syncthreads();

        if (warp_id == 0) {
            float sq = 0.0f;
            for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) sq += s_q[i] * s_q[i];
            sq = warp_reduce_sum(sq);
            float n = rsqrtf(sq + 1e-6f) * (1.0f / 11.313708498984761f);
            n = SHFL_SYNC(0xffffffff, n, 0);
            for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) s_q[i] *= n;
        }
        if (warp_id == 1) {
            float sq = 0.0f;
            for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) sq += s_k[i] * s_k[i];
            sq = warp_reduce_sum(sq);
            float n = rsqrtf(sq + 1e-6f);
            n = SHFL_SYNC(0xffffffff, n, 0);
            for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) s_k[i] *= n;
        }
        if (threadIdx.x == 0) {
            g_beta[h] = fast_sigmoid(g_beta[h]);
            const float a_log_val = H2F(__ldg(w.a_log + h));
            const float dt_b = H2F(__ldg(w.dt_bias + h));
            const float x = g_alpha[h] + dt_b;
            const float sp = (x > 20.0f) ? x : logf(1.0f + fast_exp(x));
            g_alpha[h] = fast_exp(-fast_exp(a_log_val) * sp);
        }
        __syncthreads();
        for (int c = threadIdx.x; c < DN_KEY_DIM; c += BLOCK_SIZE) {
            g_qkv[h * DN_KEY_DIM + c] = s_q[c];
            g_qkv[DN_QK_SIZE + h * DN_KEY_DIM + c] = s_k[c];
        }
        for (int c = threadIdx.x; c < DN_VALUE_DIM; c += BLOCK_SIZE)
            g_qkv[2 * DN_QK_SIZE + h * DN_VALUE_DIM + c] = s_v[c];
    }
    grid.sync();

    if (active) {
        const int h = head_group;
        for (int c = threadIdx.x; c < DN_KEY_DIM; c += BLOCK_SIZE) {
            s_q[c] = g_qkv[h * DN_KEY_DIM + c];
            s_k[c] = g_qkv[DN_QK_SIZE + h * DN_KEY_DIM + c];
        }
        for (int c = threadIdx.x; c < DN_VALUE_DIM; c += BLOCK_SIZE)
            s_v[c] = g_qkv[2 * DN_QK_SIZE + h * DN_VALUE_DIM + c];
        __syncthreads();

        if (warp_id == 0) {
            float kq = 0.0f;
            for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) kq += s_k[i] * s_q[i];
            kq = warp_reduce_sum(kq);
            if (lane_id == 0) s_kq = kq;
        }
        __syncthreads();

        const float decay = g_alpha[h], beta = g_beta[h];
        float *state = dn_state + h * DN_KEY_DIM * DN_VALUE_DIM;
        float *out_head = g_dn_out + h * DN_VALUE_DIM;
        const int j_start = split_id * J_PER_BLOCK;

#pragma unroll
        for (int jj = 0; jj < J_PER_WARP; ++jj) {
            const int j = j_start + warp_id * J_PER_WARP + jj;
            float s_regs[DN_KEY_DIM / WARP_SIZE];
            float stk = 0.0f, sqv = 0.0f;
#pragma unroll
            for (int ii = 0; ii < DN_KEY_DIM / WARP_SIZE; ++ii) {
                const int i = lane_id + ii * WARP_SIZE;
                const float sv = state[j * DN_KEY_DIM + i];
                s_regs[ii] = sv;
                stk += sv * s_k[i];
                sqv += sv * s_q[i];
            }
            stk = warp_reduce_sum(stk);
            sqv = warp_reduce_sum(sqv);
            stk = SHFL_SYNC(0xffffffff, stk, 0);
            sqv = SHFL_SYNC(0xffffffff, sqv, 0);
            const float error_j = (s_v[j] - decay * stk) * beta;
            const float o_j = decay * sqv + error_j * s_kq;
            if (lane_id == 0) out_head[j] = o_j;
#pragma unroll
            for (int ii = 0; ii < DN_KEY_DIM / WARP_SIZE; ++ii) {
                const int i = lane_id + ii * WARP_SIZE;
                state[j * DN_KEY_DIM + i] = s_regs[ii] * decay + s_k[i] * error_j;
            }
        }
    }
    grid.sync();

    // Only the owner block performs the full-head gated RMSNorm after all
    // split blocks have written their disjoint output rows.
    if (active && split_id == 0) {
        const int h = head_group;
        float *out_head = g_dn_out + h * DN_VALUE_DIM;
        __shared__ float smem_gnorm[NUM_WARPS];
        float sq = 0.0f;
        for (int i = threadIdx.x; i < DN_VALUE_DIM; i += BLOCK_SIZE) {
            const float v = out_head[i];
            sq += v * v;
        }
        sq = warp_reduce_sum(sq);
        if (lane_id == 0) smem_gnorm[warp_id] = sq;
        __syncthreads();
        if (warp_id == 0) {
            float v = (lane_id < NUM_WARPS) ? smem_gnorm[lane_id] : 0.0f;
            v = warp_reduce_sum(v);
            if (lane_id == 0) smem_gnorm[0] = rsqrtf(v / DN_VALUE_DIM + RMS_EPS);
        }
        __syncthreads();
        const float rstd = smem_gnorm[0];
        for (int i = threadIdx.x; i < DN_VALUE_DIM; i += BLOCK_SIZE) {
            const float normed = out_head[i] * rstd * H2F(__ldg(w.norm_weight + i));
            const float gate = fast_silu(g_z[h * DN_VALUE_DIM + i]);
            out_head[i] = normed * gate;
        }
    }
    grid.sync();
}
#endif

__device__ void deltanet_layer(
    AtomicGridSync &grid,
    const DeltaNetWeights &w,
    const half_t *__restrict__ input,
    half_t *__restrict__ g_residual,
    float *__restrict__ g_activations,
    float *__restrict__ g_qkv,
    float *__restrict__ g_z,
    float *__restrict__ g_beta,
    float *__restrict__ g_alpha,
    float *__restrict__ g_dn_out,
    float *__restrict__ g_mlp_inter,
    float *__restrict__ dn_state,     // [DN_NUM_HEADS, DN_KEY, DN_VAL] f32
    float *__restrict__ conv_buf,     // [DN_CONV_CH, DN_CONV_K] f32
    half_t *__restrict__ hidden_out,
    int dn_layer_idx, int layer_idx,
    half_t *__restrict__ shmem)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Phase 1: RMSNorm + projections
    half_t *s_norm = shmem;
    rmsnorm_redundant(input, w.input_layernorm_weight, s_norm, g_residual);
#if MEGAKERNEL_LAYER_DP4A
    int8_t *q_input = reinterpret_cast<int8_t *>(reinterpret_cast<char *>(shmem) + SHMEM_FLOATS_GQA * sizeof(float));
    float *q_scales = reinterpret_cast<float *>(q_input + MAX_ACT_DIM);
    layer_quantize_bf16(s_norm, q_input, q_scales, HIDDEN_SIZE);
#else
    const int8_t *q_input = nullptr;
    const float *q_scales = nullptr;
#endif
    matvec_bf16(s_norm, w.qkv_proj_weight, g_qkv, HIDDEN_SIZE, DN_CONV_CHANNELS, num_blocks, q_input, q_scales);
    matvec_bf16(s_norm, w.z_proj_weight, g_z, HIDDEN_SIZE, DN_V_SIZE, num_blocks, q_input, q_scales);
    matvec_bf16(s_norm, w.beta_proj_weight, g_beta, HIDDEN_SIZE, DN_NUM_HEADS, num_blocks, q_input, q_scales);
    matvec_bf16(s_norm, w.alpha_proj_weight, g_alpha, HIDDEN_SIZE, DN_NUM_HEADS, num_blocks, q_input, q_scales);
    grid.sync();

    // Phase 2+3: Conv1d + recurrence.  The production path uses one block
    // per head; the split path partitions each head's value rows across
    // multiple blocks so the decode recurrence can occupy more SMs.
#if MEGAKERNEL_DN_HEAD_SPLIT == 1
    if (block_id < DN_NUM_HEADS) {
        int h = block_id;
        float *layer_conv = conv_buf + dn_layer_idx * DN_CONV_CHANNELS * DN_CONV_KERNEL;

        // Conv1d + SiLU
        __shared__ float s_q[DN_KEY_DIM], s_k[DN_KEY_DIM], s_v[DN_VALUE_DIM];
        int head_ch[3] = {h*DN_KEY_DIM, DN_QK_SIZE+h*DN_KEY_DIM, 2*DN_QK_SIZE+h*DN_VALUE_DIM};
        for (int region = 0; region < 3; region++) {
            int ch_base = head_ch[region], ch_count = (region < 2) ? DN_KEY_DIM : DN_VALUE_DIM;
            float *dst = (region == 0) ? s_q : (region == 1) ? s_k : s_v;
            for (int c = threadIdx.x; c < ch_count; c += BLOCK_SIZE) {
                int ch = ch_base + c;
                float h0=layer_conv[ch*DN_CONV_KERNEL+1], h1=layer_conv[ch*DN_CONV_KERNEL+2], h2=layer_conv[ch*DN_CONV_KERNEL+3];
                layer_conv[ch*DN_CONV_KERNEL]=h0; layer_conv[ch*DN_CONV_KERNEL+1]=h1;
                layer_conv[ch*DN_CONV_KERNEL+2]=h2; layer_conv[ch*DN_CONV_KERNEL+3]=g_qkv[ch];
                float co = 0;
                for (int t = 0; t < DN_CONV_KERNEL; t++)
                    co += layer_conv[ch*DN_CONV_KERNEL+t] * H2F(__ldg(w.conv1d_weight + ch*DN_CONV_KERNEL+t));
                dst[c] = fast_silu(co);
            }
        }

        // Beta/alpha activations
        if (threadIdx.x == 0) {
            g_beta[h] = fast_sigmoid(g_beta[h]);
            float a_log_val = H2F(__ldg(w.a_log + h));
            float dt_b = H2F(__ldg(w.dt_bias + h));
            float x = g_alpha[h] + dt_b;
            float sp = (x > 20.0f) ? x : logf(1.0f + fast_exp(x));
            g_alpha[h] = fast_exp(-fast_exp(a_log_val) * sp);
        }
        __syncthreads();

        // L2 normalize Q, K
        constexpr float Q_SCALE = 1.0f / 11.313708498984761f;
        if (warp_id == 0) {
            float sq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) sq += s_q[i]*s_q[i];
            sq = warp_reduce_sum(sq); float n = rsqrtf(sq+1e-6f)*Q_SCALE;
            n = SHFL_SYNC(0xffffffff,n,0); for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) s_q[i] *= n;
        }
        if (warp_id == 1) {
            float sq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) sq += s_k[i]*s_k[i];
            sq = warp_reduce_sum(sq); float n = rsqrtf(sq+1e-6f);
            n = SHFL_SYNC(0xffffffff,n,0); for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) s_k[i] *= n;
        }
        __syncthreads();

        float decay = g_alpha[h], beta = g_beta[h];

        // k·q dot
        __shared__ float s_kq;
        if (warp_id == 0) {
            float kq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) kq += s_k[i]*s_q[i];
            kq = warp_reduce_sum(kq); if (lane_id == 0) s_kq = kq;
        }
        __syncthreads();
        float kq = s_kq;

        // Warp-cooperative recurrence (state in global memory — decode is 1 token, fine)
        float *state = dn_state + h * DN_KEY_DIM * DN_VALUE_DIM;
        float *out_head = g_dn_out + h * DN_VALUE_DIM;

        constexpr int J_PER_WARP = DN_VALUE_DIM / NUM_WARPS;
        constexpr int I_PER_LANE = DN_KEY_DIM / WARP_SIZE;

#pragma unroll
        for (int jj = 0; jj < J_PER_WARP; jj++) {
            int j = warp_id * J_PER_WARP + jj;
            float s_regs[I_PER_LANE], stk = 0, sqv = 0;
#pragma unroll
            for (int ii = 0; ii < I_PER_LANE; ii++) {
                int i = lane_id + ii * WARP_SIZE;
                float sv = state[j*DN_KEY_DIM+i]; s_regs[ii] = sv;
                stk += sv * s_k[i]; sqv += sv * s_q[i];
            }
            stk = warp_reduce_sum(stk); sqv = warp_reduce_sum(sqv);
            stk = SHFL_SYNC(0xffffffff,stk,0); sqv = SHFL_SYNC(0xffffffff,sqv,0);
            float error_j = (s_v[j] - decay * stk) * beta;
            float o_j = decay * sqv + error_j * kq;
            if (lane_id == 0) out_head[j] = o_j;
#pragma unroll
            for (int ii = 0; ii < I_PER_LANE; ii++) {
                int i = lane_id + ii * WARP_SIZE;
                state[j*DN_KEY_DIM+i] = s_regs[ii] * decay + s_k[i] * error_j;
            }
        }

        // Gated RMSNorm
        __syncthreads();
        {
            __shared__ float smem_gnorm[NUM_WARPS];
            float sq = 0; for (int i = threadIdx.x; i < DN_VALUE_DIM; i += BLOCK_SIZE) sq += out_head[i]*out_head[i];
            sq = warp_reduce_sum(sq); if (lane_id == 0) smem_gnorm[warp_id] = sq; __syncthreads();
            if (warp_id == 0) { float v = (lane_id < NUM_WARPS) ? smem_gnorm[lane_id] : 0; v = warp_reduce_sum(v); if (lane_id == 0) smem_gnorm[0] = rsqrtf(v/DN_VALUE_DIM + RMS_EPS); }
            __syncthreads(); float rstd = smem_gnorm[0];
            for (int i = threadIdx.x; i < DN_VALUE_DIM; i += BLOCK_SIZE) {
                float normed = out_head[i] * rstd * H2F(__ldg(w.norm_weight + i));
                float gate = fast_silu(g_z[h*DN_VALUE_DIM+i]);
                out_head[i] = normed * gate;
            }
        }
    } else {
        // Idle blocks: could prefetch weights
    }
    grid.sync();
#else
    deltanet_phase_split(grid, w, g_qkv, g_z, g_beta, g_alpha, g_dn_out,
                         dn_state, conv_buf, dn_layer_idx);
#endif

    // Phase 4: Out projection + residual → bf16
    {
        float *s_dn = reinterpret_cast<float *>(shmem);
        for (int i = threadIdx.x; i < DN_V_SIZE; i += BLOCK_SIZE) s_dn[i] = g_dn_out[i];
        __syncthreads();
 #if MEGAKERNEL_LAYER_DP4A
        layer_quantize_f32(s_dn, q_input, q_scales, DN_V_SIZE);
 #endif
        matvec_o_residual_bf16(s_dn, w.out_proj_weight, g_residual, hidden_out, DN_V_SIZE, HIDDEN_SIZE, num_blocks, q_input, q_scales);
    }
    grid.sync();

#if MEGAKERNEL_APPROX_SKIP_MLP || MEGAKERNEL_APPROX_SKIP_MLP_EVERY > 0
    // See the full-attention layer above; this deliberately leaves the
    // post-DeltaNet residual unmodified.
    if (MEGAKERNEL_APPROX_SKIP_MLP ||
        (MEGAKERNEL_APPROX_SKIP_MLP_EVERY > 0 &&
         (layer_idx % MEGAKERNEL_APPROX_SKIP_MLP_EVERY) == 0)) return;
#endif

    // Phase 5: Post-attn norm + MLP
    half_t *s_act = shmem;
    rmsnorm_from_bf16(hidden_out, w.post_attn_layernorm_weight, s_act, g_residual);
#if MEGAKERNEL_LAYER_DP4A
    layer_quantize_bf16(s_act, q_input, q_scales, HIDDEN_SIZE);
#endif
    matvec_gate_up_silu_bf16(s_act, w.gate_proj_weight, w.up_proj_weight,
                              g_mlp_inter, HIDDEN_SIZE, DECODE_MLP_WIDTH, num_blocks, q_input, q_scales);
    grid.sync();

    float *s_mlp = reinterpret_cast<float *>(shmem);
    for (int i = threadIdx.x; i < DECODE_MLP_WIDTH; i += BLOCK_SIZE) s_mlp[i] = g_mlp_inter[i];
    __syncthreads();
#if MEGAKERNEL_LAYER_DP4A
    layer_quantize_f32(s_mlp, q_input, q_scales, INTERMEDIATE_SIZE);
#endif
    matvec_down_residual_bf16(s_mlp, w.down_proj_weight, g_residual, hidden_out,
                               DECODE_MLP_WIDTH, HIDDEN_SIZE, num_blocks, INTERMEDIATE_SIZE,
                               q_input, q_scales);
    grid.sync();
}

// =============================================================================
// LM Head: vocab projection + argmax
// =============================================================================

template <bool BATCHED>
__global__ void lm_head_kernel(
    const float *__restrict__ hidden,
    // In the int8 build each row is HIDDEN int8 values followed by one f32
    // per-row scale.  The prefill path continues to receive the BF16 tensor;
    // only decode uses this packed projection.
#if MEGAKERNEL_LM_INT8
    const int8_t *__restrict__ weight,
#else
    const half_t *__restrict__ weight,
#endif
    float *__restrict__ block_max_vals,
    int *__restrict__ block_max_idxs,
    int *__restrict__ output_token,
    unsigned int *__restrict__ sync_counter,
    const float *__restrict__ seen_token_mask,
    float repetition_penalty)
{
    if constexpr (BATCHED) {
        int batch_id = (int)blockIdx.z;
        hidden += batch_id * HIDDEN_SIZE;
        block_max_vals += batch_id * LM_NUM_BLOCKS;
        block_max_idxs += batch_id * LM_NUM_BLOCKS;
        output_token += batch_id;
        sync_counter += batch_id;
        seen_token_mask += batch_id * VOCAB_SIZE;
    }
    __shared__ float s_hidden[HIDDEN_SIZE];
#if MEGAKERNEL_LM_DP4A
    __shared__ __align__(4) int8_t s_hidden_i8[HIDDEN_SIZE];
    __shared__ float s_hidden_scales[HIDDEN_SIZE / 32];
#endif
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LM_BLOCK_SIZE) s_hidden[i] = hidden[i];
#if MEGAKERNEL_LM_DP4A
    // Per-group scales retain the DP4A speedup without the large logit error
    // from a single fixed scale across the whole final hidden vector.
    for (int g = threadIdx.x; g < HIDDEN_SIZE / 32; g += LM_BLOCK_SIZE) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(s_hidden[g * 32 + i]));
        s_hidden_scales[g] = amax / 127.0f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LM_BLOCK_SIZE) {
        const float scale = s_hidden_scales[i / 32];
        int q = scale > 0.0f ? __float2int_rn(s_hidden[i] / scale) : 0;
        q = max(-127, min(127, q));
        s_hidden_i8[i] = (int8_t)q;
    }
#endif
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE, lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = LM_BLOCK_SIZE / WARP_SIZE;
    int rpb = (VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int rs = blockIdx.x * rpb, re = min(rs + rpb, VOCAB_SIZE);

    float local_max = -INFINITY; int local_max_idx = -1;
    for (int m = rs + warp_id; m < re; m += num_warps) {
        float sum = 0;
#if MEGAKERNEL_LM_INT8
        // Keep every row 16-byte aligned for vectorized uint4 loads.
        const int8_t *w_row = weight + m * (HIDDEN_SIZE + 16);
        const float scale = *reinterpret_cast<const float *>(w_row + HIDDEN_SIZE);
#if MEGAKERNEL_LM_DP4A
        float dot_scaled = 0.0f;
        for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
            int qw = *reinterpret_cast<const int *>(w_row + k);
            int hw = *reinterpret_cast<const int *>(s_hidden_i8 + k);
            dot_scaled += float(__dp4a(qw, hw, 0)) * s_hidden_scales[k / 32];
        }
        sum = warp_reduce_sum(dot_scaled) * scale;
#else
#pragma unroll 2
        for (int k = lane_id * 16; k < HIDDEN_SIZE; k += WARP_SIZE * 16) {
            uint4 q_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
            const int8_t *qp = reinterpret_cast<const int8_t *>(&q_u4);
#pragma unroll
            for (int i = 0; i < 16; ++i) sum += float(qp[i]) * s_hidden[k+i];
        }
#endif
#else
        const half_t *w_row = weight + m * HIDDEN_SIZE;
#pragma unroll 4
        for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
            uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
            const half_t *wp = reinterpret_cast<const half_t *>(&w_u4);
            for (int i = 0; i < 8; i++) sum += H2F(wp[i]) * s_hidden[k+i];
        }
#endif
        sum = warp_reduce_sum(sum);
#if MEGAKERNEL_LM_INT8
        sum *= scale;
#endif
        if (lane_id == 0 && repetition_penalty > 1.0f && seen_token_mask && seen_token_mask[m] > 0.0f) {
            sum = (sum > 0.0f) ? (sum / repetition_penalty) : (sum * repetition_penalty);
        }
        if (lane_id == 0 && sum > local_max) { local_max = sum; local_max_idx = m; }
    }
    local_max = SHFL_SYNC(0xffffffff, local_max, 0);
    local_max_idx = SHFL_SYNC(0xffffffff, local_max_idx, 0);

    __shared__ float wm[32]; __shared__ int wi[32];
    if (lane_id == 0) { wm[warp_id] = local_max; wi[warp_id] = local_max_idx; }
    __syncthreads();
    if (warp_id == 0) {
        float mv = (lane_id < num_warps) ? wm[lane_id] : -INFINITY;
        int mi = (lane_id < num_warps) ? wi[lane_id] : -1;
        for (int o = WARP_SIZE/2; o > 0; o /= 2) {
            float ov = SHFL_DOWN_SYNC(0xffffffff, mv, o);
            int oi = SHFL_DOWN_SYNC(0xffffffff, mi, o);
            if (ov > mv) { mv = ov; mi = oi; }
        }
        if (lane_id == 0) { block_max_vals[blockIdx.x] = mv; block_max_idxs[blockIdx.x] = mi; }
    }
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence(); atomicAdd(sync_counter, 1); }
    if (blockIdx.x == 0) {
        if (threadIdx.x == 0) { volatile unsigned int *vc = (volatile unsigned int *)sync_counter; while (*vc < (unsigned int)gridDim.x) {} __threadfence(); }
        __syncthreads();
        int tid = threadIdx.x; float bv = -INFINITY; int bi = -1;
        for (int i = tid; i < gridDim.x; i += LM_BLOCK_SIZE) { float v = block_max_vals[i]; if (v > bv) { bv = v; bi = block_max_idxs[i]; } }
        __shared__ float sv[256]; __shared__ int si[256];
        sv[tid] = bv; si[tid] = bi; __syncthreads();
        for (int s = LM_BLOCK_SIZE/2; s > 0; s >>= 1) { if (tid < s && sv[tid+s] > sv[tid]) { sv[tid] = sv[tid+s]; si[tid] = si[tid+s]; } __syncthreads(); }
        if (tid == 0) *output_token = si[0];
    }
}

// =============================================================================
// Main decode kernel
// =============================================================================

template <bool DEVICE_INPUT, bool BATCHED>
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
decode_kernel(
    const half_t *__restrict__ embed_weight,
    const half_t *__restrict__ final_norm_weight,
    const half_t *__restrict__ lm_head_weight,
    const LayerWeights *__restrict__ layer_weights,
    half_t *__restrict__ fa_k_cache,
    half_t *__restrict__ fa_v_cache,
    float *__restrict__ dn_states,
    float *__restrict__ conv_bufs,
    half_t *__restrict__ hidden_buffer,
    float *__restrict__ g_activations,
    half_t *__restrict__ g_residual,
    float *__restrict__ g_qkv_scratch,
    float *__restrict__ g_kv_scratch,
    float *__restrict__ g_attn_out,
    float *__restrict__ g_mlp_inter,
    float *__restrict__ g_z_scratch,
    float *__restrict__ g_beta_scratch,
    float *__restrict__ g_alpha_scratch,
    float *__restrict__ g_normalized,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_generation,
    float *__restrict__ seen_token_mask,
    float repetition_penalty,
    int input_token_id, const int *input_token_ptr, int position, int max_seq_len)
{
    int batch_id = BATCHED ? (int)blockIdx.z : 0;
    if constexpr (DEVICE_INPUT) {
        input_token_id = input_token_ptr[batch_id];
    }
    if constexpr (BATCHED) {
        const int fa_cache_stride = 6 * FA_NUM_KV_HEADS * max_seq_len * FA_HEAD_DIM;
        const int dn_state_stride = 18 * DN_NUM_HEADS * DN_KEY_DIM * DN_VALUE_DIM;
        const int conv_stride = 18 * DN_CONV_CHANNELS * DN_CONV_KERNEL;
        fa_k_cache += batch_id * fa_cache_stride;
        fa_v_cache += batch_id * fa_cache_stride;
        dn_states += batch_id * dn_state_stride;
        conv_bufs += batch_id * conv_stride;
        hidden_buffer += batch_id * HIDDEN_SIZE;
        g_activations += batch_id * ACTIVATIONS_STRIDE;
        g_residual += batch_id * HIDDEN_SIZE;
        g_qkv_scratch += batch_id * max(FA_QPROJ_SIZE, DN_CONV_CHANNELS);
        g_kv_scratch += batch_id * (FA_KV_SIZE * 2);
        g_attn_out += batch_id * max(FA_Q_SIZE, DN_V_SIZE);
        g_mlp_inter += batch_id * INTERMEDIATE_SIZE;
        g_z_scratch += batch_id * DN_V_SIZE;
        g_beta_scratch += batch_id * DN_NUM_HEADS;
        g_alpha_scratch += batch_id * DN_NUM_HEADS;
        g_normalized += batch_id * HIDDEN_SIZE;
        barrier_counter += batch_id;
        barrier_generation += batch_id;
        seen_token_mask += batch_id * VOCAB_SIZE;
    }
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;

    if (block_id == 0 && threadIdx.x == 0 &&
        seen_token_mask && repetition_penalty > 1.0f &&
        input_token_id >= 0 && input_token_id < VOCAB_SIZE) {
        seen_token_mask[input_token_id] = 1.0f;
    }

    AtomicGridSync grid{barrier_counter, barrier_generation, (unsigned int)num_blocks, 0};

    // Block-local workspace for normalization, MLP and attention merge.
    __shared__ __align__(16) char shmem_raw[SHMEM_FLOATS * sizeof(float)];
    half_t *shmem_bf16 = reinterpret_cast<half_t *>(shmem_raw);

    const half_t *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;

    int fa_kv_stride = FA_NUM_KV_HEADS * max_seq_len * FA_HEAD_DIM;
    int dn_state_stride = DN_NUM_HEADS * DN_KEY_DIM * DN_VALUE_DIM;

    int dn_layer_idx = 0, fa_layer_idx = 0;

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        const half_t *layer_input = (layer == 0) ? embed_row : hidden_buffer;

        if (LAYER_TYPE[layer] == 0) {
            deltanet_layer(
                grid, layer_weights[layer].dn, layer_input,
                g_residual, g_activations, g_qkv_scratch, g_z_scratch,
                g_beta_scratch, g_alpha_scratch, g_attn_out, g_mlp_inter,
                dn_states + dn_layer_idx * dn_state_stride,
                conv_bufs, hidden_buffer, dn_layer_idx, layer, shmem_bf16);
            dn_layer_idx++;
        } else {
            full_attention_layer(
                grid, layer_weights[layer].fa, layer_input,
                fa_k_cache + fa_layer_idx * fa_kv_stride,
                fa_v_cache + fa_layer_idx * fa_kv_stride,
                g_residual, g_activations, g_qkv_scratch, g_kv_scratch,
                g_attn_out, g_mlp_inter, hidden_buffer,
                layer, position, max_seq_len, shmem_bf16);
            fa_layer_idx++;
        }
    }

    // Final RMSNorm (block 0 only)
    if (block_id == 0) {
        __shared__ float smem_reduce[NUM_WARPS];
        int warp_id = threadIdx.x / WARP_SIZE, lane_id = threadIdx.x % WARP_SIZE;
        float local_sum_sq = 0;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
            float v = H2F(hidden_buffer[i]); g_activations[i] = v; local_sum_sq += v*v;
        }
        local_sum_sq = warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq; __syncthreads();
        if (warp_id == 0) { float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0; sum = warp_reduce_sum(sum); if (lane_id == 0) smem_reduce[0] = rsqrtf(sum/HIDDEN_SIZE + RMS_EPS); }
        __syncthreads(); float rstd = smem_reduce[0];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
            float wt = H2F(__ldg(final_norm_weight + i));
            g_normalized[i] = g_activations[i] * rstd * (1.0f + wt);
        }
    }
}

// =============================================================================
// C entry point
// =============================================================================

static int query_max_safe_decode_blocks_impl()
{
    // Occupancy is a launch-time property of the compiled kernel and device,
    // not of the current token position. Cache it after the first query so
    // step_many/streaming decode does not pay the CUDA occupancy API cost on
    // every token.
    static int cached_device = -1;
    static int cached_blocks = 0;
    int device_id = 0;
    if (cudaGetDevice(&device_id) == cudaSuccess &&
        device_id == cached_device && cached_blocks > 0) {
        return cached_blocks;
    }

    int sm_count = 0;
    int active_blocks_per_sm = 0;
    int max_safe_blocks = NUM_BLOCKS;

    if (cudaGetDevice(&device_id) == cudaSuccess &&
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id) == cudaSuccess &&
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm,
            decode_kernel<false, false>,
            BLOCK_SIZE,
            0) == cudaSuccess &&
        sm_count > 0 &&
        active_blocks_per_sm > 0) {
        const int resident_blocks = sm_count * active_blocks_per_sm;
        if (resident_blocks > 0) {
            max_safe_blocks = resident_blocks;
        }
    }

    max_safe_blocks = (max_safe_blocks < 1) ? 1 : max_safe_blocks;
    if (device_id >= 0) {
        cached_device = device_id;
        cached_blocks = max_safe_blocks;
    }
    return max_safe_blocks;
}

// A smaller grid can be faster while the KV scan is still short enough that
// each block has useful work. At long contexts, the full resident grid wins
// because it exposes more independent cache lines. Keep this adaptive choice
// opt-in so other GPUs retain the compiled default geometry.
static int choose_decode_blocks(int position, int max_safe_blocks)
{
    int decode_blocks = NUM_BLOCKS;
    if (decode_blocks > max_safe_blocks) decode_blocks = max_safe_blocks;
    if (g_decode_blocks_override > 0) {
        decode_blocks = g_decode_blocks_override;
    } else {
        static const int context_blocks = []() {
            const char *e = std::getenv("MEGAKERNEL_DECODE_CONTEXT_BLOCKS");
            // On the Elysium-sized 82-block Ampere build, 64 blocks gives a
            // small-context win without reducing the long-context grid. Keep
            // an explicit environment override for other launch geometries.
            return e ? std::atoi(e) : ((NUM_BLOCKS >= 82) ? 64 : 0);
        }();
        static const int context_max = []() {
            const char *e = std::getenv("MEGAKERNEL_DECODE_CONTEXT_MAX");
            return e ? std::atoi(e) : ((NUM_BLOCKS >= 82) ? 8192 : 0);
        }();
        if (context_blocks > 0 && context_max > 0 &&
            position + 1 >= MEGAKERNEL_DECODE_SPLITKV_MIN_CONTEXT &&
            position + 1 < context_max) {
            decode_blocks = context_blocks;
        }
    }
    if (decode_blocks > max_safe_blocks) decode_blocks = max_safe_blocks;
    return decode_blocks < 1 ? 1 : decode_blocks;
}

extern "C" void launch_decode(
    int input_token_id, int *output_token_id,
    const void *embed_weight, const LayerWeights *layer_weights,
    const void *final_norm_weight,
    const void *lm_head_weight,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch, void *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    float *block_max_vals, int *block_max_idxs,
    unsigned int *lm_sync_counter,
    float *seen_token_mask,
    float repetition_penalty,
    int position, int max_seq_len, cudaStream_t stream)
{
    static const bool profile = []() {
        const char *e = std::getenv("MEGAKERNEL_DECODE_PROFILE");
        return e && std::atoi(e) != 0;
    }();
    static cudaEvent_t profile_start = nullptr;
    static cudaEvent_t profile_layers = nullptr;
    static cudaEvent_t profile_done = nullptr;
    static int profile_count = 0;
    if (profile && !profile_start) {
        cudaEventCreate(&profile_start);
        cudaEventCreate(&profile_layers);
        cudaEventCreate(&profile_done);
    }
    const int max_safe_blocks = query_max_safe_decode_blocks_impl();
    int decode_blocks = choose_decode_blocks(position, max_safe_blocks);

    cudaMemsetAsync(barrier_counter, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(barrier_generation, 0, sizeof(unsigned int), stream);

    if (profile && profile_count < 8) cudaEventRecord(profile_start, stream);
    decode_kernel<false, false><<<decode_blocks, BLOCK_SIZE, 0, stream>>>(
        (const half_t *)embed_weight,
        (const half_t *)final_norm_weight,
        (const half_t *)lm_head_weight,
        layer_weights,
        (half_t *)fa_k_cache, (half_t *)fa_v_cache,
        (float *)dn_states, (float *)conv_bufs,
        (half_t *)hidden_buffer,
        (float *)g_activations, (half_t *)g_residual,
        (float *)g_qkv_scratch, (float *)g_kv_scratch,
        (float *)g_attn_out, (float *)g_mlp_inter,
        (float *)g_z_scratch, (float *)g_beta_scratch,
        (float *)g_alpha_scratch, (float *)g_normalized,
        barrier_counter, barrier_generation,
        seen_token_mask, repetition_penalty,
        input_token_id, nullptr, position, max_seq_len);

    if (profile && profile_count < 8) cudaEventRecord(profile_layers, stream);
    cudaMemsetAsync(lm_sync_counter, 0, sizeof(unsigned int), stream);

    lm_head_kernel<false><<<LM_NUM_BLOCKS, LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized,
#if MEGAKERNEL_LM_INT8
        (const int8_t *)lm_head_weight,
#else
        (const half_t *)lm_head_weight,
#endif
        block_max_vals, block_max_idxs,
        output_token_id, lm_sync_counter,
        seen_token_mask, repetition_penalty);
    if (profile && profile_count < 8) {
        cudaEventRecord(profile_done, stream);
        cudaEventSynchronize(profile_done);
        float layer_ms = 0.0f, lm_ms = 0.0f;
        cudaEventElapsedTime(&layer_ms, profile_start, profile_layers);
        cudaEventElapsedTime(&lm_ms, profile_layers, profile_done);
        std::fprintf(stderr, "[decode] blocks=%d layers=%.3f ms lm_head=%.3f ms total=%.3f ms\n",
                     decode_blocks, layer_ms, lm_ms, layer_ms + lm_ms);
        ++profile_count;
    }
}

extern "C" void set_decode_blocks_override(int blocks)
{
    g_decode_blocks_override = blocks;
}

// Device-token variant used by step_many().  The token pointer is updated by
// a device-to-device copy between steps, so a long greedy decode stays on the
// CUDA stream without synchronizing Python for every token.
extern "C" void launch_decode_device(
    const int *input_token_ptr, int *output_token_id,
    const void *embed_weight, const LayerWeights *layer_weights,
    const void *final_norm_weight,
    const void *lm_head_weight,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch, void *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    float *block_max_vals, int *block_max_idxs,
    unsigned int *lm_sync_counter,
    float *seen_token_mask,
    float repetition_penalty,
    int position, int max_seq_len, cudaStream_t stream)
{
    const int max_safe_blocks = query_max_safe_decode_blocks_impl();
    int decode_blocks = choose_decode_blocks(position, max_safe_blocks);
    cudaMemsetAsync(barrier_counter, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(barrier_generation, 0, sizeof(unsigned int), stream);
    decode_kernel<true, false><<<decode_blocks, BLOCK_SIZE, 0, stream>>>(
        (const half_t *)embed_weight,
        (const half_t *)final_norm_weight,
        (const half_t *)lm_head_weight,
        layer_weights,
        (half_t *)fa_k_cache, (half_t *)fa_v_cache,
        (float *)dn_states, (float *)conv_bufs,
        (half_t *)hidden_buffer,
        (float *)g_activations, (half_t *)g_residual,
        (float *)g_qkv_scratch, (float *)g_kv_scratch,
        (float *)g_attn_out, (float *)g_mlp_inter,
        (float *)g_z_scratch, (float *)g_beta_scratch,
        (float *)g_alpha_scratch, (float *)g_normalized,
        barrier_counter, barrier_generation,
        seen_token_mask, repetition_penalty,
        0, input_token_ptr, position, max_seq_len);
    cudaMemsetAsync(lm_sync_counter, 0, sizeof(unsigned int), stream);
    lm_head_kernel<false><<<LM_NUM_BLOCKS, LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized,
#if MEGAKERNEL_LM_INT8
        (const int8_t *)lm_head_weight,
#else
        (const half_t *)lm_head_weight,
#endif
        block_max_vals, block_max_idxs,
        output_token_id, lm_sync_counter,
        seen_token_mask, repetition_penalty);
}

extern "C" void launch_decode_many(
    const int *token_buffer, int *output_tokens, int steps,
    const void *embed_weight, const LayerWeights *layer_weights,
    const void *final_norm_weight,
    const void *lm_head_weight,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch, void *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    float *block_max_vals, int *block_max_idxs,
    unsigned int *lm_sync_counter,
    float *seen_token_mask,
    float repetition_penalty,
    int position, int max_seq_len, cudaStream_t stream)
{
    const int *current_input = token_buffer;
    for (int step = 0; step < steps; ++step) {
        launch_decode_device(
            current_input, output_tokens + step,
            embed_weight, layer_weights, final_norm_weight, lm_head_weight,
            fa_k_cache, fa_v_cache, dn_states, conv_bufs,
            hidden_buffer, g_activations, g_residual,
            g_qkv_scratch, g_kv_scratch, g_attn_out, g_mlp_inter,
            g_z_scratch, g_beta_scratch, g_alpha_scratch, g_normalized,
            barrier_counter, barrier_generation, block_max_vals, block_max_idxs,
            lm_sync_counter, seen_token_mask, repetition_penalty,
            position + step, max_seq_len, stream);
        current_input = output_tokens + step;
    }
}

// Batch decode for equal-length sequences. Each sequence has independent
// recurrent/KV state and barrier slots; the x dimension is reduced so all
// sequence groups remain resident together.
extern "C" void launch_decode_batch(
    const int *input_tokens, int *output_tokens, int batch_size,
    const void *embed_weight, const LayerWeights *layer_weights,
    const void *final_norm_weight, const void *lm_head_weight,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch, void *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    float *block_max_vals, int *block_max_idxs,
    unsigned int *lm_sync_counter, float *seen_token_mask,
    float repetition_penalty, int position, int max_seq_len,
    cudaStream_t stream)
{
    if (batch_size <= 0) return;
    const int max_safe_blocks = query_max_safe_decode_blocks_impl();
    int decode_blocks = NUM_BLOCKS;
    if (g_decode_blocks_override > 0) decode_blocks = g_decode_blocks_override;
    decode_blocks = min(decode_blocks, max_safe_blocks / batch_size);
    if (decode_blocks < 1) decode_blocks = 1;
    cudaMemsetAsync(barrier_counter, 0, sizeof(unsigned int) * batch_size, stream);
    cudaMemsetAsync(barrier_generation, 0, sizeof(unsigned int) * batch_size, stream);
    decode_kernel<true, true><<<dim3(decode_blocks, 1, batch_size), BLOCK_SIZE, 0, stream>>>(
        (const half_t *)embed_weight, (const half_t *)final_norm_weight,
        (const half_t *)lm_head_weight, layer_weights,
        (half_t *)fa_k_cache, (half_t *)fa_v_cache,
        (float *)dn_states, (float *)conv_bufs,
        (half_t *)hidden_buffer, (float *)g_activations, (half_t *)g_residual,
        (float *)g_qkv_scratch, (float *)g_kv_scratch, (float *)g_attn_out,
        (float *)g_mlp_inter, (float *)g_z_scratch, (float *)g_beta_scratch,
        (float *)g_alpha_scratch, (float *)g_normalized,
        barrier_counter, barrier_generation, seen_token_mask, repetition_penalty,
        0, input_tokens, position, max_seq_len);
    const int lm_blocks = max(1, LM_NUM_BLOCKS / batch_size);
    cudaMemsetAsync(lm_sync_counter, 0, sizeof(unsigned int) * batch_size, stream);
    lm_head_kernel<true><<<dim3(lm_blocks, 1, batch_size), LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized,
#if MEGAKERNEL_LM_INT8
        (const int8_t *)lm_head_weight,
#else
        (const half_t *)lm_head_weight,
#endif
        block_max_vals, block_max_idxs, output_tokens, lm_sync_counter,
        seen_token_mask, repetition_penalty);
}

extern "C" int query_max_safe_decode_blocks()
{
    return query_max_safe_decode_blocks_impl();
}

// Export the compile-time layout knobs so Python can reject a mismatched
// runtime configuration before launching a kernel with incompatible buffers.
extern "C" int query_compiled_num_blocks() { return NUM_BLOCKS; }
extern "C" int query_compiled_block_size() { return BLOCK_SIZE; }
extern "C" int query_compiled_lm_num_blocks() { return LM_NUM_BLOCKS; }
extern "C" int query_compiled_lm_block_size() { return LM_BLOCK_SIZE; }
extern "C" int query_compiled_lm_int8() { return MEGAKERNEL_LM_INT8; }
extern "C" int query_compiled_lm_dp4a() { return MEGAKERNEL_LM_DP4A; }
extern "C" int query_compiled_layer_int8() { return MEGAKERNEL_LAYER_INT8; }
extern "C" int query_compiled_layer_dp4a() { return MEGAKERNEL_LAYER_DP4A; }
extern "C" int query_compiled_splitkv() { return MEGAKERNEL_DECODE_SPLITKV; }
extern "C" int query_compiled_splitkv_min_context() {
    return MEGAKERNEL_DECODE_SPLITKV_MIN_CONTEXT;
}
