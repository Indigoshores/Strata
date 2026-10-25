// include/strata/prefill/moe_tile_buckets.cuh - Pass 5.3: 3-bucket variable-height MoE tile partitioner (B64, B32, B16)
#pragma once

#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace strata::prefill::fused {

constexpr int BM_B64 = 64;
constexpr int BM_B32 = 32;
constexpr int BM_B16 = 16;

struct Tables {
    int32_t* cnt;       // per expert: count [E]
    int32_t* off;       // per expert: row offset in sorted token list [E + 1]
    int32_t* fill;      // per expert: placement cursor [E]
    int32_t* ts;        // per expert: legacy BM=64 first tile [E + 1]
    int2*    tiles;     // legacy BM=64 tiles: {expert, first row}

    // 3-Bucket extensions (B64, B32, B16)
    int32_t* ts64;      // per expert: first B64 tile [E + 1]
    int2*    tiles64;   // B64 tiles: {expert, first row}
    int32_t* ts32;      // per expert: first B32 tile [E + 1]
    int2*    tiles32;   // B32 tiles: {expert, first row}
    int32_t* ts16;      // per expert: first B16 tile [E + 1]
    int2*    tiles16;   // B16 tiles: {expert, first row}
};

using BucketedTables = Tables;

__host__ __device__ inline size_t cap_legacy(int64_t n, int n_expert) {
    return (size_t) ((n + BM_B64 - 1) / BM_B64 + n_expert);
}

__host__ __device__ inline size_t cap_b64(int64_t n, int n_expert) {
    return (size_t) ((n + BM_B64 - 1) / BM_B64 + n_expert);
}

__host__ __device__ inline size_t cap_b32(int64_t n, int n_expert) {
    return (size_t) ((n + BM_B32 - 1) / BM_B32 + n_expert);
}

__host__ __device__ inline size_t cap_b16(int64_t n, int n_expert) {
    return (size_t) ((n + BM_B16 - 1) / BM_B16 + n_expert);
}

inline int pick_max_bm(int64_t n, int n_expert) {
    const char* v = std::getenv("STRATA_MOE_MAX_BM");
    if (v != nullptr && v[0] != '\0') {
        int val = std::atoi(v);
        if (val == 16 || val == 32 || val == 64) return val;
    }
    const double avg = (double) n / std::max(n_expert, 1);
    if (avg <= 16.0) return 16;
    if (avg <= 40.0) return 32;
    return 64;
}

__host__ __device__ inline size_t bucket_scratch_bytes(int64_t n, int n_expert) {
    // 7 int32 arrays: cnt(E), off(E+1), fill(E), ts(E+1), ts64(E+1), ts32(E+1), ts16(E+1)
    const size_t total_i32 = (size_t) (7 * n_expert + 5);
    const size_t off_tiles32 = (total_i32 * sizeof(int32_t) + 15) / 16 * 16;
    const size_t off_tiles16 = off_tiles32 + cap_b32(n, n_expert) * sizeof(int2);
    const size_t off_tiles64 = off_tiles16 + cap_b16(n, n_expert) * sizeof(int2);
    const size_t off_tiles_legacy = off_tiles64 + cap_b64(n, n_expert) * sizeof(int2);
    return off_tiles_legacy + cap_legacy(n, n_expert) * sizeof(int2);
}

inline Tables tables(void* scratch, int n_expert, int64_t n = 0) {
    int32_t* p = (int32_t*) scratch;
    Tables tb;
    tb.cnt   = p;
    tb.off   = p + n_expert;
    tb.fill  = p + 2 * n_expert + 1;
    tb.ts    = p + 3 * n_expert + 1;
    tb.ts64  = p + 4 * n_expert + 2;
    tb.ts32  = p + 5 * n_expert + 3;
    tb.ts16  = p + 6 * n_expert + 4;

    uint8_t* base = (uint8_t*) scratch;
    const size_t total_i32 = (size_t) (7 * n_expert + 5);
    const size_t off_tiles32 = (total_i32 * sizeof(int32_t) + 15) / 16 * 16;
    const size_t off_tiles16 = off_tiles32 + cap_b32(n, n_expert) * sizeof(int2);
    const size_t off_tiles64 = off_tiles16 + cap_b16(n, n_expert) * sizeof(int2);
    const size_t off_tiles_legacy = off_tiles64 + cap_b64(n, n_expert) * sizeof(int2);

    tb.tiles32 = (int2*) (base + off_tiles32);
    tb.tiles16 = (int2*) (base + off_tiles16);
    tb.tiles64 = (int2*) (base + off_tiles64);
    tb.tiles   = (int2*) (base + off_tiles_legacy);
    return tb;
}

inline BucketedTables bucket_tables(void* scratch, int n_expert, int64_t n = 0) {
    return tables(scratch, n_expert, n);
}

// Exclusive prefix sum over the block (blockDim a multiple of 32); *total = the block's sum
__device__ inline int block_excl_sum(int v, int* total, int* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) sh[w] = x;
    __syncthreads();
    if (w == 0) {
        int s = lane < nw ? sh[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += y;
        }
        sh[lane] = s;
    }
    __syncthreads();
    const int r = x - v + (w > 0 ? sh[w - 1] : 0);
    *total = sh[nw - 1];
    __syncthreads();   // sh is reused by the next call
    return r;
}

__global__ inline void count_kernel(const int32_t* __restrict__ ids, int64_t n, int E, int32_t* __restrict__ cnt) {
    extern __shared__ int32_t hist[];
    for (int e = threadIdx.x; e < E; e += blockDim.x) hist[e] = 0;
    __syncthreads();
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int e = ids[i];
        if ((unsigned) e < (unsigned) E) atomicAdd(&hist[e], 1);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += blockDim.x)
        if (hist[e]) atomicAdd(&cnt[e], hist[e]);
}

__global__ inline void scan_buckets_kernel(Tables tb, int E, int max_bm = 64) {
    __shared__ int sh[32];
    const int e = threadIdx.x;
    const int c = (e < E) ? tb.cnt[e] : 0;

    // Legacy tile count (BM = 64)
    const int nt_legacy = (c + BM_B64 - 1) / BM_B64;

    // Adaptive bucket tile counts
    int c64 = 0, c32 = 0, c16 = 0;
    int n_full_64 = 0, r64 = 0;
    int n_full_32 = 0, r32 = 0;

    if (max_bm == 16) {
        c16 = (c + 15) / 16;
    } else if (max_bm == 32) {
        n_full_32 = c / 32;
        r32 = c - 32 * n_full_32;
        c32 = n_full_32 + (r32 >= 17 ? 1 : 0);
        c16 = (r32 >= 1 && r32 <= 16) ? 1 : 0;
    } else {
        n_full_64 = c / 64;
        r64 = c - 64 * n_full_64;
        c64 = n_full_64 + (r64 >= 49 ? 1 : 0);
        c32 = (r64 >= 17 && r64 <= 48) ? 1 : 0;
        c16 = ((r64 >= 1 && r64 <= 16) || (r64 >= 33 && r64 <= 48)) ? 1 : 0;
    }

    int total_rows = 0;
    int total_tiles_legacy = 0;
    int total_tiles64 = 0;
    int total_tiles32 = 0;
    int total_tiles16 = 0;

    const int o    = block_excl_sum(c, &total_rows, sh);
    const int to   = block_excl_sum(nt_legacy, &total_tiles_legacy, sh);
    const int to64 = block_excl_sum(c64, &total_tiles64, sh);
    const int to32 = block_excl_sum(c32, &total_tiles32, sh);
    const int to16 = block_excl_sum(c16, &total_tiles16, sh);

    if (e < E) {
        tb.off[e]  = o;
        tb.fill[e] = o;
        tb.ts[e]   = to;
        tb.ts64[e] = to64;
        tb.ts32[e] = to32;
        tb.ts16[e] = to16;

        // Emit legacy tiles (BM = 64)
        for (int i = 0; i < nt_legacy; ++i) {
            tb.tiles[to + i] = make_int2(e, o + i * BM_B64);
        }

        if (max_bm == 16) {
            // Emit B16 tiles (BM = 16)
            for (int i = 0; i < c16; ++i) {
                tb.tiles16[to16 + i] = make_int2(e, o + i * 16);
            }
        } else if (max_bm == 32) {
            // Emit B32 tiles (BM = 32)
            for (int i = 0; i < n_full_32; ++i) {
                tb.tiles32[to32 + i] = make_int2(e, o + i * 32);
            }
            if (r32 >= 17) {
                tb.tiles32[to32 + n_full_32] = make_int2(e, o + n_full_32 * 32);
            }
            // Emit B16 tiles (BM = 16)
            if (r32 >= 1 && r32 <= 16) {
                tb.tiles16[to16] = make_int2(e, o + n_full_32 * 32);
            }
        } else {
            // Emit B64 tiles (BM = 64)
            for (int i = 0; i < n_full_64; ++i) {
                tb.tiles64[to64 + i] = make_int2(e, o + i * 64);
            }
            if (r64 >= 49) {
                tb.tiles64[to64 + n_full_64] = make_int2(e, o + n_full_64 * 64);
            }

            // Emit B32 tiles (BM = 32)
            if (r64 >= 17 && r64 <= 48) {
                tb.tiles32[to32] = make_int2(e, o + n_full_64 * 64);
            }

            // Emit B16 tiles (BM = 16)
            if (r64 >= 1 && r64 <= 16) {
                tb.tiles16[to16] = make_int2(e, o + n_full_64 * 64);
            } else if (r64 >= 33 && r64 <= 48) {
                tb.tiles16[to16] = make_int2(e, o + n_full_64 * 64 + 32);
            }
        }
    }

    if (e == 0) {
        tb.off[E]  = total_rows;
        tb.ts[E]   = total_tiles_legacy;
        tb.ts64[E] = total_tiles64;
        tb.ts32[E] = total_tiles32;
        tb.ts16[E] = total_tiles16;
    }
}

__global__ inline void place_kernel(const int32_t* __restrict__ ids, int64_t n, int k, int E, Tables tb,
                                    int32_t* __restrict__ slot, int32_t* __restrict__ src,
                                    const float* __restrict__ w = nullptr, float* __restrict__ rw = nullptr) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int e = ids[i];
    if ((unsigned) e >= (unsigned) E) { if (slot) slot[i] = 0; return; }
    const int p = atomicAdd(&tb.fill[e], 1);
    if (slot) slot[i] = p;
    if (src) src[p] = (int32_t) (i / k);
    if (rw && w) rw[p] = w[i];
}

inline void group_bucketed(const int32_t* ids, int64_t n, int k, int n_expert,
                           void* scratch, int32_t* slot, int32_t* src,
                           cudaStream_t stream = nullptr, int max_bm = 0,
                           const float* w = nullptr, float* rw = nullptr) {
    if (n_expert < 1 || n_expert > 1024) {
        std::fprintf(stderr, "group_bucketed: %d experts (the grouping takes 1 to 1024)\n", n_expert);
        std::exit(1);
    }
    const Tables tb = tables(scratch, n_expert, n);
    cudaMemsetAsync(tb.cnt, 0, (size_t) n_expert * sizeof(int32_t), stream);
    if (n > 0) {
        unsigned nb = (unsigned) std::min<int64_t>((n + 1023) / 1024, 64);
        count_kernel<<<nb, 1024, (size_t) n_expert * sizeof(int32_t), stream>>>(ids, n, n_expert, tb.cnt);
    }
    const unsigned threads = (unsigned) std::max(32, ((n_expert + 31) / 32 * 32));
    const int effective_max_bm = (max_bm == 16 || max_bm == 32 || max_bm == 64) ? max_bm : pick_max_bm(n, n_expert);
    scan_buckets_kernel<<<1, threads, 0, stream>>>(tb, n_expert, effective_max_bm);
    if (n > 0 && (slot != nullptr || src != nullptr || rw != nullptr)) {
        unsigned nb = (unsigned) ((n + 255) / 256);
        place_kernel<<<nb, 256, 0, stream>>>(ids, n, k, n_expert, tb, slot, src, w, rw);
    }
}

}  // namespace strata::prefill::fused
