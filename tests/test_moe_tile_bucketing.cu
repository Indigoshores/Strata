// tests/test_moe_tile_bucketing.cu - Pass 5.3: 3-Bucket (B64, B32, B16) Variable-Height MoE Tile Scan Test
#include "strata/prefill/moe_tile_buckets.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <iomanip>

using namespace strata::prefill::fused;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

static std::vector<int32_t> generate_routing(int mode, int64_t n, int E) {
    std::vector<int32_t> ids(n);
    if (n <= 0) return ids;

    if (mode == 0) {
        // Uniform deterministic routing
        for (int64_t i = 0; i < n; ++i) {
            ids[i] = (int32_t) ((i * 13 + 7) % E);
        }
    } else if (mode == 1) {
        // Skewed Zipfian routing
        std::vector<double> cdf(E);
        double sum = 0.0;
        for (int e = 0; e < E; ++e) {
            sum += 1.0 / std::pow(e + 1.0, 1.25);
            cdf[e] = sum;
        }
        for (int e = 0; e < E; ++e) {
            cdf[e] /= sum;
        }
        for (int64_t i = 0; i < n; ++i) {
            uint32_t h = (uint32_t) (i * 2654435761u + 1013904223u);
            double u = (double) (h % 1000000) / 1000000.0;
            auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
            int e = (int) std::distance(cdf.begin(), it);
            if (e >= E) e = E - 1;
            ids[i] = e;
        }
    } else {
        // Adversarial hotspot routing
        // expert 0 gets ~80% of tokens
        // expert E - 1 gets 37 tokens
        // 300 experts (or E/2) get 0 tokens
        // remaining experts get 1..16 tokens
        int64_t cur = 0;
        int64_t n0 = (int64_t) (0.80 * n);
        for (; cur < n0 && cur < n; ++cur) ids[cur] = 0;

        int64_t n_last = std::min<int64_t>(37, n - cur);
        for (int64_t i = 0; i < n_last; ++i) ids[cur++] = E - 1;

        int num_zero = std::min(300, std::max(0, E - 2));
        int active_start = 1;
        int active_end = E - 1 - num_zero;
        if (active_end <= active_start) active_end = active_start + 1;

        int e = active_start;
        int per_exp = 1;
        while (cur < n) {
            int take = std::min<int64_t>(per_exp, n - cur);
            for (int i = 0; i < take; ++i) ids[cur++] = e;
            e = active_start + ((e - active_start + 1) % (active_end - active_start));
            per_exp = 1 + (per_exp % 16);
        }
    }
    return ids;
}

static bool run_bucketing_test(int E, int T, int K, int mode, double* out_ratio = nullptr) {
    const int64_t n = (int64_t) T * K;
    const std::vector<int32_t> ids = generate_routing(mode, n, E);

    int32_t* d_ids = nullptr;
    int32_t* d_slot = nullptr;
    int32_t* d_src = nullptr;
    void* d_scratch = nullptr;

    const size_t scratch_sz = bucket_scratch_bytes(n, E);
    CUDA_CHECK(cudaMalloc(&d_ids, (size_t) std::max<int64_t>(1, n) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_slot, (size_t) std::max<int64_t>(1, n) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_src, (size_t) std::max<int64_t>(1, n) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_scratch, scratch_sz));

    if (n > 0) {
        CUDA_CHECK(cudaMemcpy(d_ids, ids.data(), (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice));
    }

    group_bucketed(d_ids, n, K, E, d_scratch, d_slot, d_src, nullptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Retrieve tables from GPU
    const Tables tb = tables(d_scratch, E, n);
    std::vector<int32_t> h_cnt(E);
    std::vector<int32_t> h_off(E + 1);
    std::vector<int32_t> h_ts(E + 1);
    std::vector<int32_t> h_ts64(E + 1);
    std::vector<int32_t> h_ts32(E + 1);
    std::vector<int32_t> h_ts16(E + 1);

    CUDA_CHECK(cudaMemcpy(h_cnt.data(), tb.cnt, (size_t) E * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_off.data(), tb.off, (size_t) (E + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ts.data(), tb.ts, (size_t) (E + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ts64.data(), tb.ts64, (size_t) (E + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ts32.data(), tb.ts32, (size_t) (E + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ts16.data(), tb.ts16, (size_t) (E + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost));

    const int total_legacy_tiles = h_ts[E];
    const int total_b64_tiles = h_ts64[E];
    const int total_b32_tiles = h_ts32[E];
    const int total_b16_tiles = h_ts16[E];

    std::vector<int2> h_tiles(total_legacy_tiles);
    std::vector<int2> h_tiles64(total_b64_tiles);
    std::vector<int2> h_tiles32(total_b32_tiles);
    std::vector<int2> h_tiles16(total_b16_tiles);

    if (total_legacy_tiles > 0)
        CUDA_CHECK(cudaMemcpy(h_tiles.data(), tb.tiles, (size_t) total_legacy_tiles * sizeof(int2), cudaMemcpyDeviceToHost));
    if (total_b64_tiles > 0)
        CUDA_CHECK(cudaMemcpy(h_tiles64.data(), tb.tiles64, (size_t) total_b64_tiles * sizeof(int2), cudaMemcpyDeviceToHost));
    if (total_b32_tiles > 0)
        CUDA_CHECK(cudaMemcpy(h_tiles32.data(), tb.tiles32, (size_t) total_b32_tiles * sizeof(int2), cudaMemcpyDeviceToHost));
    if (total_b16_tiles > 0)
        CUDA_CHECK(cudaMemcpy(h_tiles16.data(), tb.tiles16, (size_t) total_b16_tiles * sizeof(int2), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_ids));
    CUDA_CHECK(cudaFree(d_slot));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_scratch));

    // Invariant 1: Total rows
    if (h_off[E] != n) {
        std::fprintf(stderr, "FAIL: Total rows off[E]=%d != n=%lld\n", h_off[E], (long long) n);
        return false;
    }

    // Invariant 2: Lossless, disjoint coverage across all 3 buckets
    std::vector<int> visits(n, 0);

    // Expand B64
    for (int t = 0; t < total_b64_tiles; ++t) {
        const int e = h_tiles64[t].x;
        const int row0 = h_tiles64[t].y;
        if (e < 0 || e >= E) {
            std::fprintf(stderr, "FAIL: B64 tile %d expert out of bounds: %d\n", t, e);
            return false;
        }
        if (row0 < h_off[e] || row0 >= h_off[e + 1]) {
            std::fprintf(stderr, "FAIL: B64 tile %d row0=%d outside off[%d]=%d..%d\n", t, row0, e, h_off[e], h_off[e + 1]);
            return false;
        }
        const int nrows = std::min(BM_B64, h_off[e + 1] - row0);
        if (nrows <= 0) {
            std::fprintf(stderr, "FAIL: B64 tile %d non-positive nrows: %d\n", t, nrows);
            return false;
        }
        for (int r = 0; r < nrows; ++r) {
            const int s = row0 + r;
            if (s < 0 || s >= n) {
                std::fprintf(stderr, "FAIL: B64 row out of bounds: %d\n", s);
                return false;
            }
            visits[s]++;
        }
    }

    // Expand B32
    for (int t = 0; t < total_b32_tiles; ++t) {
        const int e = h_tiles32[t].x;
        const int row0 = h_tiles32[t].y;
        if (e < 0 || e >= E) {
            std::fprintf(stderr, "FAIL: B32 tile %d expert out of bounds: %d\n", t, e);
            return false;
        }
        if (row0 < h_off[e] || row0 >= h_off[e + 1]) {
            std::fprintf(stderr, "FAIL: B32 tile %d row0=%d outside off[%d]=%d..%d\n", t, row0, e, h_off[e], h_off[e + 1]);
            return false;
        }
        const int nrows = std::min(BM_B32, h_off[e + 1] - row0);
        if (nrows <= 0) {
            std::fprintf(stderr, "FAIL: B32 tile %d non-positive nrows: %d\n", t, nrows);
            return false;
        }
        for (int r = 0; r < nrows; ++r) {
            const int s = row0 + r;
            if (s < 0 || s >= n) {
                std::fprintf(stderr, "FAIL: B32 row out of bounds: %d\n", s);
                return false;
            }
            visits[s]++;
        }
    }

    // Expand B16
    for (int t = 0; t < total_b16_tiles; ++t) {
        const int e = h_tiles16[t].x;
        const int row0 = h_tiles16[t].y;
        if (e < 0 || e >= E) {
            std::fprintf(stderr, "FAIL: B16 tile %d expert out of bounds: %d\n", t, e);
            return false;
        }
        if (row0 < h_off[e] || row0 >= h_off[e + 1]) {
            std::fprintf(stderr, "FAIL: B16 tile %d row0=%d outside off[%d]=%d..%d\n", t, row0, e, h_off[e], h_off[e + 1]);
            return false;
        }
        const int nrows = std::min(BM_B16, h_off[e + 1] - row0);
        if (nrows <= 0) {
            std::fprintf(stderr, "FAIL: B16 tile %d non-positive nrows: %d\n", t, nrows);
            return false;
        }
        for (int r = 0; r < nrows; ++r) {
            const int s = row0 + r;
            if (s < 0 || s >= n) {
                std::fprintf(stderr, "FAIL: B16 row out of bounds: %d\n", s);
                return false;
            }
            visits[s]++;
        }
    }

    // Verify disjoint multiplicity == 1 on every slot
    for (int64_t s = 0; s < n; ++s) {
        if (visits[s] != 1) {
            std::fprintf(stderr, "FAIL: Slot %lld visited %d times (expected exactly 1) in 3-bucket coverage! (E=%d, T=%d, mode=%d)\n",
                         (long long) s, visits[s], E, T, mode);
            return false;
        }
    }

    // Invariant 3: Legacy tiles coverage check (multiplicity == 1)
    std::vector<int> legacy_visits(n, 0);
    for (int t = 0; t < total_legacy_tiles; ++t) {
        const int e = h_tiles[t].x;
        const int row0 = h_tiles[t].y;
        const int nrows = std::min(BM_B64, h_off[e + 1] - row0);
        for (int r = 0; r < nrows; ++r) {
            legacy_visits[row0 + r]++;
        }
    }
    for (int64_t s = 0; s < n; ++s) {
        if (legacy_visits[s] != 1) {
            std::fprintf(stderr, "FAIL: Slot %lld visited %d times in legacy tiles! (E=%d, T=%d)\n",
                         (long long) s, legacy_visits[s], E, T);
            return false;
        }
    }

    // Invariant 4: Sub-range slicing [0, E/2) and [E/2, E)
    const int mid = E / 2;
    const int slice0_b64 = h_ts64[mid] - h_ts64[0];
    const int slice1_b64 = h_ts64[E] - h_ts64[mid];
    if (slice0_b64 + slice1_b64 != total_b64_tiles) {
        std::fprintf(stderr, "FAIL: B64 sub-range sum mismatch: %d + %d != %d\n", slice0_b64, slice1_b64, total_b64_tiles);
        return false;
    }
    for (int t = 0; t < slice0_b64; ++t) {
        if (h_tiles64[t].x >= mid) return false;
    }
    for (int t = slice0_b64; t < total_b64_tiles; ++t) {
        if (h_tiles64[t].x < mid) return false;
    }

    const int slice0_b32 = h_ts32[mid] - h_ts32[0];
    const int slice1_b32 = h_ts32[E] - h_ts32[mid];
    if (slice0_b32 + slice1_b32 != total_b32_tiles) {
        std::fprintf(stderr, "FAIL: B32 sub-range sum mismatch: %d + %d != %d\n", slice0_b32, slice1_b32, total_b32_tiles);
        return false;
    }
    for (int t = 0; t < slice0_b32; ++t) {
        if (h_tiles32[t].x >= mid) return false;
    }
    for (int t = slice0_b32; t < total_b32_tiles; ++t) {
        if (h_tiles32[t].x < mid) return false;
    }

    const int slice0_b16 = h_ts16[mid] - h_ts16[0];
    const int slice1_b16 = h_ts16[E] - h_ts16[mid];
    if (slice0_b16 + slice1_b16 != total_b16_tiles) {
        std::fprintf(stderr, "FAIL: B16 sub-range sum mismatch: %d + %d != %d\n", slice0_b16, slice1_b16, total_b16_tiles);
        return false;
    }
    for (int t = 0; t < slice0_b16; ++t) {
        if (h_tiles16[t].x >= mid) return false;
    }
    for (int t = slice0_b16; t < total_b16_tiles; ++t) {
        if (h_tiles16[t].x < mid) return false;
    }

    // Invariant 5: Effective padded WMMA M-rows reduction
    const int64_t wmma_legacy = (int64_t) total_legacy_tiles * 64;
    const int64_t wmma_bucketed = (int64_t) total_b64_tiles * 64 + (int64_t) total_b32_tiles * 32 + (int64_t) total_b16_tiles * 16;
    const double ratio = (wmma_bucketed > 0) ? ((double) wmma_legacy / (double) wmma_bucketed) : 1.0;
    if (out_ratio) *out_ratio = ratio;

    return true;
}

int main() {
    std::puts("======================================================================");
    std::puts("  Strata Pass 5.3: MoE 3-Bucket Variable-Height Tile Partitioner Test ");
    std::puts("  Buckets: B64 (BM=64), B32 (BM=32), B16 (BM=16)                      ");
    std::puts("======================================================================");

    const std::vector<int> experts = {8, 64, 512};
    const std::vector<int> tokens = {1, 6, 13, 16, 17, 31, 32, 33, 48, 49, 63, 64, 65, 662, 1637, 5184};
    const std::vector<int> modes = {0, 1, 2}; // Uniform, Zipfian, Adversarial Hotspot
    const char* mode_names[] = {"Uniform", "Zipfian", "Adversarial Hotspot"};

    int cases_tested = 0;

    for (int E : experts) {
        const int K = (E == 8) ? 8 : 10;
        for (int mode : modes) {
            for (int T : tokens) {
                double ratio = 1.0;
                bool ok = run_bucketing_test(E, T, K, mode, &ratio);
                if (!ok) {
                    std::fprintf(stderr, "*** TEST FAILED at E=%d, T=%d, K=%d, mode=%s ***\n",
                                 E, T, K, mode_names[mode]);
                    return 1;
                }
                cases_tested++;

                // Verify >= 3.0x padded row reduction at T = 662, E = 512, K = 10
                if (E == 512 && T == 662 && mode == 0) {
                    if (ratio < 3.0) {
                        std::fprintf(stderr, "*** FAILED: T=662 ratio %.2fx < 3.0x gate! ***\n", ratio);
                        return 1;
                    }
                }
            }
        }
    }

    std::printf("\n[PASS] Verified %d test configurations across E={8,64,512}, T={1..5184}, 3 routing modes.\n", cases_tested);
    std::puts("[PASS] Disjointness: 100% of routed slots covered with multiplicity == 1 (zero overlap, zero gaps).");
    std::puts("[PASS] Losslessness: zero missed rows, zero duplicate rows across all intervals.");
    std::puts("[PASS] Sub-range slicing: ts64, ts32, ts16 slices match exact expert ownership.");

    // Report padding reduction for key sequence lengths (E=512, K=10, Uniform routing)
    std::puts("\n--- WMMA Padded M-Rows Reduction Benchmark (E=512, K=10) ---");
    const int benchmark_T[] = {662, 1637, 5184};
    for (int T : benchmark_T) {
        double ratio = 0.0;
        run_bucketing_test(512, T, 10, 0, &ratio);
        std::printf("  T = %4d : %6.2fx reduction in effective padded WMMA M-rows (vs legacy BM=64)\n", T, ratio);
    }
    std::puts("------------------------------------------------------------\n");

    std::puts("test_moe_tile_bucketing: ALL CHECKS PASSED!");
    return 0;
}
