// tests/test_moe_variable_tiles.cu - Pass 5.4: 3-Bucket (B64, B32, B16) Variable-Height MoE WMMA Parity & Speedup Test
#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_fused_iq.hpp"
#include "strata/prefill/moe_tile_buckets.cuh"
#include "ggml.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace strata::prefill;
using namespace strata::prefill::fused;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

static void set_var_bm_env(int val) {
#ifdef _WIN32
    if (val == 0) _putenv("STRATA_MOE_VAR_BM=0");
    else _putenv("STRATA_MOE_VAR_BM=1");
#else
    if (val == 0) setenv("STRATA_MOE_VAR_BM", "0", 1);
    else setenv("STRATA_MOE_VAR_BM", "1", 1);
#endif
}

constexpr int N = 2560;
constexpr int FF = 640;
constexpr int K = 10;
constexpr uint64_t CANARY_VAL = 0xDEADBEEFCAFEBABEULL;
constexpr size_t CANARY_WORDS = 1024;
constexpr size_t CANARY_BYTES = CANARY_WORDS * sizeof(uint64_t);

static void fill_matrix(uint8_t* m, ggml_type t, int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t bs = ggml_type_size(t);
    const size_t blck_size = (size_t) ggml_blck_size(t);
    const size_t nb = (size_t) rows * (cols / blck_size);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    const float lo = (t == GGML_TYPE_Q8_0) ? 0.0001f :
                     (t == GGML_TYPE_IQ3_S) ? 0.0004f :
                     (t == GGML_TYPE_IQ4_XS) ? 0.00005f :
                     (t == GGML_TYPE_IQ4_NL) ? 0.0002f : 0.0008f;
    std::uniform_real_distribution<float> sc_dist(lo, 6.0f * lo);
    for (size_t i = 0; i < nb * bs; ++i) m[i] = (uint8_t) byte_dist(rng);
    for (size_t b = 0; b < nb; ++b) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(sc_dist(rng));
        std::memcpy(m + b * bs, &h, 2);
    }
}

struct ExpertPool {
    ggml_type gu_type;
    ggml_type d_type;
    size_t gu_row;
    size_t d_row;
    size_t up_off;
    size_t down_off;
    size_t bytes;
    std::vector<uint8_t*> d_blobs;

    ExpertPool(ggml_type gu, ggml_type d, std::mt19937& rng)
        : gu_type(gu), d_type(d), d_blobs(16, nullptr) {
        gu_row = ggml_row_size(gu, N);
        d_row = ggml_row_size(d, FF);
        up_off = gu_row * FF;
        down_off = 2 * up_off;
        bytes = down_off + d_row * N;

        std::vector<uint8_t> h_blob(bytes);
        for (int i = 0; i < 16; ++i) {
            fill_matrix(h_blob.data(), gu_type, 2 * FF, N, rng);
            fill_matrix(h_blob.data() + down_off, d_type, N, FF, rng);
            CUDA_CHECK(cudaMalloc(&d_blobs[i], bytes));
            CUDA_CHECK(cudaMemcpy(d_blobs[i], h_blob.data(), bytes, cudaMemcpyHostToDevice));
        }
    }

    ~ExpertPool() {
        for (int i = 0; i < 16; ++i) {
            if (d_blobs[i]) cudaFree(d_blobs[i]);
        }
    }

    fused::NativeGeom geom() const {
        fused::NativeGeom g;
        g.gu_type = (int) gu_type;
        g.d_type = (int) d_type;
        g.gu_row = gu_row;
        g.d_row = d_row;
        g.up_off = up_off;
        g.down_off = down_off;
        return g;
    }
};

static std::vector<int32_t> generate_routing(int mode, int64_t n, int E) {
    std::vector<int32_t> ids(n);
    if (n <= 0) return ids;

    if (mode == 0) {
        // Uniform deterministic
        for (int64_t i = 0; i < n; ++i) {
            ids[i] = (int32_t) ((i * 13 + 7) % E);
        }
    } else if (mode == 1) {
        // Zipfian distribution
        std::vector<double> cdf(E);
        double sum = 0.0;
        for (int e = 0; e < E; ++e) {
            sum += 1.0 / std::pow(e + 1.0, 1.25);
            cdf[e] = sum;
        }
        for (int e = 0; e < E; ++e) cdf[e] /= sum;
        for (int64_t i = 0; i < n; ++i) {
            uint32_t h = (uint32_t) (i * 2654435761u + 1013904223u);
            double u = (double) (h % 1000000) / 1000000.0;
            auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
            int e = (int) std::distance(cdf.begin(), it);
            if (e >= E) e = E - 1;
            ids[i] = e;
        }
    } else {
        // Adversarial hotspot
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

static void init_canary_buffer(uint8_t* d_alloc, size_t payload_bytes) {
    std::vector<uint64_t> canary(CANARY_WORDS, CANARY_VAL);
    CUDA_CHECK(cudaMemcpy(d_alloc, canary.data(), CANARY_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_alloc + CANARY_BYTES + payload_bytes, canary.data(), CANARY_BYTES, cudaMemcpyHostToDevice));
}

static bool verify_canaries(const uint8_t* d_alloc, size_t payload_bytes, const char* name) {
    std::vector<uint64_t> prefix(CANARY_WORDS);
    std::vector<uint64_t> postfix(CANARY_WORDS);
    CUDA_CHECK(cudaMemcpy(prefix.data(), d_alloc, CANARY_BYTES, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(postfix.data(), d_alloc + CANARY_BYTES + payload_bytes, CANARY_BYTES, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < CANARY_WORDS; ++i) {
        if (prefix[i] != CANARY_VAL) {
            std::fprintf(stderr, "CANARY OVERWRITE: %s prefix[%zu] = 0x%llx (expected 0x%llx)\n",
                         name, i, (unsigned long long) prefix[i], (unsigned long long) CANARY_VAL);
            return false;
        }
        if (postfix[i] != CANARY_VAL) {
            std::fprintf(stderr, "CANARY OVERWRITE: %s postfix[%zu] = 0x%llx (expected 0x%llx)\n",
                         name, i, (unsigned long long) postfix[i], (unsigned long long) CANARY_VAL);
            return false;
        }
    }
    return true;
}

struct ParityResult {
    double cos_sim = 1.0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    bool canaries_ok = true;
};

static ParityResult run_single_parity(const ExpertPool& pool, int E, int T, int mode,
                                      const float* d_x, const int32_t* d_ids,
                                      void* d_scratch, int32_t* d_slot, int32_t* d_src,
                                      uint8_t* d_xa,
                                      uint8_t* d_ha_alloc, size_t ha_bytes,
                                      uint8_t* d_dm_alloc, size_t dm_bytes,
                                      cudaStream_t stream) {
    const int64_t n = (int64_t) T * K;
    const fused::NativeGeom ng = pool.geom();
    uint8_t* d_ha = d_ha_alloc + CANARY_BYTES;
    float* d_dm = reinterpret_cast<float*>(d_dm_alloc + CANARY_BYTES);

    fused::quantize_act_native(d_x, T, N, d_xa, stream);
    fused::group(d_ids, n, K, E, d_scratch, d_slot, d_src, stream);

    // 1. Run baseline STRATA_MOE_VAR_BM=0
    set_var_bm_env(0);
    init_canary_buffer(d_ha_alloc, ha_bytes);
    init_canary_buffer(d_dm_alloc, dm_bytes);
    CUDA_CHECK(cudaMemsetAsync(d_dm, 0, dm_bytes, stream));

    for (int e0 = 0; e0 < E; e0 += fused::kMaxBatch) {
        fused::Batch b;
        b.e0 = e0;
        b.e1 = std::min(E, e0 + fused::kMaxBatch);
        for (int e = b.e0; e < b.e1; ++e) b.blob[e - b.e0] = pool.d_blobs[e & 15];
        fused::experts_native(b, ng, E, n, d_scratch, d_xa, d_src, d_ha, d_dm, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ParityResult res;
    if (!verify_canaries(d_ha_alloc, ha_bytes, "ha (baseline)") ||
        !verify_canaries(d_dm_alloc, dm_bytes, "dm (baseline)")) {
        res.canaries_ok = false;
        return res;
    }

    const size_t total_floats = (size_t) n * N;
    std::vector<float> h_base(total_floats);
    CUDA_CHECK(cudaMemcpy(h_base.data(), d_dm, dm_bytes, cudaMemcpyDeviceToHost));

    // 2. Run variable BM STRATA_MOE_VAR_BM=1
    set_var_bm_env(1);
    init_canary_buffer(d_ha_alloc, ha_bytes);
    init_canary_buffer(d_dm_alloc, dm_bytes);
    CUDA_CHECK(cudaMemsetAsync(d_dm, 0, dm_bytes, stream));

    for (int e0 = 0; e0 < E; e0 += fused::kMaxBatch) {
        fused::Batch b;
        b.e0 = e0;
        b.e1 = std::min(E, e0 + fused::kMaxBatch);
        for (int e = b.e0; e < b.e1; ++e) b.blob[e - b.e0] = pool.d_blobs[e & 15];
        fused::experts_native(b, ng, E, n, d_scratch, d_xa, d_src, d_ha, d_dm, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!verify_canaries(d_ha_alloc, ha_bytes, "ha (var_bm)") ||
        !verify_canaries(d_dm_alloc, dm_bytes, "dm (var_bm)")) {
        res.canaries_ok = false;
        return res;
    }

    std::vector<float> h_var(total_floats);
    CUDA_CHECK(cudaMemcpy(h_var.data(), d_dm, dm_bytes, cudaMemcpyDeviceToHost));

    // Compute metrics
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < total_floats; ++i) {
        const double va = h_base[i];
        const double vb = h_var[i];
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
        const float diff = std::fabs(h_base[i] - h_var[i]);
        if (diff > res.max_abs_diff) res.max_abs_diff = diff;
        const float denom = std::max(std::fabs(h_base[i]), std::fabs(h_var[i]));
        if (denom > 1e-6f) {
            res.max_rel_diff = std::max(res.max_rel_diff, diff / denom);
        }
    }
    if (norm_a > 0.0 && norm_b > 0.0) {
        res.cos_sim = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }
    return res;
}

static void run_speedup_benchmark(const ExpertPool& pool, int E, int T,
                                  cudaStream_t stream) {
    const int64_t n = (int64_t) T * K;
    const fused::NativeGeom ng = pool.geom();

    std::mt19937 rng(42);
    std::vector<float> h_x((size_t) T * N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : h_x) v = nd(rng);
    const std::vector<int32_t> ids = generate_routing(1 /* Zipfian */, n, E);

    float* d_x = nullptr;
    int32_t* d_ids = nullptr;
    int32_t* d_slot = nullptr;
    int32_t* d_src = nullptr;
    void* d_scratch = nullptr;
    uint8_t* d_xa = nullptr;
    uint8_t* d_ha = nullptr;
    float* d_dm = nullptr;

    const size_t xa_bytes = fused::act_bytes(T, N);
    const size_t ha_bytes = fused::act_bytes(n, FF);
    const size_t dm_bytes = (size_t) n * N * sizeof(float);
    const size_t scratch_sz = fused::group_bytes(n, E);

    CUDA_CHECK(cudaMalloc(&d_x, (size_t) T * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ids, (size_t) n * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_slot, (size_t) n * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_src, (size_t) n * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_scratch, scratch_sz));
    CUDA_CHECK(cudaMalloc(&d_xa, xa_bytes));
    CUDA_CHECK(cudaMalloc(&d_ha, ha_bytes));
    CUDA_CHECK(cudaMalloc(&d_dm, dm_bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), (size_t) T * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ids, ids.data(), (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice));

    fused::quantize_act_native(d_x, T, N, d_xa, stream);
    fused::group(d_ids, n, K, E, d_scratch, d_slot, d_src, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    const int warmup = 5;
    const int iters = 20;

    // Benchmark Baseline (STRATA_MOE_VAR_BM=0)
    set_var_bm_env(0);
    for (int i = 0; i < warmup; ++i) {
        for (int e0 = 0; e0 < E; e0 += fused::kMaxBatch) {
            fused::Batch b;
            b.e0 = e0; b.e1 = std::min(E, e0 + fused::kMaxBatch);
            for (int e = b.e0; e < b.e1; ++e) b.blob[e - b.e0] = pool.d_blobs[e & 15];
            fused::experts_native(b, ng, E, n, d_scratch, d_xa, d_src, d_ha, d_dm, stream);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    for (int i = 0; i < iters; ++i) {
        for (int e0 = 0; e0 < E; e0 += fused::kMaxBatch) {
            fused::Batch b;
            b.e0 = e0; b.e1 = std::min(E, e0 + fused::kMaxBatch);
            for (int e = b.e0; e < b.e1; ++e) b.blob[e - b.e0] = pool.d_blobs[e & 15];
            fused::experts_native(b, ng, E, n, d_scratch, d_xa, d_src, d_ha, d_dm, stream);
        }
    }
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms_base_total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_base_total, ev_start, ev_stop));
    const float ms_base = ms_base_total / (float) iters;

    // Benchmark Hybrid (STRATA_MOE_VAR_BM=1)
    set_var_bm_env(1);
    for (int i = 0; i < warmup; ++i) {
        for (int e0 = 0; e0 < E; e0 += fused::kMaxBatch) {
            fused::Batch b;
            b.e0 = e0; b.e1 = std::min(E, e0 + fused::kMaxBatch);
            for (int e = b.e0; e < b.e1; ++e) b.blob[e - b.e0] = pool.d_blobs[e & 15];
            fused::experts_native(b, ng, E, n, d_scratch, d_xa, d_src, d_ha, d_dm, stream);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    for (int i = 0; i < iters; ++i) {
        for (int e0 = 0; e0 < E; e0 += fused::kMaxBatch) {
            fused::Batch b;
            b.e0 = e0; b.e1 = std::min(E, e0 + fused::kMaxBatch);
            for (int e = b.e0; e < b.e1; ++e) b.blob[e - b.e0] = pool.d_blobs[e & 15];
            fused::experts_native(b, ng, E, n, d_scratch, d_xa, d_src, d_ha, d_dm, stream);
        }
    }
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms_var_total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_var_total, ev_start, ev_stop));
    const float ms_var = ms_var_total / (float) iters;

    const float speedup_pct = (ms_base - ms_var) / ms_base * 100.0f;
    std::printf("  [Perf] T = %4d | Baseline (BM=64): %7.3f ms | Hybrid (BM=16/32/64): %7.3f ms | Speedup: %+6.2f%%\n",
                T, ms_base, ms_var, speedup_pct);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    cudaFree(d_x);
    cudaFree(d_ids);
    cudaFree(d_slot);
    cudaFree(d_src);
    cudaFree(d_scratch);
    cudaFree(d_xa);
    cudaFree(d_ha);
    cudaFree(d_dm);
}

int main() {
    std::printf("================================================================================\n");
    std::printf("  test_moe_variable_tiles - Pass 5.4 3-Bucket WMMA Parity & Speedup Gate\n");
    std::printf("================================================================================\n");

    std::mt19937 rng(1337);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Verify GPU capability
    if (!fused::built()) {
        std::printf("Strata fused MoE kernels not built; exiting with code 0.\n");
        return 0;
    }

    std::printf("Initializing Synthetic Expert Pools (IQ4_XS/IQ4_NL, IQ4_XS/Q8_0, IQ3_S/IQ4_NL)...\n");
    ExpertPool pool_iq4xs_iq4nl(GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, rng);
    ExpertPool pool_iq4xs_q80(GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0, rng);
    ExpertPool pool_iq3s_iq4nl(GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, rng);

    // Test token counts
    const std::vector<int> boundary_T = {1, 6, 13, 16, 17, 31, 32, 33, 48, 49, 63, 64, 65, 662, 1637};
    const int max_T = 1637;
    const int64_t max_n = (int64_t) max_T * K;
    const int max_E = 512;

    // Allocate scratch & canary buffers
    float* d_x = nullptr;
    int32_t* d_ids = nullptr;
    int32_t* d_slot = nullptr;
    int32_t* d_src = nullptr;
    void* d_scratch = nullptr;
    uint8_t* d_xa = nullptr;
    uint8_t* d_ha_alloc = nullptr;
    uint8_t* d_dm_alloc = nullptr;

    const size_t max_xa_bytes = fused::act_bytes(max_T, N);
    const size_t max_ha_bytes = fused::act_bytes(max_n, FF);
    const size_t max_dm_bytes = (size_t) max_n * N * sizeof(float);
    const size_t max_scratch_sz = fused::group_bytes(max_n, max_E);

    CUDA_CHECK(cudaMalloc(&d_x, (size_t) max_T * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ids, (size_t) max_n * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_slot, (size_t) max_n * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_src, (size_t) max_n * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_scratch, max_scratch_sz));
    CUDA_CHECK(cudaMalloc(&d_xa, max_xa_bytes));
    CUDA_CHECK(cudaMalloc(&d_ha_alloc, max_ha_bytes + 2 * CANARY_BYTES));
    CUDA_CHECK(cudaMalloc(&d_dm_alloc, max_dm_bytes + 2 * CANARY_BYTES));

    std::vector<float> h_x((size_t) max_T * N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : h_x) v = nd(rng);
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), (size_t) max_T * N * sizeof(float), cudaMemcpyHostToDevice));

    bool all_passed = true;
    int total_tests = 0;

    std::printf("\n--- Section 1: Boundary Tokens & Routing Parity (IQ4_XS + IQ4_NL) ---\n");
    for (int E : {64, 512}) {
        for (int mode : {0, 1, 2}) {
            const char* mode_name = (mode == 0) ? "Uniform" : (mode == 1) ? "Zipfian" : "Hotspot";
            for (int T : boundary_T) {
                const int64_t n = (int64_t) T * K;
                const std::vector<int32_t> ids = generate_routing(mode, n, E);
                CUDA_CHECK(cudaMemcpy(d_ids, ids.data(), (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice));

                const size_t cur_ha_bytes = fused::act_bytes(n, FF);
                const size_t cur_dm_bytes = (size_t) n * N * sizeof(float);

                ParityResult res = run_single_parity(pool_iq4xs_iq4nl, E, T, mode,
                                                     d_x, d_ids, d_scratch, d_slot, d_src, d_xa,
                                                     d_ha_alloc, cur_ha_bytes,
                                                     d_dm_alloc, cur_dm_bytes,
                                                     stream);

                total_tests++;
                const bool pass = res.canaries_ok && (res.cos_sim >= 0.999999);
                if (!pass) {
                    all_passed = false;
                    std::printf("  [FAIL] E=%3d mode=%-8s T=%4d: cos=%.8f max_abs=%.6f canary=%s\n",
                                E, mode_name, T, res.cos_sim, res.max_abs_diff, res.canaries_ok ? "OK" : "CORRUPT");
                } else if (T == 1 || T == 17 || T == 49 || T == 65 || T == 662 || T == 1637) {
                    std::printf("  [PASS] E=%3d mode=%-8s T=%4d: cos=%.8f max_abs=%.6f canary=OK\n",
                                E, mode_name, T, res.cos_sim, res.max_abs_diff);
                }
            }
        }
    }

    std::printf("\n--- Section 2: Secondary Quant Formats Parity Checks ---\n");
    for (const auto& entry : {
        std::make_pair(&pool_iq4xs_q80, "IQ4_XS + Q8_0"),
        std::make_pair(&pool_iq3s_iq4nl, "IQ3_S + IQ4_NL")
    }) {
        const ExpertPool& p = *entry.first;
        const char* name = entry.second;
        for (int T : {1, 17, 49, 662}) {
            const int64_t n = (int64_t) T * K;
            const std::vector<int32_t> ids = generate_routing(1 /* Zipfian */, n, 512);
            CUDA_CHECK(cudaMemcpy(d_ids, ids.data(), (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice));

            const size_t cur_ha_bytes = fused::act_bytes(n, FF);
            const size_t cur_dm_bytes = (size_t) n * N * sizeof(float);

            ParityResult res = run_single_parity(p, 512, T, 1,
                                                 d_x, d_ids, d_scratch, d_slot, d_src, d_xa,
                                                 d_ha_alloc, cur_ha_bytes,
                                                 d_dm_alloc, cur_dm_bytes,
                                                 stream);
            total_tests++;
            const bool pass = res.canaries_ok && (res.cos_sim >= 0.999999);
            if (!pass) {
                all_passed = false;
                std::printf("  [FAIL] %-16s T=%4d: cos=%.8f max_abs=%.6f canary=%s\n",
                            name, T, res.cos_sim, res.max_abs_diff, res.canaries_ok ? "OK" : "CORRUPT");
            } else {
                std::printf("  [PASS] %-16s T=%4d: cos=%.8f max_abs=%.6f canary=OK\n",
                            name, T, res.cos_sim, res.max_abs_diff);
            }
        }
    }

    // Clean up parity allocations
    cudaFree(d_x);
    cudaFree(d_ids);
    cudaFree(d_slot);
    cudaFree(d_src);
    cudaFree(d_scratch);
    cudaFree(d_xa);
    cudaFree(d_ha_alloc);
    cudaFree(d_dm_alloc);

    std::printf("\n--- Section 3: GPU Latency Speedup Benchmark (E=512, top_k=10, Qwen 177B) ---\n");
    run_speedup_benchmark(pool_iq4xs_iq4nl, 512, 662, stream);
    run_speedup_benchmark(pool_iq4xs_iq4nl, 512, 1637, stream);
    run_speedup_benchmark(pool_iq4xs_iq4nl, 512, 5184, stream);

    CUDA_CHECK(cudaStreamDestroy(stream));

    std::printf("\n================================================================================\n");
    if (all_passed) {
        std::printf("  ALL %d PARITY CHECKS PASSED (cos >= 0.999999, 0 canary overwrites)!\n", total_tests);
        std::printf("================================================================================\n");
        return 0;
    } else {
        std::fprintf(stderr, "  SOME PARITY CHECKS FAILED!\n");
        std::printf("================================================================================\n");
        return 1;
    }
}
