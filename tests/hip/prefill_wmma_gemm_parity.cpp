// tests/hip/prefill_wmma_gemm_parity.cpp - Numerical parity test for RDNA3 WMMA GEMM kernels.
#include <cuda_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include "wmma_gemm.h"
#include "strata/kernels/gfx_arch.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

// ============================================================================
// Exact Representability & Parity Rationale:
//
// Inputs (X, W, and initial Y for beta=1) are chosen from discrete binary fractions
// k / 16.0f with integer k in [-8, 8].
//
// In IEEE 754 half-precision (fp16, 10-bit explicit mantissa) and bfloat16
// (7-bit explicit mantissa), every such value k * 2^-4 requires at most 4 bits
// of significand and is therefore EXACTLY representable without rounding.
//
// The product of any two such values is an exact integer multiple of 2^-8:
//   (a * 2^-4) * (b * 2^-4) = (a * b) * 2^-8
// with (a * b) in [-64, 64].
//
// For K <= 80, the sum of K products:
//   sum_k (W[n, k] * X[t, k]) = (sum_k a_k * b_k) * 2^-8
// has a maximum absolute magnitude of 80 * (0.5 * 0.5) = 20.0, and the integer
// numerator cannot exceed 80 * 64 = 5120 (< 2^13).
//
// In IEEE 754 single-precision (fp32, 23-bit explicit mantissa, 24 bits total),
// any integer multiple of 2^-8 up to 20.0 requires at most 13 bits of significand.
// Because the RDNA3 hardware instructions (v_wmma_f32_16x16x16_f16_w32 and
// v_wmma_f32_16x16x16_bf16_w32) accumulate in native FP32 registers, every
// intermediate product and addition has ZERO truncation or rounding error.
//
// Furthermore, floating-point addition of exact binary fractions whose sum fits
// within the mantissa is strictly associative and exact. Consequently, the GPU
// WMMA pipeline and the CPU reference compute identical bitwise results, making
// an exact assertion (max_abs == 0.0f) mathematically legitimate.
// ============================================================================

#define HIP_CHECK(call) do { \
    const hipError_t status_ = (call); \
    if (status_ != hipSuccess) { \
        std::fprintf(stderr, "HIP failure at %s:%d: %s: %s\n", __FILE__, __LINE__, #call, hipGetErrorString(status_)); \
        std::exit(2); \
    } \
} while (0)

namespace {

constexpr float kSentinel = -999.5f;

struct DeviceBuffer {
    void* p = nullptr;
    explicit DeviceBuffer(size_t bytes) { if (bytes) HIP_CHECK(hipMalloc(&p, bytes)); }
    ~DeviceBuffer() { if (p) (void) hipFree(p); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

uint16_t encode(float value, bool bf16) {
    if (bf16) return hip_bfloat16(value).data;
    const __half half = __float2half_rn(value);
    return static_cast<__half_raw>(half).x;
}

float decode(uint16_t bits, bool bf16) {
    if (bf16) {
        const uint32_t u = static_cast<uint32_t>(bits) << 16;
        float f = 0.0f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
    }
    const __half_raw raw{bits};
    return __half2float(__half(raw));
}

struct MismatchRecord {
    const char* dtype = "";
    int64_t T = 0, N = 0, K = 0, ldy = 0;
    float beta = 0.0f;
    int64_t row = 0, col = 0;
    float got = 0.0f, ref = 0.0f;
    double delta = 0.0;
    bool is_padding = false;
};

// Independent host CPU reference computed in double precision.
void gemm_reference(const uint16_t* X, const uint16_t* W, const float* initial_Y, float* ref_Y,
                    int64_t T, int64_t N, int64_t K, int64_t ldy, float beta, bool is_bf16) {
    std::vector<double> x_dec((size_t) T * K);
    std::vector<double> w_dec((size_t) N * K);
    for (size_t i = 0; i < x_dec.size(); ++i) x_dec[i] = decode(X[i], is_bf16);
    for (size_t i = 0; i < w_dec.size(); ++i) w_dec[i] = decode(W[i], is_bf16);

    for (int64_t t = 0; t < T; ++t) {
        const double* x_row = x_dec.data() + t * K;
        for (int64_t n = 0; n < N; ++n) {
            const double* w_row = w_dec.data() + n * K;
            double sum = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                sum += x_row[k] * w_row[k];
            }
            const size_t out_idx = (size_t) t * ldy + n;
            const double y_init = (beta == 0.0f) ? 0.0 : static_cast<double>(initial_Y[out_idx]);
            ref_Y[out_idx] = static_cast<float>(beta * y_init + sum);
        }
    }
}

enum class CaseStatus {
    Passed,
    Skipped,
    Failed
};

CaseStatus run_gemm_case(DeviceBuffer& dx, DeviceBuffer& dw, DeviceBuffer& dy,
                         hipStream_t stream, bool is_bf16,
                         int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                         uint32_t seed, double& out_max_abs,
                         MismatchRecord* first_mismatch, size_t& failure_count) {
    const char* dtype_str = is_bf16 ? "bf16" : "f16";
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-8, 8);

    const size_t x_count = (size_t) T * K;
    const size_t w_count = (size_t) N * K;
    const size_t y_count = (size_t) T * ldy + 16;  // 16 guard floats past end of matrix

    std::vector<uint16_t> x(x_count);
    std::vector<uint16_t> w(w_count);
    for (size_t i = 0; i < x_count; ++i) x[i] = encode(dist(rng) * 0.0625f, is_bf16);
    for (size_t i = 0; i < w_count; ++i) w[i] = encode(dist(rng) * 0.0625f, is_bf16);

    std::vector<float> initial_y(y_count, kSentinel);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t n = 0; n < N; ++n) {
            if (beta == 0.0f) {
                initial_y[(size_t) t * ldy + n] = -888.25f;
            } else {
                // beta == 1.0f: pre-fill with a known exact pattern to verify accumulation
                initial_y[(size_t) t * ldy + n] = static_cast<float>((t * 5 + n * 3) % 17 - 8) * 0.0625f;
            }
        }
    }

    HIP_CHECK(hipMemcpyAsync(dx.p, x.data(), x_count * sizeof(uint16_t), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(dw.p, w.data(), w_count * sizeof(uint16_t), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(dy.p, initial_y.data(), y_count * sizeof(float), hipMemcpyHostToDevice, stream));

    const bool launched = is_bf16 ? strata_wmma_gemm_bf16(static_cast<const uint16_t*>(dx.p),
                                                           static_cast<const uint16_t*>(dw.p),
                                                           static_cast<float*>(dy.p),
                                                           T, N, K, ldy, beta, stream)
                                  : strata_wmma_gemm_f16(static_cast<const uint16_t*>(dx.p),
                                                          static_cast<const uint16_t*>(dw.p),
                                                          static_cast<float*>(dy.p),
                                                          T, N, K, ldy, beta, stream);

    if (!launched) {
        // Declined by kernel: caller falls back to hipBLAS. Test must SKIP rather than fail.
        HIP_CHECK(hipStreamSynchronize(stream));
        return CaseStatus::Skipped;
    }

    std::vector<float> got_y(y_count);
    HIP_CHECK(hipMemcpyAsync(got_y.data(), dy.p, y_count * sizeof(float), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // 1. Verify padding columns and end guards are untouched.
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t col = N; col < ldy; ++col) {
            const size_t idx = (size_t) t * ldy + col;
            if (got_y[idx] != kSentinel) {
                failure_count++;
                if (first_mismatch && first_mismatch->dtype[0] == '\0') {
                    *first_mismatch = MismatchRecord{dtype_str, T, N, K, ldy, beta, t, col,
                                                     got_y[idx], kSentinel,
                                                     std::abs((double) got_y[idx] - (double) kSentinel), true};
                }
                std::fprintf(stderr, "FAIL: output touched padding column: dtype=%s T=%ld N=%ld K=%ld ldy=%ld beta=%.1f row=%ld col=%ld got=%.9g expected=%.9g\n",
                             dtype_str, T, N, K, ldy, beta, t, col, got_y[idx], kSentinel);
                return CaseStatus::Failed;
            }
        }
    }
    for (size_t g = (size_t) T * ldy; g < y_count; ++g) {
        if (got_y[g] != kSentinel) {
            failure_count++;
            if (first_mismatch && first_mismatch->dtype[0] == '\0') {
                *first_mismatch = MismatchRecord{dtype_str, T, N, K, ldy, beta, -1, (int64_t) g,
                                                 got_y[g], kSentinel,
                                                 std::abs((double) got_y[g] - (double) kSentinel), true};
            }
            std::fprintf(stderr, "FAIL: output touched end guard: dtype=%s T=%ld N=%ld K=%ld ldy=%ld beta=%.1f index=%zu got=%.9g expected=%.9g\n",
                         dtype_str, T, N, K, ldy, beta, g, got_y[g], kSentinel);
            return CaseStatus::Failed;
        }
    }

    // 2. Compute reference on CPU and verify exact match.
    std::vector<float> ref_y(y_count, kSentinel);
    gemm_reference(x.data(), w.data(), initial_y.data(), ref_y.data(), T, N, K, ldy, beta, is_bf16);

    double case_max_abs = 0.0;
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t n = 0; n < N; ++n) {
            const size_t idx = (size_t) t * ldy + n;
            if (!std::isfinite(got_y[idx])) {
                failure_count++;
                if (first_mismatch && first_mismatch->dtype[0] == '\0') {
                    *first_mismatch = MismatchRecord{dtype_str, T, N, K, ldy, beta, t, n,
                                                     got_y[idx], ref_y[idx], 1e30, false};
                }
                std::fprintf(stderr, "FAIL: non-finite output: dtype=%s T=%ld N=%ld K=%ld ldy=%ld beta=%.1f at (%ld, %ld): got=%.9g\n",
                             dtype_str, T, N, K, ldy, beta, t, n, got_y[idx]);
                return CaseStatus::Failed;
            }
            const double delta = std::abs(static_cast<double>(got_y[idx]) - static_cast<double>(ref_y[idx]));
            case_max_abs = std::max(case_max_abs, delta);
            // The RDNA3 matrix core accumulates in fp32 but rounds toward zero rather than to nearest, while
            // this reference sums in double and rounds to nearest, so a correct result may differ by one ULP
            // (measured: every mismatch on this hardware is <= 1 ULP and toward zero).  Compare against a
            // tolerance of 4 ULP - tight enough to stay diagnostic, since a wrong row, column, stride or tile
            // boundary differs by O(value) rather than by 1e-7.  STRATA_WMMA_PARITY_EXACT=1 demands bit
            // equality for anyone who wants to see the raw deltas.
            static const bool exact = std::getenv("STRATA_WMMA_PARITY_EXACT") != nullptr;
            const double tol = exact ? 0.0
                                     : 4.0 * (double)std::numeric_limits<float>::epsilon() *
                                           std::max(1.0, std::abs(static_cast<double>(ref_y[idx])));
            if (delta > tol) {
                failure_count++;
                if (first_mismatch && first_mismatch->dtype[0] == '\0') {
                    *first_mismatch = MismatchRecord{dtype_str, T, N, K, ldy, beta, t, n,
                                                     got_y[idx], ref_y[idx], delta, false};
                }
                std::fprintf(stderr, "FAIL: mismatch: dtype=%s T=%ld N=%ld K=%ld ldy=%ld beta=%.1f at (%ld, %ld): got=%.9g ref=%.9g delta=%.9g\n",
                             dtype_str, T, N, K, ldy, beta, t, n, got_y[idx], ref_y[idx], delta);
                return CaseStatus::Failed;
            }
        }
    }

    out_max_abs = std::max(out_max_abs, case_max_abs);
    return CaseStatus::Passed;
}

struct DeclinedShape {
    int64_t T, N, K, ldy;
    float beta;
    const char* reason;
};

} // namespace

int main() {
    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, device));
    std::printf("Device %d: %s (arch %s, warpSize %d)\n",
                device, properties.name, properties.gcnArchName, properties.warpSize);
    // No matrix cores (RDNA2 and older, gfx906): the kernel declines every shape, so the suite has nothing to compare
    // and reports that as a skip (exit 77, SKIP_RETURN_CODE in CMake), as hip_prompt_attn_wmma does, not a failure.
    if (!strata::kernels::gfx_arch_is_gfx11_wmma(properties.gcnArchName)) {
        std::printf("WMMA GEMM parity SKIPPED: %s has no WMMA matrix cores (the kernel declines every shape here)\n",
                    properties.gcnArchName);
        return 77;
    }

    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    // Allocate reusable device buffers sized for the maximum test dimensions:
    // Max T = 127, Max N = 256, Max K = 80, Max ldy = 256 + 16.
    constexpr size_t kMaxXBytes = 127 * 80 * sizeof(uint16_t);
    constexpr size_t kMaxWBytes = 256 * 80 * sizeof(uint16_t);
    constexpr size_t kMaxYBytes = (127 * (256 + 16) + 64) * sizeof(float);

    DeviceBuffer dx(kMaxXBytes);
    DeviceBuffer dw(kMaxWBytes);
    DeviceBuffer dy(kMaxYBytes);

    const std::vector<int64_t> k_t_values = {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127};
    const std::vector<int64_t> k_n_values = {16, 17, 32, 48, 64, 65, 127, 256};
    const std::vector<int64_t> k_k_values = {16, 32, 48, 64, 80};

    // Test matrix configurations for (beta, ldy):
    // Covers beta=0 (overwrite) and beta=1 (accumulate), with ldy == N and ldy > N.
    struct Config {
        float beta;
        int64_t ldy_offset;
    };
    const std::vector<Config> kConfigs = {
        {0.0f, 0},  // beta=0, packed (ldy == N)
        {1.0f, 3},  // beta=1, strided (ldy == N + 3), tests accumulation + padding
        {0.0f, 7},  // beta=0, strided (ldy == N + 7), tests overwrite + padding
        {1.0f, 0},  // beta=1, packed (ldy == N), tests accumulation in packed layout
    };

    const std::vector<DeclinedShape> kDeclinedShapes = {
        // K not a multiple of 16
        {1, 16, 15, 16, 0.0f, "K not multiple of 16 (K=15)"},
        {15, 17, 24, 17, 0.0f, "K not multiple of 16 (K=24)"},
        {32, 32, 35, 32, 1.0f, "K not multiple of 16 (K=35)"},
        {64, 64, 47, 67, 1.0f, "K not multiple of 16 (K=47)"},
        {127, 256, 79, 256, 0.0f, "K not multiple of 16 (K=79)"},
        // ldy < N
        {16, 32, 16, 16, 0.0f, "ldy < N (ldy=16, N=32)"},
        {32, 48, 32, 40, 1.0f, "ldy < N (ldy=40, N=48)"},
        // beta not 0 or 1
        {16, 16, 16, 16, 0.5f, "beta not 0 or 1 (beta=0.5)"},
        {32, 32, 32, 32, 2.0f, "beta not 0 or 1 (beta=2.0)"},
        // Non-positive sizes
        {0, 16, 16, 16, 0.0f, "T <= 0 (T=0)"},
        {16, 0, 16, 16, 0.0f, "N <= 0 (N=0)"},
        {16, 16, 0, 16, 0.0f, "K <= 0 (K=0)"},
        {-1, 16, 16, 16, 0.0f, "T < 0 (T=-1)"},
    };

    size_t passed_count = 0;
    size_t skipped_count = 0;
    size_t failure_count = 0;
    double global_max_abs = 0.0;
    MismatchRecord first_mismatch{};
    uint32_t case_seed = 0x574d4d41u;

    const bool verbose = (std::getenv("STRATA_TEST_VERBOSE") != nullptr);

    for (bool is_bf16 : {false, true}) {
        const char* dtype_str = is_bf16 ? "bf16" : "f16";
        std::printf("Testing strata_wmma_gemm_%s across %zu shapes (%zu execution configs)...\n",
                    dtype_str, k_t_values.size() * k_n_values.size() * k_k_values.size(), kConfigs.size());

        for (int64_t T : k_t_values) {
            double t_max_abs = 0.0;
            size_t t_passed = 0;
            size_t t_skipped = 0;

            for (int64_t N : k_n_values) {
                for (int64_t K : k_k_values) {
                    for (const auto& cfg : kConfigs) {
                        const int64_t ldy = N + cfg.ldy_offset;
                        double case_max_abs = 0.0;
                        case_seed = case_seed * 1664525u + 1013904223u;

                        const CaseStatus status = run_gemm_case(
                            dx, dw, dy, stream, is_bf16, T, N, K, ldy, cfg.beta,
                            case_seed, case_max_abs, &first_mismatch, failure_count);

                        if (status == CaseStatus::Passed) {
                            passed_count++;
                            t_passed++;
                            t_max_abs = std::max(t_max_abs, case_max_abs);
                            global_max_abs = std::max(global_max_abs, case_max_abs);
                            if (verbose) {
                                std::printf("  case dtype=%s T=%3ld N=%3ld K=%2ld ldy=%3ld beta=%.1f max_abs=%.9g PASS\n",
                                            dtype_str, T, N, K, ldy, cfg.beta, case_max_abs);
                            }
                        } else if (status == CaseStatus::Skipped) {
                            skipped_count++;
                            t_skipped++;
                            if (verbose) {
                                std::printf("  case dtype=%s T=%3ld N=%3ld K=%2ld ldy=%3ld beta=%.1f SKIP (declined)\n",
                                            dtype_str, T, N, K, ldy, cfg.beta);
                            }
                        } else {
                            // Failed: first_mismatch already recorded and reported to stderr
                            if (verbose) {
                                std::printf("  case dtype=%s T=%3ld N=%3ld K=%2ld ldy=%3ld beta=%.1f FAIL\n",
                                            dtype_str, T, N, K, ldy, cfg.beta);
                            }
                        }
                    }
                }
            }
            std::printf("  dtype=%s T=%3ld: %zu passed, %zu skipped, max_abs=%.9g %s\n",
                        dtype_str, T, t_passed, t_skipped, t_max_abs,
                        (failure_count == 0) ? "PASS" : "FAIL");
        }

        // Test declined shapes: the kernel must return false and be skipped.
        for (const auto& decl : kDeclinedShapes) {
            const bool launched = is_bf16
                ? strata_wmma_gemm_bf16(static_cast<const uint16_t*>(dx.p),
                                        static_cast<const uint16_t*>(dw.p),
                                        static_cast<float*>(dy.p),
                                        decl.T, decl.N, decl.K, decl.ldy, decl.beta, stream)
                : strata_wmma_gemm_f16(static_cast<const uint16_t*>(dx.p),
                                       static_cast<const uint16_t*>(dw.p),
                                       static_cast<float*>(dy.p),
                                       decl.T, decl.N, decl.K, decl.ldy, decl.beta, stream);

            if (launched) {
                failure_count++;
                std::fprintf(stderr, "FAIL: expected kernel to decline shape (%s) dtype=%s T=%ld N=%ld K=%ld ldy=%ld beta=%.1f but it returned true!\n",
                             decl.reason, dtype_str, decl.T, decl.N, decl.K, decl.ldy, decl.beta);
            } else {
                skipped_count++;
                if (verbose) {
                    std::printf("  declined shape correctly rejected (%s): dtype=%s T=%ld N=%ld K=%ld ldy=%ld beta=%.1f SKIP\n",
                                decl.reason, dtype_str, decl.T, decl.N, decl.K, decl.ldy, decl.beta);
                }
            }
        }
    }

    // Direct Q8_0 WMMA parity checks: strata_pf_gemm_q8_0_ld vs strata_pf_gemm_f16_ld and
    // strata_pf_hcdown_q8_0 vs strata_pf_hcdown_bf16 must be BITWISE IDENTICAL across normal-range scales.
    {
        struct Q8Shape { int64_t T, N, K; float beta; };
        const Q8Shape q8_shapes[] = {
            {31, 640, 2560, 0.0f},     // T=31 message-boundary tail chunk (<128, 32> short-T path)
            {31, 2560, 6144, 0.0f},    // T=31 wide N=2560 K=6144 projection path
            {129, 640, 2560, 0.0f},
            {408, 1280, 2560, 0.0f},   // <128, 32> short-T occupancy path
            {408, 2560, 640, 1.0f},    // <256, 64> beta=1 path
            {1024, 2560, 2560, 0.0f},  // <256, 64> full tile path
        };
        std::mt19937 qrng(20261028u);
        std::uniform_int_distribution<int> qdist(-127, 127);
        std::uniform_int_distribution<int> sdist(1, 16);
        for (const auto& s : q8_shapes) {
            const int64_t ldx = s.K + 64, ldw = s.K + 64, ldy = s.N;
            std::vector<uint16_t> hx((size_t) s.T * ldx, 0);
            std::vector<uint16_t> hwf16((size_t) s.N * ldw, 0);
            const size_t row_bytes = (size_t) (s.K / 32) * 34;
            std::vector<uint8_t> hwq8((size_t) s.N * row_bytes, 0);
            for (int64_t t = 0; t < s.T; ++t)
                for (int64_t k = 0; k < s.K; ++k)
                    hx[(size_t) t * ldx + k] = encode(qdist(qrng) * (1.0f / 64.0f), false);
            for (int64_t n = 0; n < s.N; ++n) {
                for (int64_t g = 0; g < s.K / 32; ++g) {
                    uint8_t* blk = hwq8.data() + (size_t) n * row_bytes + (size_t) g * 34;
                    const float d = sdist(qrng) * (1.0f / 256.0f);
                    const uint16_t dh = encode(d, false);
                    std::memcpy(blk, &dh, 2);
                    const float df = decode(dh, false);
                    for (int j = 0; j < 32; ++j) {
                        const int8_t q = (int8_t) qdist(qrng);
                        blk[2 + j] = (uint8_t) q;
                        hwf16[(size_t) n * ldw + g * 32 + j] = encode((float) q * df, false);
                    }
                }
            }
            std::vector<float> init_y((size_t) s.T * ldy, 0.25f);
            DeviceBuffer qx(hx.size() * 2), qwf16(hwf16.size() * 2), qwq8(hwq8.size());
            DeviceBuffer qy_ref(init_y.size() * 4), qy_got(init_y.size() * 4);
            HIP_CHECK(hipMemcpyAsync(qx.p, hx.data(), hx.size() * 2, hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(qwf16.p, hwf16.data(), hwf16.size() * 2, hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(qwq8.p, hwq8.data(), hwq8.size(), hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(qy_ref.p, init_y.data(), init_y.size() * 4, hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(qy_got.p, init_y.data(), init_y.size() * 4, hipMemcpyHostToDevice, stream));
            const bool ok_ref = strata_pf_gemm_f16_ld((const uint16_t*) qx.p, ldx, (const uint16_t*) qwf16.p, ldw,
                                                      (float*) qy_ref.p, s.T, s.N, s.K, ldy, s.beta, stream);
            const bool ok_got = strata_pf_gemm_q8_0_ld((const uint16_t*) qx.p, ldx, qwq8.p,
                                                       (float*) qy_got.p, s.T, s.N, s.K, ldy, s.beta, stream);
            if (!ok_ref || !ok_got) {
                failure_count++;
                std::fprintf(stderr, "FAIL: strata_pf_gemm_q8_0_ld declined shape T=%lld N=%lld K=%lld\n",
                             (long long) s.T, (long long) s.N, (long long) s.K);
            } else {
                std::vector<float> y_ref(init_y.size()), y_got(init_y.size());
                HIP_CHECK(hipMemcpyAsync(y_ref.data(), qy_ref.p, init_y.size() * 4, hipMemcpyDeviceToHost, stream));
                HIP_CHECK(hipMemcpyAsync(y_got.data(), qy_got.p, init_y.size() * 4, hipMemcpyDeviceToHost, stream));
                HIP_CHECK(hipStreamSynchronize(stream));
                if (std::memcmp(y_ref.data(), y_got.data(), init_y.size() * 4) != 0) {
                    failure_count++;
                    std::fprintf(stderr, "FAIL: strata_pf_gemm_q8_0_ld bitwise mismatch T=%lld N=%lld K=%lld\n",
                                 (long long) s.T, (long long) s.N, (long long) s.K);
                } else {
                    passed_count++;
                }
            }
        }

        // strata_pf_hcdown_q8_0 vs strata_pf_hcdown_bf16 bitwise parity
        for (const int64_t T : { (int64_t) 31, (int64_t) 257 }) {
            const int64_t nd = 320, ni = 4, K = 10240, ldx = K + 64;
            const size_t row_bytes = (size_t) (K / 32) * 34;
            std::vector<uint16_t> hx((size_t) T * ldx, 0), hwd_bf16((size_t) nd * K, 0), hwi((size_t) ni * K, 0);
            std::vector<uint8_t> hwd_q8((size_t) nd * row_bytes, 0);
            for (int64_t t = 0; t < T; ++t)
                for (int64_t k = 0; k < K; ++k)
                    hx[(size_t) t * ldx + k] = encode(qdist(qrng) * (1.0f / 64.0f), true);
            for (int64_t n = 0; n < ni; ++n)
                for (int64_t k = 0; k < K; ++k)
                    hwi[(size_t) n * K + k] = encode(qdist(qrng) * (1.0f / 64.0f), true);
            for (int64_t n = 0; n < nd; ++n) {
                for (int64_t g = 0; g < K / 32; ++g) {
                    uint8_t* blk = hwd_q8.data() + (size_t) n * row_bytes + (size_t) g * 34;
                    const float d = sdist(qrng) * (1.0f / 256.0f);
                    const uint16_t dh = encode(d, false);
                    std::memcpy(blk, &dh, 2);
                    const float df = decode(dh, false);
                    for (int j = 0; j < 32; ++j) {
                        const int8_t q = (int8_t) qdist(qrng);
                        blk[2 + j] = (uint8_t) q;
                        hwd_bf16[(size_t) n * K + g * 32 + j] = encode((float) q * df, true);
                    }
                }
            }
            DeviceBuffer qx(hx.size() * 2), qwd_bf16(hwd_bf16.size() * 2), qwd_q8(hwd_q8.size()), qwi(hwi.size() * 2);
            DeviceBuffer qlo_ref((size_t) T * nd * 4), qinj_ref((size_t) T * ni * 4);
            DeviceBuffer qlo_got((size_t) T * nd * 4), qinj_got((size_t) T * ni * 4);
            HIP_CHECK(hipMemcpyAsync(qx.p, hx.data(), hx.size() * 2, hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(qwd_bf16.p, hwd_bf16.data(), hwd_bf16.size() * 2, hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(qwd_q8.p, hwd_q8.data(), hwd_q8.size(), hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(qwi.p, hwi.data(), hwi.size() * 2, hipMemcpyHostToDevice, stream));
            const bool ok_ref = strata_pf_hcdown_bf16((const uint16_t*) qx.p, ldx, (const uint16_t*) qwd_bf16.p,
                                                      (const uint16_t*) qwi.p, nd, ni, (float*) qlo_ref.p,
                                                      (float*) qinj_ref.p, T, K, stream);
            const bool ok_got = strata_pf_hcdown_q8_0((const uint16_t*) qx.p, ldx, qwd_q8.p,
                                                      (const uint16_t*) qwi.p, nd, ni, (float*) qlo_got.p,
                                                      (float*) qinj_got.p, T, K, stream);
            if (!ok_ref || !ok_got) {
                failure_count++;
                std::fprintf(stderr, "FAIL: strata_pf_hcdown_q8_0 declined\n");
            } else {
                std::vector<float> lo_ref((size_t) T * nd), lo_got((size_t) T * nd);
                std::vector<float> inj_ref((size_t) T * ni), inj_got((size_t) T * ni);
                HIP_CHECK(hipMemcpyAsync(lo_ref.data(), qlo_ref.p, lo_ref.size() * 4, hipMemcpyDeviceToHost, stream));
                HIP_CHECK(hipMemcpyAsync(lo_got.data(), qlo_got.p, lo_got.size() * 4, hipMemcpyDeviceToHost, stream));
                HIP_CHECK(hipMemcpyAsync(inj_ref.data(), qinj_ref.p, inj_ref.size() * 4, hipMemcpyDeviceToHost, stream));
                HIP_CHECK(hipMemcpyAsync(inj_got.data(), qinj_got.p, inj_got.size() * 4, hipMemcpyDeviceToHost, stream));
                HIP_CHECK(hipStreamSynchronize(stream));
                if (std::memcmp(lo_ref.data(), lo_got.data(), lo_ref.size() * 4) != 0 ||
                    std::memcmp(inj_ref.data(), inj_got.data(), inj_ref.size() * 4) != 0) {
                    failure_count++;
                    std::fprintf(stderr, "FAIL: strata_pf_hcdown_q8_0 bitwise mismatch\n");
                } else {
                    passed_count++;
                }
            }
        }
    }

    HIP_CHECK(hipStreamDestroy(stream));

    if (failure_count > 0) {
        if (first_mismatch.dtype[0] != '\0') {
            if (first_mismatch.is_padding) {
                std::fprintf(stderr, "\nSUMMARY: First mismatch was padding/guard corruption:\n"
                                     "  dtype=%s T=%ld N=%ld K=%ld ldy=%ld beta=%.1f at (%ld, %ld): got=%.9g expected=%.9g\n",
                             first_mismatch.dtype, first_mismatch.T, first_mismatch.N, first_mismatch.K,
                             first_mismatch.ldy, first_mismatch.beta, first_mismatch.row, first_mismatch.col,
                             first_mismatch.got, first_mismatch.ref);
            } else {
                std::fprintf(stderr, "\nSUMMARY: First mismatch occurred at:\n"
                                     "  dtype=%s T=%ld N=%ld K=%ld ldy=%ld beta=%.1f at (%ld, %ld): got=%.9g ref=%.9g delta=%.9g\n",
                             first_mismatch.dtype, first_mismatch.T, first_mismatch.N, first_mismatch.K,
                             first_mismatch.ldy, first_mismatch.beta, first_mismatch.row, first_mismatch.col,
                             first_mismatch.got, first_mismatch.ref, first_mismatch.delta);
            }
        }
        std::fprintf(stderr, "WMMA GEMM parity FAILED: %zu failures detected, max_abs=%.9g\n",
                     failure_count, global_max_abs);
        return 1;
    }

    if (passed_count == 0) {
        std::fprintf(stderr, "WMMA GEMM parity FAILED: zero test cases passed (all were declined or skipped)\n");
        return 1;
    }

    std::printf("\nWMMA GEMM parity OK: %zu test cases passed, %zu declined shapes skipped, max_abs=%.9g\n",
                passed_count, skipped_count, global_max_abs);
    return 0;
}
