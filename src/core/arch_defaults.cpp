// src/core/arch_defaults.cpp - see include/strata/core/arch_defaults.hpp.
#include "strata/core/arch_defaults.hpp"

#include "strata/kernels/gfx_arch.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch) {
    std::vector<std::pair<std::string, std::string>> t;
    if (strata::kernels::gfx_arch_is_gfx12(gcn_arch)) {
        // gfx1200 / gfx1201 (RDNA4): only the PROMPT half of the gfx1151 table.  Reported exact (identical answers) on an R9700
        // in #1478 and re-checked bitwise here (notes-142-amd.md); the decode half measured slower on that card, so it stays out.
        // STRATA_HCD_EXACT is left out: it needs the gfx1151 hipBLASLt tuning table, finds none on gfx12, says so on every start
        // and does nothing (measured: with and without it identical, 0.0 to +0.1%).
        // STRATA_GDN_CONVL2 is AMD-only on purpose: it is not bitwise on CUDA 13.3 sm_120 (#1522), and this table is never
        // consulted by a CUDA build.
        const char* off12 = std::getenv("STRATA_GFX12_DEFAULTS");
        if (off12 != nullptr && off12[0] == '0') return t;
        t = {
            {"STRATA_GDN_HEAD", "1"},
            {"STRATA_GDN_PP", "2"},
            {"STRATA_GDN_CONVL2", "1"},
            {"STRATA_GDN_NOY", "1"},
            {"STRATA_CVEC_FUSE", "1"},
            {"STRATA_DEQUANT_WIDE", "1"},
        };
        return t;
    }
    if (!strata::kernels::gfx_arch_is_gfx1151(gcn_arch)) return t;
    const char* off = std::getenv("STRATA_GFX1151_DEFAULTS");
    if (off != nullptr && off[0] == '0') return t;
    // Exact (bitwise) on Aurora, each with its measured gain in aurora_s23.md:
    t = {
        // prompt
        {"STRATA_GDN_HEAD", "1"},        // the GDN recurrence, four lanes a column + the grid-stride norm (S23)
        {"STRATA_GDN_PP", "2"},          //   two columns per lane: 64K prompt +2.9% (stream A, round 2)
        {"STRATA_GDN_CONVL2", "1"},      // conv + q/k L2 norm in one kernel: 64K -0.3 s (stream A)
        {"STRATA_GDN_NOY", "1"},         // no dead FP32 store of the norm (stream A)
        {"STRATA_CVEC_FUSE", "1"},       // a steered layer's write, control vector and next norm in one pass (S23)
        {"STRATA_HCD_EXACT", "1"},       // the HC down GEMM in hipBLASLt solution 1176's k order: 64K +1.8% (stream S)
        {"STRATA_PF_PAD", "1"},          // padded GEMM row strides (acts with STRATA_PF_GEMM; inert without it) (S23)
        {"STRATA_DEQUANT_WIDE", "1"},    // coalesced 16-byte uint4 wide dequantization for Q4_K/Q5_K/Q6_K/IQ4_NL/IQ4_XS (bitwise exact)
        {"STRATA_TOPK_REG_MIN_BLOCKS", "4096"},   // the register top-k from 16K cells, not 28K: same ids; per 2,048 queries the ref takes 1.0 / 2.4 / 3.9 ms at 16K / 20K / 24K cells, the scan 0.87 / 0.93 / 1.00
        // decode
        {"STRATA_Q8_PACKED", "1"},       // the packed Q8_0 decode layout: +2.3% (S26)
        {"STRATA_Q6_PACKED", "1"},       // the packed Q6_K heads (S26)
        {"STRATA_MMVF_ROWS", "1"},       // bf16 multi-row GEMV, 4 rows per block (S25)
        {"STRATA_ATTN_LANECELL", "1"},   // decode attention scores, one cell per thread (S25)
        {"STRATA_EXPERT_V2", "1"},       // the grouped decode experts, IQ3_S gate/up + IQ4_NL down (S26)
        {"STRATA_TSUM", "1"},            // several warp sums as one transposed butterfly (S26)
        {"STRATA_LFUSE", "1"},           // fewer launches around the shared expert and the KV append (S26)
        {"STRATA_GDN_SPLIT", "1"},       // the GDN step over 4 blocks per head (S25/S26)
        {"STRATA_QFUSE", "1"},           // activation q8_1 images written by their producers (S26)
        {"STRATA_PLE_BATCH", "1"},       // the verify window's PLE key / value projections at once (S25)
#if defined(_WIN32)
        {"STRATA_PF_Z_STREAM", "0"},     // Windows WDDM: avoid 72 cross-stream event barriers/chunk in GDN prefill
        {"STRATA_PF_SH_STREAM", "0"},    // Windows WDDM: avoid 96 cross-stream event barriers/chunk in shared expert (-813 ms at 16K)
        {"STRATA_SH_STREAM", "0"},       // Windows WDDM: single-stream decode yields +16.4% decode speed (Phase 1)
#else
        {"STRATA_PF_Z_STREAM", "1"},     // non-Windows: GDN Z-projection side-stream overlap (Gufo #758 da4d8b82)
        {"STRATA_PF_SH_STREAM", "1"},    // non-Windows: overlap shared expert GEMMs on side_stream for T >= 1024
        {"STRATA_SH_STREAM", "1"},       // the shared expert on its own stream: decode +1.8% / +6.7% (UD-Q4_K_XL) (140-m)
#endif

        {"STRATA_GDN_CHUNKED", "1"},     // chunked DeltaNet recurrence on gfx11 matrix cores (Gufo #758 29e900c9)
        {"STRATA_WDDM_FLUSH", "0"},      // Pass 5B.4: 0 avoids unnecessary synchronous WDDM flush overhead
        {"STRATA_MOE_MAX_BM", "64"},     // Pass 5B.4: grouped MoE max block multiple
        {"STRATA_FUSE_HITS", "1"},       // graph fusion: fuse hits
        {"STRATA_FUSE_WAIT", "1"},       // graph fusion: fuse wait
        {"STRATA_HEAD_MIX_MULTI", "1"},  // graph fusion: multi-head mix
        {"STRATA_QFUSE_HC", "1"},        // graph fusion: Q-fuse HC
        {"STRATA_ADMIT_MERGE_FIRST", "1"},// merge trailing prompt header tokens into first verify window
        {"STRATA_CKPT_POOL", "1"},       // reuse conversation checkpoint vector buffers
        {"STRATA_PF_SWITCH_MIN_T", "16"}, // warmup WMMA GEMM / hipBLASLt plans for chunked prefill (including T=31 tail chunk)

        {"STRATA_PF_FUSED", "1"},        // zero-sync GPU fused experts prefill
        {"STRATA_PF_F16", "1"},          // FP16 WMMA matrix-core experts prefill (-25.0% MoE wall time on gfx1151, 105/105 parity)
        {"STRATA_REFILL_PARTS", "1"},    // eliminate CPU staging buffer assembly during refill
        {"STRATA_EAGER_PARK", "1"},      // eager background conversation parking
        {"STRATA_PF_DM_F16", "1"},       // FP16 MoE down-projection scratch bandwidth reduction
        {"STRATA_PREFILL_STREAM_MIN", "16"}, // stream all experts for T >= 16 on gfx1151
        {"STRATA_PF_MOE_SCATTER", "0"},  // combine directly via m.Dm (avoids atomicAdd contention)
        {"STRATA_PF_HC_WMMA_Q8", "0"},   // disable prefill HC WMMA Q8 fallback
        {"STRATA_EXPDEC", "0"},          // expert decode: 0 preserves full WGP occupancy on gfx1151
        {"STRATA_EXP_MODE", "8"},        // fused LDS gate+up+SwiGLU+q8_1 + LDS down across IQ3_S/IQ4_XS/IQ4_NL/Q8_0 (Pass 6E)
        {"STRATA_S27_FUSE_GU", "1"},     // fuse gate & up decode projection (Pass 6C)
        {"STRATA_EXPDEC_FUSE_GU", "1"},  // helper when STRATA_EXPDEC=1 is explicitly tested
        {"STRATA_FAST_LOAD", "1"},       // fast tensor loading
        {"STRATA_BULK_RING_MIB", "96"},  // staging ring size (MiB)
        {"STRATA_BULK_THREADS", "16"},   // bulk load worker threads
        {"STRATA_PREFETCH_WINDOW", "192"},// prefetch window size
        {"STRATA_VERIFY_COHERENT", "1"}, // coherent verify window reads
        {"STRATA_MTP_DEVICE_CHAIN", "1"},// GPU-side MTP draft validation chain
        {"STRATA_WDDM_COPY_KERNEL", "1"},// GPU copy kernel for WDDM staging
    };
    return t;
}

std::vector<std::string> apply_arch_defaults(const char* gcn_arch) {
    std::vector<std::string> set;
    for (const auto& kv : arch_default_env(gcn_arch)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, whatever it is
#if defined(_WIN32)
        _putenv_s(kv.first.c_str(), kv.second.c_str());
#else
        setenv(kv.first.c_str(), kv.second.c_str(), 0);
#endif
        set.push_back(kv.first);
    }
    if (!set.empty()) {
        const bool g12 = strata::kernels::gfx_arch_is_gfx12(gcn_arch);
        std::fprintf(stderr, "strata: %s: %zu exact speed switches on by default (%s=0 turns "
                             "them off; a switch you set is kept): ",
                     g12 ? "gfx12 (RDNA4)" : "gfx1151 (Strix Halo)", set.size(),
                     g12 ? "STRATA_GFX12_DEFAULTS" : "STRATA_GFX1151_DEFAULTS");
        for (size_t i = 0; i < set.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", set[i].c_str() + 7);
        std::fprintf(stderr, "\n");
    }
    return set;
}

}  // namespace strata::core
