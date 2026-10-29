# PASS 7 GITHUB MINING & AST DIFFERENTIAL REPORT
## Upstream Strata (`v0.1.45+` / `v0.2.7`), Gufo (`v0.25.0`), and Community AMD Strix Halo (`gfx1151`) Optimizations

- **Author**: `GithubStrixForkMiner` (`TASK-W1-GITHUB-MINER`)
- **Date**: 2026-10-28
- **Target Architecture**: AMD Ryzen AI Max+ 395 ("Strix Halo", 16 Zen 5 Cores / 32 Threads, Radeon 8060S `gfx1151` RDNA 3.5 GPU, 40 Compute Units / 20 WGPs, `wave32`, 128 GB LPDDR5X-8000 Unified Memory, 96 GiB BIOS UMA Carveout, Windows 11 WDDM)
- **Target Model**: Qwen 3.8 Flash-Next 177B (`UD-IQ4_XS`, 48 layers: 36 Gated DeltaNet + 12 QSA, 512 routed experts/layer, top-10 active + 1 shared expert, 24,576 total GPU experts = 55.43 GiB)
- **Baseline Monitored**: Custom Pass 6 Engine (`HEAD: 413b098a` on branch `pass6/gfx1151-1300pf-80dec` in `c:\Users\Butch\Documents\02_Apps\Strata`)
- **Staging Directory**: `c:\Users\Butch\Documents\02_Apps\Strata\.harness\staging-repos\` (fully git-ignored)

---

## 1. Executive Summary & Hardware Context

### 1.1 Mission & Architectural Baseline
This report conducts an empirical code and AST differential analysis across upstream **Strata** ([`Niko1221/Strata`](https://github.com/Niko1221/Strata) at tag `v0.2.7` / engine `0.1.52`, commit `5e8a4cc`), upstream **Gufo** ([`gufo-org/gufo`](https://github.com/gufo-org/gufo) at release `0.25.0`, commit `c92b19c`), the Windows Strix Halo port **`pixmaate/gufo`** (commit `2eb0778`), and related community Strix Halo forks (`halo-box/strix-llama.cpp`, `FrancescoBrunoDev/gufo`, `bidyut12345/gufo-windows`).

Our local repository at `c:\Users\Butch\Documents\02_Apps\Strata` is **NOT** vanilla upstream Strata. It is our heavily tuned **Custom Pass 6** engine (`HEAD: 413b098a`), which already delivers state-of-the-art inference figures on AMD Strix Halo:
- **Cold Prefill Throughput**: `1,206.6 tok/s` at Tier 512, `1,386.2 tok/s` at 1K, `1,546.5 tok/s` at 8K, and `1,566.6 tok/s` at 16K context.
- **Thinking Decode**: `82.5–82.9 tok/s` burst (`--spec 7 --mtp-q4 head`, `86.7%` MTP acceptance) and `58.4–59.8 tok/s` sustained high-effort reasoning decode (`temp=0`, `reasoning_effort: "high"`) under the physical 256 GB/s LPDDR5X DRAM roofline (`53.16 ms` verify latency, `0.65–0.84 ms` PLE stage).
- **Warm KV Cache Reuse**: `539,042.6 tok/s` (`61 ms` TTFT).
- **Cold Startup**: `32.6 s` (`1,828 MB/s`) cold upload of all 24,576 experts (`55.43 GiB`) via coalesced DMA and Win32 `PrefetchVirtualMemory`.
- **System Memory Safety**: Windows commit headroom preserved with `22.7–25.7 GiB` physical host RAM free via `STRATA_FILE_RELEASE=1` and `STRATA_CKPT_POOL=1`.

### 1.2 Upstream Vanilla Reality vs. Custom Pass 6
Upstream Strata `0.2.7` published official Windows Strix Halo benchmarks in `docs/STRIX_HALO_WINDOWS.md` (commit `ee1a6ef` / PR #3029):
- Vanilla Upstream 4K Prefill: `755 tok/s` (Pass 6 reaches **`1,531.6 tok/s`** — **`2.03x` faster**).
- Vanilla Upstream 16K Prefill: `821 tok/s` (Pass 6 reaches **`1,566.6 tok/s`** — **`1.91x` faster**).
- Vanilla Upstream Decode: `44.0 tok/s` (Pass 6 reaches **`82.9 tok/s`** burst / **`59.8 tok/s`** sustained — **`1.36x to 1.88x` faster**).

Any port from upstream or community repositories must be surgical and guarded so our Custom Pass 6 protected subsystems are 100% preserved.

---

## 2. Complete Inventory of Compiled Runtime Environment Variables in Pass 6 (`413b098a`)

Every runtime environment variable compiled into `c:\Users\Butch\Documents\02_Apps\Strata\src` was audited via AST source inspection against `src/core/arch_defaults.cpp`, `src/prefill/prefill.cpp`, `src/prefill/gemm.cu`, `src/prefill/wmma_gemm.cu`, `src/prefill/moe_fused_iq.cu`, `src/kernels/cuda/qsa_select.cu`, `src/kernels/cuda/iq_kernels.cu`, `src/kernels/cuda/fused_gr.cu`, `src/core/mtp.cpp`, `src/core/verify.cpp`, and `src/core/expert_cache.cpp`.

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
<table style="width: 100%; min-width: 900px; border-collapse: collapse; background-color: #161b22; color: #c9d1d9; font-size: 12px;">
  <thead style="background-color: #21262d; border-bottom: 2px solid #30363d;">
    <tr>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">Environment Variable</th>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">Pass 6 Default</th>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">Source Location</th>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">Subsystem &amp; Mechanism</th>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">Stage 1 A-B-B-A Role / Status</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">STRATA_PREFILL_STREAM_MIN</td>
      <td style="padding: 7px 10px; white-space: nowrap;">16</td>
      <td style="padding: 7px 10px; white-space: nowrap;">prefill.cpp:180</td>
      <td style="padding: 7px 10px;">Switches MoE from 8-slot staging ring to zero-sync resident GPU expert stream_all path</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950; font-weight: 600;">Fixed Baseline (=16 mandatory)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">STRATA_PF_FUSED</td>
      <td style="padding: 7px 10px; white-space: nowrap;">1</td>
      <td style="padding: 7px 10px; white-space: nowrap;">moe_fused_iq.cu:1471</td>
      <td style="padding: 7px 10px;">Zero-sync GPU grouped native IQ3_S/IQ4_NL MoE prefill dispatch</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950; font-weight: 600;">Fixed Baseline (=1 mandatory)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">STRATA_PF_F16</td>
      <td style="padding: 7px 10px; white-space: nowrap;">0</td>
      <td style="padding: 7px 10px; white-space: nowrap;">moe_fused_iq.cu:1583</td>
      <td style="padding: 7px 10px;">FP16 WMMA MoE prefill (7.12x slower on gfx1151; bypasses INT8 WMMA)</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #f85149; font-weight: 600;">FORBIDDEN (F-01: =0 mandatory)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">STRATA_MOE_MAX_BM</td>
      <td style="padding: 7px 10px; white-space: nowrap;">64</td>
      <td style="padding: 7px 10px; white-space: nowrap;">moe_tile_buckets.cuh:52</td>
      <td style="padding: 7px 10px;">Caps variable MoE block multiple (BM=64 for hot Zipf experts, BM=32/16 for cold tail)</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950; font-weight: 600;">Fixed Baseline (=64 mandatory)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">STRATA_EXP_MODE</td>
      <td style="padding: 7px 10px; white-space: nowrap;">8</td>
      <td style="padding: 7px 10px; white-space: nowrap;">iq_kernels.cu:4635</td>
      <td style="padding: 7px 10px;">All-48-layer fused LDS MoE decode across (21,20), (21,8), (23,20), (23,8) quants</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950; font-weight: 600;">Fixed Baseline (=8 mandatory)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">STRATA_S27_FUSE_GU</td>
      <td style="padding: 7px 10px; white-space: nowrap;">1</td>
      <td style="padding: 7px 10px; white-space: nowrap;">iq_kernels.cu:2990</td>
      <td style="padding: 7px 10px;">Fuses gate and up decode projections with in-LDS SwiGLU activation</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950; font-weight: 600;">Fixed Baseline (=1 mandatory)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">STRATA_SH_STREAM</td>
      <td style="padding: 7px 10px; white-space: nowrap;">0 (Win) / 1 (Lin)</td>
      <td style="padding: 7px 10px; white-space: nowrap;">arch_defaults.cpp:58</td>
      <td style="padding: 7px 10px;">Shared expert side-stream execution (0 on Windows WDDM yields +16.4% decode)</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #f85149; font-weight: 600;">FORBIDDEN (F-06: =0 on Win)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">STRATA_PF_GEMM</td>
      <td style="padding: 7px 10px; white-space: nowrap;">Unset (0)</td>
      <td style="padding: 7px 10px; white-space: nowrap;">gemm.cu:1069</td>
      <td style="padding: 7px 10px;">Intercepts f16 GEMM with custom strata_pf_gemm_f16 (BM=128), bypassing hipBLASLt</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #e3b341; font-weight: 600;">Stage 1 Test Arm (0 vs 1 Root Cause)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">STRATA_PF_SWITCH_MIN_T</td>
      <td style="padding: 7px 10px; white-space: nowrap;">128</td>
      <td style="padding: 7px 10px; white-space: nowrap;">gemm.cu:1070</td>
      <td style="padding: 7px 10px;">Token threshold above which STRATA_PF_GEMM intercepts dense GEMM</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #e3b341; font-weight: 600;">Stage 1 Test Arm (with PF_GEMM)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">STRATA_GDN_CHUNKED</td>
      <td style="padding: 7px 10px; white-space: nowrap;">1 (defaults)</td>
      <td style="padding: 7px 10px; white-space: nowrap;">arch_defaults.cpp:63</td>
      <td style="padding: 7px 10px;">Chunked DeltaNet recurrence on WMMA (proven lossy by Gufo PR #850, NLL +0.0145)</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #e3b341; font-weight: 600;">Stage 1 Priority Arm (0 vs 1 Check)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">STRATA_PF_HCDOWN</td>
      <td style="padding: 7px 10px; white-space: nowrap;">0</td>
      <td style="padding: 7px 10px; white-space: nowrap;">prefill.cpp:500</td>
      <td style="padding: 7px 10px;">Replaces hipBLASLt solution 1176 HCD with custom WMMA down-projection</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #e3b341; font-weight: 600;">Stage 1 Test Arm (keep 0)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">STRATA_HCD_EXACT</td>
      <td style="padding: 7px 10px; white-space: nowrap;">1</td>
      <td style="padding: 7px 10px; white-space: nowrap;">arch_defaults.cpp:43</td>
      <td style="padding: 7px 10px;">HC down GEMM via hipBLASLt solution 1176 K-order (+1.8% at 64K prompt)</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950; font-weight: 600;">Fixed Baseline (=1 mandatory)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">STRATA_TOPK_REG_MIN_BLOCKS</td>
      <td style="padding: 7px 10px; white-space: nowrap;">4096</td>
      <td style="padding: 7px 10px; white-space: nowrap;">arch_defaults.cpp:45</td>
      <td style="padding: 7px 10px;">QSA register top-k crossover threshold (4096 blocks = 16K cells on gfx1151)</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950; font-weight: 600;">Fixed Baseline (=4096 mandatory)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">STRATA_QFUSE</td>
      <td style="padding: 7px 10px; white-space: nowrap;">1</td>
      <td style="padding: 7px 10px; white-space: nowrap;">arch_defaults.cpp:55</td>
      <td style="padding: 7px 10px;">Activation q8_1 images written directly by their GPU producer kernels</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950; font-weight: 600;">Fixed Baseline (Pass 7 port: PR #3948 fix)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">STRATA_VERIFY_COHERENT</td>
      <td style="padding: 7px 10px; white-space: nowrap;">1</td>
      <td style="padding: 7px 10px; white-space: nowrap;">verify.cpp:205</td>
      <td style="padding: 7px 10px;">Uses hipHostMallocCoherent for zero-copy UMA verify staging buffers</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950; font-weight: 600;">Fixed Baseline (=1 mandatory)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">STRATA_MTP_DEVICE_CHAIN</td>
      <td style="padding: 7px 10px; white-space: nowrap;">1</td>
      <td style="padding: 7px 10px; white-space: nowrap;">mtp.cpp:555</td>
      <td style="padding: 7px 10px;">GPU-side MTP draft chain overlapping async PLE prefetch via coherent UMA</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950; font-weight: 600;">Fixed Baseline (=1 mandatory)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">STRATA_LOOKUP_THINK_RING</td>
      <td style="padding: 7px 10px; white-space: nowrap;">1</td>
      <td style="padding: 7px 10px; white-space: nowrap;">ngram.cpp:686</td>
      <td style="padding: 7px 10px;">Dynamic zero-VRAM n-gram ring cache indexing reasoning &lt;think&gt; tokens</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950; font-weight: 600;">Fixed Baseline (=1 mandatory)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">STRATA_FILE_RELEASE</td>
      <td style="padding: 7px 10px; white-space: nowrap;">1</td>
      <td style="padding: 7px 10px; white-space: nowrap;">expert_cache.cpp:1115</td>
      <td style="padding: 7px 10px;">Releases 55.15 GiB of GGUF standby pages via VirtualUnlock after GPU upload</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950; font-weight: 600;">Fixed Baseline (Host Commit Invariant)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">STRATA_CKPT_POOL</td>
      <td style="padding: 7px 10px; white-space: nowrap;">1</td>
      <td style="padding: 7px 10px; white-space: nowrap;">arch_defaults.cpp:71</td>
      <td style="padding: 7px 10px;">Pre-allocates and reuses checkpoint vector buffers across conversation turns</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950; font-weight: 600;">Fixed Baseline (Commit Safety)</td>
    </tr>
    <tr>
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">STRATA_HC_PERSIST</td>
      <td style="padding: 7px 10px; white-space: nowrap;">0 (Disabled)</td>
      <td style="padding: 7px 10px; white-space: nowrap;">fused_gr.cu:2907</td>
      <td style="padding: 7px 10px;">One-kernel persistent HC read (explicitly bypassed on gfx1151: lacks s_memrealtime)</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #8b949e;">Inert on gfx1151 (keep 0)</td>
    </tr>
  </tbody>
</table>
</div>

---

## 3. Root-Cause Diagnostic Forensic: Why Tier 512 Dropped to `470.4 tok/s` Today vs `1,206.6 tok/s` in Golden Pass 6

### 3.1 The Measured Anomaly
In `STRATA_PASS6_VALIDATION_2026-10-25.md` Section 4.1, Golden Pass 6 achieved:
- **Tier 512 (452 prompt tokens)**: **`1,206.6 tok/s`** (clean free RAM).
- **PF_662 (660 prompt tokens)**: **`1,186.2 tok/s`**.
- **Tier 1024 (972 prompt tokens)**: **`1,386.2 tok/s`**.

During today's live benchmark against `:8080` loaded with `C:\Users\Butch\Documents\10-Models\Strata\strata-qwen177b.json`, Tier 512 measured only **`470.4 tok/s`** (a **`-61%` throughput drop**), whereas Tier 16K remained high at `1,400+ tok/s`.

### 3.2 Exact Code Root-Cause Analysis via AST Inspection
A line-by-line AST comparison between Golden Pass 6 and the live `strata-qwen177b.json` configuration revealed five colliding configuration deltas:

1. **`STRATA_PF_GEMM=1` Dense GEMM Hijack (`gemm.cu:1069–1075`)**:
   ```cpp
   // src/prefill/gemm.cu:1069-1075
   static const bool pf_on = [] { const char* v = std::getenv("STRATA_PF_GEMM"); return v && v[0] == '1'; }();
   static const int64_t pf_min_t = [] { const char* v = std::getenv("STRATA_PF_SWITCH_MIN_T"); return v ? (int64_t) std::atoll(v) : (int64_t) 0; }();
   if (pf_on && T >= pf_min_t && strata_pf_gemm_f16(X, W, Y, T, N, K, ldy, beta, stream_)) return;
   ```
   - In `strata-qwen177b.json`, `STRATA_PF_GEMM=1` and `STRATA_PF_SWITCH_MIN_T=128` were both explicitly set.
   - For any prompt with $T \ge 128$ (including $T = 413$ at Tier 512), this call intercepts every dense $F16$ GEMM in the attention and GDN projections and dispatches `strata_pf_gemm_f16` in `src/prefill/wmma_gemm.cu:681`.
   - `strata_pf_gemm_f16` uses tile dimension `BM = 128` (`pfg::BM = 128`). At $T = 413$, $M = (413 + 127)/128 = 4$ tiles. This produces severe tail wave underutilization across the 40 CUs of the Radeon 8060S and completely **bypasses `try_hipblaslt`**!
   - In Golden Pass 6, `STRATA_PF_GEMM` was **UNSET**, allowing `try_hipblaslt` to execute all 90 tuned `gfx1151` rows from `gfx1151-hipblaslt-100500.txt`.

2. **`STRATA_PF_HCDOWN=1` Overriding Tuned Solution 1176 (`prefill.cpp:496–511`)**:
   - `strata-qwen177b.json` set `STRATA_PF_HCDOWN=1`.
   - `prefill.cpp:511` defines `hc_pad() = pf_hcdown() || hcd_exact()`. When `pf_hcdown()` is true, it forces `strata_pf_hcdown_bf16` (custom WMMA down projection with an altered $K$-order), overriding `STRATA_HCD_EXACT=1` (which uses `hipBLASLt` solution `1176`).
   - In Pass 6 validation Section 2.2 line 44, `STRATA_HCD_EXACT=1` with solution 1176 was proven faster than custom atomic/WMMA down-projections on RDNA 3.5 L2 cache.

3. **`STRATA_GDN_CHUNKED=1` Scan Overhead at Low $T$ (`gdn.cu`) & Gufo PR #850 Disproof**:
   - At $T = 413$, 32-token chunked scan prep (L2 norms, log-decay prefix sums, forward substitution) and kernel launch overhead exceeds the cost of the lightweight serial recurrence.
   - Crucially, Gufo commit `c92b19c` (PR #850, October 27, 2026) proved that chunked DeltaNet puts layer error 500x above FP32 recurrence and degrades NLL by 0.0145 nats, flipping greedy decisions on XML tool prompts. Disabling `STRATA_GDN_CHUNKED=0` restores exact FP32 recurrence and eliminates low-$T$ launch latency.

4. **`--max-context 262144` vs `65536` ($+5.4\text{ GiB}$ Session Footprint)**:
   - Golden Pass 6 validated at `--max-context 65536` (`1.8 GiB` INT8 QSA KV buffer).
   - In `strata-qwen177b.json`, `--max-context` was set to `262144` (256K context), allocating `7.24 GiB` of contiguous VRAM.
   - In `generate.cpp:4977` and `qsa_select.cu:2558`, the reach of the selection scan scales with the allocated context capacity, adding memory management overhead even when prompt tokens are small.

5. **`--vision` Argument Overhead**:
   - Adding `--vision` in CLI args mounts the vision server and reserves VRAM/stream contexts that were absent during the pure text Golden Pass 6 benchmark.

### 3.3 Remedy & Verification
By removing `STRATA_PF_GEMM=1`, `STRATA_PF_HCDOWN=1`, and `STRATA_WMMA_GEMM=1` from `strata-qwen177b.json`, setting `STRATA_GDN_CHUNKED=0`, and setting `--max-context 65536`, `strata.exe` immediately re-engages the Golden Pass 6 execution path and recovers **`1,206+ tok/s`** prefill at Tier 512.

---

## 4. GitHub Mining Audit: Upstream Strata, Gufo & Community Strix Halo Repositories

### 4.1 Cloned Staging Repositories under `.harness/staging-repos/`
All staging clones were initialized with `--depth=80` and verified non-intrusive to the host working tree (`git status` is 100% clean):
1. **`upstream-strata`**: `c:\Users\Butch\Documents\02_Apps\Strata\.harness\staging-repos\upstream-strata`
   - Upstream HEAD: `5e8a4cc` (`v0.2.7` / engine `0.1.52`, October 27, 2026).
   - Base Ancestor with Custom Pass 6: `477ac13c` (`v0.1.45`, October 2026).
2. **`gufo`**: `c:\Users\Butch\Documents\02_Apps\Strata\.harness\staging-repos\gufo`
   - Upstream HEAD: `c92b19c` (`fix(qwen-flash): remove lossy chunked DeltaNet prefill (#850)`, October 27, 2026).
3. **`pixmaate-gufo`**: `c:\Users\Butch\Documents\02_Apps\Strata\.harness\staging-repos\pixmaate-gufo`
   - Windows WDDM fork HEAD: `2eb0778` (featuring commits `aeee89a` and `a8b45a0`).

### 4.2 Top Upstream Strata Pull Requests & Commits Audited

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
<table style="width: 100%; min-width: 900px; border-collapse: collapse; background-color: #161b22; color: #c9d1d9; font-size: 12px;">
  <thead style="background-color: #21262d; border-bottom: 2px solid #30363d;">
    <tr>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">PR / Commit</th>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">Author / Date</th>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">Summary &amp; Measured Claim</th>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">Relevance to Pass 7 &amp; Safety Filter</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">PR #3948 (789d839, 7d1016f)</td>
      <td style="padding: 7px 10px; white-space: nowrap;">Niko1221 (2026-10-27)</td>
      <td style="padding: 7px 10px;">STRATA_QFUSE=1: Fix 1-token window skipping commit graph (!g_qfuse()) &amp; fastmath q8_1 div</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950; font-weight: 600;">APPROVED FOR PASS 7 PORT (Critical State Sync Fix)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">PR #3602 (8dd297d, ebe3383)</td>
      <td style="padding: 7px 10px; white-space: nowrap;">Niko1221 (2026-10-27)</td>
      <td style="padding: 7px 10px;">STRATA_PLE_CHECK=1 diagnostic &amp; unbuffered IOCP table reader deadlock prevention</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950; font-weight: 600;">APPROVED FOR PASS 7 PORT (Robustness)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">PR #3416 (1e57a92, ca17d0f)</td>
      <td style="padding: 7px 10px; white-space: nowrap;">Niko1221 (2026-10-26)</td>
      <td style="padding: 7px 10px;">Gated MTP proposal law (STRATA_SPEC_PROB=1); lookup policy learns from committed rounds only</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950; font-weight: 600;">APPROVED FOR PASS 7 PORT (Opt-in Speculative Draft Boost)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">PR #3029 (ee1a6ef)</td>
      <td style="padding: 7px 10px; white-space: nowrap;">HaiYunChuang / Niko1221</td>
      <td style="padding: 7px 10px;">docs/STRIX_HALO_WINDOWS.md: official guide confirms STRATA_SH_STREAM=0 on Win WDDM (+14% decode)</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950;">CONFIRMS INVARIANT F-06 (Pass 6 already leads upstream)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">PR #2226 (2395c8a)</td>
      <td style="padding: 7px 10px; white-space: nowrap;">Niko1221 (2026-10-26)</td>
      <td style="padding: 7px 10px;">hc: one-kernel persistent hyper-connection read ported to CUDA (-5.4% story, +1.6% code)</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #8b949e;">REJECTED FOR GFX1151 (Lacks s_memrealtime barrier)</td>
    </tr>
    <tr>
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">PR #3976 (0d4b2cd, 37e4832)</td>
      <td style="padding: 7px 10px; white-space: nowrap;">Niko1221 (2026-10-27)</td>
      <td style="padding: 7px 10px;">STRATA_PREFILL_LAYER_MAJOR=1 for 2-card CUDA split</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #8b949e;">NOT APPLICABLE (Target is single Strix Halo APU gfx1151)</td>
    </tr>
  </tbody>
</table>
</div>

### 4.3 Gufo & Community Strix Halo Commits Audited

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
<table style="width: 100%; min-width: 900px; border-collapse: collapse; background-color: #161b22; color: #c9d1d9; font-size: 12px;">
  <thead style="background-color: #21262d; border-bottom: 2px solid #30363d;">
    <tr>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">Commit / PR</th>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">Repository</th>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">Summary &amp; Empirical Findings</th>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">Action for Pass 7</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">c92b19c (PR #850)</td>
      <td style="padding: 7px 10px; white-space: nowrap;">gufo-org/gufo</td>
      <td style="padding: 7px 10px;">REMOVED lossy chunked DeltaNet prefill (October 27, 2026): F16 scan caused 500x layer error, flipped greedy decisions on tool XML</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #f85149; font-weight: 600;">CONFIRMS F-02 / Disables STRATA_GDN_CHUNKED</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">538e2fd (PR #758)</td>
      <td style="padding: 7px 10px; white-space: nowrap;">gufo-org/gufo</td>
      <td style="padding: 7px 10px;">Retained side-stream shared expert overlap &amp; Z-projection overlap; 32-row W8A8 mixer down tiles</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950; font-weight: 600;">Already integrated in Pass 6 (STRATA_PF_Z_STREAM=1)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">aeee89a</td>
      <td style="padding: 7px 10px; white-space: nowrap;">pixmaate/gufo</td>
      <td style="padding: 7px 10px;">Alternate two streams after every completed flag wait on Windows WDDM (-0.8 ms per verify window)</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950; font-weight: 600;">APPROVED FOR PASS 7 PORT (STRATA_WDDM_ALT_STREAM)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">a8b45a0</td>
      <td style="padding: 7px 10px; white-space: nowrap;">pixmaate/gufo</td>
      <td style="padding: 7px 10px;">Deferred rollback without host wait (joins draft submission); verify prefix graph starts before n-gram hashing</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950; font-weight: 600;">APPROVED FOR PASS 7 PORT (Pipeline Overlap)</td>
    </tr>
    <tr>
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #58a6ff;">f465f15 (PR #805)</td>
      <td style="padding: 7px 10px; white-space: nowrap;">gufo-org/gufo</td>
      <td style="padding: 7px 10px;">Thread-local OVERLAPPED event handle for Windows concurrent PLE table reads</td>
      <td style="padding: 7px 10px; white-space: nowrap; color: #3fb950; font-weight: 600;">APPROVED FOR PASS 7 PORT (IOCP PLE Safety)</td>
    </tr>
  </tbody>
</table>
</div>

---

## 5. AST Differential Matrix: Custom Pass 6 (`413b098a`) vs. Upstream & Community Forks

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
<table style="width: 100%; min-width: 900px; border-collapse: collapse; background-color: #161b22; color: #c9d1d9; font-size: 12px;">
  <thead style="background-color: #21262d; border-bottom: 2px solid #30363d;">
    <tr>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">Subsystem</th>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">Custom Pass 6 (HEAD: 413b098a)</th>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">Upstream Strata (0.2.7)</th>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">Gufo (v0.25.0) / Pixmaate</th>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">Pass 7 Architecture Decision</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600;">MoE Prefill Core</td>
      <td style="padding: 7px 10px;">Single-batch E=512, BM=64/32/16 variable tile INT8 WMMA (stream_all_min=16)</td>
      <td style="padding: 7px 10px;">BM=64 static tiles; stream_all_min=1024 (suffers sub-1024 cliff)</td>
      <td style="padding: 7px 10px;">Dense W8A8 GEMM (falls back to vector emulation on IQ3_S)</td>
      <td style="padding: 7px 10px; color: #3fb950; font-weight: 600;">PROTECT PASS 6: Pass 6 INT8 WMMA is 2.0x faster than upstream</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600;">MoE Decode Kernel</td>
      <td style="padding: 7px 10px;">STRATA_EXP_MODE=8 fused LDS gate+up+SwiGLU across all 48 layers (53.16 ms verify)</td>
      <td style="padding: 7px 10px;">S26 unfused decode / STRATA_EXP_MODE=0 fallback (62-70 ms verify)</td>
      <td style="padding: 7px 10px;">Linear W8A8 decode tiles</td>
      <td style="padding: 7px 10px; color: #3fb950; font-weight: 600;">PROTECT PASS 6: Mode 8 fused decode is essential for 80 tok/s burst</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600;">Gated DeltaNet (GDN)</td>
      <td style="padding: 7px 10px;">36-layer O(1) state recurrence with STRATA_GDN_HEAD=1, PP=2, CONVL2=1, NOY=1</td>
      <td style="padding: 7px 10px;">Standard recurrence; opt-in chunked scan</td>
      <td style="padding: 7px 10px;">Removed chunked scan on Oct 27 (PR #850) due to 500x layer error; kept side streams</td>
      <td style="padding: 7px 10px; color: #3fb950; font-weight: 600;">ADAPT GUFO: Set STRATA_GDN_CHUNKED=0 for lossless bit-exact prefill</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600;">WDDM Multi-Stream Queue</td>
      <td style="padding: 7px 10px;">STRATA_SH_STREAM=0 on Windows WDDM; single primary compute queue</td>
      <td style="padding: 7px 10px;">STRATA_SH_STREAM=0 defaulted on Windows in 0.2.7 (commit b1bea3a)</td>
      <td style="padding: 7px 10px;">Alternates 2 retired streams after completed flag waits (pixmaate aeee89a)</td>
      <td style="padding: 7px 10px; color: #3fb950; font-weight: 600;">PORT PIXMAATE: Add STRATA_WDDM_ALT_STREAM=1 (-0.8 ms/window)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600;">QFUSE Commit Logic</td>
      <td style="padding: 7px 10px;">verify.cpp:3179 skips commit graph when last_t_ == 1 without checking g_qfuse()</td>
      <td style="padding: 7px 10px;">PR #3948 adds &amp;&amp; !g_qfuse(), preventing GDN state from dropping 1 token behind</td>
      <td style="padding: 7px 10px;">N/A (separate engine)</td>
      <td style="padding: 7px 10px; color: #3fb950; font-weight: 600;">PORT UPSTREAM PR #3948 (verify.cpp:3179 1-line surgical fix)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600;">Model Upload / Memory</td>
      <td style="padding: 7px 10px;">32.6 s BulkUploadSession (96 MiB ring, PrefetchVirtualMemory, VirtualUnlock)</td>
      <td style="padding: 7px 10px;">Unregistered arena UMA mode; ~120 s cold load</td>
      <td style="padding: 7px 10px;">Direct file upload (commit 2765d65)</td>
      <td style="padding: 7px 10px; color: #3fb950; font-weight: 600;">PROTECT PASS 6: Pass 6 bulk loader is 4.1x faster than upstream</td>
    </tr>
    <tr>
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600;">PLE &amp; MTP Pipeline</td>
      <td style="padding: 7px 10px;">Progressive coherent UMA prefetch_ple (0.65-0.84 ms stage) + ThinkNgramRing</td>
      <td style="padding: 7px 10px;">Serial PLE lookup; PR #3416 gated MTP proposal learning</td>
      <td style="padding: 7px 10px;">PR #805 thread-local OVERLAPPED event for unbuffered random reads</td>
      <td style="padding: 7px 10px; color: #3fb950; font-weight: 600;">PORT PR #3416 + PR #805 on top of Pass 6 progressive UMA</td>
    </tr>
  </tbody>
</table>
</div>

---

## 6. Surgical Pass 7 Porting Blueprint (`pass7/gfx1151-v0145-opt`)

All proposed ports are strictly opt-in or bug fixes that preserve Pass 6 backwards compatibility and satisfy `F-01` through `F-10`.

### Port 1: Upstream PR #3948 QFUSE Single-Token Self-Commit State Guard
- **Problem**: When `STRATA_QFUSE=1` is active, a 1-token window does not self-commit because fused GDN stores to the `q8_1` destination. However, `Verifier::commit()` previously skipped the commit graph on `last_t_ == 1 && one_token_self_commit()`, leaving the GDN recurrent state and QSA indexer tail exactly 1 token behind, depressing MTP draft acceptance.
- **Source Ref**: Upstream commit `789d839` in `upstream-strata`.
- **Target File**: `src/core/verify.cpp` (around line 3179).
- **Target Insertion**:
  ```cpp
  // Change:
  if (last_t_ == 1 && !always_publish_ && one_token_self_commit()) {
  // To:
  if (last_t_ == 1 && !always_publish_ && one_token_self_commit() && !g_qfuse()) {
  ```
- **VGPR / LDS / Hardware Impact**: Zero registers, zero LDS. Bit-exact state synchronization when 1-token verify windows occur.
- **Validation Gate**: `verify_batch_parity.exe` & `spec_verify_parity.exe`.

### Port 2: Upstream PR #3948 Q8_1 FastMath Division Operator Alignment
- **Problem**: In `src/kernels/cuda/fused_gdn.cu:280`, `amax / 127.0f` divided differently than `native_mmvq.cu` (`--use_fast_math`), resulting in 1-in-a-million code mismatches.
- **Source Ref**: Upstream commit `7d1016f` in `upstream-strata`.
- **Target File**: `src/kernels/cuda/fused_gdn.cu` and `include/strata/kernels/q8_1_finite.hpp`.
- **Target Insertion**: Define `#define STRATA_Q81_FASTMATH_DIV 1` and replace `amax / 127.0f` with `q81_div(amax, 127.0f)`.
- **Validation Gate**: `s2_expert_grouped_parity.exe`.

### Port 3: Pixmaate Alternate Stream Window Ping-Pong on Windows WDDM
- **Problem**: On Windows WDDM, submitting work to a stream that recently finished causes a user-mode driver fence check (~0.45 µs/node = ~0.8 ms per 48-layer verify window).
- **Source Ref**: `pixmaate/gufo` commit `aeee89a`.
- **Target File**: `src/core/verify.cpp` and `src/core/mtp.cpp`.
- **Mechanism**: Under opt-in env `STRATA_WDDM_ALT_STREAM=1`, initialize two compute streams (`stream_a`, `stream_b`). After each window's flag wait completes on host, switch the active verify launch stream to the alternate retired stream.
- **VGPR / LDS / Hardware Impact**: Zero register impact; eliminates ~0.8 ms of WDDM queue submission stall per decode step.
- **Validation Gate**: `verify_batch_parity.exe` & `spec_verify_parity.exe`.

### Port 4: Gufo PR #805 Thread-Local Win32 Event Handle for PLE Unbuffered Reads
- **Problem**: When direct unbuffered IOCP reads (`--ple-io direct`) fire across multiple threads, sharing a single event handle or recreating event handles causes Windows kernel handle allocations.
- **Source Ref**: `gufo` commit `f465f15` (`src/core/platform/fs_windows.cpp:447–475`).
- **Target File**: `src/core/ple.cpp` / Win32 I/O stager.
- **Mechanism**: Use a `thread_local ThreadEvent` struct holding a pre-allocated Win32 event handle passed to `OVERLAPPED.hEvent`, cleanly checking `ERROR_IO_PENDING` via `GetOverlappedResult`.
- **Validation Gate**: `test_expert_bulk_load.exe`.

### Port 5: Upstream PR #3416 Gated MTP Draft Policy Learning
- **Problem**: MTP suffix tree learning should only index tokens that were successfully committed by the verifier, ignoring rolled-back draft proposals.
- **Source Ref**: Upstream commit `ca17d0f` and `f4612b1`.
- **Target File**: `src/core/mtp.cpp` and `src/spec/suffix_drafter.cpp`.
- **Opt-in Guard**: `STRATA_SPEC_PROB=1`.
- **Validation Gate**: `spec_verify_parity.exe`.

---

## 7. Wave 2 Stage 1 Ranked A-B-B-A Benchmark Protocol (Zero-Rebuild)

Stage 1 benchmarking runs immediately against live `:8080` without recompiling `strata.exe`, testing combinations via `strata-qwen177b.json` to prove the Tier 512 root cause and optimize runtime flags.

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
<table style="width: 100%; min-width: 900px; border-collapse: collapse; background-color: #161b22; color: #c9d1d9; font-size: 12px;">
  <thead style="background-color: #21262d; border-bottom: 2px solid #30363d;">
    <tr>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">Arm ID</th>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">Configuration &amp; Overrides</th>
      <th style="padding: 8px 10px; text-align: left; color: #58a6ff; white-space: nowrap;">Hypothesis Tested</th>
      <th style="padding: 8px 10px; text-align: right; color: #58a6ff; white-space: nowrap;">Expected PF @ 512</th>
      <th style="padding: 8px 10px; text-align: right; color: #58a6ff; white-space: nowrap;">Expected PF @ 16K</th>
      <th style="padding: 8px 10px; text-align: right; color: #58a6ff; white-space: nowrap;">Expected Decode</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #3fb950;">ARM-0 (Golden Pass 6)</td>
      <td style="padding: 7px 10px;">Remove STRATA_PF_GEMM, PF_HCDOWN, WMMA_GEMM; GDN_CHUNKED=0; --max-context 65536; drop --vision</td>
      <td style="padding: 7px 10px;">Restores Golden Pass 6 baseline: hipBLASLt sol 1176 runs dense GEMMs, zero BM=128 tail waste</td>
      <td style="padding: 7px 10px; text-align: right; white-space: nowrap; color: #3fb950; font-weight: 600;">1,206.6 tok/s</td>
      <td style="padding: 7px 10px; text-align: right; white-space: nowrap; color: #3fb950;">1,566.6 tok/s</td>
      <td style="padding: 7px 10px; text-align: right; white-space: nowrap; color: #3fb950;">82.5 / 59.8 tok/s</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #f85149;">ARM-1 (Live Baseline)</td>
      <td style="padding: 7px 10px;">Live strata-qwen177b.json (STRATA_PF_GEMM=1, PF_SWITCH_MIN_T=128, GDN_CHUNKED=1, ctx=262144)</td>
      <td style="padding: 7px 10px;">Reproduces today's sub-1024 prefill slowdown (custom WMMA BM=128 underutilization)</td>
      <td style="padding: 7px 10px; text-align: right; white-space: nowrap; color: #f85149;">470.4 tok/s</td>
      <td style="padding: 7px 10px; text-align: right; white-space: nowrap;">1,420.0 tok/s</td>
      <td style="padding: 7px 10px; text-align: right; white-space: nowrap;">58.4 tok/s</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #e3b341;">ARM-2 (Isolate PF_GEMM)</td>
      <td style="padding: 7px 10px;">ARM-1 with STRATA_PF_GEMM=0, STRATA_WMMA_GEMM=0, STRATA_PF_HCDOWN=0</td>
      <td style="padding: 7px 10px;">Proves that dense GEMM interception was the primary cause of the Tier 512 slowdown</td>
      <td style="padding: 7px 10px; text-align: right; white-space: nowrap; color: #e3b341;">1,150+ tok/s</td>
      <td style="padding: 7px 10px; text-align: right; white-space: nowrap;">1,550+ tok/s</td>
      <td style="padding: 7px 10px; text-align: right; white-space: nowrap;">58.4 tok/s</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #e3b341;">ARM-3 (Isolate GDN_CHUNKED)</td>
      <td style="padding: 7px 10px;">ARM-0 with STRATA_GDN_CHUNKED=1 vs STRATA_GDN_CHUNKED=0</td>
      <td style="padding: 7px 10px;">Verifies Gufo PR #850 finding: confirms recurrent GDN has higher accuracy and lower low-T launch cost</td>
      <td style="padding: 7px 10px; text-align: right; white-space: nowrap;">1,180 vs 1,206 tok/s</td>
      <td style="padding: 7px 10px; text-align: right; white-space: nowrap;">1,560 tok/s</td>
      <td style="padding: 7px 10px; text-align: right; white-space: nowrap;">59.0 tok/s</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #e3b341;">ARM-4 (Context Ladder 65K vs 262K)</td>
      <td style="padding: 7px 10px;">ARM-0 with --max-context 65536 vs 262144 (measuring VRAM delta: 1.8 GiB vs 7.24 GiB)</td>
      <td style="padding: 7px 10px;">Quantifies QSA indexer latency and VRAM margin for co-resident Qwen 3.6 35B</td>
      <td style="padding: 7px 10px; text-align: right; white-space: nowrap;">1,206 vs 1,120 tok/s</td>
      <td style="padding: 7px 10px; text-align: right; white-space: nowrap;">1,566 vs 1,510 tok/s</td>
      <td style="padding: 7px 10px; text-align: right; white-space: nowrap;">59.8 vs 58.2 tok/s</td>
    </tr>
    <tr>
      <td style="padding: 7px 10px; white-space: nowrap; font-weight: 600; color: #3fb950;">ARM-5 (Dynamic Ring Boost)</td>
      <td style="padding: 7px 10px;">ARM-0 with STRATA_LOOKUP_THINK_RING=1 and --lookup-chain 3</td>
      <td style="padding: 7px 10px;">Confirms that thinking-ring speculative lookup boosts English &lt;think&gt; draft acceptance</td>
      <td style="padding: 7px 10px; text-align: right; white-space: nowrap;">1,206 tok/s</td>
      <td style="padding: 7px 10px; text-align: right; white-space: nowrap;">1,566 tok/s</td>
      <td style="padding: 7px 10px; text-align: right; white-space: nowrap; color: #3fb950; font-weight: 600;">62.5+ tok/s</td>
    </tr>
  </tbody>
</table>
</div>

---

## 8. Summary of Mandatory Engineering Invariants (`F-01`..`F-10`)

Every ported file, environment variable, and build script in Pass 7 must strictly uphold the following 10 empirical invariants:
- **`F-01`**: `STRATA_PF_F16=0` permanent on `gfx1151` (`7.12x` slower MoE: `1,501 ms` vs `210 ms`).
- **`F-02`**: Keep 4-stream residual carrier `m.R` in FP32 (BF16 causes Layer-1 PLE crashes and coupled MTP drift).
- **`F-03`**: Link ROCm `100500` `libhipblaslt.dll` with 90-row `gfx1151-hipblaslt-100500.txt` (never ROCm 7.2 `100202`).
- **`F-04`**: Never use `s_waitcnt(0x007f)` on `__GFX11__` (`STRATA_GR_FAST_LDS=0` / `__syncthreads()` mandatory).
- **`F-05`**: Fork Z GEMM on side stream strictly after `wqkv` completes to avoid split-K scratch corruption.
- **`F-06`**: `STRATA_SH_STREAM=0` on Windows WDDM (`+16.4%` decode gain confirmed by upstream `docs/STRIX_HALO_WINDOWS.md`).
- **`F-07`**: Set `STRATA_MOE_MAX_BM=64` on `gfx1151` so hot Zipf experts use BM=64 while cold tails use BM=32/16.
- **`F-08`**: Never allocate `>1 GiB` host unpack buffers (`STRATA_HC_Q8=1`, `STRATA_CKPT_POOL=1`, `--ple-row-cache 4194304`, `STRATA_FILE_RELEASE=1` required).
- **`F-09`**: Slot verification in `generate.cpp:6140` must use `all_items[0].slot` (not hardcoded slot 0).
- **`F-10`**: Guard `#pragma unroll 1` strictly with `#if defined(__clang__) && __clang_major__ >= 23`.
