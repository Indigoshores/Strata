# AMD Strix Halo (Ryzen AI Max+ 395 / gfx1151) Optimization Deep Dive

This document provides an exhaustive technical analysis of the hardware architecture, memory subsystem, execution switches, and kernel optimizations for **AMD Strix Halo** (`gfx1151`) within the [Niko1221/Strata](https://github.com/Niko1221/Strata.git) inference engine.

---

## 1. Hardware Architecture & System Profile

AMD Strix Halo represents a paradigm shift in local LLM inference by integrating a massive GPU onto a single unified memory bus with low-latency CPU access:

- **Host Processor**: AMD Ryzen AI Max+ 395 (16 Zen 5 physical cores, 32 execution threads, AVX-512 support).
- **Integrated GPU Accelerator**: AMD Radeon 8060S (40 RDNA 3.5 Compute Units, 2560 stream processors, target `gfx1151`, PCI Device ID `1002:1586`).
- **Memory Architecture**: 128 GB Unified LPDDR5X-8533 on a 256-bit wide bus delivering **273 GB/s** peak memory bandwidth shared dynamically between CPU and GPU.
- **Wavefront Size**: Native Wave32 execution mode.
- **Matrix Hardware**: Hardware WMMA (Wave Matrix Multiply Accumulate) matrix cores natively supporting INT4, INT8, BF16, and FP16 dot products.
- **Software Stack**: AMD ROCm 7.x (ROCm 7.14 / ROCm 10.2 nightly) with hipBLAS, rocBLAS, and hipBLASLt.

---

## 2. Unified Memory Management (UMA / GTT Subsystem)

Unlike discrete GPUs (e.g. RX 7900 XTX with dedicated GDDR6 VRAM over PCIe), Strix Halo shares one continuous pool of physical RAM:

### Linux GTT Architecture
- The Linux kernel exposes unified memory via the GTT (Graphics Translation Table) pool.
- With kernel boot parameters `ttm.pages_limit=29360128 ttm.page_pool_size=29360128`, up to **112 GiB** of RAM is mapped to the GPU, leaving the remaining memory for the OS.
- Strata's `device_free_bytes()` reads `MemAvailable` minus a headroom reserve (`STRATA_UMA_HEADROOM_GIB=6`, default 6 GiB) instead of conventional `MemFree`, preventing premature out-of-memory errors.

### Windows WDDM Budget Subsystem
- On Windows 10/11, the display driver model (WDDM 3.x) assigns a dedicated VRAM carve-out (typically 2–16 GB in BIOS) plus a shared memory budget (typically 50% to 85% of total system RAM).
- In `src/platform/memory.cpp`, Strata queries **DXGI 1.4** via `IDXGIAdapter3::QueryVideoMemoryInfo` with `DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL`.
- This ensures `--expert-cache auto` scales dynamically without exceeding the OS video budget, avoiding catastrophic page-thrashing where decode throughput drops from 54 tok/s to 30 tok/s.

### Zero-Copy Host Page Locking
- In `src/platform/memory.cpp`, resident weights are pinned using Win32 `SetProcessWorkingSetSizeEx` combined with `VirtualLock` in 1 GiB chunks:
  $$\text{WorkingSet}_{\text{min}} = \text{CurrentWS} + \text{ModelBytes} + 512\text{ MiB}$$
- This guarantees that Windows never pages expert blobs to disk during active inference rounds.

---

## 3. The 18 Architectural Defaults on gfx1151

When Strata initializes on GPU 0, `src/core/arch_defaults.cpp` automatically applies 18 specialized runtime environment switches if `gfx1151` is detected. 

> [!IMPORTANT]
> **Zero Divergence Guarantee**: Every single one of these 18 switches was verified to produce **byte-identical output tokens** (matching token IDs bit-for-bit against the unoptimized reference implementation).

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
  <table style="width: 100%; min-width: 600px; border-collapse: collapse; background-color: #161b22; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 14px;">
    <thead>
      <tr style="background-color: #21262d; border-bottom: 2px solid #30363d; text-align: left;">
        <th style="padding: 12px 16px; color: #58a6ff; white-space: nowrap;">Switch Name</th>
        <th style="padding: 12px 16px; color: #58a6ff; white-space: nowrap;">Default</th>
        <th style="padding: 12px 16px; color: #58a6ff; white-space: nowrap;">Pipeline Phase</th>
        <th style="padding: 12px 16px; color: #58a6ff; white-space: nowrap;">Performance Impact & Mechanism</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_GDN_HEAD</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #58a6ff;">Prompt</span></td>
        <td style="padding: 12px 16px;">Gated Delta Net recurrence: 4 SIMD lanes per column + grid-stride norm fusion.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_GDN_PP</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>2</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #58a6ff;">Prompt</span></td>
        <td style="padding: 12px 16px;">Two columns per lane in GDN recurrence; improves 64K prompt processing by <strong>+2.9%</strong>.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_GDN_CONVL2</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #58a6ff;">Prompt</span></td>
        <td style="padding: 12px 16px;">Fuses 1D short convolution and query/key L2 normalization into a single launch (-0.3s at 64K).</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_GDN_NOY</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #58a6ff;">Prompt</span></td>
        <td style="padding: 12px 16px;">Eliminates dead FP32 global memory store of normalization output.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_CVEC_FUSE</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #58a6ff;">Prompt</span></td>
        <td style="padding: 12px 16px;">Fuses steered layer write, control vector injection, and subsequent layer normalization in one pass.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_HCD_EXACT</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #58a6ff;">Prompt</span></td>
        <td style="padding: 12px 16px;">Hyper-connection down projection (N 320, K 10240) in a WMMA kernel matching hipBLASLt sol 1176/1177 (<strong>+1.8%</strong> at 64K).</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_PF_PAD</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #58a6ff;">Prompt</span></td>
        <td style="padding: 12px 16px;">Pads GEMM row strides to 128-byte cache line boundaries for optimal memory coalescing.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_Q8_PACKED</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #3fb950;">Decode</span></td>
        <td style="padding: 12px 16px;">Packed Q8_0 weights layout; accelerates decode throughput by <strong>+2.3%</strong>.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_Q6_PACKED</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #3fb950;">Decode</span></td>
        <td style="padding: 12px 16px;">Packed Q6_K heads layout for high-efficiency memory bandwidth utilization.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_MMVF_ROWS</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #3fb950;">Decode</span></td>
        <td style="padding: 12px 16px;">BF16 multi-row GEMV kernel computing 4 matrix rows per threadblock simultaneously.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_ATTN_LANECELL</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #3fb950;">Decode</span></td>
        <td style="padding: 12px 16px;">Attention score calculation mapped to 1 cell per thread for maximum register caching.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_EXPERT_V2</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #3fb950;">Decode</span></td>
        <td style="padding: 12px 16px;">Grouped MoE decode experts: IQ3_S gate/up projections paired with IQ4_NL down projections.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_TSUM</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #3fb950;">Decode</span></td>
        <td style="padding: 12px 16px;">Multiple warp reduction sums executed in a single transposed butterfly shuffle instruction.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_LFUSE</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #3fb950;">Decode</span></td>
        <td style="padding: 12px 16px;">Fuses kernel launches around the shared expert and the KV cache append operation.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_GDN_SPLIT</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #3fb950;">Decode</span></td>
        <td style="padding: 12px 16px;">Distributes the GDN state step across 4 threadblocks per attention head.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_QFUSE</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #3fb950;">Decode</span></td>
        <td style="padding: 12px 16px;">Activation Q8_1 images written directly into cache by their upstream producers.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_PLE_BATCH</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #3fb950;">Decode</span></td>
        <td style="padding: 12px 16px;">Batches the PLE key and value projections for an entire verification window at once.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_SH_STREAM</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>0</code> (Win) / <code>1</code> (Lin)</td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #3fb950;">Decode</span></td>
        <td style="padding: 12px 16px;">Runs shared expert on a secondary HIP stream on Linux (<strong>+1.8% to +6.7%</strong>). On Windows WDDM, defaults to <code>0</code> (`arch_defaults.cpp:58` / Invariant <code>F-06</code>) for <strong>+16.4%</strong> faster decode.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_TOPK_REG_MIN_BLOCKS</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>4096</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #58a6ff;">Prompt</span></td>
        <td style="padding: 12px 16px;">19th bit-exact default in <code>v0.1.45</code>: activates QSA register top-k from 16K cells (4,096 blocks) instead of 28K (3.90 ms -&gt; 1.00 ms at 24K).</td>
      </tr>
    </tbody>
  </table>
</div>

---

## 4. Opt-In Matrix Core (WMMA) & Pass 7 Production Switches

These switches leverage the hardware RDNA 3.5 WMMA matrix accelerators and Pass 6/7 custom kernels (`strata-qwen177b.json`). Because floating-point summation order differs from scalar accumulation, WMMA switches are opt-in and enforced fail-closed by `docs-strix-go/validator.go`:

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
  <table style="width: 100%; min-width: 600px; border-collapse: collapse; background-color: #161b22; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 14px;">
    <thead>
      <tr style="background-color: #21262d; border-bottom: 2px solid #30363d; text-align: left;">
        <th style="padding: 12px 16px; color: #58a6ff; white-space: nowrap;">Opt-In Switch</th>
        <th style="padding: 12px 16px; color: #58a6ff; white-space: nowrap;">Value</th>
        <th style="padding: 12px 16px; color: #58a6ff; white-space: nowrap;">Acceleration Domain</th>
        <th style="padding: 12px 16px; color: #58a6ff; white-space: nowrap;">Description &amp; Numeric Impact</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_PF_FUSED</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #e3b341;">Prompt MoE</span></td>
        <td style="padding: 12px 16px;">Dispatches prompt experts directly to INT8 WMMA matrix cores with variable tiles (<code>BM=64/32/16</code>, <code>STRATA_MOE_MAX_BM=64</code>). Note: <code>STRATA_PF_F16</code> must remain <code>0</code> (Invariant <code>F-01</code>).</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_PREFILL_STREAM_MIN</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>16</code> (<code>&lt;= 128</code>)</td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #3fb950;">Sub-1K Cliff Guard</span></td>
        <td style="padding: 12px 16px;">Eliminates the 1,024-token MMQ fallback cliff when all 24,576 experts are resident in UMA VRAM.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_PF_GEMM</code> / <code>STRATA_WMMA_GEMM</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #e3b341;">Prompt Projections</span></td>
        <td style="padding: 12px 16px;">Computes FP16 projections on <code>128x256</code> WMMA GEMM above <code>STRATA_PF_SWITCH_MIN_T</code> and <code>64x64_4w</code> WMMA below it.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_PF_HCDOWN</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #e3b341;">Hyper-Connection</span></td>
        <td style="padding: 12px 16px;">Pass 6 custom WMMA hyper-connection down projection on the prompt path.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_HC_UPMIX</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #e3b341;">Hyper-Connection</span></td>
        <td style="padding: 12px 16px;">Fuses hyper-connection up projection directly with its activation epilogue.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_PA_FAST</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #e3b341;">Attention</span></td>
        <td style="padding: 12px 16px;">Executes prompt attention using single FP16 query and probability buffers.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_HIP_WMMA</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #e3b341;">Attention</span></td>
        <td style="padding: 12px 16px;">Hardware matrix core execution of scaled dot-product attention tiles.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_SELECT_WMMA</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #e3b341;">MoE Routing</span></td>
        <td style="padding: 12px 16px;">Accelerates block scorer and top-10 expert candidate ranking using WMMA dot products.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_HC_Q8</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #e3b341;">GGUF Reader</span></td>
        <td style="padding: 12px 16px;">Reads hyper-connection projections directly from the GGUF's native Q8_0 weights.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_PF_SWITCH_MIN_T</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>128</code> (<code>16..256</code>)</td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #e3b341;">WMMA Crossover</span></td>
        <td style="padding: 12px 16px;">Production threshold (`128` in `strata-qwen177b.json`; upstream Linux docs use `4096`).</td>
      </tr>
      <tr>
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><code>STRATA_EXP_MODE=8</code> / <code>STRATA_WDDM_ALT_STREAM=1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>8</code> / <code>1</code></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #3fb950;">Pass 7 Decode</span></td>
        <td style="padding: 12px 16px;">All-48-layer fused LDS decode MoE + alternating WDDM verify commit stream (<code>78.9-86.2 tok/s</code> engine decode with <code>--spec 7</code>).</td>
      </tr>
    </tbody>
  </table>
</div>

---

## 5. hipBLASLt Calibration Tables

To avoid falling back to slow generic hipBLAS GEMM, Strata pre-calibrates the exact GEMM solution IDs for `gfx1151`.

### Table Location:
- `tools/hip/gfx1151-hipblaslt-100500.txt` (for ROCm 7.x / hipBLASLt 1.5.0)
- `tools/hip/gfx1151-hipblaslt-100401.txt` (for ROCm 7.14.1 / hipBLASLt 1.4.1)

### Enabling the Table:
```powershell
$env:STRATA_HIPBLASLT_TUNING = "C:\Users\Butch\Documents\02_Apps\Strata\tools\hip\gfx1151-hipblaslt-100500.txt"
```

When active, Strata confirms during startup:
```text
prefill gemm: hipBLASLt tuning enabled (90 rows, gfx1151, version 100500)
```

---

## 6. Windows-Specific Quirks & Production Rules

When running on Windows 11 with Strix Halo:

1. **Local DLL Precedence Rule**: Windows search order prioritizes the application directory over `C:\Windows\System32`. Certain AMD Adrenalin drivers install an incompatible `amdhip64_7.dll` in System32 that segfaults on prompt submission. **Always place `amdhip64_7.dll` and `amd_comgr.dll` directly in the same folder as `strata.exe`**.
2. **Stale HIP Error Reset**: In `src/prefill/gemm.cu`, Windows hipBLAS occasionally leaves `hipErrorInvalidValue` set after non-fatal BF16 GEMMs. The Windows engine explicitly clears this stale error flag with `hipGetLastError()` following each successful GEMM.
3. **Host Pointer Alias**: On Windows, `hipHostGetDevicePointer` returns the host virtual address directly. Direct device-to-device DMA into mapped host memory is bypassed in favor of synchronous pinned host copies.
