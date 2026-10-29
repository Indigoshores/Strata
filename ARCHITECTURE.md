# ARCHITECTURE.md — Strata

Technical Architecture Blueprint for High-Performance MoE GPU Inference & AMD Strix Halo (`gfx1151`) Serving.
Standard: Aleksey Kladov (@matklad) Architecture Standard & OpenAI Harness Engineering.

---

## 1. Bird's Eye View

`Strata` is a high-throughput Mixture-of-Experts (MoE) GPU inference system engineered to run **Qwen 3.8 Flash-Next 177B** (and its Coder, Swift 1.5, and Unsloth `UD-IQ4_XS` / `UD-Q4_K_XL` variants) on single-workstation and consumer hardware—specifically optimized for AMD ROCm / HIP on the 256-bit unified memory architecture of **AMD Ryzen AI Max+ 395 ("Strix Halo", Radeon 8060S `gfx1151`)**, alongside NVIDIA CUDA and Intel oneAPI SYCL backends.

```
[Client / Antigravity IDE Request (OpenAI / Anthropic SSE)]
                           │
                           ▼
[Production Go Server (strata-server.exe :8080) / Python Reference (serve/)]
├── Chat Tokenizer & Tail-Role Checkpoint (--tail-role-token 8678)
├── Dynamic Thinking Effort (low | medium | high | xhigh, effort_position="end")
└── Dual-Stream SSE Emitter (delta.reasoning_content + delta.content)
                           │
                           ▼ (Native Engine Process / IPC)
[C++ / HIP / CUDA / SYCL Compute Engine (strata.exe — Pass 7)]
├── Prefill Tier (src/prefill/): Fused INT8 WMMA MoE + hipBLASLt 100500 + S23 WMMA GEMM
├── Hybrid Sequence Mixers: 36-Layer Gated DeltaNet (GDN) + 12-Layer QSA Sparse Attention
├── Decode Verify (src/core/verify.cpp): STRATA_EXP_MODE=8 Fused LDS MoE + WDDM Alt-Stream
├── Speculative Pipeline: 7-Token Coupled MTP (--spec 7, --mtp-q4 head) + ThinkNgramRing
└── Prefix Lookup Engine (PLE): 28.8 GiB IQ4_NL Direct NVMe IOCP + 4M RAM Row Cache
                           │
                           ▼
[AMD Strix Halo 128 GB Unified LPDDR5X-8533 Memory (96 GiB UMA VRAM + 31.65 GiB Host RAM)]
├── Port 8080: Qwen 3.8 Flash-Next 177B (24,576 / 24,576 Routed Experts Resident = 55.43 GiB)
└── Port 8081: Co-Resident Qwen 3.6 35B UD-IQ3_S (llama-server.exe = 15.52 GiB Commit)
```

---

## 2. Codemap (Coarse-Grained Module Map)

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
  <table style="width: 100%; min-width: 600px; border-collapse: collapse; background-color: #161b22; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 13px;">
    <thead>
      <tr style="background-color: #21262d; border-bottom: 1px solid #30363d;">
        <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Subsystem</th>
        <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Location / Path</th>
        <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Core Responsibilities &amp; Key Symbols</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">Core Runtime &amp; Verify</td>
        <td style="padding: 8px 12px; white-space: nowrap;"><code>src/core/</code></td>
        <td style="padding: 8px 12px;"><code>arch_defaults.cpp</code> (<code>gfx1151</code> switches), <code>expert_cache.cpp</code> (24,576-slot UMA cache + <code>BulkUploadSession</code>), <code>verify.cpp</code> (<code>Verifier::commit</code>, <code>!g_qfuse()</code> guard, <code>STRATA_WDDM_ALT_STREAM</code>), <code>mtp.cpp</code>, <code>ple.cpp</code>.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">Prompt Prefill Engine</td>
        <td style="padding: 8px 12px; white-space: nowrap;"><code>src/prefill/</code></td>
        <td style="padding: 8px 12px;"><code>prefill.cpp</code> (<code>Prefill::run</code>, <code>stream_all_min</code>), <code>moe_fused_iq.cu</code> (variable-tile <code>BM=64/32/16</code> INT8 WMMA MoE), <code>wmma_gemm.cu</code> (<code>strata_pf_gemm_f16</code>, <code>64x64_4w</code>), <code>gemm.cu</code> (<code>try_hipblaslt</code>).</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">GPU Compute Kernels</td>
        <td style="padding: 8px 12px; white-space: nowrap;"><code>src/kernels/cuda/</code></td>
        <td style="padding: 8px 12px;">Compiled for both CUDA and AMD HIP: <code>gdn.cu</code> / <code>fused_gdn.cu</code> (36-layer GDN), <code>qsa.cu</code> / <code>qsa_select.cu</code> (12-layer QSA), <code>iq_kernels.cu</code> (<code>STRATA_EXP_MODE=8</code> fused LDS decode MoE), <code>fused_gr.cu</code> (hyper-connections).</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">Platform &amp; Direct I/O</td>
        <td style="padding: 8px 12px; white-space: nowrap;"><code>src/platform/</code></td>
        <td style="padding: 8px 12px;"><code>memory.cpp</code> (DXGI 1.4 <code>QueryVideoMemoryInfo</code>, <code>VirtualLock</code>, <code>PrefetchVirtualMemory</code>, <code>VirtualUnlock</code>), <code>direct_file.cpp</code> (unbuffered Win32 IOCP NVMe reads).</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">Speculative Drafters</td>
        <td style="padding: 8px 12px; white-space: nowrap;"><code>src/spec/</code>, <code>src/ngram/</code></td>
        <td style="padding: 8px 12px;"><code>suffix_drafter.cpp</code> (<code>--lookup-chain 3</code>), <code>ngram.cpp</code> (<code>STRATA_LOOKUP_THINK_RING=1</code> zero-VRAM thinking ring).</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">Public Engine Headers</td>
        <td style="padding: 8px 12px; white-space: nowrap;"><code>include/strata/</code></td>
        <td style="padding: 8px 12px;"><code>core/verify.hpp</code>, <code>kernels/q8_1_finite.hpp</code> (<code>q81_div</code> fastmath helper), <code>prefill/kernels.hpp</code>, <code>prefill/moe_fused_iq.hpp</code>.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">Reference Python Server</td>
        <td style="padding: 8px 12px; white-space: nowrap;"><code>serve/</code></td>
        <td style="padding: 8px 12px;">Upstream Python ASGI server (<code>server.py</code>, <code>frontend.py</code>, <code>mcp.py</code>, <code>structured.py</code>).</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">Go Validation Suite</td>
        <td style="padding: 8px 12px; white-space: nowrap;"><code>docs-strix-go/</code></td>
        <td style="padding: 8px 12px;">Pure Go 1.26 CLI (<code>strata-tracker.exe</code>): <code>validator.go</code> (25-check fail-closed config &amp; log auditor), <code>benchmark.go</code> (zero-reuse SSE context ladder), <code>thinking_verify.go</code>.</td>
      </tr>
      <tr>
        <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">Documentation System</td>
        <td style="padding: 8px 12px; white-space: nowrap;"><code>docs/</code></td>
        <td style="padding: 8px 12px;">Anchored by <a href="docs/index.md">docs/index.md</a>: Pass 7 validation &amp; mining reports, validation framework spec, and 38 upstream hardware/kernel guides.</td>
      </tr>
    </tbody>
  </table>
</div>

---

## 3. Codenames & Domain Vocabulary

- **Strix Halo (`gfx1151`)**: AMD Ryzen AI Max+ 395 APU with 16 Zen 5 cores, 40 RDNA 3.5 compute units (`wave32`, hardware WMMA), and `128 GB` LPDDR5X-8533 unified memory (`96 GiB` BIOS UMA carve-out + `31.65 GiB` Windows 11 host RAM).
- **Disjoint Expert Cache (`R4 Hit Path`)**: Architecture holding all `24,576` routed experts (`55.43 GiB` for `UD-IQ4_XS`) resident in unified GPU memory on 128 GB Strix Halo (`zero-doorbell graph`), or splitting hot experts on GPU + cold experts on AVX-512 CPU threads on smaller-VRAM machines.
- **Hybrid 36 GDN + 12 QSA Topology**: Qwen 3.8 Flash-Next 177B's 48 layers alternate 36 **Gated Delta Net (GDN)** linear-recurrence layers ($\mathcal{O}(1)$ state per head) with 12 **Quantized Sparse Attention (QSA)** layers using WMMA block scoring (`STRATA_SELECT_WMMA=1`).
- **Prefix Lookup Engine (`PLE`)**: `28.8 GiB` `IQ4_NL` per-layer n-gram token embedding shard streamed from NVMe via unbuffered Win32 IOCP (`FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`, `--ple-io direct --ple-inflight 1024`) into a 4M-row RAM cache (`--ple-row-cache 4194304`).
- **Coupled Speculative Verification (`--spec 7`)**: Combines the Q4_0 Multi-Token Prediction draft head (`--mtp-q4 head`), suffix-tree prompt lookup (`--lookup-chain 3`), dynamic thinking n-gram ring (`STRATA_LOOKUP_THINK_RING=1`), and Gumbel-max exponential coupling (`STRATA_SPEC_GUMBEL=1`).
- **Tail-Role Checkpoint (`--tail-role-token 8678`)**: Places the conversation KV checkpoint immediately before the trailing `<|im_start|>system\nReasoning effort is set to ...` turn (`vocab.json["system"] = 8678`), preserving `95.5%` to `100%` warm KV prefix reuse when toggling `Low -> Medium -> XHigh` mid-conversation.

---

## 4. Architectural Invariants & Layering Boundaries

1. **Fail-Closed Prefill Switch Invariant**: In `strata-qwen177b.json`, `STRATA_PREFILL_STREAM_MIN=16` (`<= 128`), `STRATA_PF_FUSED=1`, `STRATA_PF_GEMM=1`, `STRATA_WMMA_GEMM=1`, `STRATA_PF_HCDOWN=1`, `STRATA_PF_PAD=1`, and `STRATA_PF_SWITCH_MIN_T=128` (`16..256`) are enforced fail-closed by `docs-strix-go/validator.go` and `prefill_switches_regression_test.go`.
2. **Negative Invariants on `gfx1151` (`F-01`..`F-10`)**:
   - **NEVER set `STRATA_PF_F16=1`** on `gfx1151` (`F-01`: causes a `7.12x` MoE prefill regression vs INT8 WMMA).
   - **NEVER set `STRATA_SH_STREAM=1` on Windows WDDM** (`F-06`: keep `0` or omitted; secondary stream sync costs `-16.4%` decode on WDDM).
   - **NEVER use `s_waitcnt(0x007f)` on `__GFX11__`** (`F-04`: `STRATA_GR_FAST_LDS=0` / `__syncthreads()` mandatory).
   - **NEVER omit `STRATA_FILE_RELEASE=1` or `STRATA_CKPT_POOL=1`** (`F-08`: releases `55.15 GiB` of GGUF standby pages via `VirtualUnlock` after GPU upload, keeping `>20 GiB` OS RAM free alongside Port `8081`).
   - **NEVER use `reasoning_effort="none"`** as a default or benchmark profile; default is `"medium"` with `effort_position="end"`.
3. **Windows ROCm DLL Precedence Invariant**: Local `amdhip64_7.dll`, `amd_comgr.dll`, and `hipblaslt.dll` (`v100500` paired with 90-row `gfx1151-hipblaslt-100500.txt`) in `engine/rocm/bin` must take precedence over `C:\Windows\System32`.

---

## 5. Cross-Cutting Concerns & Verification Gates

- **Dual-Server Shared UMA Coexistence**: Port `8080` (`strata.exe` serving Qwen 3.8 177B) and Port `8081` (`llama-server.exe` serving Qwen 3.6 35B `UD-IQ3_S`) run concurrently at `33%` OS RAM load (`20.91–21.15 GiB` free physical RAM, `16.4 GiB` free UMA VRAM).
- **Automated Verification Gates**:
  ```powershell
  # 1. Go Validator & Fail-Closed Switch Regression Suite
  go -C C:\Users\Butch\Documents\02_Apps\Strata\docs-strix-go test -v ./...

  # 2. Static Config + 1.84M-Line Engine Telemetry Audit
  .\docs-strix-go\strata-tracker.exe validate

  # 3. Live End-to-End Thinking & Port 8080/8081 Coexistence Gate
  .\docs-strix-go\strata-tracker.exe verify-thinking
  ```
