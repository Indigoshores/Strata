# Gufo AI Inference Engine: Architecture, Strix Halo Benchmarks & Strata Porting Analysis

**Document Purpose**: Definitive technical guide, architectural comparison, and upstream cross-reference between **Gufo** ([`gufo-org/gufo`](https://github.com/gufo-org/gufo), cloned at [`c:/Users/Butch/Documents/02_Apps/gufo`](file:///c:/Users/Butch/Documents/02_Apps/gufo)) and **Strata** ([`Niko1221/Strata`](https://github.com/Niko1221/Strata), located at [`c:/Users/Butch/Documents/02_Apps/Strata`](file:///c:/Users/Butch/Documents/02_Apps/Strata)).  
**Target Hardware**: AMD Ryzen AI Max+ 395 ("Strix Halo", Radeon 8060S `gfx1151`, 40 RDNA 3.5 CUs, up to `128 GiB` LPDDR5X-8533 Unified Memory).  
**Status**: UP-TO-DATE (Gufo Release `v0.25.0` / Strata Pass 7 `2026-10-28`).

---

## 1. Executive Summary & Overview

**Gufo** is an open-source, vertical local inference engine designed specifically for AMD Strix Halo (`gfx1151`) on Linux and Windows. While **Strata** focuses deeply on massive Mixture-of-Experts (MoE) LLM throughput—specifically **Qwen 3.8 Flash-Next 177B (`UD-IQ4_XS`)** via Disjoint Expert Caching, Prefix Lookup Engine (PLE), and WMMA kernels—**Gufo** positions itself as a comprehensive multi-modal engine covering text, vision, audio (ASR/TTS), and video generation pipelines on Strix Halo.

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
  <table style="width: 100%; min-width: 700px; border-collapse: collapse; background-color: #161b22; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 13px;">
    <thead>
      <tr style="background-color: #21262d; border-bottom: 2px solid #30363d;">
        <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Dimension</th>
        <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Gufo (gufo-org/gufo)</th>
        <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Strata (Niko1221/Strata Custom Pass 7)</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 9px 12px; font-weight: 600; white-space: nowrap;">Primary Objective</td>
        <td style="padding: 9px 12px;">Multi-modal vertical workstation engine (Text, Audio, Vision, Video)</td>
        <td style="padding: 9px 12px;">Extreme-throughput MoE LLM serving (Qwen 3.8 177B &amp; 35B co-resident)</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 9px 12px; font-weight: 600; white-space: nowrap;">Target Models</td>
        <td style="padding: 9px 12px;">Qwen 3.8 Flash-Next, Qwen 3.8 27B, DeepSeek V4 Flash, Qwen3-ASR/TTS, EmbeddingGemma-2, MiniMax H3</td>
        <td style="padding: 9px 12px;">Qwen 3.8 Flash-Next 177B (UD-IQ4_XS / UD-Q4_K_XL), Qwen 3.6 35B (co-resident on :8081)</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 9px 12px; font-weight: 600; white-space: nowrap;">Speculative Decoding</td>
        <td style="padding: 9px 12px;">MTP, DFlash2, DSpark</td>
        <td style="padding: 9px 12px;">7-token coupled MTP (<code>--spec 7</code>, <code>--mtp-q4 head</code>), Suffix Lookup-3, ThinkNgramRing</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 9px 12px; font-weight: 600; white-space: nowrap;">Toolchain (Windows)</td>
        <td style="padding: 9px 12px;">TheRock 10.0.0 SDK (<code>clang-cl</code>) + MSVC runtime + vcpkg + CMake</td>
        <td style="padding: 9px 12px;">ROCm 10.2 / <code>amdhip64_7.dll</code> + hipBLASLt 100500 + Pure Go 1.26 supervisor suite</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 9px 12px; font-weight: 600; white-space: nowrap;">DeltaNet (GDN) Policy</td>
        <td style="padding: 9px 12px; color: #3fb950; font-weight: 600;">PR #850 removed lossy chunked scan; uses exact FP32 recurrence</td>
        <td style="padding: 9px 12px; color: #3fb950; font-weight: 600;">Ported Gufo PR #850: <code>STRATA_GDN_CHUNKED=0</code> for lossless recurrence</td>
      </tr>
      <tr>
        <td style="padding: 9px 12px; font-weight: 600; white-space: nowrap;">WDDM Stream Strategy</td>
        <td style="padding: 9px 12px;">Alternating verify queue stream (<code>pixmaate aeee89a</code>)</td>
        <td style="padding: 9px 12px;">Ported to Pass 7: <code>STRATA_WDDM_ALT_STREAM=1</code> (-0.8 ms per verify window)</td>
      </tr>
    </tbody>
  </table>
</div>

---

## 2. Models Supported in Gufo (`docs/models/`)

Gufo maintains dedicated documentation, benchmark scripts, and quality contracts for 8 distinct model architectures:

1. **Qwen 3.8 Flash-Next** ([`docs/models/qwen3.8-flash-next/`](file:///c:/Users/Butch/Documents/02_Apps/gufo/docs/models/qwen3.8-flash-next/README.md)):
   - **Quantization**: Unsloth `UD-Q4_K_XL` + `MTP Q8_0`.
   - **Performance**: Up to `1,801.01 tok/s` prefill (Autoregressive); `116.93 tok/s` single-user decode and `245.81 aggregated tok/s` on 8 concurrent requests with MTP.
2. **Qwen 3.8 27B** ([`docs/models/qwen3.8-27b/`](file:///c:/Users/Butch/Documents/02_Apps/gufo/docs/models/qwen3.8-27b/README.md)):
   - **Quantization**: Unsloth `UD-Q4_K_XL` / `UD-Q8_K_XL` and `DFlash2 Q4_K_M`.
   - **Performance**: `648.89 tok/s` prefill; `71.55 tok/s` single-user decode and `128.42 aggregated tok/s` across 8 requests.
3. **DeepSeek V4 Flash** ([`docs/models/deepseek-v4-flash/`](file:///c:/Users/Butch/Documents/02_Apps/gufo/docs/models/deepseek-v4-flash/README.md)):
   - **Quantization**: antirez Flash 0731 `IQ2XXS` + `DSpark`.
   - **Performance**: `484.62 tok/s` prefill; `26.62 tok/s` single-user decode and `54.74 aggregated tok/s` across 8 requests.
4. **EmbeddingGemma 2** ([`docs/models/embeddinggemma-2/`](file:///c:/Users/Butch/Documents/02_Apps/gufo/docs/models/embeddinggemma-2/README.md)):
   - High-throughput text and image embeddings: up to `41.5k tokens/s` over HTTP; `13.0 ms` for a 404-token request.
5. **Qwen3-ASR 1.7B** ([`docs/models/qwen3-asr/`](file:///c:/Users/Butch/Documents/02_Apps/gufo/docs/models/qwen3-asr/README.md)):
   - Speech-to-text recognition running at `15.27x` realtime speed in pure BF16.
6. **Qwen3-TTS 1.7B** ([`docs/models/qwen3-tts/`](file:///c:/Users/Butch/Documents/02_Apps/gufo/docs/models/qwen3-tts/README.md)):
   - Speech synthesis and voice cloning (CustomVoice, VoiceDesign, Base) up to `2.54x` realtime; `201 ms` to first audio chunk.
7. **Qwen-Image-2.1** ([`docs/models/qwen-image-2.1/`](file:///c:/Users/Butch/Documents/02_Apps/gufo/docs/models/qwen-image-2.1/README.md)):
   - BF16 image generation and editing diffusion pipeline.
8. **MiniMax H3** ([`docs/models/minimax-h3/`](file:///c:/Users/Butch/Documents/02_Apps/gufo/docs/models/minimax-h3/README.md)):
   - BF16 text-to-video and audio pipeline (`FL2VA`).

---

## 3. Key Technological Innovations & Cross-Pollination with Strata

During the Pass 7 GitHub mining audit ([`docs/PASS7_GITHUB_MINING_REPORT.md`](file:///c:/Users/Butch/Documents/02_Apps/Strata/docs/PASS7_GITHUB_MINING_REPORT.md)), Strata evaluated and incorporated critical architectural fixes from Gufo and its community forks:

### 1. Removal of Lossy Chunked DeltaNet Prefill (Gufo PR #850)
- **Gufo Discovery**: On October 27, 2026, Gufo merged PR #850 (`commit c92b19c`). They demonstrated that attempting to run 32-token chunked DeltaNet scan on WMMA matrix cores puts layer numerical error **`500x` above exact FP32 recurrence**, inflating NLL by `+0.0145 nats` and causing greedy decision flips on structured XML tool-calling prompts.
- **Strata Adoption**: Strata validated this finding on AMD Strix Halo `gfx1151`, confirmed that `STRATA_GDN_CHUNKED=1` was responsible for a `-61%` prefill drop at short prompt lengths ($T = 413$), and permanently locked `STRATA_GDN_CHUNKED=0` for exact, lossless FP32 recurrence.

### 2. Windows WDDM Queue Ping-Pong (`pixmaate/gufo` commit `aeee89a`)
- **Mechanism**: Submitting subsequent compute work to a stream that recently finished on Windows WDDM triggers a kernel user-mode driver fence check (~0.45 µs/node = ~0.8 ms per 48-layer verify window). `pixmaate/gufo` introduced alternating between two pre-allocated verify streams (`stream_a` and `stream_b`).
- **Strata Adoption**: Integrated into Strata Pass 7 under opt-in environment flag `STRATA_WDDM_ALT_STREAM=1` in [`src/core/verify.cpp`](file:///c:/Users/Butch/Documents/02_Apps/Strata/src/core/verify.cpp), eliminating ~0.8 ms of WDDM driver submission stall per decode step.

### 3. Thread-Local OVERLAPPED Handles for Windows NVMe I/O (Gufo PR #805)
- **Mechanism**: When issuing unbuffered asynchronous NVMe reads across multiple threads, sharing a single event handle or repeatedly calling `CreateEventW` causes handle table contention and kernel transitions. Gufo PR #805 established a `thread_local` Win32 event handle pattern.
- **Strata Adoption**: Applied to Strata's Prefix Lookup Engine (PLE) in [`src/platform/direct_file.cpp`](file:///c:/Users/Butch/Documents/02_Apps/Strata/src/platform/direct_file.cpp) for the 28.8 GiB n-gram direct I/O path.

---

## 4. Building & Running Gufo on Windows (`gfx1151`)

To build and run Gufo natively on Windows from the local clone at [`c:/Users/Butch/Documents/02_Apps/gufo`](file:///c:/Users/Butch/Documents/02_Apps/gufo):

### Prerequisites
1. **TheRock 10.0.0 SDK for `gfx1151`**: `therock-dist-windows-gfx1151-10.0.0.tar.gz` extracted to `C:\TheRock\build` with `$env:HIP_PATH = "C:\TheRock\build"`.
   > [!WARNING]
   > Later SDK builds (10.1.0 through 10.2.0a nightlies) compile correct kernels that stall on `gfx1151`, causing decode to run hundreds of times slower. TheRock 10.0.0 is the validated toolchain for Gufo Windows.
2. **Visual Studio 2022** with Desktop C++ workload.
3. **vcpkg** with `$env:VCPKG_ROOT = "C:\vcpkg"`.

### Build Commands
```powershell
cd c:\Users\Butch\Documents\02_Apps\gufo

# 1. Patch SDK math headers (idempotent, MSVC 14.40+ conflict fix)
python tools\windows\patch_sdk_math.py $env:HIP_PATH

# 2. Configure and build release binaries via vs-env
tools\windows\vs-env.cmd cmake --preset windows-release
tools\windows\vs-env.cmd cmake --build --preset windows-release --parallel 8

# Output binaries:
# build\windows-release\gufo.exe
# build\windows-release\gufo-server.exe
```

---

## 5. Architectural Cross-Reference Matrix

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
  <table style="width: 100%; min-width: 600px; border-collapse: collapse; background-color: #161b22; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 13px;">
    <thead>
      <tr style="background-color: #21262d; border-bottom: 1px solid #30363d;">
        <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Capability</th>
        <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Gufo (c:/Users/Butch/Documents/02_Apps/gufo)</th>
        <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Strata (c:/Users/Butch/Documents/02_Apps/Strata)</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">Text MoE 177B Prefill</td>
        <td style="padding: 8px 12px;">Dense W8A8 GEMM (falls back to vector emulation on IQ3_S)</td>
        <td style="padding: 8px 12px; color: #3fb950; font-weight: 600;">1,356.7 tok/s (INT8 WMMA variable tile BM=64/32/16) — 2.0x faster</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">Text MoE 177B Decode</td>
        <td style="padding: 8px 12px;">Linear W8A8 decode tiles</td>
        <td style="padding: 8px 12px; color: #3fb950; font-weight: 600;">78.9–86.2 tok/s (STRATA_EXP_MODE=8 fused LDS decode MoE)</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">Multi-Modal Support</td>
        <td style="padding: 8px 12px; color: #3fb950; font-weight: 600;">Full (ASR, TTS, Diffusion Image Gen, Video)</td>
        <td style="padding: 8px 12px;">Vision server integration (<code>--vision</code> via <code>strata-vision.exe</code>)</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">Co-Resident Model Serving</td>
        <td style="padding: 8px 12px;">Single process multi-model execution</td>
        <td style="padding: 8px 12px; color: #3fb950; font-weight: 600;">Dual server in 128 GB UMA (Port 8080 177B + Port 8081 35B)</td>
      </tr>
      <tr>
        <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">Validation Suite</td>
        <td style="padding: 8px 12px;">Nix hosted checks + <code>check-pr</code> MSVC suite</td>
        <td style="padding: 8px 12px; color: #3fb950; font-weight: 600;">Pure Go 1.26 CLI (<code>strata-tracker.exe</code>: validate, audit, bench)</td>
      </tr>
    </tbody>
  </table>
</div>
