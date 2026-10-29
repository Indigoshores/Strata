# LLM.md — Strata

Operational Fast Router for AI Agents.
- **Architectural Blueprint**: [ARCHITECTURE.md](file:///c:/Users/Butch/Documents/02_Apps/Strata/ARCHITECTURE.md)
- **Documentation Index**: [docs/index.md](file:///c:/Users/Butch/Documents/02_Apps/Strata/docs/index.md)
- **Strix Halo & Go Toolchain Hub**: [docs-strix-go/README.md](file:///c:/Users/Butch/Documents/02_Apps/Strata/docs-strix-go/README.md)
- **Human Overview**: [README.md](file:///c:/Users/Butch/Documents/02_Apps/Strata/README.md)

---

## 1. PROJECT IDENTITY & ROLE
- **Type**: Multi-tier High-Performance MoE GPU Inference & Serving System.
- **Components**:
  1. **Native Compute Engine (`src/`, `include/`, `sycl/`)**: C++/HIP/CUDA/SYCL GPU kernel engine (`strata.exe`) for Qwen 3.8 Flash-Next 177B (`UD-IQ4_XS`).
  2. **Python Reference Server (`serve/`)**:Upstream OpenAI/Anthropic-compatible ASGI server and MCP gateway.
  3. **Strix Halo Go Toolchain (`docs-strix-go/`)**: Pure Go 1.26 validation, config auditing, A-B-B-A benchmarking, and upstream tracking CLI (`strata-tracker.exe`).
- **Canonical AI Model**: Gemini 3.8 Flash (`gemini-3.8-flash`).
- **Primary Target Hardware**: AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`, 40 RDNA 3.5 CUs, `128 GB` LPDDR5X-8533 Unified Memory: `96 GiB` BIOS UMA VRAM + `31.65 GiB` Host RAM).

---

## 2. COARSE-GRAINED CODEMAP
- **C++/HIP/CUDA/SYCL Engine (`src/`, `include/`, `sycl/`)**:
  - `src/core/`: Disjoint expert cache (`expert_cache.cpp`), verify window (`verify.cpp`), MTP drafter (`mtp.cpp`), architecture defaults (`arch_defaults.cpp`).
  - `src/prefill/`: Prefill scheduler (`prefill.cpp`), WMMA GEMM (`wmma_gemm.cu`), fused IQ MoE (`moe_fused_iq.cu`), hipBLASLt dispatch (`gemm.cu`).
  - `src/kernels/cuda/`: Gated DeltaNet (`gdn.cu`, `fused_gdn.cu`), Quantized Sparse Attention (`qsa.cu`, `qsa_select.cu`), hyper-connections (`fused_gr.cu`), decode MoE (`iq_kernels.cu`).
  - `include/strata/`: Internal C++ headers and `q8_1_finite.hpp` numeric helpers.
- **Python Reference Server (`serve/`)**:
  - `server.py`, `frontend.py`, `mcp.py`, `structured.py`, `responses.py`.
- **Strix Halo Go Toolchain (`docs-strix-go/`)**:
  - `main.go`, `validator.go`, `benchmark.go`, `thinking_verify.go`: Pure Go 1.26 validation CLI (`strata-tracker.exe`).
- **Production Runtime (`C:\Users\Butch\Documents\10-Models\Strata\`)**:
  - `strata-qwen177b.json`: Active Pass 9.1 Champion runtime configuration (`strata.exe` `a621538b...`, Port `8080`, `33,594 MiB` / `32.81 GiB` free UMA VRAM + `16.1–21.0 GiB` free OS RAM for Port `8081` `Qwen 3.6 35B` coexistence).

---

## 3. BUILD, TEST & VALIDATION COMMANDS
```powershell
# 1. Pure Go 1.26 Validation & Regression Tests
go -C C:\Users\Butch\Documents\02_Apps\Strata\docs-strix-go test -v ./...

# 2. Build & Run Full 5-Phase Strix Halo Validation Suite
go -C C:\Users\Butch\Documents\02_Apps\Strata\docs-strix-go build -o strata-tracker.exe .
.\docs-strix-go\strata-tracker.exe validate -out .\docs-strix-go\validation-report.json

# 3. Live Zero-Reuse Context Ladder SSE Benchmark (reasoning='none' is strictly forbidden)
.\docs-strix-go\strata-tracker.exe bench -url http://127.0.0.1:8080 -contexts 512,1024,2048,4096,8192,16384 -gen 256 -temp 0 -reasoning medium

# 4. Verify Thinking Alignment, KV Prefix Reuse & Port 8080/8081 Coexistence
.\docs-strix-go\strata-tracker.exe verify-thinking
```

---

## 4. HARD INVARIANTS (PASS 9.1 CHAMPION)
1. **Single-Pass FP16 WMMA Prefill Guards**: `STRATA_NO_TURN_SPLIT=1`, `STRATA_PF_F16=1`, `STRATA_PF_F16_MIN_T=0`, `STRATA_PF_MOE_SCATTER=0`, `STRATA_PF_DM_F16=1`, `STRATA_GDN_CHUNKED=1`, `STRATA_GDN_CHUNKED_MIN_T=128`, `STRATA_PREFILL_STREAM_MIN=16`, and `STRATA_PF_SWITCH_MIN_T=16` MUST remain active in `strata-qwen177b.json` (`39/39 PASS` on `audit-config`).
2. **Forbidden Switches on `gfx1151` Windows WDDM**: `STRATA_SH_STREAM=0` (`STRATA_PF_SH_STREAM=0`, `STRATA_PF_Z_STREAM=0`), `STRATA_PF_MOE_SCATTER=0` (`=1` overrides `Dm` FP16 and adds `+80.5 GB` DRAM traffic at 16K), and never allocate the `26.82 GiB` PLE table in VRAM (must stream via Direct NVMe I/O).
3. **Thinking & Coupled MTP Spec-7 Invariant**: Default `reasoning_effort="medium"`, `effort_position="end"`, `--tail-role-token 8678`, `--prefill 18432`, `--spec 7`, `--lookup-chain 3`, and `--mtp-q4 head`; `reasoning_effort="none"` is forbidden in benchmarks.
4. **Unified Memory Headroom & Coexistence**: Port `8080` (`strata.exe`, 177B, `24,576` GPU expert slots in `55.43 GiB` VRAM) leaves `33,594 MiB` (`32.81 GiB`) free UMA VRAM and `16.1–21.0 GiB` free OS RAM (`STRATA_FILE_RELEASE=1`, `2.02 GiB` working set) to co-reside with `Qwen 3.6 35B` on Port `8081`.
