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
  - `strata-qwen177b.json`: Active Pass 7 production runtime configuration (Port `8080`, co-resident with `llama-server.exe` Qwen 3.6 35B on Port `8081`).

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

## 4. HARD INVARIANTS (PASS 7 PRODUCTION)
1. **Fail-Closed Prefill Guards**: `STRATA_PREFILL_STREAM_MIN=16` (`<= 128`), `STRATA_PF_FUSED=1`, `STRATA_PF_GEMM=1`, `STRATA_WMMA_GEMM=1`, `STRATA_PF_HCDOWN=1`, `STRATA_PF_PAD=1`, and `STRATA_PF_SWITCH_MIN_T=128` (`16..256`) MUST remain active in `strata-qwen177b.json`.
2. **Forbidden Switches on `gfx1151` Windows WDDM**: `STRATA_PF_F16=0` (or absent; `=1` causes a `7.12x` MoE slowdown), `STRATA_SH_STREAM=0` on Windows WDDM (`+16.4%` decode gain vs `1`), and `STRATA_HC_PERSIST=0` (inert on `gfx1151`).
3. **Thinking & Tail-Role Checkpoint Invariant**: Default `reasoning_effort="medium"`, `effort_position="end"`, `--tail-role-token 8678`, `--spec 7`, `--mtp-q4 head`, and `--coupled-draft`; `reasoning_effort="none"` is forbidden in benchmarks.
4. **Dual-Server UMA Coexistence**: Port `8080` (`strata.exe`, 177B, `24,576` GPU expert slots) and Port `8081` (`llama-server.exe`, Qwen 3.6 35B) co-reside in `128 GB` UMA with `STRATA_FILE_RELEASE=1` preserving `>20 GiB` free OS RAM.
