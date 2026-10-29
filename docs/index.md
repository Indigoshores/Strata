# Documentation Index — Strata

Clean System of Record for the Strata MoE GPU Inference Engine & AMD Strix Halo (`gfx1151`) Optimization Suite.
Standard: OpenAI Harness Engineering & Matklad `ARCHITECTURE.md` Progressive Disclosure.

---

## 1. Root Architecture & Operational Routers
- [LLM.md](../LLM.md) — Operational fast router ($\le 120$ lines) for AI agents: project identity, codemap, build/validation commands, and Pass 7 hard invariants.
- [ARCHITECTURE.md](../ARCHITECTURE.md) — Canonical Matklad system blueprint covering the Pass 7 C++/HIP engine, 36 GDN + 12 QSA hybrid architecture, 24,576-expert UMA cache, and negative invariants (`F-01`..`F-10`).
- [AGENTS.md](../AGENTS.md) — Upstream repository contributor guidelines and setup entrypoints.
- [README.md](../README.md) — Human-facing project overview, quickstart, and localized READMEs (`README.de.md`, `README.es.md`, `README.fr.md`, `README.ja.md`, `README.pt-BR.md`, `README.zh-CN.md`).
- [SECURITY.md](../SECURITY.md) — Security policy and vulnerability reporting (`127.0.0.1` binding invariant).

---

## 2. Qwen 3.8 Flash-Next 177B on AMD Strix Halo — Pass 9.1 Champion & Validation Dossiers
Read these documents in order for our custom Windows 11 AMD Strix Halo (`Ryzen AI Max+ 395 / Radeon 8060S gfx1151`, `128 GB` LPDDR5X) engineering work:

1. **[PASS9_1_STRATA_GUFO_SUMMARY.md](./PASS9_1_STRATA_GUFO_SUMMARY.md) (Authoritative Pass 9.1 Champion State — 2026-10-29)**:
   - Documents the active Pass 9.1 Champion (`strata.exe` `a621538b...`): single-pass FP16 WMMA MoE prefill (`743.4–1,496.7 tok/s`), coupled `--spec 7` MTP decode (`81.8–90.5 tok/s` client / `79.7–87.9 tok/s` engine), Gufo WDDM/IOCP adoptions, and `48.92 GiB` combined free Unified Memory (`32.81 GiB` UMA VRAM + `16.11 GiB` OS RAM).
2. **[STRATA_GUFO_PERFORMANCE_PATH_2026-10-29.md](./STRATA_GUFO_PERFORMANCE_PATH_2026-10-29.md) & [STRATA_GUFO_PREFILL_IMPLEMENTATION_PLAN_2026-10-29.md](./STRATA_GUFO_PREFILL_IMPLEMENTATION_PLAN_2026-10-29.md)**:
   - Architectural path and prefill kernel design analysis bridging Strata and Gufo on `gfx1151`.
3. **[GUFO.md](./GUFO.md) (Gufo AI Engine Architecture & Cross-Reference)**:
   - Comprehensive technical architecture, model coverage (Qwen 3.8 27B/Flash-Next, DeepSeek V4 Flash, Qwen3-ASR/TTS, EmbeddingGemma-2, MiniMax H3), Windows TheRock 10.0.0 build guide, and architectural cross-pollination with Strata (PR #850 lossy DeltaNet removal, Pixmaate WDDM alternating stream, and thread-local OVERLAPPED handles).
2. **[PASS7_FINAL_VALIDATION_REPORT.md](./PASS7_FINAL_VALIDATION_REPORT.md) (Authoritative Current State — 2026-10-28)**:
   - Complete Pass 7 production validation report (`branch pass7/gfx1151-v0145-opt`, `29 PASS / 0 WARN / 0 FAIL` on `audit-config`).
   - Documents `--tail-role-token 8678` (`95.5%` warm KV reuse across `Low -> Medium -> XHigh`), `--spec 7` coupled MTP decode (`78.9–86.2 tok/s` engine / `82.1–89.8 tok/s` client decode, `1,356.7 tok/s` GPU prefill), ported commits (`789d839`, `7d1016f`, `aeee89a`), and Win32 RAM/VRAM coexistence with Qwen 3.6 35B on Port `8081`.
3. **[PASS7_GITHUB_MINING_REPORT.md](./PASS7_GITHUB_MINING_REPORT.md) (AST Differential & Root-Cause Forensics — 2026-10-28)**:
   - Exhaustive AST audit across Custom Pass 6 (`413b098a`), upstream `Niko1221/Strata` (`v0.2.7` / `0.1.52`), `gufo-org/gufo` (`v0.25.0`), and `pixmaate/gufo`.
   - Catalogs all compiled runtime environment variables and defines the **10 Mandatory Engineering Invariants (`F-01`..`F-10`)**.
4. **[STAGE1_BENCHMARK_REPORT.md](./STAGE1_BENCHMARK_REPORT.md) (Wave 2 Stage 1 Runtime Sweep & Thinking Verification)**:
   - Documents the 4-phase `verify-thinking` suite (`GET/POST /settings`, default `"medium"` thinking, `10–11 ms` warm prefix TTFT, Port `8081` coexistence) and the `--spec 7` + `temperature: 1.0` vs `0.2` latency matrix.
5. **[VALIDATION_FRAMEWORK.md](./VALIDATION_FRAMEWORK.md) (5-Phase Pure Go 1.26 Validation & Benchmark Specification)**:
   - Operational manual for `docs-strix-go/strata-tracker.exe` (`validate`, `audit-config`, `audit-log`, `bench`, `ab-matrix`, `ctest-gate`, `restart`, `verify-thinking`).
6. **[QWEN38_177B_UPGRADE_AND_BENCHMARK_REPORT.md](./QWEN38_177B_UPGRADE_AND_BENCHMARK_REPORT.md) (Historical Baseline Dossier — 2026-10-18)**:
   - Initial `v0.1.40.2` to `v0.1.45` upgrade dossier, 4,839-request `strata_engine.log` telemetry audit, and discovery of the `STRATA_PREFILL_STREAM_MIN` sub-1024 prefill cliff.

---

## 3. Pure Go 1.26 Strix Halo Toolchain (`docs-strix-go/`)
- [docs-strix-go/README.md](../docs-strix-go/README.md) — Sitemap and CLI quickstart for `strata-tracker.exe`.
- [docs-strix-go/STRIX_HALO_OPTIMIZATIONS.md](../docs-strix-go/STRIX_HALO_OPTIMIZATIONS.md) — Deep dive into `gfx1151` UMA/WDDM memory management, the 18 bit-exact architectural defaults, opt-in WMMA switches, and `hipBLASLt 100500` tuning tables.
- [docs-strix-go/STRATA_KERNEL_ARCHITECTURE.md](../docs-strix-go/STRATA_KERNEL_ARCHITECTURE.md) — Algorithmic breakdown of Disjoint MoE Expert Caching, 36-layer GDN recurrence, 12-layer QSA sparse attention, Direct NVMe IOCP PLE, and coupled MTP speculative decoding.
- [docs-strix-go/GO_STRATA_DEPLOYMENT_BLUEPRINT.md](../docs-strix-go/GO_STRATA_DEPLOYMENT_BLUEPRINT.md) — Pure Go 1.26 supervisor, Win32 `VirtualAlloc`/`VirtualLock` memory pinning, IOCP NVMe streaming, and SSE server blueprint.
- [docs-strix-go/REPO_UPDATE_TRACKER.md](../docs-strix-go/REPO_UPDATE_TRACKER.md) — Standard Operating Procedure for fetching, diffing, and verifying upstream `Niko1221/Strata` updates (`upstream` remote).
- [docs-strix-go/UPSTREAM_SEARCH_CHEATSHEET.md](../docs-strix-go/UPSTREAM_SEARCH_CHEATSHEET.md) — Pre-compiled git and AST search queries by subsystem.

---

## 4. Upstream Engine & Hardware Reference Guides (`docs/`)
- **Core Engine & Setup**:
  - [DETAILS.md](./DETAILS.md) — Complete upstream engine reference, CLI flags, API endpoints, and measured benchmarks.
  - [HOW_IT_WORKS.md](./HOW_IT_WORKS.md) — Plain-language overview of Strata's architecture.
  - [AI_SETUP.md](./AI_SETUP.md), [INSTALL.md](./INSTALL.md), [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) — Installation, non-interactive AI setup, and diagnostics.
  - [MCP_SERVER.md](./MCP_SERVER.md), [LLAMA_SWAP.md](./LLAMA_SWAP.md), [IMAGES.md](./IMAGES.md), [LOGIT_BIAS.md](./LOGIT_BIAS.md) — MCP tools, model swapping, vision input, and sampling controls.
- **Hardware Backends**:
  - **AMD ROCm / HIP**: [AMD_HIP.md](./AMD_HIP.md), [AMD_HIP_PERFORMANCE.md](./AMD_HIP_PERFORMANCE.md), [STRIX_HALO.md](./STRIX_HALO.md), [RADEON_780M.md](./RADEON_780M.md), [benchmarks/windows-zen2-gfx1030-workers.md](./benchmarks/windows-zen2-gfx1030-workers.md).
  - **NVIDIA CUDA & Multi-GPU**: [MULTI_GPU.md](./MULTI_GPU.md), [SECOND_GPU.md](./SECOND_GPU.md), [NVIDIA_V100.md](./NVIDIA_V100.md), [OLDER_GPUS.md](./OLDER_GPUS.md), [NVFP4.md](./NVFP4.md).
  - **Intel SYCL & Apple Silicon**: [INTEL.md](./INTEL.md), [INTEL_ARC.md](./INTEL_ARC.md), [A770_SPEED_SPEC.md](./A770_SPEED_SPEC.md), [MAC.md](./MAC.md).
- **Models & Quantization Formats**:
  - [MODELS.md](./MODELS.md), [QUANTS.md](./QUANTS.md), [QWEN36.md](./QWEN36.md), [UNSLOTH_Q4.md](./UNSLOTH_Q4.md), [UNSLOTH_Q6.md](./UNSLOTH_Q6.md), [ORCA.md](./ORCA.md), [ORCA_Q4_K_S.md](./ORCA_Q4_K_S.md).
- **Memory, Caching & Kernel Subsystems**:
  - [DISJOINT_EXPERT_CACHE.md](./DISJOINT_EXPERT_CACHE.md), [EXCHANGE_ROTATION.md](./EXCHANGE_ROTATION.md), [VRAM_ELASTIC.md](./VRAM_ELASTIC.md), [BATCHED_DMA.md](./BATCHED_DMA.md), [BATCHING.md](./BATCHING.md).
  - [KV_PREFETCH.md](./KV_PREFETCH.md), [MESSAGE_BOUNDARY_CACHE.md](./MESSAGE_BOUNDARY_CACHE.md), [PROMPT_CACHE_TAIL.md](./PROMPT_CACHE_TAIL.md), [PREFILL_HEAD_ALIAS.md](./PREFILL_HEAD_ALIAS.md).
  - [EXPERTDEC.md](./EXPERTDEC.md), [MMVQ_IL_TABLE.md](./MMVQ_IL_TABLE.md), [QSAROWS.md](./QSAROWS.md).
- **Community Benchmarks & Testing**:
  - [COMMUNITY_BENCHMARKS.md](./COMMUNITY_BENCHMARKS.md), [RESEARCH_RUNS.md](./RESEARCH_RUNS.md), [TEST_REQUESTS.md](./TEST_REQUESTS.md).
