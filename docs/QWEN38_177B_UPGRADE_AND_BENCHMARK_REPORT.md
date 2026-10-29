# Qwen 3.8 Flash-Next 177B (Strata) — Architecture, Live Telemetry & Upstream Upgrade Dossier (`v0.1.40.2` to `v0.1.45`)

**Document Purpose**: Historical baseline dossier (`2026-10-18`) for secondary LLM research, kernel auditing, and pre-production A/B testing from `v0.1.40.2` to `v0.1.45`.
> [!IMPORTANT]
> **Superseded by Pass 7 Validation**: The upgrade and optimization work planned in this dossier was completed in Pass 6 and Pass 7 (`branch pass7/gfx1151-v0145-opt`). For current production settings (`--spec 7`, `--tail-role-token 8678`, `STRATA_PF_F16=0`, `STRATA_SH_STREAM=0`, `STRATA_WDDM_ALT_STREAM=1`) and measured benchmarks (`78.9–86.2 tok/s` engine decode, `82.1–89.8 tok/s` client decode, `1,356.7 tok/s` GPU prefill), consult **[PASS7_FINAL_VALIDATION_REPORT.md](./PASS7_FINAL_VALIDATION_REPORT.md)** and **[PASS7_GITHUB_MINING_REPORT.md](./PASS7_GITHUB_MINING_REPORT.md)**.

**Generated**: `2026-10-18`  
**Repositories Covered**:
- **Engine Repository**: [`c:/Users/Butch/Documents/02_Apps/Strata`](file:///c:/Users/Butch/Documents/02_Apps/Strata) (Upstream: [`https://github.com/Niko1221/Strata`](https://github.com/Niko1221/Strata))
- **Orchestrator & Benchmark Suite**: [`c:/Users/Butch/Documents/02_Apps/antigravity-add-model`](file:///c:/Users/Butch/Documents/02_Apps/antigravity-add-model)
- **Production Runtime Config**: [`C:/Users/Butch/Documents/10-Models/Strata/strata-qwen177b.json`](file:///C:/Users/Butch/Documents/10-Models/Strata/strata-qwen177b.json)
- **Live Engine Telemetry Log**: [`C:/Users/Butch/Documents/10-Models/Strata/strata_engine.log`](file:///C:/Users/Butch/Documents/10-Models/Strata/strata_engine.log)

---

## 1. Executive Summary & Current Baseline State

1. **Git State**:
   - **Local Checked-Out Commit**: `e8ca9afd` (`tag: v0.1.40.2` on branch `main`)
   - **Upstream Fetched Head**: `477ac13cbe8e582c20484871ce459d20d82df09d` (`tag: v0.1.45` on `origin/main`)
   - **Intermediate Releases Available in Local Git Object Store**: `v0.1.40.3`, `v0.1.40.4`, `v0.1.41`, `v0.1.42`, `v0.1.43`, `v0.1.43.1`, `v0.1.44`, `v0.1.44.1`, `v0.1.45` (fetched via `git fetch origin --tags`).
2. **Model Topology (`Qwen 3.8 Flash-Next 177B`, Unsloth `UD-IQ4_XS`)**:
   - **Layers**: 48 total layers (**36 Gated Delta Net (GDN)** linear-recurrence layers + **12 Quantized Sparse Attention (QSA)** layers).
   - **Mixture-of-Experts (MoE)**: 512 routed experts + 1 shared expert per layer (**24,576 routed experts total**, top-10 routed per token $\approx 6\text{B}$ active parameters per token).
   - **Quantization Format (`UD-IQ4_XS`, 93.60 GB total on disk)**:
     - Routed Experts (`55.43 GiB`): `IQ3_S` (`gate`/`up`, 1280 rows) + `IQ4_NL` (`down`, 640 cols).
     - Prefix Lookup Engine (`PLE`, Shard 2, `28.8 GiB`): `IQ4_NL` n-gram token embedding table (`per_layer_token_embd.weight`), accessed via unbuffered Win32 IOCP NVMe reads (`--ple-io direct --ple-inflight 1024`) with a 4M-entry RAM cache (`--ple-row-cache 4194304`).
     - KV Cache: `--kv int8` across `--max-context 262144` (256K context window).
3. **Verified Live Telemetry (`strata_engine.log`, 4,839 Generation Requests Audited)**:
   - **Expert Residency**: All **24,576 of 24,576 routed experts (`55.43 GiB`)** are 100% resident in GPU memory (`R4 hit path ON`, `zero-doorbell graph`, with `30.0 GiB` VRAM free).
   - **Sustained Decode Distribution (`gen >= 50 tokens`)**:
     - **2,858 of 4,839 requests (59.1%)** achieved **$\ge 50.0\text{ tok/s}$ decode**.
     - **836 of 4,839 requests (17.3%)** achieved **$\ge 60.0\text{ tok/s}$ decode**.
     - **Peak Decode**: **`79.7 tok/s`** (`L1461652`, `127/133` = **95.5% draft acceptance**).
     - **Cold 8K Coding Benchmark (`L201885`–`L202413`)**: **`1,292.7–1,324.7 tok/s` prefill** and **`62.0–62.4 tok/s` decode** (`82.6%–84.3%` draft acceptance).
     - **Ultra-Long Context Live Session (`176K–187K` context, `L1832721`–`L1836966`)**: **`57.7–67.8 tok/s` decode** (`73.1%–86.8%` draft acceptance).

---

## 2. How Strata Achieves Its Speedup on Qwen 3.8 Flash-Next 177B

### A. 3-Tier Disjoint Expert Cache & Simultaneous GPU + CPU Execution
Managed by [`ExpertCache`](file:///c:/Users/Butch/Documents/02_Apps/Strata/include/strata/core/expert_cache.hpp) and [`ExpertSource`](file:///c:/Users/Butch/Documents/02_Apps/Strata/include/strata/core/expert_source.hpp), with decode verification orchestrated in [`src/core/verify.cpp`](file:///c:/Users/Butch/Documents/02_Apps/Strata/src/core/verify.cpp):
- **Always-Hot GPU / GTT Tier**: Holds all dense weights (attention/GDN mixers, gated-residual hyper-connections, routers, shared experts, output head, MTP draft head), the INT8 KV cache, and the GPU expert cache (`--expert-cache auto`), seeded at startup from `data/expert-profile.bin`.
- **Locked Host RAM Tier**: Locks expert pages into physical memory in 1 GiB segments via Win32 `SetProcessWorkingSetSizeEx` + `VirtualLock` in [`src/platform/memory.cpp`](file:///c:/Users/Butch/Documents/02_Apps/Strata/src/platform/memory.cpp).
- **Zero-Wait GPU + CPU Split during Decode Verify**: When a verify window of $T=5\text{--}6$ tokens routes to top-10 experts per token:
  - Resident GPU experts execute via grouped multi-token kernels (`STRATA_EXPERT_V2=1` in [`src/kernels/cuda/moe_fused_iq.cu`](file:///c:/Users/Butch/Documents/02_Apps/Strata/src/kernels/cuda/moe_fused_iq.cu)) while the shared expert runs concurrently on a secondary stream (`STRATA_SH_STREAM=1`).
  - When experts spill to host RAM on lower-VRAM cards, the CPU pool ([`src/kernels/cpu/pool.cpp`](file:///c:/Users/Butch/Documents/02_Apps/Strata/src/kernels/cpu/pool.cpp)) computes non-resident experts concurrently using custom AVX-512 / AVX2 multi-token IQ kernels that dequantize each expert row once in vector registers across all tokens needing that expert.

### B. Hybrid 36-Layer Gated Delta Net (GDN) + 12-Layer Quantized Sparse Attention (QSA)
- **GDN Linear Recurrence (36 of 48 layers)**: Implemented in [`src/kernels/cuda/gdn.cu`](file:///c:/Users/Butch/Documents/02_Apps/Strata/src/kernels/cuda/gdn.cu) and [`src/kernels/cuda/fused_gdn.cu`](file:///c:/Users/Butch/Documents/02_Apps/Strata/src/kernels/cuda/fused_gdn.cu), GDN maintains a fixed-size recurrent state matrix $\mathbf{S}_t = \beta_t \mathbf{S}_{t-1} + \mathbf{k}_t \mathbf{v}_t^{\top}$ per head. Its memory footprint and per-token decode latency remain strictly $\mathcal{O}(1)$ from 512 tokens to 262,144 tokens.
- **QSA Sparse Block Selection (12 of 48 layers)**: Implemented in [`src/kernels/cuda/qsa.cu`](file:///c:/Users/Butch/Documents/02_Apps/Strata/src/kernels/cuda/qsa.cu) and [`src/prefill/qsa_prompt_attn.cu`](file:///c:/Users/Butch/Documents/02_Apps/Strata/src/prefill/qsa_prompt_attn.cu), QSA scores KV blocks using a wave32 WMMA indexer (`STRATA_SELECT_WMMA=1`), selects top-$k$ blocks in registers, and runs tensor-core attention only over selected blocks.

### C. Multi-Layer Speculative Decoding (`MTP` + `--lookup-chain 3` + Coupled Gumbel Sampling)
- **Native Qwen MTP Head (`--mtp`, `--mtp-q4 all`, `--spec 5`)**: Evaluated in [`src/core/mtp.cpp`](file:///c:/Users/Butch/Documents/02_Apps/Strata/src/core/mtp.cpp), proposing up to 5 draft tokens per verify step.
- **Suffix Tree Prompt Lookup (`--lookup-chain 3`)**: Implemented in [`src/spec/suffix_drafter.cpp`](file:///c:/Users/Butch/Documents/02_Apps/Strata/src/spec/suffix_drafter.cpp), matching prompt/output n-grams to draft multi-token chains with zero neural overhead during code editing and JSON/tool generation.
- **Gumbel-Max Coupled Drafts (`STRATA_SPEC_GUMBEL=1`, commit `6381df3`)**: Implemented in [`src/kernels/cuda/sampler.cu`](file:///c:/Users/Butch/Documents/02_Apps/Strata/src/kernels/cuda/sampler.cu), seeding exponential variates deterministically by `(seed, pos, token_id)` so draft and target models share identical random draws at temperature $\gt 0$.

### D. The Sub-1024 Prefill Cliff & `STRATA_PREFILL_STREAM_MIN=128` Fix
In [`src/prefill/prefill.cpp`](file:///c:/Users/Butch/Documents/02_Apps/Strata/src/prefill/prefill.cpp#L140-L143) and [`Prefill::run` (`src/prefill/prefill.cpp`)](file:///c:/Users/Butch/Documents/02_Apps/Strata/src/prefill/prefill.cpp#L2609-L2610):
```cpp
inline int64_t stream_all_min() {
    static const int64_t v = [] { const char* e = std::getenv("STRATA_PREFILL_STREAM_MIN"); return e ? (int64_t) std::atoll(e) : (int64_t) 1024; }();
    return v;
}
// In Prefill::run:
const bool fused_nat = (use_mmq || fused_only) && stream_all && no_peer && lay.native && fused::native_supported(mmq_gt, mmq_dt);
```
- **Root Cause Found During Live Benchmarking**: `fused_nat` (`STRATA_PF_FUSED=1` GPU-side expert grouping with zero host synchronization) requires `stream_all == true` (`T >= stream_all_min()`, which defaults to `1024`). Consequently, any prompt chunk with $T \lt 1,024$ tokens falls back to host-grouped MMQ (`cudaStreamSynchronize` on every MoE layer), running at only **`363.6–528.9 tok/s`**. As soon as $T \ge 1,024$ (`2048` tier), `STRATA_PF_FUSED=1` engages and prefill doubles to **`1,102.6–1,324.7 tok/s`**.
- **Applied Fix**: Added `"STRATA_PREFILL_STREAM_MIN": "128"` to [`strata-qwen177b.json`](file:///C:/Users/Butch/Documents/10-Models/Strata/strata-qwen177b.json#L26-L28), [`cmd/patcher/main.go`](file:///c:/Users/Butch/Documents/02_Apps/antigravity-add-model/cmd/patcher/main.go#L531-L533), and [`src/localModelService.ts`](file:///c:/Users/Butch/Documents/02_Apps/antigravity-add-model/src/localModelService.ts#L886-L935). Because 100% of experts are resident in GPU memory (`0` bytes streamed), lowering `STRATA_PREFILL_STREAM_MIN` to `128` enables GPU-grouped `STRATA_PF_FUSED=1` on all interactive agent turns of $128\text{--}1,023$ tokens.

---

## 3. Live Empirical Benchmarks & Telemetry (`2026-10-18`)

### A. Live Context Ladder (`cmd/bench` & `cmd/strata-server/benchmark`)

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
<table style="width: 100%; min-width: 600px; border-collapse: collapse; background-color: #161b22; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 13px;">
  <thead>
    <tr style="background-color: #21262d; border-bottom: 1px solid #30363d;">
      <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Context Tier</th>
      <th style="padding: 10px 12px; text-align: right; color: #58a6ff; white-space: nowrap;">Prompt Tokens</th>
      <th style="padding: 10px 12px; text-align: right; color: #58a6ff; white-space: nowrap;">Engine Prefill</th>
      <th style="padding: 10px 12px; text-align: right; color: #58a6ff; white-space: nowrap;">Old Pre-MTP Decode</th>
      <th style="padding: 10px 12px; text-align: right; color: #58a6ff; white-space: nowrap;">Live MTP Decode</th>
      <th style="padding: 10px 12px; text-align: right; color: #58a6ff; white-space: nowrap;">Drafts Accepted</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">512 (Short)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">484 tok</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap; color: #f85149;">363.6 tok/s (Pre-Fix MMQ)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">38.4 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap; color: #e3b341; font-weight: 600;">55.7 tok/s (+45.1%)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap; color: #3fb950;">53 / 80 (66.3%)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">1024 (1K)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">1,004 tok</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap; color: #f85149;">528.9 tok/s (Pre-Fix MMQ)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">39.9 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap; color: #3fb950; font-weight: 600;">52.3 tok/s (+31.1%)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">52 / 84 (61.9%)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">2048 (2K)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">2,053 tok</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap; color: #3fb950;">1,102.6 tok/s (Fused)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">49.8 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap; color: #3fb950; font-weight: 600;">49.1 tok/s (-1.4%)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">51 / 83 (61.4%)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">4096 (4K)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">4,048 tok</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap; color: #3fb950;">1,216.8 tok/s (Fused)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">37.9 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap; color: #3fb950; font-weight: 600;">52.9 tok/s (+39.6%)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap; color: #3fb950;">51 / 75 (68.0%)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">8192 (8K Structured)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">7,636 tok</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap; color: #e3b341; font-weight: 600;">1,324.7 tok/s (Fused)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">35.4 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap; color: #e3b341; font-weight: 600;">62.4 tok/s (+76.3%)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap; color: #3fb950;">209 / 248 (84.3%)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">16384 (16K)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">16,176 tok</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap; color: #3fb950;">1,322.1 tok/s (Fused)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">40.7 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap; color: #3fb950; font-weight: 600;">52.3 tok/s (+28.5%)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap; color: #3fb950;">52 / 78 (66.7%)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">186K (Live Agent Turn)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">186,073 tok</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">185,296 reused</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">43.2 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap; color: #e3b341; font-weight: 600;">67.8 tok/s (+56.9%)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap; color: #3fb950;">103 / 122 (84.4%)</td>
    </tr>
    <tr>
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">46.8K (Peak Decode)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">46,800 tok</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">46,368 reused</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">50.2 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap; color: #e3b341; font-weight: 600;">79.7 tok/s (+58.8%)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap; color: #3fb950;">127 / 133 (95.5%)</td>
    </tr>
  </tbody>
</table>
</div>

### B. Benchmark Harness Fixes Applied in `antigravity-add-model`
During live testing, [`cmd/bench/main.go`](file:///c:/Users/Butch/Documents/02_Apps/antigravity-add-model/cmd/bench/main.go) initially reported `Gen=0` (`0.0 t/s`) because `strata-server` defaults to `reasoning_effort="high"` (`generationThinking = true` in [`cmd/strata-server/api/server.go`](file:///c:/Users/Butch/Documents/02_Apps/antigravity-add-model/cmd/strata-server/api/server.go)) and emits `<think>` tokens inside `delta.reasoning_content` while `delta.content` remains empty during deliberation.
- **Fix 1**: Updated [`runSingleBenchmark` (`cmd/bench/main.go`)](file:///c:/Users/Butch/Documents/02_Apps/antigravity-add-model/cmd/bench/main.go#L260-L283) and [`main` (`cmd/strata-server/benchmark/main.go`)](file:///c:/Users/Butch/Documents/02_Apps/antigravity-add-model/cmd/strata-server/benchmark/main.go#L29-L148) to count both `delta.content` and `delta.reasoning_content` chunks, fall back to `usage.completion_tokens`, and expose a `-reasoning` flag (`high` | `none`).
- **Fix 2**: Updated the `baselines` map in [`runLadder` (`cmd/bench/main.go`)](file:///c:/Users/Butch/Documents/02_Apps/antigravity-add-model/cmd/bench/main.go#L147-L157) from the legacy pre-MTP values (`35.4–50.2 tok/s`) to the verified MTP + `--lookup-chain 3` baselines (`52.3–62.4 tok/s`).

---

## 4. Upstream GitHub Releases (`v0.1.40.2` to `v0.1.45`), Merge Branches & Pull Requests

### A. Upstream Release Tag Progression (`https://github.com/Niko1221/Strata`)
The local repository (`v0.1.40.2`, `e8ca9afd`) is 9 release tags behind `origin/main` (`v0.1.45`, `477ac13c`):
- `v0.1.40.2` (`e8ca9afd`) — Current local HEAD
- `v0.1.40.3` / `v0.1.40.4` — Early Strix Halo & RDNA3/4 point releases
- `v0.1.41` / `v0.1.42` / `v0.1.43` / `v0.1.43.1` — `STRATA_PF_F16=1`, `STRATA_TOPK_REG_MIN_BLOCKS=4096`, `STRATA_HC_PERSIST=1` A/B benchmarks
- `v0.1.44` / `v0.1.44.1` — Multi-GPU uneven layer-split auto (`#2644`), thinking budget `answer_room` (`d0e990ba`), `pick_slot` checkpoint routing (`#2518`)
- `v0.1.45` (`477ac13c`) — Latest upstream release integrating 10 performance PR merge branches:
  - `f-145-pp-rdna34` (`b277412e`) — AMD RDNA3 (`gfx1100`), RDNA3.5 Strix Halo (`gfx1151`), and RDNA4 (`gfx1201`) kernel group
  - `f-145-pp-cache` (`cbc6e3d8`) — Disjoint Expert Cache, `STRATA_EXCHANGE_ROTATE`, `STRATA_PCIE_BALANCE`, `STRATA_ADAPT_NOWAIT`
  - `f-145-pp-cpu` (`62b7d251`) — AVX2/AVX-512 singleton expert rows, register accumulators, `--lookup-chain` window cap fix (`ea08bd6e`)
  - `f-145-pp-serve` (`2f3a5d90`) — Serving layer & slot/checkpoint management
  - `f-145-pp-batch` (`e8b1edbc`) & `f-145-pp-batch-1441` (`da6d8d92`) — Continuous batching & verify window improvements
  - `f-145-pp-cuda86` (`5a6f4ed5`), `f-145-pp-sm75` (`df6262f2`), `f-145-pp-rdna2` (`21c57972`), `f-145-pp-sycl` (`926b0bbe`)

### B. High-Impact Merged PRs & Commits in `v0.1.45` (`origin/main`)

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
<table style="width: 100%; min-width: 600px; border-collapse: collapse; background-color: #161b22; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 13px;">
  <thead>
    <tr style="background-color: #21262d; border-bottom: 1px solid #30363d;">
      <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">PR / Commit SHA</th>
      <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Switch / Subsystem</th>
      <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Target Arch</th>
      <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Measured Speedup / Impact</th>
      <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Precision / Safety</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600; color: #e3b341;">PR #2018 (edf61f18)</td>
      <td style="padding: 8px 12px; white-space: nowrap;">VGPR-capped fused IQ experts</td>
      <td style="padding: 8px 12px; white-space: nowrap;">gfx1151 (Clang 23)</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #3fb950;">-10% to -15% layer time on IQ3_S / IQ4_NL / IQ4_XS</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #3fb950;">Bit-exact (Auto under PF_FUSED=1)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600; color: #e3b341;">bd5cc1f2 / feeb8dc8</td>
      <td style="padding: 8px 12px; white-space: nowrap;">STRATA_PF_F16=1</td>
      <td style="padding: 8px 12px; white-space: nowrap;">gfx11 WMMA</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #3fb950;">UD-IQ4_XS gate/up -15.3%, down -22.9% prefill</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #e3b341;">Opt-in FP16 WMMA (KL 0.0005 nats)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">PR #2299 (1b1bd4b4)</td>
      <td style="padding: 8px 12px; white-space: nowrap;">STRATA_GDN_CHUNKED=1</td>
      <td style="padding: 8px 12px; white-space: nowrap;">gfx11 WMMA</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #3fb950;">1.6x faster GDN recurrence (16-tok WY chunks)</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #e3b341;">Opt-in FP16 WMMA (+1.1% to +2.5% E2E)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">PR #2016 (4cf36daf)</td>
      <td style="padding: 8px 12px; white-space: nowrap;">STRATA_TOPK_REG_MIN_BLOCKS=4096</td>
      <td style="padding: 8px 12px; white-space: nowrap;">gfx1151 Default</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #3fb950;">QSA top-k: 3.90 ms -&gt; 1.00 ms per 2K queries at 24K</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #3fb950;">Bit-exact (19th default in arch_defaults)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">PR #1749 (82922f55)</td>
      <td style="padding: 8px 12px; white-space: nowrap;">Wide HIP Top-K kernel</td>
      <td style="padding: 8px 12px; white-space: nowrap;">All AMD HIP</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #3fb950;">22.0 ms -&gt; 0.93 ms at 524K (+8.7% at 131K, +34% at 500K)</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #3fb950;">Bit-exact (Replaces fallback radix sort)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">PR #2017 (acc94b21)</td>
      <td style="padding: 8px 12px; white-space: nowrap;">WMMA prompt attn read prefetch</td>
      <td style="padding: 8px 12px; white-space: nowrap;">gfx11 / gfx12</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #3fb950;">Issues chunk KV reads before WMMA compute loop</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #3fb950;">Bit-exact (Always on with HIP_WMMA=1)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">PR #2622 (36381540)</td>
      <td style="padding: 8px 12px; white-space: nowrap;">PLE gather_batch key reuse</td>
      <td style="padding: 8px 12px; white-space: nowrap;">All Backends</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #3fb950;">Deduplicates prefetched NVMe PLE rows in verify window</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #3fb950;">Bit-exact (Default ON)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">Commit ea08bd6e</td>
      <td style="padding: 8px 12px; white-space: nowrap;">--lookup-chain window cap fix</td>
      <td style="padding: 8px 12px; white-space: nowrap;">All Backends</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #3fb950;">Sizes lookup chain from active spec cap instead of stale mtp_max_t</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #3fb950;">Bit-exact (Default ON)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">PR #2226 (6cbc15a8)</td>
      <td style="padding: 8px 12px; white-space: nowrap;">STRATA_HC_PERSIST=1</td>
      <td style="padding: 8px 12px; white-space: nowrap;">AMD HIP</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #3fb950;">-6.6% / -4.4% ms per verify window at 4K / 32K</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #e3b341;">Opt-in persistent kernel (bounded 1e-4)</td>
    </tr>
    <tr>
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">Commit d0e990ba</td>
      <td style="padding: 8px 12px; white-space: nowrap;">answer_room thinking budget</td>
      <td style="padding: 8px 12px; white-space: nowrap;">Serve API</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #3fb950;">Closes &lt;think&gt; early to reserve 25% of max_tokens for answer</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #3fb950;">Prevents empty content on capped requests</td>
    </tr>
  </tbody>
</table>
</div>

### C. Open Pull Requests & Community Forks to Evaluate

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
<table style="width: 100%; min-width: 600px; border-collapse: collapse; background-color: #161b22; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 13px;">
  <thead>
    <tr style="background-color: #21262d; border-bottom: 1px solid #30363d;">
      <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">PR / Fork</th>
      <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Branch / Ref</th>
      <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Key Optimization</th>
      <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Measured Result</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600; color: #e3b341;">Open PR #2997 (bsorensen110)</td>
      <td style="padding: 8px 12px; white-space: nowrap;">perf/rdna4-d1-hc-q8-decode (d09a7b37)</td>
      <td style="padding: 8px 12px; white-space: nowrap;">STRATA_GR_TSUM=1 + gr_sync_lds() + decode HC_Q8=1</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #3fb950;">-16.2% GR read latency at T=6; VGPRs 109-&gt;95 (2 CTAs/WGP)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">Open PR #2996 (LostDakota)</td>
      <td style="padding: 8px 12px; white-space: nowrap;">serve-report-dropped-tool-calls (506bdb0c)</td>
      <td style="padding: 8px 12px; white-space: nowrap;">report_dropped_tool_calls: true in server config</td>
      <td style="padding: 8px 12px; white-space: nowrap;">Reports truncated tool call names &amp; char counts (#211)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">Fork: libratechw/Strata</td>
      <td style="padding: 8px 12px; white-space: nowrap;">Upstream gfx1151 contributor</td>
      <td style="padding: 8px 12px; white-space: nowrap;">Strix Halo WMMA, GDN chunked WY (#2299), Clang 23 IQ (#2018)</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #3fb950;">Primary source of all gfx1151 kernel speedups in v0.1.45</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">Fork: bsorensen110/Strata</td>
      <td style="padding: 8px 12px; white-space: nowrap;">perf/rdna4-* branches</td>
      <td style="padding: 8px 12px; white-space: nowrap;">LDS-only barriers (s_waitcnt lgkmcnt(0); s_barrier), HC_PERSIST</td>
      <td style="padding: 8px 12px; white-space: nowrap; color: #3fb950;">Eliminates 1,284 L0 cache invalidations per layer on HIP</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">Fork: eddoursul/Strata</td>
      <td style="padding: 8px 12px; white-space: nowrap;">PR #245 / Unsloth packs</td>
      <td style="padding: 8px 12px; white-space: nowrap;">Split-artifact loader &amp; mixed-format Unsloth Q4/IQ4 kernels</td>
      <td style="padding: 8px 12px; white-space: nowrap;">Enables UD-IQ4_XS and UD-Q4_K_XL native execution</td>
    </tr>
    <tr>
      <td style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">Fork: MistVVK/XeStrata</td>
      <td style="padding: 8px 12px; white-space: nowrap;">209-star multi-arch fork</td>
      <td style="padding: 8px 12px; white-space: nowrap;">PcieAuto in-window share refinement (ported in b0e88eb1)</td>
      <td style="padding: 8px 12px; white-space: nowrap;">Dynamic PCIe/CPU expert split stabilization</td>
    </tr>
  </tbody>
</table>
</div>

#### Deep-Dive on Open PR `#2997` (`bsorensen110:perf/rdna4-d1-hc-q8-decode`)
For another LLM evaluating whether to cherry-pick PR `#2997` (`https://github.com/Niko1221/Strata/pull/2997`, commit `d09a7b37f4ce5fe4d9d1ec80b1f6a1000813ef38`) on top of `v0.1.45`:
1. **Why It Matters for `STRATA_HC_Q8=1`**: Your production [`strata-qwen177b.json`](file:///C:/Users/Butch/Documents/10-Models/Strata/strata-qwen177b.json#L25) already sets `"STRATA_HC_Q8": "1"`. In `v0.1.40.2`, `gr_up_q8_q32_kernel` and `gr_down_q8_fast_kernel` in `src/kernels/cuda/fused_gr.cu` use legacy `warp_sum` butterfly reductions and `__syncthreads()` inside the verify loop (`T = 1..6`).
2. **What PR `#2997` Changes**:
   - Replaces `warp_sum` with `s26ts::tsum` (`v_permlane16_swap_b32` + DPP row butterfly across power-of-two `P = pow2_ceil(T)` tokens) and preloads `R[i]`, `w_norm[i]`, `bo_prev`, and `inj_prev` before the `lo[k]` dot loop.
   - Replaces `__syncthreads()` with `gr_sync_lds()` (`s_waitcnt lgkmcnt(0); s_barrier` on `gfx11` and `s_wait_dscnt 0x0; s_barrier_signal -1; s_barrier_wait -1` on `gfx12`), avoiding `960 + 324 = 1,284` Shader Engine L0 vector cache invalidations (`global_inv scope:SCOPE_SE`) per layer.
   - Drops VGPR count in `gr_up_q8_q32_kernel` from `53/69/78/89/99/109` (`T=1..6`) to **`45/59/68/77/86/95 VGPRs`**, keeping `T=5, 6` (your exact `--spec 5` window size!) under the 96-VGPR boundary required for **2 CTAs/WGP** (16 waves/SIMD instead of 8).

---

## 5. Production Configuration (`strata-qwen177b.json`) — Current vs Candidate `v0.1.45`

### A. Active Production Configuration (Updated `2026-10-18` with `STRATA_PREFILL_STREAM_MIN=128`)
File: [`C:/Users/Butch/Documents/10-Models/Strata/strata-qwen177b.json`](file:///C:/Users/Butch/Documents/10-Models/Strata/strata-qwen177b.json)

```json
{
  "exe": "C:\\Users\\Butch\\Documents\\10-Models\\Strata\\engine\\strata.exe",
  "cwd": "C:\\Users\\Butch\\Documents\\10-Models\\Strata",
  "model_name": "Qwen 3.8 Flash-Next 177B (Strata)",
  "backend": "hip",
  "port": 8080,
  "host": "127.0.0.1",
  "open_browser": false,
  "sampling": {
    "temperature": 0.2,
    "top_p": 0.95
  },
  "tokenizer": "C:\\Users\\Butch\\Documents\\10-Models\\Strata\\data\\packs\\unsloth-ud-iq4_xs\\tokenizer",
  "lib_dirs": [
    "C:\\Users\\Butch\\Documents\\10-Models\\Strata\\engine\\rocm\\bin"
  ],
  "env": {
    "STRATA_HIPBLASLT_TUNING": "C:\\Users\\Butch\\Documents\\10-Models\\Strata\\tools\\hip\\gfx1151-hipblaslt-100500.txt",
    "STRATA_PF_FUSED": "1",
    "STRATA_PF_GEMM": "1",
    "STRATA_HC_UPMIX": "1",
    "STRATA_PA_FAST": "1",
    "STRATA_HIP_WMMA": "1",
    "STRATA_SELECT_WMMA": "1",
    "STRATA_HC_Q8": "1",
    "STRATA_PF_SWITCH_MIN_T": "256",
    "STRATA_PREFILL_STREAM_MIN": "128",
    "STRATA_SH_STREAM": "1",
    "STRATA_PF_HCDOWN": "1",
    "STRATA_WMMA_GEMM": "1"
  },
  "args": [
    "--pack", "C:\\Users\\Butch\\Documents\\10-Models\\Strata\\data\\packs\\unsloth-ud-iq4_xs",
    "--native", "C:\\Users\\Butch\\Documents\\10-Models\\models\\unsloth-UD-IQ4_XS\\Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf",
    "--ple-gguf", "C:\\Users\\Butch\\Documents\\10-Models\\models\\unsloth-UD-IQ4_XS\\Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf",
    "--expert-profile", "C:\\Users\\Butch\\Documents\\10-Models\\Strata\\data\\expert-profile.bin",
    "--expert-cache", "auto",
    "--no-prefill-borrow",
    "--mmap-experts",
    "--ple-io", "direct",
    "--ple-inflight", "1024",
    "--ple-row-cache", "4194304",
    "--prefill", "16384",
    "--max-context", "262144",
    "--kv", "int8",
    "--spec", "5",
    "--spec-min-p", "0.05",
    "--lookup-chain", "3",
    "--mtp", "C:\\Users\\Butch\\Documents\\10-Models\\Strata\\data\\mtp\\rt",
    "--mtp-q4", "all"
  ]
}
```

### B. Candidate `v0.1.45` Environment Additions for Pre-Production A/B Testing
Once `strata.exe` is rebuilt from `v0.1.45` (`477ac13c`), test adding these four opt-in variables one at a time in `strata-qwen177b.json`:
1. `"STRATA_PF_F16": "1"` — Runs `UD-IQ4_XS` (`IQ3_S` gate/up + `IQ4_NL` down) prompt experts on `gfx1151`'s FP16 WMMA matrix cores (`-15.3%` gate/up time, `-22.9%` down time).
2. `"STRATA_GDN_CHUNKED": "1"` — Runs the 36-layer DeltaNet prompt recurrence in 16-token WY chunks on WMMA matrix cores (`1.6x` faster GDN recurrence).
3. `"STRATA_SPEC_GUMBEL": "1"` — Enables Gumbel-max coupled draft sampling at `temperature: 0.2` (`+9%` sampled decode speed via higher draft acceptance).
4. `"STRATA_HC_PERSIST": "1"` *(and `"STRATA_GR_TSUM": "1"` if PR `#2997` is cherry-picked)* — Persistent hyper-connection read kernel / transposed butterfly + LDS barrier on decode verify (`-4.4%` to `-6.6%` verify window latency).

---

## 6. Step-by-Step Verification & Testing Protocol for Reviewing LLMs

### Phase 1: Inspect & Fast-Forward Local `Strata` Clone to `v0.1.45`
```powershell
# 1. Verify current HEAD (v0.1.40.2) vs fetched origin/main (v0.1.45)
git -C C:\Users\Butch\Documents\02_Apps\Strata status -sb
git -C C:\Users\Butch\Documents\02_Apps\Strata log HEAD..origin/main --oneline -n 30

# 2. Inspect diffs on critical Strix Halo / prefill / decode files before merging
git -C C:\Users\Butch\Documents\02_Apps\Strata diff v0.1.40.2..v0.1.45 -- src/core/arch_defaults.cpp src/prefill/prefill.cpp src/kernels/cuda/moe_fused_iq.cu docs/STRIX_HALO.md

# 3. Create a safe test branch from origin/main (v0.1.45) before fast-forwarding main
git -C C:\Users\Butch\Documents\02_Apps\Strata checkout -b test/v0.1.45-upgrade v0.1.45
```

### Phase 2: Optional Cherry-Pick of Open PR `#2997` (`STRATA_GR_TSUM=1` + `gr_sync_lds`)
```powershell
# Fetch PR #2997 into a dedicated branch for isolated A/B testing
git -C C:\Users\Butch\Documents\02_Apps\Strata fetch origin pull/2997/head:pr-2997-hc-q8-tsum
```

### Phase 3: Build & Run CTest Parity Verification Gates
```powershell
# Build Windows HIP engine (ensure -DSTRATA_MMQ_KQUANTS=ON and -DSTRATA_Q6K_EXPERTS=ON per PR #2515)
# Then run parity self-tests before swapping strata.exe in production:
ctest --test-dir build-hip -R "gr_parity|hip_prefill_fused_iq|hc_persist_parity" --output-on-failure
```

### Phase 4: Run Live End-to-End Benchmarks in `antigravity-add-model`
```powershell
# 1. Run context ladder benchmark (512 -> 16384 tokens) with reasoning tokens captured
go -C C:\Users\Butch\Documents\02_Apps\antigravity-add-model run ./cmd/bench -contexts 512,1024,2048,4096,8192,16384 -gen 64 -temp 0 -reasoning high

# 2. Run 256-token structured code generation benchmark (verifies >60 tok/s MTP + lookup-chain-3 acceptance)
go -C C:\Users\Butch\Documents\02_Apps\antigravity-add-model\cmd\strata-server run ./benchmark
```

### Phase 5: Production Acceptance Criteria
- **Sub-1K Prefill (`512` & `1024` tiers)**: Must increase from `363.6–528.9 tok/s` (MMQ fallback) to **`>= 950–1,100 tok/s`** once `STRATA_PREFILL_STREAM_MIN=128` is active after engine restart.
- **2K–16K Prefill**: Must sustain **`>= 1,100–1,350+ tok/s`** (`+10% to +18%` further gain expected with `STRATA_PF_F16=1` and `STRATA_GDN_CHUNKED=1` in `v0.1.45`).
- **Sustained Decode**: Must hold **`>= 50.0 tok/s` minimum** on prose deliberation and **`60.0–68.0+ tok/s`** on structured coding / tool-calling workloads (`>= 75%` draft acceptance).
