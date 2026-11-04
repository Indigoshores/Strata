# Qwen 3.8 Flash-Next Prefill Ports — Test Branch Plan (`feat/qwen38-prefill-ports`)

> [!IMPORTANT]
> This branch tests prefill-only optimizations ported from verified upstream/Gufo sources while **strictly protecting the Arm D decode pipeline** on AMD Strix Halo (Ryzen AI Max+ 395 / Radeon 8060S / gfx1151, 96.0 GiB UMA carveout, Windows 11).
> Production Arm D (UD-IQ4_XS, Port 8080, `--conversation-cache-mib 3072`, `--short-read 15`, post-READY `0x55550000` CCD1 pinning) is the untouchable baseline.

## 1. Verified Baseline (Arm D, measured 2026-11-04, `Cached=0`)

| Metric | Value | Source |
|:---|:---|:---|
| Cold Boot Prefill (512..16384) | 855.5 - 1,426.1 tok/s (mean 1,257.9) | armD.json cold_sweep |
| Warm Engine Prefill (512..16384) | 979.6 - 1,454.1 tok/s (mean 1,330.5) | prefill_diag.json |
| Cold Sustained Decode (128 tok) | 48.7 - 62.2 tok/s (mean 56.6) | armD.json cold_sweep |
| Cold Sustained Decode (384 tok unseen) | 48.8 - 54.8 tok/s | true_cold_decode.json |
| Warm Burst Decode (32 tok, 90% MTP) | 75.2 - 83.1 tok/s (mean 79.3) | prefill_diag.json |
| Multi-Agent 16K Resume TTFT | 794 ms (23.9x vs 18,957 ms cold) | armD.json multi_agent_slot |

## 2. Hard Invariants (NEVER modify on this branch)

1. **Decode path untouched**: no changes to `mul_mat_vec` / N=1 GEMV, MTP draft-verification loop, or acceptance logic.
2. **Affinity**: post-READY single-CCD pin `0x55550000` (CCD1 8 physical Zen 5) stays; no Linux pthread affinity logic.
3. **Streams**: `STRATA_SH_STREAM=0` (upstream STRIX_HALO.md confirms SH_STREAM=1 is 6.9% slower on Windows gfx1151 since 0.2.7).
4. **Memory**: `--expert-cache auto --mmap-experts` (99.65 GiB commit limit; do NOT hardcode 58G from 128 GB-machine docs).
5. **No N=64 chunked DeltaNet on gfx1151**: upstream v0.2.9 applied the 64-token threshold to **gfx12 Radeon only**; our Arm B test failed bit-exact greedy parity at 117 tok AND slowed prefill (311.2 -> 303.7 tok/s). Keep `STRATA_GDN_CHUNKED_MIN_T=128`.
6. **No Q4_K_XL co-residency**: ~77-80 GiB footprint + Qwen 3.6 35B (~24-26 GiB) exceeds the 96 GiB carveout; Issue #4106 shows streamed Q4_K_XL collapses to 437-474 pp / 32-44 tg.

## 3. Test Arms (one variable per arm, all behind `batch >= 16` prefill guards)

## Arm P1 — `STRATA_PREFILL_STREAM_MIN` sweep (config-only, zero code risk)
- Upstream STRIX_HALO.md: with all experts in UMA, MMQ is the slow part; N=128 beat N=1024 for ~1K-token agent turns (3.1-6.0 s -> 0.9-2.5 s).
- Production currently runs `STRATA_PREFILL_STREAM_MIN=16`. Sweep {16, 64, 128, 256} on the 1K/3.3K/6.6K agent-turn probes + full 6-tier sweep.
- Gate: warm prefill mean must not drop; cold 512-tier (855.5 tok/s) is the regression canary.

## Arm P2 — PLE n-gram async pre-staging (cold-prefill gap)
- Target: cold-vs-warm 512-tier gap (855.5 vs 979.6 tok/s, -12.7%) = Layer-2 PLE row-cache cold stall.
- Port Gufo-style pre-touch: issue non-blocking `PrefetchVirtualMemory` / `ReadFileScatter` hints for the n-gram shard slice during Stage-1 dense layers (existing hooks: `--ple-io direct`, `--ple-inflight 1024`, `--ple-row-cache 1048576`, `STRATA_PREFETCH_WINDOW=192`).
- Gate: cold 512/1024-tier prefill >= warm values; decode paths byte-identical.

## Arm P3 — Gufo-style dense prefill work-group grid
- Target: warm prefill plateau (~1,450 tok/s at 4K-16K). Gufo claims 1,702-1,801 pp (Q4_K_XL, 128 GB machine — treat as hypothesis).
- Port 2D WMMA tile dispatch (256-512 token tiles, `dim3 block(32,8)`) strictly inside the `batch >= 32` prefill GEMM; decode GEMV translation unit untouched.
- Gate: bit-exact greedy parity + warm prefill improvement at 4K/8K/16K tiers.

## Arm A4 — `STRATA_SELECT_WMMA` bit-exact audit at long context (audit-only, no port)
- Upstream v0.2.9 notes SELECT_WMMA block scorer is NOT bit-exact (~10 of 2,051 cells flip past ~5K tokens). Production has it ON.
- Run greedy parity probes at 6.6K / 13K / 26K tokens with SELECT_WMMA=1 vs =0; measure prefill delta.
- Output: documented accept-risk decision for production (keep =1 with known divergence, or flip =0 if prefill cost < 1%).

## 4. Regression Gates (fail-closed, every arm)

1. **Bit-exact greedy SHA-256 parity** at 88 (`0e8e4a7b65`), 512 (`d37e5996a3`), 1024 (`a65f4babca`) checkpoints vs Arm D.
2. **Cold sustained decode >= 48 tok/s** (128-384 tok, `Cached=0`, unseen prompts).
3. **Warm burst decode >= 75 tok/s** (32 tok, 90% MTP accept regime).
4. **Affinity check**: engine log shows post-READY pin to `0x55550000` before any benchmark run.
5. Any gate failure -> revert the arm's last patch immediately; never merge a failing arm.

## 5. Benchmark Harness

- `ab_bench.exe` (pure Go 1.26, scratch/ab_bench in Antigravity conversation brain): Win32 process/affinity manager + 6-tier cold sweep + warm diag + parity checkpoints.
- Test instance runs on **Port 8081** with a dedicated `strata-qwen38-pf.json` config; Arm D Port 8080 stays online.
- hipBLASLt table: `tools/hip/gfx1151-hipblaslt-100500.txt` (Arm D calibrated version restored on this branch; the pre-branch copy is kept as `.bak_prebranch`).

## 6. Provenance (verified 2026-11-04, headless browser audit)

- Niko1221/Strata issues #3032, #3398, #4106: REAL (gfx1151 community suites; SH_STREAM=0 +12-15%; Q4_K_XL streamed collapse).
- #3098: actually a bench-port PR (misattributed in Gemini doc).
- gufo.blog + gufo-org/gufo: REAL (1,702.93 pp / 65.25 tg on Q4_K_XL, 128 GB machine).
- v0.2.9 release notes: REAL numbers (8K UD-IQ4_XS 794 -> 1,243) but the mechanism is PF_F16 + HC_Q8 + SELECT_WMMA on Windows gfx1151 (already active in Arm D), NOT the gfx12-only N=64 chunk threshold.
- xenodeve/Strata-xeno: REAL fork but NVIDIA multi-GPU; short-prompt bypass already covered natively by `--short-read 15`.
