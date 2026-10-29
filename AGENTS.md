# AGENTS.md

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Contributing a change or report

- Search the open issues and pull requests first, and add to a thread that already covers your point.
- Open an issue with the form that fits (bug report, feature request or question).
- One change per pull request. Say what it changes and what it leaves alone.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).

## AMD Strix Halo (gfx1151, 128 GB UMA) - Pass 9.1 Champion Configuration

Pass 9.1 establishes single-pass FP16 WMMA MoE prefill and coupled `--spec 7` MTP speculative decode on AMD Strix Halo (`gfx1151`, 128 GB UMA) using `Qwen 3.8 Flash-Next 177B (UD-IQ4_XS)` (`strata.exe` `SHA256: a621538b75e570888cf11d763391d82a224dd540f537a43d1e281b30466ac491`):
- **Champion Switches Locked**:
  - `STRATA_NO_TURN_SPLIT=1`: Disables conversational turn splitting, preserving 1-pass `PP` prompt execution (`0` turn-split checkpoints).
  - `STRATA_PF_F16=1` + `STRATA_PF_F16_MIN_T=0`: Enables persistent-WGP FP16 WMMA MoE prefill across all prompt sizes (`T >= 16`).
  - `STRATA_PF_MOE_SCATTER=0` + `STRATA_PF_DM_F16=1`: Disables `moe_accum` scatter override so `uint16_t` `Dm` (`-80.5 GB` DRAM at 16K) and `native_f16_kernel` MoE down-projections execute.
  - `STRATA_GDN_CHUNKED=1` + `STRATA_GDN_CHUNKED_MIN_T=128` + `STRATA_PREFILL_STREAM_MIN=16`: Resolves sub-1024 prefill cliff and runs 16-token WMMA WY GDN recurrence.
  - `--prefill 18432` + `--spec 7` + `--lookup-chain 3`: Single-chunk prefill covering 17,612-token prompts with 7-token coupled MTP + lookup speculative verification.
- **Empirical Warmed Context Ladder Performance (gen 256, reasoning medium, with MTP draft prefill active)**:
  - 512 tok (408 prompt): Prefill 743.4 tok/s (`TTFT = 649.0 ms`), Decode 88.0 tok/s (Client) / 85.6 tok/s (Engine), 77.5% draft acceptance
  - 1024 tok (1,020 prompt): Prefill 1,035.7 tok/s (`TTFT = 1,086.0 ms`), Decode 85.7 tok/s (Client) / 83.4 tok/s (Engine), 77.0% draft acceptance
  - 2048 tok (2,098 prompt): Prefill 1,231.5 tok/s (`TTFT = 1,808.4 ms`), Decode 81.8 tok/s (Client) / 79.7 tok/s (Engine), 73.4% draft acceptance
  - 4096 tok (4,254 prompt): Prefill 1,367.8 tok/s (`TTFT = 3,214.8 ms`), Decode 90.5 tok/s (Client) / 87.9 tok/s (Engine), 84.2% draft acceptance
  - 8192 tok (8,732 prompt): Prefill 1,449.1 tok/s (`TTFT = 6,136.2 ms`), Decode 87.2 tok/s (Client) / 84.6 tok/s (Engine), 82.8% draft acceptance
  - 16384 tok (17,612 prompt): Prefill 1,496.7 tok/s (`PP = 1,499.6 tok/s`, `TTFT = 11,879.1 ms`), Decode 85.1 tok/s (Client) / 82.7 tok/s (Engine), 83.7% draft acceptance
- **Unified Memory Headroom (128 GB LPDDR5X UMA: 96 GiB GPU Carveout + 31.65 GiB OS RAM)**:
  - **UMA GPU VRAM**: `33,594 MiB` (`32.81 GiB`) free at handshake (`33,454 MiB` / `32.67 GiB` during decode with all `24,576` experts in `55.43 GiB` VRAM), leaving room to run `Qwen3.6-35B-A3B` concurrently.
  - **Host Physical OS RAM**: `STRATA_FILE_RELEASE=1` hands `55.15 GiB` of expert mmap pages back to Windows (`strata.exe` working set `2.02 GiB`, `16.1–21.0 GiB` OS RAM free; `~48.9+ GiB` combined free unified memory).
- **Validation**:
  - `audit-config`: 39 PASS, 0 WARN, 0 FAIL.
  - `verify-thinking`: Overall Verdict: true (multi-turn KV cache prefix preservation verified across effort levels).
  - All Go 1.26 tests pass 100% via `go test -p 1 -v ./...` (including `STRATA_NO_TURN_SPLIT` binary symbol staging guard).

<!-- MEMOHARNESS:BEGIN -->
# Curated MemoHarness Rules
- **AVOID PAST FAILURE [pass91_champion_lock]**: Keep Pass 9.1 (`strata.exe` `a621538b...`, `strata-server.exe` `3b24e099...`) as active champion; Pass 10/11 experimental builds regressed sub-1024 prefill (`640 vs 757.8 tok/s`) or mid-tier decode and are archived off-path (`.bak_pass10` / `.bak_pass11`).
<!-- MEMOHARNESS:END -->
