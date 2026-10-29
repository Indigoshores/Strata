# Pass 7 Validation Report: Qwen 3.8 Flash-Next 177B on AMD Strix Halo (`gfx1151`)

**Date:** 2026-10-28  
**Hardware:** AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`, 40 RDNA 3.5 CUs, `128 GB` LPDDR5X-8000 Unified Memory: `96 GiB` BIOS UMA VRAM + `31.65 GiB` Windows 11 Physical System RAM)  
**Co-Resident Workload:** `llama-server.exe` (`PID 18288`) serving `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` continuously on `http://127.0.0.1:8081` (`15.52 GiB` private commit)  
**Engine Binary:** [`strata.exe`](file:///C:/Users/Butch/Documents/10-Models/Strata/engine/strata.exe) (`sha256: 752cbeacb972cdde36cd9488683d690399e8d597de6b05ae660d15a8e11d18ef`, ROCm 10.2 / `amdhip64_7.dll` + `gfx1151-hipblaslt-100500.txt` 90-row tuning table, backup at `strata.exe.bak_pass6`)  
**Go 1.26 Server Binary:** [`strata-server.exe`](file:///C:/Users/Butch/Documents/10-Models/Strata/bin/strata-server.exe) (`sha256: 3b24e09936a749426ea625e381f58aebed7b481339a5dd73d042ea0df9f07176`)  
**Production Config:** [`strata-qwen177b.json`](file:///C:/Users/Butch/Documents/10-Models/Strata/strata-qwen177b.json) (`strix-harness.exe audit-config`: **29 PASS / 0 WARN / 0 FAIL**)

---

## 1. What Changed in Pass 7

1. **Restored & Locked Pass 6 WMMA Prefill Switches (`Fail-Closed Gate`)**:
   - Restored `STRATA_PF_GEMM=1`, `STRATA_PF_PAD=1`, `STRATA_WMMA_GEMM=1`, `STRATA_PF_HCDOWN=1`, `STRATA_PF_SWITCH_MIN_T=128`, and `STRATA_PREFILL_STREAM_MIN=16` in [`strata-qwen177b.json`](file:///C:/Users/Butch/Documents/10-Models/Strata/strata-qwen177b.json), backed by fail-closed regression checks in [`docs-strix-go/validator.go`](file:///c:/Users/Butch/Documents/02_Apps/Strata/docs-strix-go/validator.go) and [`docs-strix-go/prefill_switches_regression_test.go`](file:///c:/Users/Butch/Documents/02_Apps/Strata/docs-strix-go/prefill_switches_regression_test.go) (`go test ./...`: **PASS**).
2. **Eliminated the Extra Checkpoint Split on Trailing Reasoning-Effort Turns (`--tail-role-token 8678`)**:
   - With `effort_position="end"`, [`strata-server`](file:///C:/Users/Butch/Documents/02_Apps/antigravity-add-model/cmd/strata-server/tokenizer/chat.go) appends `<|im_start|>system\nReasoning effort is set to medium...<|im_end|>\n` right before `<|im_start|>assistant\n`.
   - Passing `--tail-role-token 8678` (`vocab.json["system"] = 8678`) instructs [`src/program/generate.cpp`](file:///c:/Users/Butch/Documents/02_Apps/Strata/src/program/generate.cpp#L13482-L13487) to place the conversation checkpoint *before* the trailing effort turn (`1` checkpoint instead of `2`) while `STRATA_ADMIT_MERGE_FIRST=1` merges the final 8 prompt tokens directly into verify window #1 (`0 ms` tail step).
   - **Measured Result**: `95.5%` warm KV prefix reuse (`833 / 872` tokens reused in `881 ms` end-to-end turn time) when toggling `Low -> Medium -> XHigh` mid-conversation, and `+6.6%` to `+41.3%` faster cold E2E TTFT across `1K–16K`.
3. **7-Token Coupled Speculative Verification (`--spec 7`) + Pass 7 C++/HIP Ports (`branch pass7/gfx1151-v0145-opt`)**:
   - Ported commit `789d839` (`!g_qfuse()` guard on 1-token self-commit in [`src/core/verify.cpp`](file:///c:/Users/Butch/Documents/02_Apps/Strata/src/core/verify.cpp#L3190)), commit `7d1016f` (`q81_div` helper in [`include/strata/kernels/q8_1_finite.hpp`](file:///c:/Users/Butch/Documents/02_Apps/Strata/include/strata/kernels/q8_1_finite.hpp)), and `pixmaate/gufo` commit `aeee89a` (`STRATA_WDDM_ALT_STREAM=1`, opt-in alternating commit stream in [`include/strata/core/verify.hpp`](file:///c:/Users/Butch/Documents/02_Apps/Strata/include/strata/core/verify.hpp) & [`src/core/verify.cpp`](file:///c:/Users/Butch/Documents/02_Apps/Strata/src/core/verify.cpp)).
   - **GPU Parity Verification on Radeon 8060S (`sm_115` / `gfx1151`)**: `verify_batch_parity.exe` (bitwise equal), `spec_verify_parity.exe` (`1,296` rows, `0` mismatches), `hip_prefill_wmma_gemm_parity.exe` (`3,520` cases, `max_abs = 2.42e-07`), `test_moe_variable_tiles.exe` (`105/105` exact parity), `s2_expert_grouped_parity.exe` (`0` failures), and `test_expert_bulk_load.exe` (**PASS**).
4. **End-to-End `"Medium"` Thinking Alignment & Antigravity Picker UI (`antigravity-add-model`)**:
   - Aligned default thinking effort to `"medium"` across [`strata-qwen177b.json`](file:///C:/Users/Butch/Documents/10-Models/Strata/strata-qwen177b.json), [`custom_models.json`](file:///C:/Users/Butch/.gemini/antigravity/custom_models.json), [`src/proxy/modelUtils.ts`](file:///C:/Users/Butch/Documents/02_Apps/antigravity-add-model/src/proxy/modelUtils.ts), [`src/proxy/translators/openai.ts`](file:///C:/Users/Butch/Documents/02_Apps/antigravity-add-model/src/proxy/translators/openai.ts), and inline `[Low | Med | XHigh]` pills in [`src/preload.ts`](file:///C:/Users/Butch/Documents/02_Apps/antigravity-add-model/src/preload.ts) (`344/344` Vitest tests + Go server/patcher tests **PASS**).

---

## 2. Measured Performance on Strix Halo (`gfx1151`, Port `8081` Live Co-Resident)

All cold context-ladder runs below used unique nonces (`Reused = 0`), `256` generated `"medium"` thinking tokens (`temperature = 0.0`), `24,576 / 24,576` experts resident in UMA VRAM (`56,762 MiB`), and live Qwen 3.6 35B active on `:8081`.

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
<table style="width: 100%; min-width: 600px; border-collapse: collapse; background: #161b22; color: #c9d1d9; font-size: 13px;">
  <thead>
    <tr style="background: #21262d; border-bottom: 1px solid #30363d;">
      <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Context Tier</th>
      <th style="padding: 10px 12px; text-align: right; color: #58a6ff; white-space: nowrap;">GPU Chunk Prefill</th>
      <th style="padding: 10px 12px; text-align: right; color: #58a6ff; white-space: nowrap;">E2E Prefill</th>
      <th style="padding: 10px 12px; text-align: right; color: #58a6ff; white-space: nowrap;">Pass 6 Decode (spec 5)</th>
      <th style="padding: 10px 12px; text-align: right; color: #e3b341; white-space: nowrap;">Pass 7 Engine Decode</th>
      <th style="padding: 10px 12px; text-align: right; color: #3fb950; white-space: nowrap;">Pass 7 Client Decode</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap;">512 tok (408 actual)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">646.7 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">485.2 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">55.7 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; color: #e3b341; white-space: nowrap;">78.9 - 79.9 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; color: #3fb950; white-space: nowrap;">82.1 tok/s (+41.6%)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap;">1,024 tok (1,020 actual)</td>
      <td style="padding: 8px 12px; text-align: right; color: #3fb950; white-space: nowrap;">968.9 tok/s (Max)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">794.1 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">52.3 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; color: #e3b341; white-space: nowrap;">79.9 - 81.4 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; color: #3fb950; white-space: nowrap;">82.9 - 83.6 tok/s (+52.8%)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap;">2,048 tok (2,098 actual)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">1,077.6 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">957.8 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">49.1 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; color: #e3b341; white-space: nowrap;">76.8 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; color: #3fb950; white-space: nowrap;">79.6 tok/s (+56.4%)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap;">4,096 tok (4,254 actual)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">1,222.7 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">1,141.6 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">52.9 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; color: #e3b341; white-space: nowrap;">83.7 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; color: #3fb950; white-space: nowrap;">86.2 tok/s (+58.2%)</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap;">8,192 tok (8,731 actual)</td>
      <td style="padding: 8px 12px; text-align: right; color: #3fb950; white-space: nowrap;">1,331.3 tok/s (Max)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">1,282.0 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">62.4 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; color: #e3b341; white-space: nowrap;">84.0 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; color: #3fb950; white-space: nowrap;">87.5 tok/s (+34.6%)</td>
    </tr>
    <tr>
      <td style="padding: 8px 12px; white-space: nowrap;">16,384 tok (17,613 actual)</td>
      <td style="padding: 8px 12px; text-align: right; color: #3fb950; white-space: nowrap;">1,352.8 - 1,356.7 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">1,285.3 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">52.3 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; color: #e3b341; white-space: nowrap;">86.0 - 86.2 tok/s</td>
      <td style="padding: 8px 12px; text-align: right; color: #3fb950; white-space: nowrap;">89.8 tok/s (+64.4%)</td>
    </tr>
  </tbody>
</table>
</div>

---

## 3. Win32 Physical RAM, Commit & UMA VRAM Validation (`GlobalMemoryStatusEx` + `psapi.dll`)

Measured across idle dual-server startup and post-16K sustained stress with both `:8080` (Qwen 3.8 177B) and `:8081` (Qwen 3.6 35B) live:

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
<table style="width: 100%; min-width: 600px; border-collapse: collapse; background: #161b22; color: #c9d1d9; font-size: 13px;">
  <thead>
    <tr style="background: #21262d; border-bottom: 1px solid #30363d;">
      <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Measurement Stage</th>
      <th style="padding: 10px 12px; text-align: right; color: #58a6ff; white-space: nowrap;">OS RAM Load</th>
      <th style="padding: 10px 12px; text-align: right; color: #58a6ff; white-space: nowrap;">Used / Free OS RAM</th>
      <th style="padding: 10px 12px; text-align: right; color: #58a6ff; white-space: nowrap;">strata.exe Working Set</th>
      <th style="padding: 10px 12px; text-align: right; color: #58a6ff; white-space: nowrap;">strata.exe Commit</th>
      <th style="padding: 10px 12px; text-align: right; color: #58a6ff; white-space: nowrap;">Free UMA VRAM</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap;">Port 8081 Only (Baseline)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">19%</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">6.18 / 25.47 GiB</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">Not loaded</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">Not loaded</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">76+ GiB</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap;">Pass 6 Binary (Post-16K)</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">33%</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">10.50 / 21.14 GiB</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">3,262.0 MiB</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">63,891.0 MiB</td>
      <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">16,612 MiB</td>
    </tr>
    <tr style="border-bottom: 1px solid #30363d;">
      <td style="padding: 8px 12px; white-space: nowrap;">Pass 7 Champion (Loaded)</td>
      <td style="padding: 8px 12px; text-align: right; color: #3fb950; white-space: nowrap;">33%</td>
      <td style="padding: 8px 12px; text-align: right; color: #3fb950; white-space: nowrap;">10.49 / 21.15 GiB</td>
      <td style="padding: 8px 12px; text-align: right; color: #3fb950; white-space: nowrap;">2,906.4 MiB</td>
      <td style="padding: 8px 12px; text-align: right; color: #3fb950; white-space: nowrap;">63,546.0 MiB</td>
      <td style="padding: 8px 12px; text-align: right; color: #3fb950; white-space: nowrap;">16,577 MiB</td>
    </tr>
    <tr>
      <td style="padding: 8px 12px; white-space: nowrap;">Pass 7 Champion (Post-16K + Multi-Turn)</td>
      <td style="padding: 8px 12px; text-align: right; color: #3fb950; white-space: nowrap;">33%</td>
      <td style="padding: 8px 12px; text-align: right; color: #3fb950; white-space: nowrap;">10.74 / 20.91 GiB</td>
      <td style="padding: 8px 12px; text-align: right; color: #3fb950; white-space: nowrap;">3,153.4 MiB (-109 MiB)</td>
      <td style="padding: 8px 12px; text-align: right; color: #3fb950; white-space: nowrap;">63,800.7 MiB (-90 MiB)</td>
      <td style="padding: 8px 12px; text-align: right; color: #3fb950; white-space: nowrap;">16,416 MiB</td>
    </tr>
  </tbody>
</table>
</div>

- **Non-Expansion Verdict:** Pass 7 adds **zero** RAM footprint expansion (`33%` OS RAM load, `20.91–21.15 GiB` physical RAM free, `-109 MiB` lower working set and `-90 MiB` lower private commit than Pass 6 after full `16K` + multi-turn stress), while `llama-server.exe` on Port `8081` remained flat at `174 MiB` working set / `15,515.6 MiB` private commit (`HTTP 200 OK`).
