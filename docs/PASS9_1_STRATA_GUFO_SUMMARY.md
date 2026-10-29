# Strata Pass 9.1: Architectural Evolution, Gufo Borrowed Features & Empirical Benchmark Comparison

**Target System**: AMD Ryzen AI Max+ 395 ("Strix Halo", Radeon 8060S `gfx1151`, 40 RDNA 3.5 CUs / 20 WGPs, 128 GB LPDDR5X Unified Memory, Windows 11 WDDM)  
**Primary Model**: Qwen 3.8 Flash-Next 177B (`UD-IQ4_XS`, 48 layers: 36 GDN linear attention + 12 QSA sparse attention, 512 routed experts/layer, top-10 active, 24,576 total GPU experts)  
**Production Binary Hash**: `a621538b75e570888cf11d763391d82a224dd540f537a43d1e281b30466ac491`

---

## 1. Executive Summary

**Strata Pass 9.1** represents the hardened production champion configuration on AMD Strix Halo. It successfully bridges high-throughput single-pass prefill with sustained speculative decoding by coupling:
1. **Single-pass FP16 WMMA MoE prefill** across all prompt sizes (`STRATA_PF_F16=1`, `STRATA_PF_F16_MIN_T=0`).
2. **Coupled 7-token MTP speculative decoding** (`--spec 7`, `--mtp-q4 head`, `lookup-chain 3`) with draft prefill integration on the 49th MTP layer.
3. **Driver stall reduction and state synchronization** borrowed directly from the upstream and community **Gufo** inference engine.

---

## 2. Modifications Applied to Original (Vanilla) Strata

Vanilla upstream Strata (`v0.1.45`–`v0.2.7`) was designed primarily for discrete NVIDIA CUDA GPUs and generic ROCm setups, leaving substantial performance on the table for AMD Strix Halo's unified memory architecture:

- **Single-Pass Prompt Execution (`STRATA_NO_TURN_SPLIT=1`)**:
  - *Vanilla*: Forced conversational multi-turn chunking and intermediate turn checkpoints, adding CPU-GPU round-trips.
  - *Pass 9.1*: Eliminates turn-splitting overhead, executing prompts in a single unified compute pass (`PP = 1,499.6 tok/s` at 16K).
- **Persistent-WGP FP16 WMMA MoE Prefill (`STRATA_PF_F16=1`, `STRATA_PF_F16_MIN_T=0`)**:
  - *Vanilla*: Used discrete GEMM calls or INT8 fallbacks that suffered from launch overhead or underutilized RDNA 3.5 dual-issue matrix cores.
  - *Pass 9.1*: Keeps 20 Workgroup Processors (WGPs) persistently loaded with persistent work-stealing queues across all prompt tokens ($T \ge 16$).
- **FP16 Down-Projection Without Scatter Overhead (`STRATA_PF_MOE_SCATTER=0`, `STRATA_PF_DM_F16=1`)**:
  - *Vanilla*: Performed atomic scatter-accumulate operations into FP32 accumulators.
  - *Pass 9.1*: Replaces scatter with native `uint16_t` down-projections (`Dm`), cutting DRAM traffic by **80.5 GB** at 16K context.
- **GDN Chunked Recurrence Tuning (`STRATA_GDN_CHUNKED=1`, `STRATA_GDN_CHUNKED_MIN_T=128`, `STRATA_PREFILL_STREAM_MIN=16`)**:
  - *Vanilla*: Ran naive sequential recurrence or unbounded chunk sizes that collapsed small-context prefill.
  - *Pass 9.1*: Bypasses chunking overhead for $T < 128$ while using 16-token WMMA WY recurrence for long sequences.
- **Win32 Memory Management & Page Release (`STRATA_FILE_RELEASE=1`, `STRATA_CKPT_POOL=1`)**:
  - *Vanilla*: Maintained large memory-mapped file standby lists, exhausting Windows commit charge.
  - *Pass 9.1*: Issues Win32 `VirtualUnlock` immediately after bulk uploading 24,576 experts (`55.43 GiB`) into VRAM, keeping **22.7–25.7 GiB** host RAM free.
- **Expanded Context & Lookup Window (`--prefill 18432`, `--max-context 262144`, `--mtp-window 8192`)**:
  - Enables full single-chunk prefill covering prompts up to 17,612 tokens and beyond.

---

## 3. Features and Innovations Borrowed from Gufo

During architectural mining of upstream **Gufo** (`gufo-org/gufo` release `v0.25.0` / `v0.26.0`) and the Windows Strix Halo community port **`pixmaate/gufo`**, Strata integrated key mechanisms:

### A. Windows WDDM Queue Ping-Pong (`pixmaate/gufo` `aeee89a`)
- **Mechanism**: Submitting successive compute kernels to the same command queue on Windows WDDM incurs a kernel driver fence query (~0.45 µs/node = ~0.8 ms stall across a 48-layer verification pass).
- **Adoption in Strata**: Integrated via `STRATA_WDDM_ALT_STREAM=1` in `src/core/verify.cpp`, ping-ponging between two pre-allocated verify streams (`stream_a` and `stream_b`) to overlap fence handshakes and recover ~0.8 ms per verify window.

### B. Pipeline Overlap and Deferred Rollback (`pixmaate/gufo` `a8b45a0`)
- **Mechanism**: Decouples speculative verification rollback from the host synchronization barrier, allowing target token commit and prefix verification graphs to launch ahead of token hashing.
- **Adoption in Strata**: Streamlines MTP proposal handling and n-gram table lookups (`STRATA_LOOKUP_THINK_RING=1`).

### C. Thread-Local OVERLAPPED Handles for NVMe I/O (Gufo PR #805)
- **Mechanism**: Multi-threaded unbuffered asynchronous file reads on Windows create handle-table contention if event handles are shared or repeatedly recreated via `CreateEventW`.
- **Adoption in Strata**: Applied in `src/platform/direct_file.cpp` to the Prefix Lookup Engine (PLE) for unbuffered 28.8 GiB n-gram direct disk streaming.

### D. Exact GDN Recurrence Guard (Gufo PR #850 Analysis)
- **Discovery**: Gufo PR #850 proved that lossy FP16 chunked DeltaNet scan introduced numerical errors up to 500x higher than FP32 recurrence, corrupting greedy output on tool/XML prompts.
- **Adoption in Strata**: Enabled strict gating so that GDN chunking is restricted only to safe sequence lengths (`MIN_T=128`), preserving numerical parity and multi-turn KV cache consistency (`verify-thinking: true`).

---

## 4. Benchmark Results: Context Prefill & Decode Comparison

The following tables benchmark **Strata Pass 9.1** against:
- **Vanilla Upstream Strata** (`0.2.7` / `0.1.26`–`0.1.45` baseline)
- **Strata Pass 6** (Raw prefill champion without MTP overhead, unspeculated decode)
- **Gufo** (`0.25.0` / `0.26.0` on Strix Halo `gfx1151`, UD-Q4_K_XL with MTP)

### Table 1: Prompt Processing / Prefill Throughput (tokens/s)

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
  <table style="width: 100%; min-width: 650px; border-collapse: collapse; background-color: #161b22; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 13px;">
    <thead>
      <tr style="background-color: #21262d; border-bottom: 2px solid #30363d;">
        <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Context Tier</th>
        <th style="padding: 10px 12px; text-align: right; color: #58a6ff; white-space: nowrap;">Vanilla Strata (0.2.7)</th>
        <th style="padding: 10px 12px; text-align: right; color: #58a6ff; white-space: nowrap;">Strata Pass 6 (No MTP)</th>
        <th style="padding: 10px 12px; text-align: right; color: #58a6ff; white-space: nowrap;">Gufo (v0.25.0)*</th>
        <th style="padding: 10px 12px; text-align: right; color: #e3b341; white-space: nowrap;">Strata Pass 9.1 (Active)</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; font-weight: 600; white-space: nowrap;">512 tok</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">~450 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">1,206.6 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">-</td>
        <td style="padding: 8px 12px; text-align: right; color: #3fb950; font-weight: 600; white-space: nowrap;">743.4 tok/s (649.0 ms)</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; font-weight: 600; white-space: nowrap;">1024 tok</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">~530 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">1,386.2 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">-</td>
        <td style="padding: 8px 12px; text-align: right; color: #3fb950; font-weight: 600; white-space: nowrap;">1,035.7 tok/s (1,086.0 ms)</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; font-weight: 600; white-space: nowrap;">2048 tok</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">680 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">1,102.6 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">1,734.96 tok/s*</td>
        <td style="padding: 8px 12px; text-align: right; color: #3fb950; font-weight: 600; white-space: nowrap;">1,231.5 tok/s (1,808.4 ms)</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; font-weight: 600; white-space: nowrap;">4096 tok</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">755.0 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">1,531.6 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">-</td>
        <td style="padding: 8px 12px; text-align: right; color: #3fb950; font-weight: 600; white-space: nowrap;">1,367.8 tok/s (3,214.8 ms)</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; font-weight: 600; white-space: nowrap;">8192 tok</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">790.0 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">1,546.5 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">-</td>
        <td style="padding: 8px 12px; text-align: right; color: #3fb950; font-weight: 600; white-space: nowrap;">1,449.1 tok/s (6,136.2 ms)</td>
      </tr>
      <tr>
        <td style="padding: 8px 12px; font-weight: 600; white-space: nowrap;">16384 tok</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">821.0 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">1,566.6 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">-</td>
        <td style="padding: 8px 12px; text-align: right; color: #3fb950; font-weight: 600; white-space: nowrap;">1,496.7 tok/s (PP: 1,499.6 tok/s)</td>
      </tr>
    </tbody>
  </table>
</div>

*\*Note on Prefill Comparisons*:  
- **Pass 6 vs Pass 9.1**: Pass 6 executed only the 48 backbone layers without speculative decoding. Pass 9.1 prefill includes `draft_kv` calculation across the 49th MTP layer plus the initial 7-token MTP draft rollout (`mtp.draft_first`), which adds ~190–525 ms of upfront speculative priming to unlock fast decode.  
- **Gufo 2048**: Gufo measured autoregressive prefill with thinking disabled and greedy sampling under Linux ROCm.

---

### Table 2: Output / Decode Throughput (tokens/s) & Speculative Acceptance

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
  <table style="width: 100%; min-width: 650px; border-collapse: collapse; background-color: #161b22; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 13px;">
    <thead>
      <tr style="background-color: #21262d; border-bottom: 2px solid #30363d;">
        <th style="padding: 10px 12px; text-align: left; color: #58a6ff; white-space: nowrap;">Context Tier</th>
        <th style="padding: 10px 12px; text-align: right; color: #58a6ff; white-space: nowrap;">Vanilla Strata (0.2.7)</th>
        <th style="padding: 10px 12px; text-align: right; color: #58a6ff; white-space: nowrap;">Strata Pass 6 (Unspec.)</th>
        <th style="padding: 10px 12px; text-align: right; color: #58a6ff; white-space: nowrap;">Gufo (v0.25.0)</th>
        <th style="padding: 10px 12px; text-align: right; color: #e3b341; white-space: nowrap;">Strata Pass 9.1 (Active)</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; font-weight: 600; white-space: nowrap;">512 tok</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">44.0 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">59.8 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">35.8 tok/s (mixed)*</td>
        <td style="padding: 8px 12px; text-align: right; color: #3fb950; font-weight: 600; white-space: nowrap;">88.0 tok/s (77.5% acc)</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; font-weight: 600; white-space: nowrap;">1024 tok</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">44.0 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">56.4 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">-</td>
        <td style="padding: 8px 12px; text-align: right; color: #3fb950; font-weight: 600; white-space: nowrap;">85.7 tok/s (77.0% acc)</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; font-weight: 600; white-space: nowrap;">2048 tok</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">43.5 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">52.1 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">35.79 tok/s (mixed) / 116.9 (rep)*</td>
        <td style="padding: 8px 12px; text-align: right; color: #3fb950; font-weight: 600; white-space: nowrap;">81.8 tok/s (73.4% acc)</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; font-weight: 600; white-space: nowrap;">4096 tok</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">42.8 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">48.2 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">-</td>
        <td style="padding: 8px 12px; text-align: right; color: #3fb950; font-weight: 600; white-space: nowrap;">90.5 tok/s (84.2% acc)</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 8px 12px; font-weight: 600; white-space: nowrap;">8192 tok</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">41.0 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">43.6 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">-</td>
        <td style="padding: 8px 12px; text-align: right; color: #3fb950; font-weight: 600; white-space: nowrap;">87.2 tok/s (82.8% acc)</td>
      </tr>
      <tr>
        <td style="padding: 8px 12px; font-weight: 600; white-space: nowrap;">16384 tok</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">38.0 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">38.4 tok/s</td>
        <td style="padding: 8px 12px; text-align: right; white-space: nowrap;">-</td>
        <td style="padding: 8px 12px; text-align: right; color: #3fb950; font-weight: 600; white-space: nowrap;">85.1 tok/s (83.7% acc)</td>
      </tr>
    </tbody>
  </table>
</div>

*\*Note on Decode Comparisons*:  
- **Pass 9.1**: Measured live with `gen = 256`, `reasoning_effort: "medium"`, temperature 0.2, top_p 0.95 across realistic non-repetitive reasoning prompts on Windows WDDM. Decode speed remains flat across the entire context ladder (~82–90 tok/s) due to the $O(1)$ GDN recurrence complexity.  
- **Gufo**: 35.79 tok/s on diverse/mixed text; 116.93 tok/s was achieved only on highly repetitive synthetic tokens where speculative acceptance reached near 100%. Under realistic reasoning conditions, Pass 9.1's coupled MTP + Prefix Lookup Engine delivers more than **2.3x** the sustained single-user decode speed of Gufo.
