# Strata Kernel & Inference Engine Architecture

This document breaks down the core algorithmic breakthroughs and kernel designs that enable [Niko1221/Strata](https://github.com/Niko1221/Strata.git) to execute massive 177B Mixture-of-Experts (MoE) models at high speed on consumer and workstation hardware.

---

## 1. Disjoint MoE Expert Caching & Asynchronous Overlap

Traditional MoE implementations require all weights to reside entirely in high-bandwidth GPU memory. A 177-billion parameter model with 48 layers and 24,576 total experts requires over 90 GB of memory:

```
[ Model Layer ] -> Top-10 Router
                      |
        +-------------+-------------+
        |                           |
[ VRAM Expert Cache ]        [ Pinned Host RAM ]
  (~3,000 Hot Experts)         (All 24,576 Experts)
        |                           |
  GPU Execution               CPU Execution
 (Radeon 8060S / HIP)        (Ryzen AI Max Zen 5 AVX-512)
        \                           /
         +------------+------------+
                      |
            Combined Output State
```

### Key Technical Mechanisms:
1. **Disjoint Partitioning**: At any time, GPU VRAM holds the top-k most frequently routed experts, while system RAM holds the entire pool of 24,576 experts (page-locked via `VirtualLock` on Windows).
2. **True Concurrent Execution**: When a token routes to experts not present in VRAM, the GPU dispatches its resident experts on HIP while the CPU simultaneously executes the missing experts in place using multi-threaded AVX-512 / AVX2 kernels. Neither device waits on the other.
3. **Adaptive Conversation Cache**: During interactive sessions, Strata tracks expert routing frequencies. In the background (`--adapt-async 1`), a dedicated helper thread swaps cold VRAM experts back to RAM and promotes hot experts to VRAM without stalling decode cycles.
4. **Exchange Rotation (`STRATA_EXCHANGE_ROTATE=1`)**: Eliminates buffer copying by simply swapping pointer ownership between host and device page-locked exchange slots.

---

## 2. Gated Delta Net (GDN) Linear Recurrence

Instead of traditional softmax self-attention which scales with quadratic memory and compute $O(N^2)$, the Qwen 3.8 Flash-Next architecture employs Gated Delta Net (GDN) linear attention with $O(1)$ memory complexity per decode step:

$$\mathbf{S}_t = \beta_t \mathbf{S}_{t-1} + \mathbf{k}_t \mathbf{v}_t^T, \quad \mathbf{o}_t = \mathbf{q}_t \mathbf{S}_t$$

### Kernel Optimizations:
- **`STRATA_GDN_HEAD` & `STRATA_GDN_PP=2`**: Parallelizes state updates across 4 SIMD lanes per column with 2 columns per lane, achieving **+2.9%** speedup on 64K context prompts.
- **`STRATA_GDN_CONVL2`**: Fuses the 1D short temporal convolution and query/key L2 normalization into a single memory pass, eliminating redundant global memory round-trips.
- **`STRATA_GDN_SPLIT`**: Distributes the decode-time GDN state update across 4 threadblocks per attention head.

---

## 3. Prefix Lookup Engine (PLE) & Direct NVMe I/O

The Prefix Lookup Engine (PLE) leverages a pre-computed 28.8 GB n-gram table to ground predictions and accelerate repeated phrase lookups.

### The Unbuffered Async I/O Subsystem (`src/platform/direct_file.cpp`):
Reading rows through standard OS memory-mapped files (`mmap`) causes kernel page-cache contention, memory inflation, and page faults:

- **Direct I/O Solution**: Strata opens the n-gram file using Win32 `CreateFileW` with flags:
  ```cpp
  FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS
  ```
- **I/O Completion Ports (IOCP)**: An asynchronous thread pool (`STRATA_IO_THREADS`, default 4 on Windows) issues queued reads directly into aligned memory buffers allocated via `VirtualAlloc`.
- **Performance**: Achieves **122.3 ms** average seek-and-read latency directly from NVMe SSD at queue depth, with zero bytes of OS file cache bloat.

---

## 4. Multi-Token Prediction (MTP) & Speculative Decoding

Strata incorporates a dual-mode speculative decoding pipeline:

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
  <table style="width: 100%; min-width: 600px; border-collapse: collapse; background-color: #161b22; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 14px;">
    <thead>
      <tr style="background-color: #21262d; border-bottom: 2px solid #30363d; text-align: left;">
        <th style="padding: 12px 16px; color: #58a6ff; white-space: nowrap;">Speculative Drafter</th>
        <th style="padding: 12px 16px; color: #58a6ff; white-space: nowrap;">Draft Length</th>
        <th style="padding: 12px 16px; color: #58a6ff; white-space: nowrap;">Verification Mechanism</th>
        <th style="padding: 12px 16px; color: #58a6ff; white-space: nowrap;">Measured Yield & Acceleration</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;">MTP Layer (Draft Head)</td>
        <td style="padding: 12px 16px; white-space: nowrap;">5 to 7 tokens (<code>--spec 7</code>)</td>
        <td style="padding: 12px 16px;">Full 48-layer parallel verification pass (<code>--mtp-q4 head</code>, <code>--coupled-draft</code>)</td>
        <td style="padding: 12px 16px;"><strong>74% to 95.5%</strong> acceptance; <strong>78.9 to 86.2 tok/s</strong> engine decode (<strong>82.1 to 89.8 tok/s</strong> client decode) on Strix Halo Pass 7.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;">Prompt Lookup &amp; Think Ring</td>
        <td style="padding: 12px 16px; white-space: nowrap;">Up to 7 tokens (<code>--lookup-chain 3</code>)</td>
        <td style="padding: 12px 16px;">N-gram match against prompt context + dynamic <code>&lt;think&gt;</code> ring (<code>STRATA_LOOKUP_THINK_RING=1</code>)</td>
        <td style="padding: 12px 16px;"><strong>+6% to +11%</strong> speedup on code edits, tool calls, and deliberative reasoning chains.</td>
      </tr>
    </tbody>
  </table>
</div>

### Gumbel-Max Draft Coupling (`STRATA_SPEC_GUMBEL=1`)
When generating at non-zero temperature, standard sampling causes the draft and target models to diverge. Strata implements Gumbel-max coupling:
$$t^* = \arg\max_i \left( \frac{p_i}{E_i} \right)$$
where $E_i \sim \text{Exp}(1)$ is exponential noise pseudo-randomly keyed by token ID, seed, and position. This forces the drafter and verifier to agree on identical candidate tokens, raising acceptance from 52.8% to **59.9%** (+9% net throughput).

---

## 5. KV Cache Streaming & Rotated Quantization

Long-context inference (32K to 256K tokens) normally saturates VRAM with KV cache tensors:

- **KV Streaming (`--kv-resident 32768`)**: Only the active attention window (e.g. 32K cells) resides in VRAM. Older KV context is stored in system RAM and streamed over the UMA bus as needed, freeing up multiple gigabytes of VRAM for hundreds of additional cached MoE experts.
- **Int8 KV Cache (`--kv int8`)**: Standard production baseline, zero perplexity degradation.
- **Hadamard-Rotated 4-Bit KV (`--kv q4_0` / `--kv k8v4`)**: Multiplies key/value vectors by an orthogonal Hadamard matrix prior to 4-bit quantization, eliminating outliers and reducing KV memory consumption by 23–50%.
