**Gufo, UD-IQ4_XS, and Strata Pass 9: prefill analysis and proposed implementation**

Date: October 29, 2026. Target: Windows, AMD Strix Halo gfx1151, 128 GB UMA, Qwen3.8-Flash-Next 177B, UD-IQ4_XS, single interactive user.

Status: plan for review. No engine, server, model, launcher, or benchmark code was changed for this assessment. No GPU benchmark or service restart was performed. Proposed speed improvements remain unmeasured.

**Recommendation.** Optimize Strata first while preserving the reported Pass 9.1 champion as the control. Gufo is a useful source of prefill mechanisms and a possible second implementation of the same weights, but its Flash-Next production paths need substantial quantization work before they can run UD-IQ4_XS. Do not modify or requantize the model to make it fit Gufo: implement support for its existing tensor formats.

1. **What was actually inspected**

   MemoHarness adaptation returned no additional rules. Native gograph initialization succeeded for both repositories. Definitions, references, bidirectional call queries, diff risk, source searches, and smell/complexity queries were attempted. No graphify or Python was executed.

   Current Strata HEAD is `a04606c1`, whose subject identifies Pass 9.2. It has existing uncommitted engine changes matching the Pass 10 mechanisms described in AGENTS.md. The reported Pass 9.1 binary is therefore a separate baseline; the current source tree cannot simply be called Pass 9.1. The older performance-path document's `e5fd0b14` observation is stale for this checkout.

   Gufo HEAD is `78d741856d0063825482660e1b1081184419c7a8`, release 0.26.0. Its working tree had graph-cache directories but no tracked source changes when checked.

   Important navigation limit: this gograph server exposes `call_graph`, `review_context`, and `inspect_quality` rather than the older individual tool names. It does not expose `read_file_at_lines`. Gufo's `WmmaExperts` was searchable in source but absent from its symbol graph; references for `RoutedF16Gemm` were empty despite visible calls. Strata's `.cu`/`.cuh` searches returned no matches and a reparse did not supply kernel symbols. Some C++ calls were attributed only to a file at line zero. Global quality results mostly ranked unrelated files. These results do not establish a complete HIP/C++ blast-radius or complexity audit.

   Findings below rely on native indexed source excerpts, the existing working diff, the supplied Pass 9/10 record, and revision-pinned upstream documentation. Before code edits, obtain complete bounded kernel reads and repair or supplement native graph coverage without treating missing graph edges as proof of no callers. A broad source-excerpt request was stopped when it failed to return promptly; it is not evidence.

2. **Why Gufo's prefill is fast**

   Its advantage comes from specialized execution across the whole prefill pipeline. Merely enabling FP16 or changing a flag does not reproduce that pipeline.

   **Compact routed experts.** `executor.cpp` builds expert-specific work from actual routing counts and compacts assignments into buckets padded to 16 rows. Its routed tile policy includes 16- and 48-row tiles, with a paired 128-row gate/up route and 64-row down route for suitable work. This reuses weight tiles across tokens assigned to the same expert and limits padding work. Source anchors: [routing policy and dispatch](../../gufo/src/models/qwen38_flash_next/kernels/rocm/executor.cpp), lines 125-141, 1790-1838, and 2795-2905.

   **Weight dequantization inside matrix-core staging.** The routed kernel stages quantized weights and converts them to FP16 near their consumption; FP32 accumulators span K. This avoids a separate full-model FP16 copy and reduces blockwise arithmetic inside the multiply loop. Source anchors: [routed kernel](../../gufo/src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp), lines 5035-5065 and 6314-6418. The selected source route, rather than every generic quantization header in the tree, determines format support.

   **Dense projections have their own shape policy.** Gufo uses Q8/int8 matrix paths and a custom FP16 WMMA path selected by matrix geometry. Its FP16 route is restricted to Q8_0 weights and relevant dimensions; it is not a generic fast path for every quantized dense tensor. Source anchors: [dense dispatch](../../gufo/src/models/qwen38_flash_next/kernels/rocm/executor.cpp), lines 114-123 and 1680-1735; [dense kernels](../../gufo/src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp), lines 6497-6505 and 7204-7249. The existence of a hipBLASLt wrapper alone does not prove that a measured call uses it.

   **Fused producers and consumers.** Wide paths combine projection/postprocessing work, retain useful FP16 intermediates, and avoid an FP32 mix where the next consumers only need the half representation. Strata's current dirty tree already contains a related optimization, so it must not be proposed as wholly new. Source anchor: [half-only mix decision](../../gufo/src/models/qwen38_flash_next/kernels/rocm/executor.cpp), lines 3308-3315.

   **Model-specific recurrence and sparse attention.** The production source has a row-split GDN recurrence and WMMA causal attention. These reduce work within this hybrid model; generic dense-transformer explanations miss those components. Source anchors: [GDN](../../gufo/src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp), lines 1632-1645; [attention selection](../../gufo/src/models/qwen38_flash_next/kernels/rocm/executor.cpp), lines 2604-2647.

   **Overlap is platform dependent.** Gufo retains side-stream work for shared experts and GDN Z projections. Strata already contains both mechanisms and deliberately disables them on Windows. Its [architecture defaults](../src/core/arch_defaults.cpp), lines 60-65, distinguish WDDM from non-Windows execution. Enabling Gufo's Linux-oriented schedule wholesale would undo an existing Windows decision.

   Gufo's [published benchmark](https://github.com/gufo-org/gufo/blob/78d74185/docs/models/qwen3.8-flash-next/BENCHMARKS.md) reports 1,801.01 tokens/s AR prefill and 1,734.96 with predictor catch-up at about pp2048, depth zero, on gfx1151. It used UD-Q4_K_XL, shared-Q8 MTP, greedy sampling, and thinking off; the measured revision was `fec4759d`. Strata's supplied Pass 9.1 record uses UD-IQ4_XS, medium reasoning, spec 7, and different rendered prompts. These figures do not establish a matched engine speed ratio.

   Gufo's [experiment record](https://github.com/gufo-org/gufo/blob/78d74185/docs/models/qwen3.8-flash-next/EXPERIMENTS.md) helps rank work: one historical pp2048 profile spent roughly 388 ms in routed experts, 350 in dense FP16 GEMMs, and 133 in HC combines. These are Gufo measurements, not a Strata profile. The same record says chunked FP16 DeltaNet and BF16 residual compression were removed after quality regressions. Its October 27 retained row-split configuration measured about 1,853-1,855 pp2048 tokens/s in a separate ROCm 10.1 experiment. Do not mix that experiment with the HTTP benchmark table or assume its Windows applicability.

3. **Can Gufo run this UD-IQ4_XS model?**

   **Current answer: the inspected Flash-Next production route is incompatible; a port is feasible in principle but not yet qualified.** Gufo already implements the model architecture. Its [model contract](https://github.com/gufo-org/gufo/blob/78d74185/docs/models/qwen3.8-flash-next/README.md) qualifies UD-Q4_K_XL, not this IQ package.

   The package name is not a uniform per-tensor type. Strata explicitly identifies IQ3_S gate/up and IQ4_NL down expert pairs in [the expert benchmark](../src/prefill/mmq_expert_bench.cpp), lines 7 and 118-119, and [architecture defaults](../src/core/arch_defaults.cpp), line 53. Other source comments identify Q8_0 down exceptions. A complete inventory of the actual GGUF shards is still required; those examples are not an exhaustive tensor manifest.

   Confirmed Gufo boundaries:

   - [Weight binding](../../gufo/src/models/qwen38_flash_next/weights.cpp), lines 19-38 and 141-144: format sizing and expert allowlists cover the existing K-quant/Q5_1/Q8 paths; they do not provide the IQ3_S/IQ4_NL expert combination.
   - [Device model](../../gufo/src/models/qwen38_flash_next/kernels/rocm/device_model.cpp), lines 360-366: even Q6_K accepted by a CPU-facing binder is explicitly rejected by production HIP. Loader acceptance is insufficient evidence of execution support.
   - [Fast expert selection](../../gufo/src/models/qwen38_flash_next/kernels/rocm/executor.cpp), lines 2795-2804: WMMA selection requires Q4_K/Q5_K gate weights and Q5_1/Q8_0 down weights, with matching gate/up formats.
   - [MMQ dispatch](../../gufo/src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.hip.cpp), lines 361-362 and 725-733: production selection and explicit instantiations are restricted. IQ definitions in imported headers do not make them reachable.
   - [Decode dispatch](../../gufo/src/models/qwen38_flash_next/kernels/rocm/mmq/mmvq.hip.cpp), lines 230-245 and 797-890: narrow/decode and grouped expert routes also require work. Prefill-only loading would leave normal generation and MTP verification incomplete.
   - Dense/embedding/output/PLE tensors and predictor tensors need separate auditing. Gufo's Q8-specific dense route cannot be assumed to accelerate the IQ3_S dense tensors present in Strata's supported family.

   A proper Gufo port must proceed in this order:

   1. Inventory every target and predictor tensor: name, role, GGML type, dimensions, row/block bytes, shard, and alignment. Check tokenizer, metadata, tied weights, and predictor compatibility.
   2. Add exact format sizing, binding, CPU reference decoding, and device-storage support only for observed formats. Reject remaining unsupported tensors with their names and roles.
   3. Implement correctness-first GPU GEMV, routed MMQ, embedding/head, and other required paths. Pass narrow decode and small-batch verification before promoting a prefill fast path.
   4. Add tile-local IQ3_S/IQ4_NL and any required IQ4_XS decoders to Gufo's routed FP16 WMMA kernels. Preserve codebooks, signs, scales, row strides, tails, and accumulation contracts. Audit format assumptions in fused gate/up and down epilogues.
   5. Extend dense fast paths only where the manifest and profile justify it. Reuse existing exact decoders where possible, but do not transplant a whole unrelated model backend.
   6. Qualify AR first, then predictor catch-up, MTP verification/rollback, prefix reuse, cancellation, restore, and tools. Use the original UD-Q4_K_XL model as a regression control.

   Requantization from IQ to K-quants would change the requested model. Expanding all 177B parameters to FP16 would take roughly 354 GB in decimal units before runtime state, beyond 128 GB UMA. Neither is the proposed solution. Small transient dequantized tiles are compatible with keeping the original quantized weights.

4. **What Pass 9 already has, and what the current tree adds**

   The supplied Pass 9.1 record already enables persistent FP16 WMMA MoE, FP16 down-output storage, one-pass prompt processing, a low prefill-stream threshold, and chunked GDN. Its measured ladder reaches about 743 tokens/s at 408 prompt tokens, 1,232 at 2,098, and 1,497 at 17,612. These are reported historical measurements, not new results from this assessment.

   Source-visible dispatch supports that general structure: [prefill.cpp](../src/prefill/prefill.cpp), lines 3755 and 3833/3868, selects FP16 Dm and calls native experts. The complete `.cu` kernel implementation remains subject to the graph/read limitation above.

   Existing uncommitted Pass 10 candidates include:

   - FP16 activation quantization from strided `mixed_h` and conditional omission of unused FP32 mix writes: `prefill.cpp`, lines 3242-3274 and 3800-3804; changes in `kernels.cu` and `moe_fused_iq.cu`.
   - `ActStage` addressing/register changes in `moe_f16_common.cuh`.
   - PLE copy-stream overlap and buffer-lifetime events in `prefill.cpp`, around lines 2759-2798 and 3151-3159.
   - MTP graph/startup and post-token scheduling changes in other dirty files. These have different objectives from target prefill throughput.

   The supplied Pass 10 ladder is not uniformly faster than Pass 9.1: for example, 2K prefill is reported higher while 512 and 16K are lower. Prompt lengths and speculative acceptance also differ. That makes isolated, matched candidate qualification necessary; a blanket Pass 10 speedup claim is unsupported.

5. **Proposed Strata work, in priority order**

   **P0 — qualify the control and find the actual bottleneck.** Identify the Pass 9.1 binary/runtime/config/model and a source state that reproduces it. Preserve the dirty tree. Do not assume HEAD reconstructs the champion. Record actual dispatch, prompt/new/reused tokens, target prefill wall time, predictor catch-up, PLE work, and useful TTFT separately. Measure the full prompt phase including overlap; summing kernel durations across streams can overcount wall time.

   First run a literal XML/tool prompt and recurrence/reference checks with chunked GDN on and off in separate processes. Strata's defaults cite Gufo #758 for this mechanism; Gufo subsequently removed its implementation. That connection justifies a test, but does not prove Strata has the same defect. Any failure enters the required issue-to-prompt triage before edits.

   **P1 — qualify existing Pass 10 memory reductions individually.** Test FP16-source quantization, dead FP32 mix writes, and ActStage changes as separate candidates, then test their combination. Compare quantized activation bytes and outputs against Pass 9.1 on real, padded, and tail rows. Omitting an unread write can be exact; quantizing an already-rounded FP16 value instead of an FP32 value is a distinct numerical question and must not be declared exact from its name. Preserve debug, fallback, peer, and alternate backend consumers.

   **P2 — improve dense and HC work at actual IQ model shapes.** Profile these alongside MoE. Establish whether current calls use custom WMMA, MMQ, rocBLAS, or hipBLASLt, including library identity and selected solutions. Evaluate narrowly selected fused projections/epilogues and shape dispatch using the existing native projection contract. Gufo's Q8_0 fast path provides design ideas; it cannot be pasted over IQ3_S tensors. Avoid creating a large persistent dequantized weight copy without a measured UMA budget.

   **P3 — optimize persistent expert tiling from observed routing.** Collect per-expert counts, live and padded rows, tile count, achieved bandwidth, LDS use, register spills, and queue utilization. Test at most one narrow persistent specialization and the existing wide specialization initially. Keep the persistent queue and expert staging model. An average `tokens * 10 / 512` estimate is not a utilization measurement; routing skew and tails matter. Existing records reject the nonpersistent variable-BM candidate, so do not repeat it without a new explanation and profile.

   **P4 — tune long-prompt chunk size if memory pressure or padding warrants it.** Keep the Pass 9.1 single-chunk setting as control. On one fixed 16K prompt, compare only two selected capacities, for example 4096 and 18432, in isolated runs. Smaller chunks may reduce scratch/traffic pressure but reread weights and add launches. Check full token/state continuity and prefix reuse. Retain smaller chunks only after a complete prompt-time win.

   **P5 — recurrence, attention, and scheduling only when their profile warrants it.** Prefer exact scheduling/layout improvements before changing recurrence arithmetic. Preserve Windows single-stream defaults; reconsider side streams only with direct WDDM pass-span evidence. For deep cached contexts, profile QSA selection/attention separately from zero-depth prefill. Do not reduce reasoning effort, selected attention work, or residual precision to obtain a nominal prefill gain.

   First-token emission deferral, startup graph capture, and speculative draft policy remain separate latency/decode work. They may help overall experience, but they do not establish faster target prompt processing.

6. **Execution plan using subagents after review**

   These are proposed work packages, not subagent definitions or invocations. Expand each approved package into the required 200-line structured prompt before dispatch. For native Gemini subagents, explicitly use `Model: "flash"`, canonical `gemini-3.8-flash`; never inherit the local model. If that native route is unavailable, report it rather than silently substituting an inherited local worker.

   The coordinator owns integration, source identity, baseline protection, and GPU scheduling. Use at most three workers concurrently. GPU experiments run serially, regardless of how many workers are active. Inspect repository-local rules before assigning any Gufo writes. Use isolated worktrees/staged builds and keep the production launcher and champion binary untouched.

   **Agent A — baseline, manifest, and measurements.** Initially read-only. Owns the target/predictor format inventory, benchmark protocol, source/binary/runtime identity, and control qualification. After implementation approval, it may add Go 1.26 benchmark/report tooling in a narrowly assigned location under `docs-strix-go/`. It may not change kernels or promote a binary. Return the actual tensor manifest, profile ranking, control state/quality results, and a bounded experiment recommendation. Dependencies: none; its manifest gates Gufo support and its profile gates optimization selection.

   **Agent B — one Strata prefill candidate at a time.** Owns only the approved subset of `src/prefill/prefill.cpp`, `kernels.cu`, `moe_fused_iq.cu`, `moe_f16_common.cuh`, and necessary declaration/tests. Start with qualification of existing Pass 10 changes; do not overwrite them with a clean-branch implementation. Return consumer/caller maps, exact numerical contract, focused test evidence, compiler/register evidence where relevant, and matched timing. Dense and expert candidates touching the same files are sequential assignments. No decode, launcher, or production-binary changes.

   **Agent C — Gufo compatibility prototype, optional separate branch.** Begins with A's manifest and a full graph/source trace. Owns only Flash-Next weight binding, required CPU/device format plumbing, GPU dispatch, and focused tests. Deliver a correctness-first AR milestone before IQ WMMA optimization. No global loosening of format validation, no model conversion, and no performance claim from loader success. Start actual Gufo implementation only if this optional track is selected; compatibility research can proceed alongside Strata work without consuming the GPU.

   **Agent D — independent reviewer, replacing a completed worker.** Owns no implementation files. Audits A/B/C evidence, verifies disabled-path compatibility and backend coverage, and reruns only meaningful affected checks. Separates deterministic tests, runtime correctness, measured speed, and deployment status. Rejects unmatched speed comparisons and missing MTP/cache cases. The coordinator integrates only reviewed candidates with an explicit control comparison.

   Every invoked worker must return the AGENTS.md JSON contract: task ID and status, files changed with before/after identities, exact commands and exit codes/output tails, tests, acceptance evidence, deviations, and blockers. Compute artifact identity through an approved non-exploratory mechanism; do not use forbidden shell hash commands.

7. **Verification and experiment budget**

   Resolve actual build directories, presets, model paths, and test registration during P0. The angle-bracket paths below are placeholders, not commands claimed to have been run.

   - Build a staged HIP candidate with `cmake --build <isolated-hip-build> --parallel 2`. Build CUDA/SYCL too wherever shared files affect them; unavailable backend coverage remains pending before review/promotion.
   - Run the registered HIP parity set with `ctest --test-dir <isolated-hip-build> --output-on-failure -R '^hip_prefill_fused_(moe|iq|iq_f16)$'`. Add the registered GDN/dense checks relevant to the selected change after verifying their names.
   - From `docs-strix-go/`, run `go test -p 1 -v ./...` and build the tracker when that module changes. Its configuration/static guards support runtime checks; they do not substitute for GPU execution.
   - For Gufo, confirm the existing `windows-release` preset and use `cmake --build --preset windows-release --parallel 2`, then its registered CTest operator/route checks. Exercise `gufo bench --validate-prefill <N>` using the real model and explicit options; a comparison to its sequential path is supplemented by independent arithmetic checks.
   - Use 1, 7, 8, 9, 15, 16, 17, threshold-adjacent, and nonmultiple tile rows in focused operator tests. Include sparse/skewed routing, empty experts, maximum occupancy, padded activation strides, and mixed-format exceptions. Do not require a full-model run at every operator size.
   - Same-arithmetic layout/scheduling candidates require byte-identical relevant outputs and target state. FP16/quantization candidates require explicit, predeclared tolerances and tool/semantic checks in addition to sequential/reference logits. Do not compare an FP16 candidate only against an int8 approximation and call that exactness.
   - End-to-end cases cover literal/tool text, code, nonrepetitive prose, long-to-short reset, prefix reuse, cancellation/resume, and MTP rollback/continuation. Check final recurrent/KV state as well as output where the changed path can affect them.

   Initial timing screen: one candidate against one control, C1, zero reused prefix, approximately 512 and 2048 actual input tokens, code and prose. Use one excluded warmup per arm/shape and one A-B-B-A block per workload: 16 timed requests, 128 generated tokens, explicit medium reasoning and identical sampling. Time target prefill directly as the primary metric; also retain predictor catch-up, useful TTFT, output length, completion time, and decode acceptance. Limit timed inference to 15 minutes excluding loading; stop and report partial evidence at the limit. Profile captures are separate from retained uninstrumented timing runs.

   Use raw identical input token IDs where possible to isolate engine compute, then a matched HTTP/task check. Different chat renderings are a system comparison and must be labeled. Record clocks/temperature, residency, free UMA, DLL/toolchain, actual dispatch, process settings, and output/state checks. Run engines sequentially if safe memory headroom does not permit coexistence; repeat the control after the swap. Never run competing GPU tests in parallel.

   Advance a prefill candidate only after a repeated reduction of at least 5% in median target prompt time on its intended workload, with correctness passing and no material completion/decode regression. This is a proposed screening threshold. Smaller results may be useful but need a stronger noise estimate before adoption. Four observations do not establish tail latency. Add a 16K or deep-context case only for the candidate whose mechanism requires it; do not launch a full model/context/speculation matrix.

   Illustrative targets, not forecasts: 10% higher prefill throughput at the supplied 16K rate would move approximately 1,497 to 1,646 tokens/s and save about 1.07 s of pure prompt compute over 17,612 tokens. A 20% throughput gain would approach 1,796 tokens/s and save about 1.96 s. End-to-end TTFT also includes other work, so these are not TTFT guarantees.

8. **Review choices and completion conditions**

   Recommended first implementation scope: P0 followed by P1 and the single highest-value P2/P3 candidate supported by the Strata profile. Keep the Gufo port as an optional separately reviewed track. This gives the existing IQ model the shortest route to a measurable improvement and avoids paying for a second engine port before its benefit is known.

   A Strata candidate is complete only with a reproducible control, focused correctness/state checks, relevant backend builds, independent review, and retained matched timing artifacts. A Gufo port is complete only when the entire actual format manifest executes correctly through AR and MTP/cache paths, with existing Q4_K_XL behavior protected. Neither track is complete merely because it loads, compiles, or improves a microbenchmark.

   No production adoption is included in this planning turn. Current statuses: static dispatch/compatibility findings supported; complete HIP AST audit incomplete; original GGUF manifest not yet inspected; same-model engine winner not verified; new speed gains not verified; deployment unchanged.

   Related prior assessment: [STRATA_GUFO_PERFORMANCE_PATH_2026-10-29.md](STRATA_GUFO_PERFORMANCE_PATH_2026-10-29.md). This plan narrows the emphasis to actual prefill and adds fresh local source/compatibility findings; it does not overwrite that document or its unrelated latency/decode proposals.
