# Strata and Gufo: recommended performance path

Assessment date: October 29, 2026. Target: Windows Strix Halo gfx1151, 128 GB UMA, Qwen3.8 Flash-Next 177B, single-user interactive use with medium reasoning. Treat concurrency as a separate objective.

**Recommendation:** retain the reported Strata Pass 9.1 champion as the control. First qualify its arithmetic and state behavior, then separate first-token delivery improvements from sustained decode optimization. Evaluate Gufo as a matched challenger before investing in a migration. Existing numbers establish no comparable engine winner.

**Evidence boundary**

- MemoHarness adaptation completed with no additional rules returned.
- The required native gograph initialization failed with `Transport closed`. Native review-context, call-graph, references, and quality queries also failed. No successful AST or source blast-radius audit was obtained; implementation locations below come from documentation and require source confirmation before edits.
- Read bounded spans of the supplied walkthrough, repository documentation, and the production launcher JSON using a temporary Go reader. No Python, graphify, directory sweep, GPU benchmark, service restart, or production change was performed.
- Observed Git state: Strata HEAD `e5fd0b14`; its branch changed from `pass7/gfx1151-v0145-opt` to `main` during this review, with the same observed HEAD. This task performed no checkout. Gufo is `78d74185`, release 0.26.0, branch `main`. Strata has existing untracked documentation/tooling and graph caches; Gufo has an untracked graph cache. Preserve them.
- Fresh static check: `docs-strix-go/strata-tracker.exe audit-config` exited 0 with **39 PASS / 0 WARN / 0 FAIL**. This checks configuration and supporting files; it does not certify GPU dispatch, loaded DLL identity, numerical quality, current benchmark speed, or correspondence between production binary and source commit.
- The walkthrough reports restoration of binary identity `a621538b75e570888cf11d763391d82a224dd540f537a43d1e281b30466ac491`. That identity was supplied by the document, not recalculated or independently verified here.

**What the measurements support**

The [supplied walkthrough](C:/Users/Butch/.gemini/antigravity/brain/9b4d0dbe-556d-4a02-95b5-741c87e814f6/walkthrough.md) reports the Pass 9.1 IQ4_XS control at approximately 743-1,497 prompt tokens/s and 82-91 client decode tokens/s over its context ladder, with medium reasoning, generation length 256, spec 7, and 73-84% draft acceptance. These are historical reported results, not fresh measurements.

Gufo's [benchmark record](https://github.com/gufo-org/gufo/blob/78d74185/docs/models/qwen3.8-flash-next/BENCHMARKS.md) uses UD-Q4_K_XL, shared-Q8 MTP, greedy sampling, thinking off, approximately pp2048/tg128, and an older measured revision (`fec4759d`). At depth zero it reports:

- AR prefill: 1,801.01 tokens/s; MTP prefill including predictor catch-up: 1,734.96 tokens/s.
- MTP decode: 35.79 tokens/s on mixed text, 116.93 on repetitive text. The latter is not a general-purpose decode rate.
- Eight-request repetitive decode: 245.81 as the sum of individual request decode rates. It is neither single-user speed nor necessarily completed tokens divided by a common batch wall interval.
- AR peak HIP memory: 88.78 GiB in its stated pp2048/tg128 test; this is not an MTP memory measurement or proof that the existing second model can remain co-resident.

Different weights, predictor precision, reasoning, token histories, generation length, runtime, and timing scopes prevent a valid direct speed ratio. The local [Gufo dossier](GUFO.md) contains both the 1,801 headline and a separate assertion that Strata is twice as fast; neither comparison establishes a matched winner. Its 0.25.0 status also predates the local 0.26.0 checkout.

**Resolve these evidence conflicts first**

1. [ARCHITECTURE.md](../ARCHITECTURE.md) still forbids `STRATA_PF_F16=1` and describes Pass 7. The newer walkthrough, AGENTS.md, launcher JSON, and fresh audit instead use FP16 MoE prefill. Treat kernel version, persistent scheduling, and measured build identity as part of every tuning rule. A historical flag name alone does not identify an execution path.
2. The launcher currently enables `STRATA_GDN_CHUNKED=1`, minimum 128. Gufo's [retained experiment record](https://github.com/gufo-org/gufo/blob/78d74185/docs/models/qwen3.8-flash-next/EXPERIMENTS.md) documents removing its F16 chunked DeltaNet after a 299-token tool/XML prompt changed a literal echo into a greeting. This is evidence about Gufo's implementation, not proof of the same bug in Strata. Compare Strata's actual recurrence and tool behavior against its sequential path and an independent arithmetic reference before declaring the champion quality-qualified. A configuration audit or a thinking-mode check alone cannot establish this.
3. Candidate 3 bundled reset, FP16 drafting, snapshot, and parking changes. Its reported 19 ms TTFT improvement cannot be attributed entirely to parking. Separate those mechanisms; its degraded acceptance invalidates a blanket claim of zero decode regression.
4. The walkthrough attributes historical Windows dense-GEMM regressions to an unavailable hipBLASLt state. The older mining report attributes Pass 6 speed to active tuned hipBLASLt dispatch. Different builds may explain this, but current handle initialization and route selection remain unverified. A valid tuning file and `ROCBLAS_USE_HIPBLASLT=1` do not prove that the measured calls use its solutions.

**Priority 1: lower visible latency with the smallest arithmetic risk**

Candidate: emit the first already-committed target token before `mtp.draft_first`, following the walkthrough's Mod A. Trace the complete engine IPC, Go server, and HTTP/SSE flush path before editing `src/program/generate.cpp`.

- Confirm this is a committed target sample, not an unverified proposal. Preserve RNG consumption, token framing, reasoning/content routing, UTF-8 assembly, cancellation, and subsequent draft/verification order.
- Measure first byte, first nonempty text/reasoning delta, second token, maximum inter-token gap, and last token. A role-only or empty SSE event does not count as a delivered token.
- The walkthrough estimates 95-125 ms of earlier delivery. This is an unverified opportunity. It moves draft work after token one; it does not by itself remove GPU work or shorten completion time, and may expose a token-one-to-token-two pause.
- Keep the engine timing scope fixed. A decode metric that starts at the earlier first token can fall even when total compute is unchanged.

This is the first small implementation candidate after source navigation recovers. Success means earlier useful output with acceptable stream smoothness and unchanged total work/quality, rather than a claimed throughput increase.

**Priority 2: improve sustained decode by reducing speculative cost per committed token**

Use Gufo as a source of measured mechanisms: adaptive draft length, exact target verification, conservative vocabulary fallback, and gated context copying. Its [model contract](https://github.com/gufo-org/gufo/blob/78d74185/docs/models/qwen3.8-flash-next/README.md) describes these behaviors. Strata already configures a reduced draft vocabulary, device drafting, an 8192-token MTP window, and lookup chain 3; do not count those as new opportunities without showing a missing capability.

- Profile draft body/head, target verification, rollback, CPU sampling, and PLE separately. Optimize the largest measured contribution.
- Compare fixed widths 3, 5, and 7 only on one affected workload before designing an adaptive controller. Include short output budgets and low-acceptance prose. Preserve exact p/q acceptance and residual correction; sampled policy decisions must retain the required seeded-replay contract.
- Rank by committed tokens per complete draft/verify/rollback cycle, not acceptance percentage alone. Record lookup contributions separately because changing width or copying changes the denominator.
- Retain the control's draft arithmetic. The walkthrough reports regressions from FP16 draft-KV catch-up; neither faster isolated GEMM nor higher acceptance on one repetitive prompt establishes a win.
- Investigate draft-KV projection reuse only after proving identical input, scale/rounding, lifetime, and stream dependencies. The walkthrough's `reuse_xq` claim remains source- and parity-unverified here.

Illustrative arithmetic, not a forecast: at 88 tokens/s, a genuine 10% decode-rate improvement saves about 263 ms across 255 post-first-token intervals. For a long answer this can matter more than a 100 ms improvement to first-token delivery alone.

**Priority 3: reduce actual prefill time for long uncached prompts**

For the reported 16K tier, approximately 11.879 s of TTFT dominates a 256-token completion. A genuine 10% reduction of that whole pre-token interval would save about 1.188 s; 10% faster decode saves about 272 ms. These estimates assume unchanged work and output length.

- First profile real routed-expert token counts and padding in the persistent FP16 path. A persistent BM32 specialization for short prompts is worth a bounded experiment only if padding and occupancy explain wall time. Do not infer utilization from a uniform average over experts; verify expert count, top-k, and routing skew.
- Keep the persistent work queue and launch count when testing smaller tiles. The walkthrough's nonpersistent variable-BM candidate reportedly lost about 41 ms; smaller tiles are not automatically faster when register, LDS, scheduling, and dequantization costs change.
- For dense projections, prove the current Windows hipBLASLt route and benchmark actual shapes, including awkward tails. Then compare custom WMMA with matched library solutions. Do not disable the fast route globally by raising thresholds before a valid fallback is demonstrated.
- AMD now [documents hipBLASLt distribution on Windows](https://rocmdocs.amd.com/projects/hipBLASLt/en/latest/install/install.html), but that does not establish compatibility or performance of the local DLL, tuning indices, or model shapes. Treat the runtime, library, and tuning database as a matched set.
- Profile at real cache traffic and production allocations. Isolated matrices that fit in cache can misstate full-model bandwidth and dequantization cost.

For interactive multi-turn use, validate existing prefix reuse before optimizing uncached prefill. Keep cold new-prompt work separate from reused-prefix accounting. Report new tokens, reused tokens, and restore time, rather than calling reused-token throughput GPU prefill speed.

**Priority 4: reset and snapshot traffic, after state correctness is established**

The walkthrough's Mods B/C may remove real memory traffic, but have a larger state-safety surface than emitting a committed token.

- Replace a proposed `live_peak_len + spec + 16` bound only after proving the written extent of every affected buffer: target and draft KV, QSA pooled indices, warmup, graph replay, rejected proposals, cancellation, snapshot restore, and capacity growth. Initial and grown allocations must be initialized before any reader can observe them. Token position and pooled-buffer indices may require different bounds.
- Exercise long request -> short reset, rejected draft tails, session switching, warmup -> first request, cancellation, and restore -> continuation. Compare with full clearing and independent numerical/semantic checks.
- For parking moved after DONE, measure immediate next-request admission and engine-ready time as well as client completion. Post-DONE work may simply become the next request's queue delay.
- For batched copies, prove stream-event ordering and host-buffer lifetime before reducing synchronization.
- Keep reset, parking, copy batching, and draft quantization in separate candidates. Their benefits cannot be safely added from the bundled retrospective.

**When Gufo becomes the better investment**

Choose Gufo if matched Windows trials show better completed-task latency/quality or if concurrent serving is the real objective. Its batch planner and shared-work verification merit evaluation for C2/C4, after C1. Preserve per-request latency and compute a common-wall aggregate; do not compare a summed-rate C8 table with Strata C1.

Its [Windows build contract](https://github.com/gufo-org/gufo/blob/78d74185/BUILD-WINDOWS.md) identifies TheRock 10.0.0 as the validated fast SDK and reports severe stalls with later tested Windows SDKs. Its [ROCm 10.1 measurements](https://github.com/gufo-org/gufo/blob/78d74185/docs/models/qwen3.8-flash-next/ROCM.md) are a separate Linux/container configuration. A Linux result does not qualify a Windows upgrade. A Linux trial is an optional later platform experiment, not a prerequisite for the first Strata improvements.

Gufo's documented target is UD-Q4_K_XL. Before an engine-only A/B, establish a target quantization and model revision both engines support; do not assume IQ4_XS support in Gufo. If no common model is available locally, report a system-choice comparison including model size and quality, with the quantization difference explicit. Measure total UMA/host headroom with the second server present; neither a smaller expert-only allocation nor Gufo's AR-only HIP peak proves safe coexistence.

**Bounded validation sequence**

1. Recover native gograph access without repeating a full cold parse of the external optimization tree. Confirm definitions, callers, callees, references, risk, and complexity for only the selected mechanism. No engine implementation is approved by this documentary assessment.
2. Establish source/binary/runtime/model/config identity and retain an unchanged control. Save benchmark settings and actual routes. Confirm GDN/tool quality, multi-turn thinking, and cache behavior before performance comparisons.
3. Initial engine screen: C1, two prompt sizes (about 512 and 2048 new tokens), code and nonrepetitive prose, 128 outputs, explicit medium reasoning and identical sampling. Use one excluded warmup per shape and one A-B-B-A block: 16 timed requests total. Set a 15-minute inference budget excluding loading; stop at the budget and retain partial evidence. No C8 or full context ladder at this stage.
4. Use identical weights, rendered tokens, output budget, cache history, toolchain/runtime where possible, and co-resident-model conditions. Keep unrelated GPU inference idle. If two full models cannot coexist safely, run engines sequentially with a control repeated after the swap; do not load both just to alternate requests.
5. Record prompt/new/reused tokens, TTFT, first-to-second gap, maximum inter-token gap, full completion time, engine prompt time, decode timing scope, draft/accepted/lookup counts, actual output length, memory headroom, and errors. Include a literal tool/XML case and a small multi-turn continuation check.
6. For an implementation candidate, change one mechanism. First run its focused correctness/state check, then the affected A-B-B-A timing control. Add a 16K prefill or 32K cached-depth case only when the profile or affected behavior justifies it. Profile separately from retained uninstrumented timings.
7. Predeclare a useful win: for a throughput candidate, at least 5% lower median completion time in the target workload, without correctness failures or a consistent control slowdown. This is a proposed screening threshold, not an established repository gate. First-token candidates instead target useful TTFT and smoothness with unchanged completion work. Small/noisy screens remain inconclusive; expand only affected cases. Four observations do not establish tail-latency statistics.
8. Promote only after numerical/semantic and state gates pass and a repeated matched timing result survives. For changed proposal algorithms, distinguish greedy parity, sampled distribution correctness, and seeded replay; equal sampled output across different engines is not an engine-equivalence requirement. Do not change reasoning effort or reduce precision merely to make a speed claim.

Applied method: [Gufo optimize-kernel skill](../../gufo/.agents/skills/optimize-kernel/SKILL.md), specifically focused profiles, independent numerical checks, matched complete-request measurements, and visibly inconclusive noisy results. Its Python profiling tools were not executed.

**Current disposition:** Strata is the lower-disruption primary optimization path; Gufo is the matched challenger and source of targeted mechanisms. First-token emission is the first small latency candidate, draft/verification cost is the first sustained-decode target, and persistent prefill/dispatch work follows measured prompt cost. Source-level feasibility, live engine superiority, new speed gains, and production adoption are **not_verified**.
