package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

func main() {
	cmd := "help"
	if len(os.Args) > 1 {
		cmd = os.Args[1]
	}

	switch cmd {
	case "validate", "audit":
		runValidateCmd(os.Args[2:])
	case "audit-config":
		runAuditConfigCmd(os.Args[2:])
	case "audit-log":
		runAuditLogCmd(os.Args[2:])
	case "bench":
		runBenchCmd(os.Args[2:])
	case "ab-matrix":
		runABMatrixCmd(os.Args[2:])
	case "ctest-gate":
		runCTestGateCmd(os.Args[2:])
	case "strix":
		runStrixStatusCmd()
	case "switches":
		runSwitchesCmd()
	case "search":
		runSearchCmd(os.Args[2:])
	case "kernels":
		runKernelsCmd()
	case "check-upstream":
		runCheckUpstreamCmd()
	case "blueprint":
		runBlueprintCmd()
	case "server-info":
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		info, _ := QueryLiveServerInfo(ctx, DefaultServerURL, "C:\\Users\\Butch\\Documents\\10-Models\\Strata")
		raw, _ := json.MarshalIndent(info, "", "  ")
		fmt.Println(string(raw))
	case "restart-bench":
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
		defer cancel()
		if _, err := RestartStrataServer(ctx, DefaultServerURL, "C:\\Users\\Butch\\Documents\\10-Models\\Strata"); err != nil {
			fmt.Fprintf(os.Stderr, "restart error: %v\n", err)
			os.Exit(1)
		}
		runBenchCmd(os.Args[2:])
	case "restart":
		runRestartCmd(os.Args[2:])
	case "stop":
		runStopCmd(os.Args[2:])
	case "mem-status":
		runMemStatusCmd(os.Args[2:])
	case "verify-thinking":
		runVerifyThinkingCmd(os.Args[2:])
	default:
		printUsage()
	}
}

func printUsage() {
	fmt.Println("Strata Tracker & Qwen 3.8 Flash-Next 177B Validation Framework (Go 1.26)")
	fmt.Println("Usage:")
	fmt.Println("  strata-tracker validate        Run full 5-phase Qwen 3.8 177B validation (Config + Log + A/B Matrix + Gates)")
	fmt.Println("  strata-tracker audit-config    Validate strata-qwen177b.json against C++ AST & Strix Halo invariants")
	fmt.Println("  strata-tracker audit-log       Audit strata_engine.log (1.84M+ lines) for prefill/decode/MTP & AC-1..AC-5")
	fmt.Println("  strata-tracker bench           Run live Zero-Reuse Context Ladder HTTP/SSE benchmark (512..16384 tokens)")
	fmt.Println("  strata-tracker ab-matrix       Generate 7-arm A-B-B-A optimization matrix (v0.1.40.2 -> v0.1.45 + PR #2997)")
	fmt.Println("  strata-tracker ctest-gate      List CTest HIP/CUDA kernel numerical parity verification gates")
	fmt.Println("  strata-tracker strix           Inspect all Strix Halo (gfx1151) commits, files & configs")
	fmt.Println("  strata-tracker switches        List all 18 default, 11 active opt-in & 6 v0.1.45 candidate switches")
	fmt.Println("  strata-tracker search <terms>  Search commit history and repository for terms")
	fmt.Println("  strata-tracker kernels         Catalog all HIP / CUDA / SYCL kernels and hardware targets")
	fmt.Println("  strata-tracker check-upstream  Query GitHub API for latest releases and commits")
	fmt.Println("  strata-tracker blueprint       Display Go version strata.exe architecture blueprint")
	fmt.Println("  strata-tracker restart         Restart Strata server on Port 8080 (optional: -env \"K=V,...\")")
	fmt.Println("  strata-tracker stop            Stop strata-server.exe, strata.exe & strata-vision.exe")
	fmt.Println("  strata-tracker mem-status      Report Win32 host commit, RAM & target process commits")
	fmt.Println("  strata-tracker verify-thinking Verify thinking effort, KV cache reuse and Port 8080/8081 coexistence")
}

func runRestartCmd(args []string) {
	fs := flag.NewFlagSet("restart", flag.ExitOnError)
	serverURL := fs.String("url", DefaultServerURL, "Strata server URL")
	runtimeDir := fs.String("dir", "C:\\Users\\Butch\\Documents\\10-Models\\Strata", "Strata runtime directory")
	envFlag := fs.String("env", "", "Comma-separated KEY=VALUE environment overrides")
	_ = fs.Parse(args)

	var extraEnv []string
	if *envFlag != "" {
		for _, item := range strings.Split(*envFlag, ",") {
			item = strings.TrimSpace(item)
			if item != "" {
				extraEnv = append(extraEnv, item)
			}
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 6*time.Minute)
	defer cancel()

	dur, err := RestartStrataServer(ctx, *serverURL, *runtimeDir, extraEnv...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "restart error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Restart completed successfully in %.1fs\n", dur.Seconds())
}

func runValidateCmd(args []string) {
	fs := flag.NewFlagSet("validate", flag.ExitOnError)
	cfgPath := fs.String("config", DefaultProdConfigPath, "Path to strata-qwen177b.json")
	logPath := fs.String("log", DefaultEngineLogPath, "Path to strata_engine.log")
	repoRoot := fs.String("repo", DefaultRepoRoot, "Path to Strata repository root")
	jsonOut := fs.Bool("json", false, "Emit JSON report to stdout")
	outFile := fs.String("out", "", "Optional path to write JSON validation report")
	_ = fs.Parse(args)

	report, err := RunFullValidation(*cfgPath, *logPath, *repoRoot)
	if err != nil {
		fmt.Fprintf(os.Stderr, "validation error: %v\n", err)
		os.Exit(1)
	}

	if *outFile != "" {
		raw, _ := json.MarshalIndent(report, "", "  ")
		if err := os.WriteFile(*outFile, raw, 0644); err != nil {
			fmt.Fprintf(os.Stderr, "failed writing %s: %v\n", *outFile, err)
		}
	}

	if *jsonOut {
		raw, _ := json.MarshalIndent(report, "", "  ")
		fmt.Println(string(raw))
		return
	}

	printHumanValidationReport(report)
}

func runAuditConfigCmd(args []string) {
	fs := flag.NewFlagSet("audit-config", flag.ExitOnError)
	cfgPath := fs.String("config", DefaultProdConfigPath, "Path to strata-qwen177b.json")
	repoRoot := fs.String("repo", DefaultRepoRoot, "Path to Strata repository root")
	jsonOut := fs.Bool("json", false, "Emit JSON output")
	_ = fs.Parse(args)

	rep, err := AuditConfig(*cfgPath, *repoRoot)
	if err != nil {
		fmt.Fprintf(os.Stderr, "audit-config failed: %v\n", err)
		os.Exit(1)
	}
	if *jsonOut {
		raw, _ := json.MarshalIndent(rep, "", "  ")
		fmt.Println(string(raw))
		return
	}
	printConfigAuditSection(rep)
}

func runAuditLogCmd(args []string) {
	fs := flag.NewFlagSet("audit-log", flag.ExitOnError)
	logPath := fs.String("log", DefaultEngineLogPath, "Path to strata_engine.log")
	jsonOut := fs.Bool("json", false, "Emit JSON output")
	_ = fs.Parse(args)

	rep, err := AuditEngineLog(*logPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "audit-log failed: %v\n", err)
		os.Exit(1)
	}
	if *jsonOut {
		raw, _ := json.MarshalIndent(rep, "", "  ")
		fmt.Println(string(raw))
		return
	}
	printLogAuditSection(rep)
}

func runBenchCmd(args []string) {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	serverURL := fs.String("url", DefaultServerURL, "Strata HTTP server URL")
	logPath := fs.String("log", DefaultEngineLogPath, "Path to strata_engine.log for hardware metric correlation")
	contextsStr := fs.String("contexts", "512,1024,2048,4096,8192,16384", "Comma-separated context tiers")
	genLimit := fs.Int("gen", 256, "Max completion tokens per tier")
	temp := fs.Float64("temp", 0.0, "Sampling temperature")
	reasoning := fs.String("reasoning", "high", "Reasoning effort (high | medium | low; 'none' is strictly forbidden)")
	jsonOut := fs.Bool("json", false, "Emit JSON output")
	outFile := fs.String("out", "", "Optional path to write JSON benchmark report")
	_ = fs.Parse(args)

	validReasoning, err := ValidateReasoningEffort(*reasoning)
	if err != nil {
		fmt.Fprintf(os.Stderr, "benchmark flag error: %v\n", err)
		os.Exit(1)
	}

	var tiers []int
	for _, p := range strings.Split(*contextsStr, ",") {
		p = strings.TrimSpace(p)
		if v, err := strconv.Atoi(p); err == nil && v > 0 {
			tiers = append(tiers, v)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()

	rep, err := RunLiveBenchmark(ctx, *serverURL, *logPath, tiers, *genLimit, *temp, validReasoning)
	if err != nil {
		fmt.Fprintf(os.Stderr, "live benchmark error: %v\n", err)
		os.Exit(1)
	}

	if *outFile != "" {
		raw, _ := json.MarshalIndent(rep, "", "  ")
		_ = os.WriteFile(*outFile, raw, 0644)
	}

	if *jsonOut {
		raw, _ := json.MarshalIndent(rep, "", "  ")
		fmt.Println(string(raw))
		return
	}

	fmt.Printf("=== Live Context Ladder Benchmark (%s | reasoning=%s | temp=%.2f) ===\n", rep.ServerURL, rep.ReasoningEffort, rep.Temperature)
	for _, r := range rep.Results {
		fmt.Printf("Tier %5d tok | Prompt=%5d (Reused=%d) | Gen=%3d (R=%d, C=%d) | TTFT=%7.1f ms | ClientDec=%5.1f tok/s | EngPrefill=%6.1f tok/s | EngDec=%5.1f tok/s (%+.1f%% vs base %.1f) | Drafts=%d/%d (%.1f%%)\n",
			r.ContextTier, r.PromptTokens, r.ReusedTokens, r.GeneratedTokens, r.ReasoningTokens, r.ContentTokens,
			r.TTFTMs, r.ClientDecodeTps, r.EnginePrefillTps, r.EngineDecodeTps, r.DeltaDecodePct, r.BaselineDecodeTps,
			r.DraftsAccepted, r.DraftsTotal, r.DraftAcceptancePct)
	}
}

func runABMatrixCmd(args []string) {
	fs := flag.NewFlagSet("ab-matrix", flag.ExitOnError)
	repoRoot := fs.String("repo", DefaultRepoRoot, "Path to Strata repository root")
	baseCfgPath := fs.String("config", DefaultProdConfigPath, "Base strata-qwen177b.json path")
	writeProfiles := fs.String("write-profiles", "", "Optional directory to write ARM-00..ARM-06 candidate JSON configs")
	jsonOut := fs.Bool("json", false, "Emit JSON output")
	_ = fs.Parse(args)

	arms := BuildABMatrix(*repoRoot)
	if *writeProfiles != "" {
		if rawBase, err := os.ReadFile(*baseCfgPath); err == nil {
			_ = os.MkdirAll(*writeProfiles, 0755)
			for _, a := range arms {
				var cfg ProductionConfig
				if err := json.Unmarshal(rawBase, &cfg); err == nil {
					if cfg.Env == nil {
						cfg.Env = make(map[string]string)
					}
					for k, v := range a.EnvOverrides {
						cfg.Env[k] = v
					}
					outBytes, _ := json.MarshalIndent(cfg, "", "  ")
					_ = os.WriteFile(filepath.Join(*writeProfiles, a.ArmID+".json"), outBytes, 0644)
				}
			}
		}
	}
	if *jsonOut {
		raw, _ := json.MarshalIndent(arms, "", "  ")
		fmt.Println(string(raw))
		return
	}
	fmt.Println("=== Qwen 3.8 Flash-Next 177B A-B-B-A Optimization Matrix (v0.1.40.2 -> v0.1.45 + PR #2997) ===")
	for _, a := range arms {
		fmt.Printf("[%s] %s (%s)\n", a.ArmID, a.Name, a.GitRef)
		fmt.Printf("  Subsystem : %s\n", a.TargetSubsystem)
		fmt.Printf("  Env       : %v\n", a.EnvOverrides)
		fmt.Printf("  Gain      : %s\n", a.ExpectedGain)
		fmt.Printf("  Contract  : %s\n", a.NumericalContract)
		fmt.Printf("  Gate      : %s\n\n", a.AcceptCriteria)
	}
}

func runCTestGateCmd(args []string) {
	fs := flag.NewFlagSet("ctest-gate", flag.ExitOnError)
	jsonOut := fs.Bool("json", false, "Emit JSON output")
	_ = fs.Parse(args)

	gates := CTestGateSpecs()
	if *jsonOut {
		raw, _ := json.MarshalIndent(gates, "", "  ")
		fmt.Println(string(raw))
		return
	}
	fmt.Println("=== Mandatory CTest HIP / CUDA Kernel Parity Gates ===")
	for _, g := range gates {
		fmt.Printf("- %-30s | %-38s | %s (%s)\n", g.TargetName, g.SourceFile, g.Requirement, g.Subsystem)
	}
}

func printHumanValidationReport(rep *CombinedValidationReport) {
	fmt.Println("================================================================================")
	fmt.Println("  STRATA QWEN 3.8 FLASH-NEXT 177B — TESTING & VALIDATION FRAMEWORK REPORT")
	fmt.Printf("  Generated: %s\n", rep.GeneratedAt)
	fmt.Println("================================================================================")

	if rep.ConfigAudit != nil {
		printConfigAuditSection(rep.ConfigAudit)
	}
	if rep.LogAudit != nil {
		printLogAuditSection(rep.LogAudit)
	}
	fmt.Printf("\n[Phase 4: A-B-B-A Optimization Arms Ready: %d Arms Configured]\n", len(rep.ABMatrix))
	for _, arm := range rep.ABMatrix {
		fmt.Printf("  * %s: %s -> %s\n", arm.ArmID, arm.Name, arm.ExpectedGain)
	}
}

func printConfigAuditSection(c *ConfigAuditReport) {
	fmt.Printf("\n[Phase 1: Production Configuration Audit (%s)]\n", c.ConfigPath)
	fmt.Printf("  Model: %s | Backend: %s | Passed: %d | Warn: %d | Failed: %d\n",
		c.ModelName, c.Backend, c.PassedCount, c.WarnCount, c.FailedCount)
	fmt.Printf("  Sub-1024 Prefill Cliff Fixed (STRATA_PREFILL_STREAM_MIN<=128): %v\n", c.Sub1024CliffFixed)
	fmt.Printf("  hipBLASLt Tuning Table Valid: %v (version=%s, rows=%d)\n",
		c.TuningTableValid, c.TuningTableVersion, c.TuningTableRows)
	for _, ch := range c.Checks {
		fmt.Printf("  [%4s] %-6s %-34s : got=%s (want=%s)\n", ch.Status, ch.ID, ch.Name, ch.Actual, ch.Expected)
	}
}

func printLogAuditSection(l *LogAuditReport) {
	fmt.Printf("\n[Phase 2: Live Telemetry & Engine Log Audit (%s)]\n", l.LogPath)
	fmt.Printf("  Lines Scanned: %d | Completed Requests Audited: %d\n", l.TotalLinesScanned, l.TotalRequestsParsed)
	fmt.Printf("  Handshake: Engine=%s | Context=%d | KV=%s | ExpertSlots=%d/%d (%d MiB) | FreeVRAM=%d MiB | Spec=%d (MTP=%d, Lookup=%d)\n",
		l.Handshake.EngineVersion, l.Handshake.Context, l.Handshake.KVFormat,
		l.Handshake.ExpertSlots, TotalRoutedExperts, l.Handshake.ExpertCacheMiB,
		l.Handshake.VRAMFreeMiB, l.Handshake.SpecWindow, l.Handshake.MTPMax, l.Handshake.LookupChain)
	fmt.Printf("  Sustained Decode (gen >= 50, n=%d): P10=%.1f | P50(Med)=%.1f | Mean=%.1f | P90=%.1f | P99=%.1f | Peak=%.1f tok/s (L%d)\n",
		l.SustainedDecode.Count, l.SustainedDecode.P10, l.SustainedDecode.P50Median,
		l.SustainedDecode.Mean, l.SustainedDecode.P90, l.SustainedDecode.P99,
		l.SustainedDecode.Max, l.SustainedDecode.PeakLine)
	fmt.Printf("  Sustained Decode Thresholds: >=50 tok/s: %d/%d (%.1f%%) | >=60 tok/s: %d/%d (%.1f%%)\n",
		l.SustainedDecode.CountGe50, l.SustainedDecode.Count, l.SustainedDecode.PctGe50,
		l.SustainedDecode.CountGe60, l.SustainedDecode.Count, l.SustainedDecode.PctGe60)
	fmt.Printf("  Draft Acceptance (gen >= 50): Median=%.1f%% | Mean=%.1f%% | P90=%.1f%% | Max=%.1f%%\n",
		l.DraftAcceptancePct.P50Median, l.DraftAcceptancePct.Mean, l.DraftAcceptancePct.P90, l.DraftAcceptancePct.Max)

	fmt.Println("\n  Cold Prefill Throughput by Context Tier (Reused=0):")
	for _, t := range l.ColdPrefillTiers {
		cliffTag := ""
		if t.MMQCliffDetect {
			cliffTag = " [PRE-RESTART MMQ CLIFF DETECTED]"
		}
		fmt.Printf("    %-22s (n=%4d): Min=%6.1f | Med=%6.1f | Mean=%6.1f | P90=%6.1f | Max=%6.1f tok/s%s\n",
			t.TierName, t.SampleCount, t.MinTokPerS, t.MedianTokPerS, t.MeanTokPerS, t.P90TokPerS, t.MaxTokPerS, cliffTag)
	}

	fmt.Println("\n  Production Acceptance Gates (AC-1..AC-5):")
	for _, g := range l.AcceptanceGates {
		fmt.Printf("    [%4s] %s - %s: %s\n", g.Status, g.ID, g.Name, g.Actual)
	}
}

func runStrixStatusCmd() {
	fmt.Println("=== AMD Strix Halo (Ryzen AI Max+ 395 / Radeon 8060S gfx1151) Status ===")
	for _, rel := range []string{
		"src/core/arch_defaults.cpp",
		"src/prefill/prefill.cpp",
		"src/kernels/cuda/moe_fused_iq.cu",
		"src/kernels/cuda/gdn.cu",
		"src/kernels/cuda/qsa.cu",
		"src/kernels/cuda/fused_gr.cu",
		"tools/hip/gfx1151-hipblaslt-100401.txt",
		"tools/hip/gfx1151-hipblaslt-100500.txt",
		"docs/STRIX_HALO.md",
		"docs/QWEN38_177B_UPGRADE_AND_BENCHMARK_REPORT.md",
	} {
		full := filepath.Join(DefaultRepoRoot, filepath.FromSlash(rel))
		st, err := os.Stat(full)
		if err == nil {
			fmt.Printf("  [OK] %-48s (%d bytes)\n", rel, st.Size())
		} else {
			fmt.Printf("  [!!] %-48s (missing)\n", rel)
		}
	}
}

func runSwitchesCmd() {
	fmt.Println("=== Strix Halo (gfx1151) & Qwen 3.8 177B Runtime Switches ===")
	for _, s := range AllSwitchSpecs() {
		fmt.Printf("  %-26s = %-4s | %-16s | %-6s | %s\n", s.Name, s.ExpectedVal, s.Category, s.Phase, s.Impact)
	}
}

func runSearchCmd(args []string) {
	if len(args) == 0 {
		fmt.Println("Usage: strata-tracker search <term>")
		return
	}
	term := strings.Join(args, " ")
	out, err := exec.Command("git", "-C", DefaultRepoRoot, "log", "--all", "--grep="+term, "-i", "--oneline", "-n", "25").CombinedOutput()
	if err != nil {
		fmt.Fprintf(os.Stderr, "git search failed: %v\n", err)
		return
	}
	fmt.Print(string(out))
}

func runKernelsCmd() {
	fmt.Println("=== Strata Hardware Kernel Architecture Catalog ===")
	fmt.Println("  1. MoE Fused IQ (src/kernels/cuda/moe_fused_iq.cu)      : IQ3_S gate/up + IQ4_NL down, STRATA_EXPERT_V2=1, STRATA_PF_FUSED=1")
	fmt.Println("  2. Gated Delta Net (src/kernels/cuda/gdn.cu)            : 36-layer O(1) state recurrence, STRATA_GDN_HEAD=1, STRATA_GDN_PP=2, STRATA_GDN_CHUNKED=1")
	fmt.Println("  3. Quantized Sparse Attention (src/kernels/cuda/qsa.cu) : 12-layer WMMA block scorer & top-k selection, STRATA_SELECT_WMMA=1")
	fmt.Println("  4. Hyper-Connections (src/kernels/cuda/fused_gr.cu)     : Q8_0 verify read, STRATA_HC_Q8=1, STRATA_HCD_EXACT=1, STRATA_GR_TSUM=1")
	fmt.Println("  5. Prefix Lookup Engine (src/core/ple.cpp)              : 28.8 GiB IQ4_NL n-gram direct NVMe IOCP + 4M row cache, STRATA_PLE_BATCH=1")
	fmt.Println("  6. Speculative MTP + Suffix Drafter (src/core/mtp.cpp)  : 5-token MTP Q4_0 draft head + lookup-chain 3 + STRATA_SPEC_GUMBEL=1")
}

func runCheckUpstreamCmd() {
	client := &http.Client{Timeout: 8 * time.Second}
	resp, err := client.Get("https://api.github.com/repos/Niko1221/Strata/releases?per_page=5")
	if err != nil {
		fmt.Printf("GitHub check offline (%v); checking local git tags...\n", err)
		out, _ := exec.Command("git", "-C", DefaultRepoRoot, "tag", "--sort=-creatordate").CombinedOutput()
		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		for i := 0; i < len(lines) && i < 10; i++ {
			fmt.Printf("  tag: %s\n", lines[i])
		}
		return
	}
	defer resp.Body.Close()
	var releases []struct {
		TagName     string `json:"tag_name"`
		Name        string `json:"name"`
		PublishedAt string `json:"published_at"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&releases); err == nil {
		fmt.Println("=== Latest Upstream GitHub Releases (Niko1221/Strata) ===")
		for _, r := range releases {
			fmt.Printf("  %-12s | %s | %s\n", r.TagName, r.PublishedAt, r.Name)
		}
	}
}

func runBlueprintCmd() {
	bpPath := filepath.Join(DefaultRepoRoot, "docs-strix-go", "GO_STRATA_DEPLOYMENT_BLUEPRINT.md")
	raw, err := os.ReadFile(bpPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed reading blueprint: %v\n", err)
		return
	}
	fmt.Println(string(raw))
}
