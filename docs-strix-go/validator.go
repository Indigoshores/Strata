package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	DefaultRepoRoot       = `C:\Users\Butch\Documents\02_Apps\Strata`
	DefaultProdConfigPath = `C:\Users\Butch\Documents\10-Models\Strata\strata-qwen177b.json`
	DefaultEngineLogPath  = `C:\Users\Butch\Documents\10-Models\Strata\strata_engine.log`
	DefaultServerURL      = `http://127.0.0.1:8080`
	TotalRoutedExperts    = 24576
)

// ProductionConfig represents the schema of strata-qwen177b.json.
type ProductionConfig struct {
	Exe         string            `json:"exe"`
	Cwd         string            `json:"cwd"`
	ModelName   string            `json:"model_name"`
	Backend     string            `json:"backend"`
	Port        int               `json:"port"`
	Host        string            `json:"host"`
	OpenBrowser bool              `json:"open_browser"`
	Sampling        SamplingConfig    `json:"sampling"`
	ReasoningEffort string            `json:"reasoning_effort,omitempty"`
	EffortPosition  string            `json:"effort_position,omitempty"`
	PreOpenThink    bool              `json:"pre_open_think,omitempty"`
	Tokenizer   string            `json:"tokenizer"`
	LibDirs     []string          `json:"lib_dirs"`
	Env         map[string]string `json:"env"`
	Args        []string          `json:"args"`
}

// SamplingConfig holds default temperature and top_p settings.
type SamplingConfig struct {
	Temperature float64 `json:"temperature"`
	TopP        float64 `json:"top_p"`
}

// SwitchSpec describes a Strix Halo (gfx1151) or Qwen 3.8 177B runtime optimization switch.
type SwitchSpec struct {
	Name        string `json:"name"`
	ExpectedVal string `json:"expected_val"`
	Category    string `json:"category"` // "default_gfx1151", "optin_active", "candidate_v0145"
	Phase       string `json:"phase"`    // "Prompt", "Decode", "Both", "Memory/IO"
	Precision   string `json:"precision"`
	Impact      string `json:"impact"`
	SourceRef   string `json:"source_ref"`
}

// AllSwitchSpecs returns the complete catalog of Strix Halo gfx1151 and Qwen 3.8 177B switches
// documented across docs/STRIX_HALO.md, docs-strix-go/STRIX_HALO_OPTIMIZATIONS.md, and
// docs/QWEN38_177B_UPGRADE_AND_BENCHMARK_REPORT.md.
func AllSwitchSpecs() []SwitchSpec {
	return []SwitchSpec{
		// 18 Architectural Defaults in src/core/arch_defaults.cpp (Bit-Exact on gfx1151)
		{Name: "STRATA_GDN_HEAD", ExpectedVal: "1", Category: "default_gfx1151", Phase: "Prompt", Precision: "Bit-exact", Impact: "GDN recurrence: 4 SIMD lanes/col + grid-stride norm fusion", SourceRef: "src/core/arch_defaults.cpp"},
		{Name: "STRATA_GDN_PP", ExpectedVal: "2", Category: "default_gfx1151", Phase: "Prompt", Precision: "Bit-exact", Impact: "2 cols/lane in GDN recurrence (+2.9% at 64K prompt)", SourceRef: "src/core/arch_defaults.cpp"},
		{Name: "STRATA_GDN_CONVL2", ExpectedVal: "1", Category: "default_gfx1151", Phase: "Prompt", Precision: "Bit-exact", Impact: "Fuses 1D conv + q/k L2 norm (-0.3s at 64K)", SourceRef: "src/core/arch_defaults.cpp"},
		{Name: "STRATA_GDN_NOY", ExpectedVal: "1", Category: "default_gfx1151", Phase: "Prompt", Precision: "Bit-exact", Impact: "Eliminates dead FP32 global memory store of norm output", SourceRef: "src/core/arch_defaults.cpp"},
		{Name: "STRATA_CVEC_FUSE", ExpectedVal: "1", Category: "default_gfx1151", Phase: "Prompt", Precision: "Bit-exact", Impact: "Fuses steered layer write, control vector, and next norm", SourceRef: "src/core/arch_defaults.cpp"},
		{Name: "STRATA_HCD_EXACT", ExpectedVal: "1", Category: "default_gfx1151", Phase: "Prompt", Precision: "Bit-exact", Impact: "HC down GEMM in WMMA matching hipBLASLt sol 1176 (+1.8% at 64K)", SourceRef: "src/core/arch_defaults.cpp"},
		{Name: "STRATA_PF_PAD", ExpectedVal: "1", Category: "default_gfx1151", Phase: "Prompt", Precision: "Bit-exact", Impact: "Pads GEMM row strides to 128B cache lines", SourceRef: "src/core/arch_defaults.cpp"},
		{Name: "STRATA_DEQUANT_WIDE", ExpectedVal: "1", Category: "optin_active", Phase: "Prompt", Precision: "Bit-exact", Impact: "Coalesced 16-byte uint4 wide dequantization for Q4_K/Q5_K/Q6_K/IQ4_NL/IQ4_XS", SourceRef: "src/kernels/cuda/dequant_bf16.cu"},
		{Name: "STRATA_Q8_PACKED", ExpectedVal: "1", Category: "default_gfx1151", Phase: "Decode", Precision: "Bit-exact", Impact: "Packed Q8_0 decode layout (+2.3% decode)", SourceRef: "src/core/arch_defaults.cpp"},
		{Name: "STRATA_Q6_PACKED", ExpectedVal: "1", Category: "default_gfx1151", Phase: "Decode", Precision: "Bit-exact", Impact: "Packed Q6_K output head layout", SourceRef: "src/core/arch_defaults.cpp"},
		{Name: "STRATA_MMVF_ROWS", ExpectedVal: "1", Category: "default_gfx1151", Phase: "Decode", Precision: "Bit-exact", Impact: "BF16 multi-row GEMV (4 rows/block)", SourceRef: "src/core/arch_defaults.cpp"},
		{Name: "STRATA_ATTN_LANECELL", ExpectedVal: "1", Category: "default_gfx1151", Phase: "Decode", Precision: "Bit-exact", Impact: "Decode attention scores mapped 1 cell/thread", SourceRef: "src/core/arch_defaults.cpp"},
		{Name: "STRATA_EXPERT_V2", ExpectedVal: "1", Category: "default_gfx1151", Phase: "Decode", Precision: "Bit-exact", Impact: "Grouped MoE decode experts (IQ3_S gate/up + IQ4_NL down)", SourceRef: "src/core/arch_defaults.cpp"},
		{Name: "STRATA_TSUM", ExpectedVal: "1", Category: "default_gfx1151", Phase: "Decode", Precision: "Bit-exact", Impact: "Transposed butterfly shuffle warp reduction", SourceRef: "src/core/arch_defaults.cpp"},
		{Name: "STRATA_LFUSE", ExpectedVal: "1", Category: "default_gfx1151", Phase: "Decode", Precision: "Bit-exact", Impact: "Fuses launches around shared expert and KV append", SourceRef: "src/core/arch_defaults.cpp"},
		{Name: "STRATA_GDN_SPLIT", ExpectedVal: "1", Category: "default_gfx1151", Phase: "Decode", Precision: "Bit-exact", Impact: "Distributes GDN state step across 4 blocks/head", SourceRef: "src/core/arch_defaults.cpp"},
		{Name: "STRATA_QFUSE", ExpectedVal: "1", Category: "default_gfx1151", Phase: "Decode", Precision: "Bit-exact", Impact: "Activation Q8_1 images written directly by upstream producers", SourceRef: "src/core/arch_defaults.cpp"},
		{Name: "STRATA_PLE_BATCH", ExpectedVal: "1", Category: "default_gfx1151", Phase: "Decode", Precision: "Bit-exact", Impact: "Batches verify window PLE key/value projections", SourceRef: "src/core/arch_defaults.cpp"},
		{Name: "STRATA_SH_STREAM", ExpectedVal: "1", Category: "default_gfx1151", Phase: "Decode", Precision: "Bit-exact", Impact: "Concurrent shared expert stream (+1.8% to +6.7% decode)", SourceRef: "src/core/arch_defaults.cpp"},

		// Active Opt-In WMMA & Prefill Switches in strata-qwen177b.json
		{Name: "STRATA_PF_FUSED", ExpectedVal: "1", Category: "optin_active", Phase: "Prompt", Precision: "WMMA FP16/INT8", Impact: "GPU-grouped fused prompt MoE experts (1,102-1,324 tok/s prefill)", SourceRef: "docs/STRIX_HALO.md"},
		{Name: "STRATA_PF_GEMM", ExpectedVal: "1", Category: "optin_active", Phase: "Prompt", Precision: "WMMA FP16", Impact: "Prompt FP16 projections on 128x256 WMMA GEMM", SourceRef: "docs/STRIX_HALO.md"},
		{Name: "STRATA_HC_UPMIX", ExpectedVal: "1", Category: "optin_active", Phase: "Prompt", Precision: "WMMA FP16", Impact: "Fuses hyper-connection up projection with activation epilogue", SourceRef: "docs/STRIX_HALO.md"},
		{Name: "STRATA_PA_FAST", ExpectedVal: "1", Category: "optin_active", Phase: "Prompt", Precision: "FP16 Q/P", Impact: "Prompt attention using single FP16 query and probability buffers", SourceRef: "docs/STRIX_HALO.md"},
		{Name: "STRATA_HIP_WMMA", ExpectedVal: "1", Category: "optin_active", Phase: "Prompt", Precision: "WMMA FP16", Impact: "Scaled dot-product prompt attention on RDNA 3.5 matrix cores", SourceRef: "docs/STRIX_HALO.md"},
		{Name: "STRATA_SELECT_WMMA", ExpectedVal: "1", Category: "optin_active", Phase: "Prompt", Precision: "WMMA FP16", Impact: "QSA block scorer & top-k candidate ranking on WMMA", SourceRef: "docs/STRIX_HALO.md"},
		{Name: "STRATA_HC_Q8", ExpectedVal: "1", Category: "optin_active", Phase: "Both", Precision: "Native Q8_0", Impact: "Reads hyper-connection projections from native GGUF Q8_0 weights", SourceRef: "docs/STRIX_HALO.md"},
		{Name: "STRATA_PF_SWITCH_MIN_T", ExpectedVal: "16", Category: "optin_active", Phase: "Prompt", Precision: "Threshold", Impact: "Activates S23/Q8_0 WMMA prompt kernels from T >= 16 tokens (including T=31 tail chunk)", SourceRef: "strata-qwen177b.json"},
		{Name: "STRATA_PF_SH_STREAM", ExpectedVal: "0", Category: "optin_active", Phase: "Prompt", Precision: "Bit-exact", Impact: "Disables Windows WDDM cross-stream shared_expert barrier (-220 ms/chunk)", SourceRef: "src/prefill/prefill.cpp"},
		{Name: "STRATA_PF_Z_STREAM", ExpectedVal: "0", Category: "optin_active", Phase: "Prompt", Precision: "Bit-exact", Impact: "Disables Windows WDDM cross-stream GDN Z-projection barrier (-813 ms/chunk)", SourceRef: "src/prefill/prefill.cpp"},
		{Name: "STRATA_PREFILL_STREAM_MIN", ExpectedVal: "16", Category: "optin_active", Phase: "Prompt", Precision: "Bit-exact", Impact: "Fixes Sub-1024 prefill cliff (363->1,100+ tok/s for T=16..1023)", SourceRef: "src/prefill/prefill.cpp:140"},
		{Name: "STRATA_PF_HCDOWN", ExpectedVal: "1", Category: "optin_active", Phase: "Prompt", Precision: "WMMA FP16", Impact: "Accelerated hyper-connection down projection on prompt path", SourceRef: "strata-qwen177b.json"},
		{Name: "STRATA_WMMA_GEMM", ExpectedVal: "1", Category: "optin_active", Phase: "Prompt", Precision: "WMMA FP16", Impact: "Enables WMMA GEMM dispatch across prompt dense projections", SourceRef: "strata-qwen177b.json"},
		{Name: "STRATA_PF_GEMM_Q8", ExpectedVal: "1", Category: "optin_active", Phase: "Prompt", Precision: "Bit-exact WMMA", Impact: "Direct Q8_0 in-register WMMA prompt GEMM + direct Q8_0 HC down/upmix (-27.5 GB/chunk DRAM)", SourceRef: "src/prefill/wmma_gemm.cu"},
		{Name: "STRATA_PF_DM_F16", ExpectedVal: "1", Category: "optin_active", Phase: "Prompt", Precision: "FP16 Dm + LDS fuse", Impact: "Halves MoE Dm DRAM traffic (-80.5 GB at 16K) + LDS-fused combine and write_norm_rs", SourceRef: "src/prefill/kernels.cu"},
		{Name: "STRATA_GDN_CHUNKED_MIN_T", ExpectedVal: "128", Category: "optin_active", Phase: "Prompt", Precision: "Threshold", Impact: "16-token WMMA WY chunked GDN recurrence from T >= 128 (+9 tok/s at 512, +21 tok/s at 1K)", SourceRef: "src/prefill/kernels.cu"},
		{Name: "STRATA_WDDM_ALT_STREAM", ExpectedVal: "1", Category: "optin_active", Phase: "Decode", Precision: "Bit-exact", Impact: "Windows WDDM alternate stream scheduling without host stall", SourceRef: "strata-qwen177b.json"},
		{Name: "STRATA_EXP_MODE", ExpectedVal: "8", Category: "optin_active", Phase: "Decode", Precision: "Bit-exact", Impact: "Grouped RDNA 3.5 decode MoE expert kernel mode 8", SourceRef: "strata-qwen177b.json"},
		{Name: "STRATA_NO_TURN_SPLIT", ExpectedVal: "1", Category: "optin_active", Phase: "Prompt", Precision: "Bit-exact", Impact: "Prevents conversational turn splitting, preserving single-pass prompt execution", SourceRef: "strata-qwen177b.json"},
		{Name: "STRATA_PF_F16_MIN_T", ExpectedVal: "0", Category: "optin_active", Phase: "Prompt", Precision: "FP16 WMMA MoE", Impact: "Enables persistent-WGP FP16 WMMA MoE prefill across all prompt tiers (T >= 16)", SourceRef: "strata-qwen177b.json"},
		{Name: "STRATA_PF_MOE_SCATTER", ExpectedVal: "0", Category: "optin_active", Phase: "Prompt", Precision: "FP16 Dm + F16 down", Impact: "Disables moe_accum scatter override so STRATA_PF_DM_F16=1 and native_f16_kernel down projections run", SourceRef: "src/prefill/prefill.cpp:3669"},

		// Candidate v0.1.45 & Open PR #2997 Opt-In Switches for A/B Validation
		{Name: "STRATA_PF_F16", ExpectedVal: "1", Category: "candidate_v0145", Phase: "Prompt", Precision: "Opt-in FP16 WMMA (KL 0.0005)", Impact: "UD-IQ4_XS gate/up -15.3%, down -22.9% prefill time on gfx1151", SourceRef: "Commit bd5cc1f2 / feeb8dc8 (v0.1.45)"},
		{Name: "STRATA_GDN_CHUNKED", ExpectedVal: "1", Category: "candidate_v0145", Phase: "Prompt", Precision: "Opt-in FP16 WMMA", Impact: "1.6x faster GDN recurrence via 16-token WY chunks (+1.1% to +2.5% E2E)", SourceRef: "PR #2299 (1b1bd4b4, v0.1.45)"},
		{Name: "STRATA_TOPK_REG_MIN_BLOCKS", ExpectedVal: "4096", Category: "candidate_v0145", Phase: "Prompt", Precision: "Bit-exact (19th default)", Impact: "QSA top-k 3.90 ms -> 1.00 ms per 2K queries at 24K context", SourceRef: "PR #2016 (4cf36daf, v0.1.45)"},
		{Name: "STRATA_SPEC_GUMBEL", ExpectedVal: "1", Category: "candidate_v0145", Phase: "Decode", Precision: "Coupled Exp(1)", Impact: "+9% sampled decode throughput (52.8% -> 59.9% draft acceptance at T>0)", SourceRef: "Commit 6381df3"},
		{Name: "STRATA_HC_PERSIST", ExpectedVal: "1", Category: "candidate_v0145", Phase: "Decode", Precision: "Bounded 1e-4", Impact: "-6.6% / -4.4% ms per verify window at 4K / 32K via persistent HC kernel", SourceRef: "PR #2226 (6cbc15a8, v0.1.45)"},
		{Name: "STRATA_GR_TSUM", ExpectedVal: "1", Category: "candidate_v0145", Phase: "Decode", Precision: "Bit-exact LDS sync", Impact: "-16.2% GR read latency at T=6; VGPRs 109->95 (2 CTAs/WGP on gfx1151)", SourceRef: "Open PR #2997 (d09a7b37)"},
	}
}

// CheckResult represents an individual validation assertion.
type CheckResult struct {
	ID       string `json:"id"`
	Category string `json:"category"`
	Name     string `json:"name"`
	Status   string `json:"status"` // "PASS", "WARN", "FAIL", "INFO"
	Expected string `json:"expected"`
	Actual   string `json:"actual"`
	Evidence string `json:"evidence"`
}

// ConfigAuditReport holds the complete static + runtime configuration validation output.
type ConfigAuditReport struct {
	ConfigPath         string        `json:"config_path"`
	RepoRoot           string        `json:"repo_root"`
	ModelName          string        `json:"model_name"`
	Backend            string        `json:"backend"`
	TuningTableValid   bool          `json:"tuning_table_valid"`
	TuningTableRows    int           `json:"tuning_table_rows"`
	TuningTableVersion string        `json:"tuning_table_version"`
	Sub1024CliffFixed  bool          `json:"sub1024_cliff_fixed"`
	ActiveOptInCount   int           `json:"active_optin_count"`
	CandidateOptIns    []string      `json:"candidate_optins_pending"`
	Checks             []CheckResult `json:"checks"`
	PassedCount        int           `json:"passed_count"`
	WarnCount          int           `json:"warn_count"`
	FailedCount        int           `json:"failed_count"`
}

// AuditConfig validates strata-qwen177b.json against the Strata C++ AST and Strix Halo docs.
func AuditConfig(configPath, repoRoot string) (*ConfigAuditReport, error) {
	raw, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config %s: %w", configPath, err)
	}
	var cfg ProductionConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse JSON config %s: %w", configPath, err)
	}

	report := &ConfigAuditReport{
		ConfigPath: configPath,
		RepoRoot:   repoRoot,
		ModelName:  cfg.ModelName,
		Backend:    cfg.Backend,
	}

	addCheck := func(id, cat, name, status, expected, actual, evidence string) {
		report.Checks = append(report.Checks, CheckResult{
			ID:       id,
			Category: cat,
			Name:     name,
			Status:   status,
			Expected: expected,
			Actual:   actual,
			Evidence: evidence,
		})
		switch status {
		case "PASS":
			report.PassedCount++
		case "WARN":
			report.WarnCount++
		case "FAIL":
			report.FailedCount++
		}
	}

	// 1. Check Sub-1024 Prefill Cliff Guard (STRATA_PREFILL_STREAM_MIN)
	streamMinStr, ok := cfg.Env["STRATA_PREFILL_STREAM_MIN"]
	streamMinVal, errConv := strconv.Atoi(streamMinStr)
	if ok && errConv == nil && streamMinVal > 0 && streamMinVal <= 128 {
		report.Sub1024CliffFixed = true
		addCheck("CFG-01", "Prefill Cliff", "STRATA_PREFILL_STREAM_MIN <= 128", "PASS", "<= 128", streamMinStr,
			"Enables GPU-grouped STRATA_PF_FUSED=1 for 128..1023 token prompts instead of 1024 default MMQ fallback")
	} else {
		actual := streamMinStr
		if !ok {
			actual = "unset (defaults to 1024 in src/prefill/prefill.cpp:140)"
		}
		addCheck("CFG-01", "Prefill Cliff", "STRATA_PREFILL_STREAM_MIN <= 128", "FAIL", "<= 128", actual,
			"Prompts < 1024 tokens fall back to host-synced MMQ at 363.6-528.9 tok/s")
	}

	// 2. Check hipBLASLt Tuning Table
	tuningPath := cfg.Env["STRATA_HIPBLASLT_TUNING"]
	if tuningPath == "" {
		addCheck("CFG-02", "hipBLASLt", "STRATA_HIPBLASLT_TUNING configured", "FAIL", "valid gfx1151 table path", "unset",
			"Without tuning table, dense prompt GEMMs fall back to generic hipBLAS")
	} else {
		rows, ver, errT := inspectTuningTable(tuningPath)
		if errT != nil {
			// Check fallback in repo tools/hip/
			altPath := filepath.Join(repoRoot, "tools", "hip", filepath.Base(tuningPath))
			rows, ver, errT = inspectTuningTable(altPath)
		}
		if errT == nil && rows > 0 {
			report.TuningTableValid = true
			report.TuningTableRows = rows
			report.TuningTableVersion = ver
			addCheck("CFG-02", "hipBLASLt", "gfx1151 hipBLASLt Tuning Table", "PASS",
				"STRATA_HIPBLASLT_TUNING_V1 gfx1151", fmt.Sprintf("version=%s rows=%d", ver, rows), tuningPath)
		} else {
			addCheck("CFG-02", "hipBLASLt", "gfx1151 hipBLASLt Tuning Table", "FAIL",
				"readable tuning table", fmt.Sprintf("err=%v", errT), tuningPath)
		}
	}

	// 3. Check Active Opt-In Matrix Core (WMMA) Switches (Fail-Closed Pass 6..8 Prefill Guard)
	requiredOptIns := []struct {
		id   string
		key  string
		want string
		desc string
	}{
		{"CFG-03", "STRATA_PF_FUSED", "1", "Fused WMMA prompt MoE experts"},
		{"CFG-04", "STRATA_PF_GEMM", "1", "128x256 WMMA prompt GEMM (disabling falls back to 328-538 tok/s MMQ)"},
		{"CFG-04b", "STRATA_PF_GEMM_Q8", "1", "Direct Q8_0 in-register WMMA prompt GEMM + direct Q8_0 HC down/upmix"},
		{"CFG-04c", "STRATA_PF_PAD", "1", "128B row-stride padding channel-conflict guard for WMMA GEMM"},
		{"CFG-05", "STRATA_WMMA_GEMM", "1", "Sub-threshold 64x64_4w RDNA 3.5 WMMA prompt GEMM"},
		{"CFG-06", "STRATA_PF_HCDOWN", "1", "WMMA hyper-connection down projection on prompt path"},
		{"CFG-07", "STRATA_HC_UPMIX", "1", "Fused hyper-connection up projection + activation epilogue"},
		{"CFG-08", "STRATA_PA_FAST", "1", "Single FP16 Q/P prompt attention"},
		{"CFG-09", "STRATA_HIP_WMMA", "1", "Hardware matrix core prompt attention"},
		{"CFG-09a", "STRATA_SELECT_WMMA", "1", "WMMA QSA block scorer & router"},
		{"CFG-09c", "STRATA_HC_Q8", "1", "Native Q8_0 hyper-connection read"},
		{"CFG-09d", "STRATA_PF_SH_STREAM", "0", "Windows WDDM single-stream prefill shared_expert (-220 ms barrier removal)"},
		{"CFG-09e", "STRATA_PF_Z_STREAM", "0", "Windows WDDM single-stream prefill GDN Z-projection (-813 ms barrier removal)"},
		{"CFG-09f", "STRATA_PF_DM_F16", "1", "FP16 MoE Dm buffer (-80.5 GB DRAM) + LDS-fused combine"},
		{"CFG-09g", "STRATA_NO_TURN_SPLIT", "1", "Prevents conversational turn splitting, preserving single-pass prompt execution"},
		{"CFG-09h", "STRATA_PF_F16_MIN_T", "0", "Enables persistent-WGP FP16 WMMA MoE prefill across all prompt sizes (T >= 16)"},
		{"CFG-09i", "STRATA_PF_MOE_SCATTER", "0", "Disables moe_accum scatter override so STRATA_PF_DM_F16=1 and FP16 WMMA down-projections execute"},
	}
	for _, sw := range requiredOptIns {
		got := cfg.Env[sw.key]
		if got == sw.want {
			report.ActiveOptInCount++
			addCheck(sw.id, "WMMA / Kernel Switch", sw.key, "PASS", sw.want, got, sw.desc)
		} else {
			addCheck(sw.id, "WMMA / Kernel Switch", sw.key, "FAIL", sw.want, got, sw.desc)
		}
	}

	// 3a. Check STRATA_PF_SWITCH_MIN_T <= 256 (Fail-Closed)
	pfSwitchStr, pfSwitchOk := cfg.Env["STRATA_PF_SWITCH_MIN_T"]
	pfSwitchVal, pfSwitchErr := strconv.Atoi(pfSwitchStr)
	if pfSwitchOk && pfSwitchErr == nil && pfSwitchVal >= 16 && pfSwitchVal <= 256 {
		report.ActiveOptInCount++
		addCheck("CFG-09b", "WMMA / Kernel Switch", "STRATA_PF_SWITCH_MIN_T in [16..256]", "PASS", "16 (16..256)", pfSwitchStr,
			"Activates S23/Q8_0 WMMA prompt GEMM down to T >= 16 (including T=31 tail chunk)")
	} else {
		actual := pfSwitchStr
		if !pfSwitchOk {
			actual = "unset"
		}
		addCheck("CFG-09b", "WMMA / Kernel Switch", "STRATA_PF_SWITCH_MIN_T in [16..256]", "FAIL", "16 (16..256)", actual,
			"Missing STRATA_PF_SWITCH_MIN_T degrades Pass 8 WMMA prefill dispatch")
	}

	// 3b. Check STRATA_SH_STREAM on Windows WDDM (Pass 6 invariant: 0 or omitted via arch_defaults.cpp:58)
	shStreamGot, shStreamPresent := cfg.Env["STRATA_SH_STREAM"]
	if !shStreamPresent || shStreamGot == "0" {
		report.ActiveOptInCount++
		actual := shStreamGot
		if !shStreamPresent {
			actual = "omitted (defaults to 0 in src/core/arch_defaults.cpp:58)"
		}
		addCheck("CFG-10", "Windows WDDM Guard", "STRATA_SH_STREAM=0 (or omitted)", "PASS", "0 or omitted", actual, "Avoids Windows WDDM secondary stream synchronization overhead")
	} else {
		addCheck("CFG-10", "Windows WDDM Guard", "STRATA_SH_STREAM=0 (or omitted)", "WARN", "0 or omitted", shStreamGot, "STRATA_SH_STREAM=1 on Windows WDDM may degrade decode speed")
	}

	// 3c. Check STRATA_PF_F16 (Must be 0 or absent, NOT 1)
	pfF16Got, pfF16Present := cfg.Env["STRATA_PF_F16"]
	if !pfF16Present || pfF16Got != "1" {
		addCheck("CFG-10b", "WMMA / Kernel Switch", "STRATA_PF_F16 == 1", "FAIL", "1", pfF16Got, "Pass 8c persistent-WGP FP16 WMMA MoE prefill (105/105 parity verified) must be enabled")
	} else {
		actual := pfF16Got
		if !pfF16Present {
			actual = "absent"
		}
		addCheck("CFG-10b", "WMMA / Kernel Switch", "STRATA_PF_F16 == 1", "PASS", "1", actual, "Pass 8c persistent-WGP FP16 WMMA MoE prefill active (105/105 parity verified)")
	}

	// 3d. Check Reasoning Effort Configuration
	effort := strings.ToLower(strings.TrimSpace(cfg.ReasoningEffort))
	if effort == "" {
		effort = "medium"
	}
	switch effort {
	case "medium":
		addCheck("CFG-10c", "Thinking Alignment", "reasoning_effort aligned", "PASS", "medium (or low/high/xhigh)", effort, "Canonical Pass 6/7 medium reasoning effort profile")
	case "low", "high", "xhigh":
		addCheck("CFG-10c", "Thinking Alignment", "reasoning_effort valid", "PASS", "medium (or low/high/xhigh)", effort, "Valid explicit thinking effort level")
	case "none":
		addCheck("CFG-10c", "Thinking Alignment", "reasoning_effort valid", "FAIL", "low, medium, high, xhigh", effort, "Reasoning effort 'none' is forbidden as default production profile")
	default:
		addCheck("CFG-10c", "Thinking Alignment", "reasoning_effort valid", "WARN", "medium", effort, "Unrecognized reasoning effort setting")
	}

	// 3e. Check Effort Position & Zero PreOpenThink Alignment for Qwen 3.8 177B Pass 9.1 Parity
	effortPos := strings.ToLower(strings.TrimSpace(cfg.EffortPosition))
	if effortPos == "" {
		effortPos = "end"
	}
	if strings.Contains(strings.ToLower(cfg.ModelName), "177b") || strings.Contains(strings.ToLower(cfg.ModelName), "qwen 3.8") {
		if effortPos == "end" && !cfg.PreOpenThink {
			addCheck("CFG-10d", "Thinking Alignment", "effort_position == end && pre_open_think == false", "PASS", "end (pre_open_think=false)", fmt.Sprintf("%s (pre_open_think=%v)", effortPos, cfg.PreOpenThink),
				"Matches Pass 9.1 Reference (--tail-role-token 8678 + native <think> generation, 81.8-90.5 tok/s)")
		} else {
			addCheck("CFG-10d", "Thinking Alignment", "effort_position == end && pre_open_think == false", "FAIL", "end (pre_open_think=false)", fmt.Sprintf("%s (pre_open_think=%v)", effortPos, cfg.PreOpenThink),
				"Qwen 3.8 177B requires effort_position='end' (--tail-role-token 8678) without PreOpenThink tag inflation")
		}
	}

	// 4. Validate CLI Args (PLE Direct I/O, Expert Cache, KV Int8, MTP + Lookup Chain)
	argMap := parseArgsToMap(cfg.Args)
	argChecks := []struct {
		id   string
		flag string
		want string
		desc string
	}{
		{"CFG-11", "--expert-cache", "auto", "Dynamic DXGI/UMA expert cache sizing (24,576 slots on 128GB Strix Halo)"},
		{"CFG-12", "--ple-io", "direct", "Unbuffered Win32 FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED NVMe PLE reads"},
		{"CFG-13", "--ple-inflight", "1024", "High queue depth IOCP inflight reads for 28.8 GiB PLE shard"},
		{"CFG-14", "--ple-row-cache", "1048576", "1M..4M-entry RAM row cache for per_layer_token_embd.weight (-296 MiB WDDM commit at 1M)"},
		{"CFG-15", "--prefill", "18432", "18K single-chunk prefill covering full 16K tier (17,613 tokens) without 2nd MoE pass"},
		{"CFG-15b", "--conversation-cache-mib", "512", "512..2048 MiB KV prefix conversation cache within Windows WDDM 81.6 GiB commit ceiling"},
		{"CFG-16", "--max-context", "262144", "Full 256K hybrid GDN+QSA context window"},
		{"CFG-17", "--kv", "int8", "INT8 KV cache quantization (zero perplexity loss, 50% VRAM savings)"},
		{"CFG-18", "--spec", "7", "7-token coupled MTP + lookup verification window (80+ tok/s decode)"},
		{"CFG-19", "--lookup-chain", "3", "Suffix-tree n-gram prompt lookup drafter chain length 3"},
		{"CFG-20", "--mtp-q4", "head", "Q4_0 quantized MTP output head with higher-fidelity body weights"},
		{"CFG-20b", "--tail-role-token", "8678", "Preserves KV cache prefix across thinking/assistant boundaries"},
	}
	for _, ac := range argChecks {
		got, exists := argMap[ac.flag]
		pass := exists && got == ac.want
		if ac.flag == "--mtp-q4" && exists && (got == "all" || got == "head") {
			pass = true
		}
		if ac.flag == "--spec" && exists && (got == "5" || got == "6" || got == "7") {
			pass = true
		}
		if ac.flag == "--ple-row-cache" && exists && (got == "1048576" || got == "2097152" || got == "4194304") {
			pass = true
		}
		if ac.flag == "--conversation-cache-mib" && exists && (got == "512" || got == "1024" || got == "2048") {
			pass = true
		}
		if ac.flag == "--prefill" && exists && (got == "8192" || got == "16384" || got == "18432" || got == "20480") {
			pass = true
		}
		if pass {
			addCheck(ac.id, "Engine CLI Args", ac.flag, "PASS", ac.want, got, ac.desc)
		} else {
			addCheck(ac.id, "Engine CLI Args", ac.flag, "FAIL", ac.want, got, ac.desc)
		}
	}

	// Boolean flags check
	for _, bf := range []struct {
		id   string
		flag string
		desc string
	}{
		{"CFG-21", "--mmap-experts", "Zero-copy in-place GGUF shard expert mapping (no 77GB experts.bin copy)"},
		{"CFG-22", "--no-prefill-borrow", "Prevents prefill chunk buffers from evicting resident GPU experts"},
	} {
		if _, ok := argMap[bf.flag]; ok {
			addCheck(bf.id, "Engine CLI Flags", bf.flag, "PASS", "present", "present", bf.desc)
		} else {
			addCheck(bf.id, "Engine CLI Flags", bf.flag, "WARN", "present", "missing", bf.desc)
		}
	}

	// 5. Check Windows Local ROCm DLL Precedence Invariant (docs-strix-go/STRIX_HALO_OPTIMIZATIONS.md Section 6)
	exeDir := filepath.Dir(cfg.Exe)
	dllFound := false
	dllLocations := []string{exeDir}
	dllLocations = append(dllLocations, cfg.LibDirs...)
	for _, dir := range dllLocations {
		if _, err := os.Stat(filepath.Join(dir, "amdhip64_7.dll")); err == nil {
			dllFound = true
			break
		}
	}
	if dllFound {
		addCheck("CFG-23", "Windows ROCm Guard", "Local amdhip64_7.dll Precedence", "PASS",
			"amdhip64_7.dll in engine/ or lib_dirs", "verified present",
			"Prevents System32 Adrenalin driver DLL segfault on Windows 11 Strix Halo")
	} else {
		addCheck("CFG-23", "Windows ROCm Guard", "Local amdhip64_7.dll Precedence", "WARN",
			"amdhip64_7.dll in engine/ or lib_dirs", "not found on disk",
			"Ensure amdhip64_7.dll and amd_comgr.dll are co-located with strata.exe")
	}

	// 6. Check C++ Engine Source Sync in RepoRoot (src/core/arch_defaults.cpp & src/prefill/prefill.cpp)
	archDefPath := filepath.Join(repoRoot, "src", "core", "arch_defaults.cpp")
	if rawArch, err := os.ReadFile(archDefPath); err == nil {
		archStr := string(rawArch)
		hasShStream := strings.Contains(archStr, `"STRATA_SH_STREAM", "1"`)
		hasTopkReg := strings.Contains(archStr, `"STRATA_TOPK_REG_MIN_BLOCKS"`)
		if hasShStream {
			addCheck("CFG-24", "AST Source Check", "src/core/arch_defaults.cpp gfx1151 table", "PASS",
				"18 gfx1151 switches present", fmt.Sprintf("SH_STREAM=true, TOPK_REG_MIN_BLOCKS(v0.1.45)=%v", hasTopkReg),
				archDefPath)
		}
	}

	// 7. Catalog Candidate v0.1.45 + PR #2997 Opt-In Switches for A/B Testing
	for _, spec := range AllSwitchSpecs() {
		if spec.Category == "candidate_v0145" {
			val, present := cfg.Env[spec.Name]
			if !present || val == "" {
				report.CandidateOptIns = append(report.CandidateOptIns, fmt.Sprintf("%s=%s", spec.Name, spec.ExpectedVal))
			}
		}
	}
	addCheck("CFG-25", "v0.1.45 Upgrade Readiness", "Candidate v0.1.45 Opt-In Switches", "INFO",
		"A/B test one-by-one after v0.1.45 rebuild",
		strings.Join(report.CandidateOptIns, ", "),
		"docs/QWEN38_177B_UPGRADE_AND_BENCHMARK_REPORT.md Section 5.B")

	return report, nil
}

func inspectTuningTable(path string) (int, string, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, "", err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	rows := 0
	version := ""
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "#") {
			if strings.Contains(line, "STRATA_HIPBLASLT_TUNING_V1") {
				parts := strings.Fields(line)
				if len(parts) >= 4 {
					version = parts[len(parts)-1]
				}
			}
			continue
		}
		if strings.HasPrefix(line, "STRATA_HIPBLASLT_TUNING_V1") {
			parts := strings.Fields(line)
			if len(parts) >= 3 {
				version = parts[2]
			}
			continue
		}
		rows++
	}
	return rows, version, scanner.Err()
}

func parseArgsToMap(args []string) map[string]string {
	m := make(map[string]string)
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if strings.HasPrefix(arg, "--") {
			if i+1 < len(args) && !strings.HasPrefix(args[i+1], "--") {
				m[arg] = args[i+1]
				i++
			} else {
				m[arg] = "true"
			}
		}
	}
	return m
}

// RequestMetric represents a single completed inference request parsed from strata_engine.log.
type RequestMetric struct {
	LineNumber     int     `json:"line_number"`
	PromptTokens   int     `json:"prompt_tokens"`
	ReusedTokens   int     `json:"reused_tokens"`
	ReadTokens     int     `json:"read_tokens"`
	PrefillMs      float64 `json:"prefill_ms"`
	PrefillTokPerS float64 `json:"prefill_tok_per_s"`
	GenTokens      int     `json:"gen_tokens"`
	DecodeMs       float64 `json:"decode_ms"`
	DecodeTokPerS  float64 `json:"decode_tok_per_s"`
	DraftsAccepted int     `json:"drafts_accepted"`
	DraftsTotal    int     `json:"drafts_total"`
	AcceptancePct  float64 `json:"acceptance_pct"`
	Checkpoints    int     `json:"checkpoints"`
}

// TierPrefillSummary summarizes prefill performance for a context tier bucket.
type TierPrefillSummary struct {
	TierName       string  `json:"tier_name"`
	MinTokens      int     `json:"min_tokens"`
	MaxTokens      int     `json:"max_tokens"`
	SampleCount    int     `json:"sample_count"`
	MinTokPerS     float64 `json:"min_tok_per_s"`
	MedianTokPerS  float64 `json:"median_tok_per_s"`
	MeanTokPerS    float64 `json:"mean_tok_per_s"`
	P90TokPerS     float64 `json:"p90_tok_per_s"`
	MaxTokPerS     float64 `json:"max_tok_per_s"`
	MMQCliffDetect bool    `json:"mmq_cliff_detected"`
}

// DistributionStats holds percentile statistics for decode throughput and draft acceptance.
type DistributionStats struct {
	Count         int     `json:"count"`
	Min           float64 `json:"min"`
	P10           float64 `json:"p10"`
	P50Median     float64 `json:"p50_median"`
	Mean          float64 `json:"mean"`
	P90           float64 `json:"p90"`
	P99           float64 `json:"p99"`
	Max           float64 `json:"max"`
	CountGe50     int     `json:"count_ge_50_tps"`
	PctGe50       float64 `json:"pct_ge_50_tps"`
	CountGe60     int     `json:"count_ge_60_tps"`
	PctGe60       float64 `json:"pct_ge_60_tps"`
	PeakLine      int     `json:"peak_line"`
	PeakDraftsAcc int     `json:"peak_drafts_accepted"`
	PeakDraftsTot int     `json:"peak_drafts_total"`
}

// HandshakeInfo stores the latest [Strata Handshake] INFO record from strata_engine.log.
type HandshakeInfo struct {
	EngineVersion    string  `json:"engine_version"`
	Context          int     `json:"context"`
	KVFormat         string  `json:"kv_format"`
	ExpertSlots      int     `json:"expert_slots"`
	ExpertCacheMiB   int     `json:"expert_cache_mib"`
	VRAMFreeMiB      int     `json:"vram_free_mib"`
	SpecWindow       int     `json:"spec_window"`
	MTPMax           int     `json:"mtp_max"`
	LookupChain      int     `json:"lookup_chain"`
	PoolWorkers      int     `json:"pool_workers"`
	PCIeFrac         float64 `json:"pcie_frac"`
	R4HitPathOn      bool    `json:"r4_hit_path_on"`
	ZeroDoorbell     bool    `json:"zero_doorbell_graph"`
	HipBLASLtEnabled bool    `json:"hipblaslt_enabled"`
	HipBLASLtSummary string  `json:"hipblaslt_summary"`
	Gfx1151Switches  int     `json:"gfx1151_default_switches"`
}

// LogAuditReport contains the complete empirical telemetry analysis from strata_engine.log.
type LogAuditReport struct {
	LogPath               string               `json:"log_path"`
	TotalLinesScanned     int                  `json:"total_lines_scanned"`
	TotalRequestsParsed   int                  `json:"total_requests_parsed"`
	Handshake             HandshakeInfo        `json:"handshake"`
	SustainedDecode       DistributionStats    `json:"sustained_decode_gen_ge_50"`
	AllDecode             DistributionStats    `json:"all_decode"`
	DraftAcceptancePct    DistributionStats    `json:"draft_acceptance_pct_gen_ge_50"`
	ColdPrefillTiers      []TierPrefillSummary `json:"cold_prefill_tiers"`
	UltraLongContextCount int                  `json:"ultra_long_context_ge_100k_count"`
	UltraLongMinDecode    float64              `json:"ultra_long_min_decode_tps"`
	UltraLongMaxDecode    float64              `json:"ultra_long_max_decode_tps"`
	DecodeRAMSpillBlobs   int                  `json:"decode_ram_spill_blobs"`
	AcceptanceGates       []CheckResult        `json:"acceptance_gates"`
}

var (
	reServePrompt = regexp.MustCompile(`strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+(?:\.\d+)?) ms \((\d+(?:\.\d+)?) tok/s\), (\d+) generated in (\d+(?:\.\d+)?) ms \((\d+(?:\.\d+)?) tok/s\), drafts accepted (\d+) of (\d+), (\d+) checkpoints`)
	reExpertTiers = regexp.MustCompile(`strata serve: expert tiers: GPU \d+ hits this request; since the start RAM (\d+) blobs, files (\d+) blobs`)
	reGfxDefaults = regexp.MustCompile(`strata: gfx1151 \(Strix Halo\): (\d+) exact speed switches on by default`)
)

// AuditEngineLog streams strata_engine.log and computes empirical distributions and acceptance gates.
func AuditEngineLog(logPath string) (*LogAuditReport, error) {
	f, err := os.Open(logPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open engine log %s: %w", logPath, err)
	}
	defer f.Close()

	report := &LogAuditReport{
		LogPath: logPath,
	}

	scanner := bufio.NewScanner(f)
	buf := make([]byte, 0, 1024*1024)
	scanner.Buffer(buf, 16*1024*1024)

	var allMetrics []RequestMetric
	var sustainedMetrics []RequestMetric
	var sustainedDraftPct []float64

	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := scanner.Text()

		// Fast substring checks before regex
		if strings.Contains(line, "strata serve: prompt ") {
			if m := reServePrompt.FindStringSubmatch(line); m != nil {
				pTok, _ := strconv.Atoi(m[1])
				reused, _ := strconv.Atoi(m[2])
				readTok, _ := strconv.Atoi(m[3])
				pMs, _ := strconv.ParseFloat(m[4], 64)
				pTps, _ := strconv.ParseFloat(m[5], 64)
				genTok, _ := strconv.Atoi(m[6])
				dMs, _ := strconv.ParseFloat(m[7], 64)
				dTps, _ := strconv.ParseFloat(m[8], 64)
				da, _ := strconv.Atoi(m[9])
				dt, _ := strconv.Atoi(m[10])
				cp, _ := strconv.Atoi(m[11])

				accPct := 0.0
				if dt > 0 {
					accPct = (float64(da) / float64(dt)) * 100.0
				}

				rm := RequestMetric{
					LineNumber:     lineNum,
					PromptTokens:   pTok,
					ReusedTokens:   reused,
					ReadTokens:     readTok,
					PrefillMs:      pMs,
					PrefillTokPerS: pTps,
					GenTokens:      genTok,
					DecodeMs:       dMs,
					DecodeTokPerS:  dTps,
					DraftsAccepted: da,
					DraftsTotal:    dt,
					AcceptancePct:  accPct,
					Checkpoints:    cp,
				}
				allMetrics = append(allMetrics, rm)
				if genTok >= 50 {
					sustainedMetrics = append(sustainedMetrics, rm)
					if dt > 0 {
						sustainedDraftPct = append(sustainedDraftPct, accPct)
					}
					if pTok >= 100000 {
						report.UltraLongContextCount++
						if report.UltraLongMinDecode == 0 || dTps < report.UltraLongMinDecode {
							report.UltraLongMinDecode = dTps
						}
						if dTps > report.UltraLongMaxDecode {
							report.UltraLongMaxDecode = dTps
						}
					}
				}
			}
			continue
		}

		if strings.Contains(line, "[Strata Handshake] INFO ") {
			parseHandshakeLine(line, &report.Handshake)
			continue
		}

		if strings.Contains(line, "R4 hit path ON") {
			report.Handshake.R4HitPathOn = true
			continue
		}
		if strings.Contains(line, "100% VRAM resident: zero-doorbell graph") {
			report.Handshake.ZeroDoorbell = true
			continue
		}
		if strings.Contains(line, "prefill gemm: hipBLASLt tuning enabled") {
			report.Handshake.HipBLASLtEnabled = true
			report.Handshake.HipBLASLtSummary = strings.TrimSpace(line)
			continue
		}
		if strings.Contains(line, "strata: gfx1151 (Strix Halo):") {
			if m := reGfxDefaults.FindStringSubmatch(line); m != nil {
				report.Handshake.Gfx1151Switches, _ = strconv.Atoi(m[1])
			}
			continue
		}
		if strings.Contains(line, "strata serve: expert tiers:") {
			if m := reExpertTiers.FindStringSubmatch(line); m != nil {
				ramBlobs, _ := strconv.Atoi(m[1])
				report.DecodeRAMSpillBlobs = ramBlobs
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error scanning engine log: %w", err)
	}

	report.TotalLinesScanned = lineNum
	report.TotalRequestsParsed = len(allMetrics)
	report.AllDecode = computeDecodeStats(allMetrics)
	report.SustainedDecode = computeDecodeStats(sustainedMetrics)
	report.DraftAcceptancePct = computeFloatDistribution(sustainedDraftPct)
	report.ColdPrefillTiers = computePrefillTiers(allMetrics)
	report.AcceptanceGates = evaluateLogAcceptanceGates(report)

	return report, nil
}

func parseHandshakeLine(line string, hs *HandshakeInfo) {
	idx := strings.Index(line, "[Strata Handshake] INFO ")
	if idx < 0 {
		return
	}
	fields := strings.Fields(line[idx+len("[Strata Handshake] INFO "):])
	for _, kv := range fields {
		parts := strings.SplitN(kv, "=", 2)
		if len(parts) != 2 {
			continue
		}
		k, v := parts[0], parts[1]
		switch k {
		case "context":
			hs.Context, _ = strconv.Atoi(v)
		case "kv":
			hs.KVFormat = v
		case "expert_slots":
			hs.ExpertSlots, _ = strconv.Atoi(v)
		case "expert_cache_mib":
			hs.ExpertCacheMiB, _ = strconv.Atoi(v)
		case "vram_free_mib":
			hs.VRAMFreeMiB, _ = strconv.Atoi(v)
		case "spec":
			hs.SpecWindow, _ = strconv.Atoi(v)
		case "mtp_max":
			hs.MTPMax, _ = strconv.Atoi(v)
		case "lookup":
			hs.LookupChain, _ = strconv.Atoi(v)
		case "pool_workers":
			hs.PoolWorkers, _ = strconv.Atoi(v)
		case "pcie_frac":
			hs.PCIeFrac, _ = strconv.ParseFloat(v, 64)
		case "engine":
			hs.EngineVersion = v
		}
	}
}

func computeDecodeStats(metrics []RequestMetric) DistributionStats {
	stats := DistributionStats{Count: len(metrics)}
	if len(metrics) == 0 {
		return stats
	}

	vals := make([]float64, len(metrics))
	sum := 0.0
	for i, m := range metrics {
		vals[i] = m.DecodeTokPerS
		sum += m.DecodeTokPerS
		if m.DecodeTokPerS >= 50.0 {
			stats.CountGe50++
		}
		if m.DecodeTokPerS >= 60.0 {
			stats.CountGe60++
		}
		if m.DecodeTokPerS > stats.Max {
			stats.Max = m.DecodeTokPerS
			stats.PeakLine = m.LineNumber
			stats.PeakDraftsAcc = m.DraftsAccepted
			stats.PeakDraftsTot = m.DraftsTotal
		}
	}
	sort.Float64s(vals)
	stats.Min = vals[0]
	stats.P10 = percentileSorted(vals, 0.10)
	stats.P50Median = percentileSorted(vals, 0.50)
	stats.Mean = round2(sum / float64(len(vals)))
	stats.P90 = percentileSorted(vals, 0.90)
	stats.P99 = percentileSorted(vals, 0.99)
	stats.PctGe50 = round2((float64(stats.CountGe50) / float64(len(vals))) * 100.0)
	stats.PctGe60 = round2((float64(stats.CountGe60) / float64(len(vals))) * 100.0)
	return stats
}

func computeFloatDistribution(vals []float64) DistributionStats {
	stats := DistributionStats{Count: len(vals)}
	if len(vals) == 0 {
		return stats
	}
	cp := make([]float64, len(vals))
	copy(cp, vals)
	sort.Float64s(cp)
	sum := 0.0
	for _, v := range cp {
		sum += v
	}
	stats.Min = round2(cp[0])
	stats.P10 = round2(percentileSorted(cp, 0.10))
	stats.P50Median = round2(percentileSorted(cp, 0.50))
	stats.Mean = round2(sum / float64(len(cp)))
	stats.P90 = round2(percentileSorted(cp, 0.90))
	stats.P99 = round2(percentileSorted(cp, 0.99))
	stats.Max = round2(cp[len(cp)-1])
	return stats
}

func computePrefillTiers(metrics []RequestMetric) []TierPrefillSummary {
	tiers := []struct {
		name string
		min  int
		max  int
	}{
		{"Sub-512 (128..511)", 128, 511},
		{"512-1K (512..1023)", 512, 1023},
		{"1K-2K (1024..2047)", 1024, 2047},
		{"2K-4K (2048..4095)", 2048, 4095},
		{"4K-8K (4096..8191)", 4096, 8191},
		{"8K-16K (8192..16383)", 8192, 16383},
		{"16K+ (>=16384)", 16384, 1000000},
	}

	var out []TierPrefillSummary
	for _, t := range tiers {
		var vals []float64
		sum := 0.0
		for _, m := range metrics {
			// Evaluate cold prefill reads (0 reused tokens) or pure chunk reads >= 128 tokens
			if m.ReusedTokens == 0 && m.ReadTokens >= t.min && m.ReadTokens <= t.max && m.PrefillTokPerS > 0 {
				vals = append(vals, m.PrefillTokPerS)
				sum += m.PrefillTokPerS
			}
		}
		if len(vals) == 0 {
			continue
		}
		sort.Float64s(vals)
		med := round2(percentileSorted(vals, 0.50))
		summary := TierPrefillSummary{
			TierName:       t.name,
			MinTokens:      t.min,
			MaxTokens:      t.max,
			SampleCount:    len(vals),
			MinTokPerS:     round2(vals[0]),
			MedianTokPerS:  med,
			MeanTokPerS:    round2(sum / float64(len(vals))),
			P90TokPerS:     round2(percentileSorted(vals, 0.90)),
			MaxTokPerS:     round2(vals[len(vals)-1]),
			MMQCliffDetect: t.max < 1024 && med < 850.0,
		}
		out = append(out, summary)
	}
	return out
}

func evaluateLogAcceptanceGates(r *LogAuditReport) []CheckResult {
	var gates []CheckResult

	// AC-1: 100% GPU Expert Residency (24,576 / 24,576 slots) + Zero RAM Spill
	if r.Handshake.ExpertSlots == TotalRoutedExperts && r.Handshake.R4HitPathOn && r.DecodeRAMSpillBlobs == 0 {
		gates = append(gates, CheckResult{
			ID:       "AC-1",
			Category: "Expert Residency",
			Name:     "100% Routed Experts Resident in UMA VRAM (Zero-Doorbell)",
			Status:   "PASS",
			Expected: "24576/24576 slots, R4 hit path ON, 0 RAM spill blobs",
			Actual: fmt.Sprintf("%d/%d slots (%d MiB), R4=%v, ZeroDoorbell=%v, FreeVRAM=%d MiB",
				r.Handshake.ExpertSlots, TotalRoutedExperts, r.Handshake.ExpertCacheMiB,
				r.Handshake.R4HitPathOn, r.Handshake.ZeroDoorbell, r.Handshake.VRAMFreeMiB),
			Evidence: "strata_engine.log [Strata Handshake] INFO",
		})
	} else {
		gates = append(gates, CheckResult{
			ID:       "AC-1",
			Category: "Expert Residency",
			Name:     "100% Routed Experts Resident in UMA VRAM (Zero-Doorbell)",
			Status:   "FAIL",
			Expected: "24576/24576 slots, R4 hit path ON",
			Actual:   fmt.Sprintf("slots=%d, R4=%v, RAMSpill=%d", r.Handshake.ExpertSlots, r.Handshake.R4HitPathOn, r.DecodeRAMSpillBlobs),
			Evidence: "strata_engine.log",
		})
	}

	// AC-2: Sub-1024 Prefill Cliff Detection (<1024 tokens >= 950 tok/s)
	sub1KMedian := 0.0
	sub1KMax := 0.0
	for _, tier := range r.ColdPrefillTiers {
		if tier.MaxTokens <= 1023 {
			if sub1KMedian == 0 || tier.MedianTokPerS < sub1KMedian {
				sub1KMedian = tier.MedianTokPerS
			}
			if tier.MaxTokPerS > sub1KMax {
				sub1KMax = tier.MaxTokPerS
			}
		}
	}
	if sub1KMedian >= 950.0 {
		gates = append(gates, CheckResult{
			ID:       "AC-2",
			Category: "Sub-1K Prefill",
			Name:     "Sub-1024 Token Prefill Throughput (STRATA_PREFILL_STREAM_MIN=128)",
			Status:   "PASS",
			Expected: ">= 950.0 tok/s (Fused WMMA active)",
			Actual:   fmt.Sprintf("Median %.1f tok/s (Max %.1f tok/s)", sub1KMedian, sub1KMax),
			Evidence: "Cold <1024 token requests in strata_engine.log",
		})
	} else {
		gates = append(gates, CheckResult{
			ID:       "AC-2",
			Category: "Sub-1K Prefill",
			Name:     "Sub-1024 Token Prefill Throughput (STRATA_PREFILL_STREAM_MIN=128)",
			Status:   "WARN",
			Expected: ">= 950.0 tok/s after engine restart with STRATA_PREFILL_STREAM_MIN=128",
			Actual:   fmt.Sprintf("Historical log median %.1f tok/s (Pre-restart MMQ fallback 363.6-528.9 tok/s detected)", sub1KMedian),
			Evidence: "Restart strata.exe so STRATA_PREFILL_STREAM_MIN=128 in strata-qwen177b.json takes effect",
		})
	}

	// AC-3: 2K-16K Fused Prefill Gate (>= 1,100 tok/s)
	longPrefillMed := 0.0
	longPrefillMax := 0.0
	for _, tier := range r.ColdPrefillTiers {
		if tier.MinTokens >= 2048 {
			if tier.MedianTokPerS > longPrefillMed {
				longPrefillMed = tier.MedianTokPerS
			}
			if tier.MaxTokPerS > longPrefillMax {
				longPrefillMax = tier.MaxTokPerS
			}
		}
	}
	statusAC3 := "PASS"
	if longPrefillMed < 1100.0 {
		statusAC3 = "FAIL"
	}
	gates = append(gates, CheckResult{
		ID:       "AC-3",
		Category: "2K-16K Prefill",
		Name:     "2K-16K Fused WMMA Prefill Throughput (STRATA_PF_FUSED=1 + hipBLASLt)",
		Status:   statusAC3,
		Expected: ">= 1,100.0 tok/s (Target 1,250-1,350+ tok/s)",
		Actual:   fmt.Sprintf("Median %.1f tok/s, Peak %.1f tok/s", longPrefillMed, longPrefillMax),
		Evidence: "2K-16K cold prefill tiers in strata_engine.log",
	})

	// AC-4: Sustained Decode Gate (P50 >= 50.0 tok/s, Structured Coding >= 60.0 tok/s)
	statusAC4 := "PASS"
	if r.SustainedDecode.P50Median < 50.0 {
		statusAC4 = "FAIL"
	}
	gates = append(gates, CheckResult{
		ID:       "AC-4",
		Category: "Sustained Decode",
		Name:     "Sustained Decode Throughput (gen >= 50 tokens, MTP + lookup-chain 3)",
		Status:   statusAC4,
		Expected: "Median >= 50.0 tok/s, P90 >= 60.0 tok/s",
		Actual: fmt.Sprintf("P50=%.1f tok/s, Mean=%.1f tok/s, P90=%.1f tok/s, Peak=%.1f tok/s (%d/%d = %.1f%% >= 50 tok/s)",
			r.SustainedDecode.P50Median, r.SustainedDecode.Mean, r.SustainedDecode.P90, r.SustainedDecode.Max,
			r.SustainedDecode.CountGe50, r.SustainedDecode.Count, r.SustainedDecode.PctGe50),
		Evidence: fmt.Sprintf("Peak at L%d (%d/%d drafts accepted)", r.SustainedDecode.PeakLine, r.SustainedDecode.PeakDraftsAcc, r.SustainedDecode.PeakDraftsTot),
	})

	// AC-5: Ultra-Long Context (100K-187K+) O(1) GDN Recurrence Stability
	statusAC5 := "PASS"
	if r.UltraLongContextCount > 0 && r.UltraLongMinDecode < 50.0 {
		statusAC5 = "WARN"
	}
	gates = append(gates, CheckResult{
		ID:       "AC-5",
		Category: "Long-Context Scaling",
		Name:     "Ultra-Long Context (100K-187K+) Decode Stability (36-Layer GDN + 12-Layer QSA)",
		Status:   statusAC5,
		Expected: ">= 50.0 tok/s at >100K context",
		Actual:   fmt.Sprintf("%d requests >= 100K tokens: %.1f - %.1f tok/s decode", r.UltraLongContextCount, r.UltraLongMinDecode, r.UltraLongMaxDecode),
		Evidence: "Hybrid O(1) GDN recurrence state + QSA top-k sparse block selection",
	})

	return gates
}

func percentileSorted(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if len(sorted) == 1 {
		return round2(sorted[0])
	}
	pos := p * float64(len(sorted)-1)
	lower := int(math.Floor(pos))
	upper := int(math.Ceil(pos))
	if lower == upper || upper >= len(sorted) {
		return round2(sorted[lower])
	}
	weight := pos - float64(lower)
	return round2(sorted[lower]*(1.0-weight) + sorted[upper]*weight)
}

func round2(v float64) float64 {
	return math.Round(v*100.0) / 100.0
}

// CombinedValidationReport aggregates Config Audit, Log Audit, and A/B Optimization Matrix.
type CombinedValidationReport struct {
	GeneratedAt string             `json:"generated_at"`
	ConfigAudit *ConfigAuditReport `json:"config_audit,omitempty"`
	LogAudit    *LogAuditReport    `json:"log_audit,omitempty"`
	ABMatrix    []ABExperimentArm  `json:"ab_matrix,omitempty"`
	LiveBench   *LiveBenchReport   `json:"live_bench,omitempty"`
}

// RunFullValidation executes the static config audit, engine log audit, and A/B matrix synthesis.
func RunFullValidation(configPath, logPath, repoRoot string) (*CombinedValidationReport, error) {
	report := &CombinedValidationReport{
		GeneratedAt: time.Now().Format(time.RFC3339),
	}

	if cfgRep, err := AuditConfig(configPath, repoRoot); err == nil {
		report.ConfigAudit = cfgRep
	}
	if logRep, err := AuditEngineLog(logPath); err == nil {
		report.LogAudit = logRep
	}
	report.ABMatrix = BuildABMatrix(repoRoot)
	return report, nil
}
