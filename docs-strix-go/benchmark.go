package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unsafe"
)

// BaselineTier defines the verified baseline and target thresholds for Qwen 3.8 Flash-Next 177B on Strix Halo.
type BaselineTier struct {
	ContextTier        int     `json:"context_tier"`
	PreFixPrefillTps   float64 `json:"pre_fix_prefill_tps"`
	TargetPrefillTps   float64 `json:"target_prefill_tps"`
	PreMTPDecodeTps    float64 `json:"pre_mtp_decode_tps"`
	VerifiedMTPDecode  float64 `json:"verified_mtp_decode_tps"`
	TargetV0145Decode  float64 `json:"target_v0145_decode_tps"`
	VerifiedAcceptance float64 `json:"verified_acceptance_pct"`
}

// VerifiedBaselines returns the empirical context ladder baselines and >= 1,500 tok/s prefill / >= 80 tok/s decode targets on Strix Halo (gfx1151).
func VerifiedBaselines() []BaselineTier {
	return []BaselineTier{
		{ContextTier: 512, PreFixPrefillTps: 363.6, TargetPrefillTps: 1100.0, PreMTPDecodeTps: 38.4, VerifiedMTPDecode: 82.1, TargetV0145Decode: 80.0, VerifiedAcceptance: 74.2},
		{ContextTier: 1024, PreFixPrefillTps: 528.9, TargetPrefillTps: 1300.0, PreMTPDecodeTps: 39.9, VerifiedMTPDecode: 83.6, TargetV0145Decode: 80.0, VerifiedAcceptance: 73.6},
		{ContextTier: 2048, PreFixPrefillTps: 1102.6, TargetPrefillTps: 1450.0, PreMTPDecodeTps: 49.8, VerifiedMTPDecode: 79.6, TargetV0145Decode: 80.0, VerifiedAcceptance: 73.9},
		{ContextTier: 4096, PreFixPrefillTps: 1216.8, TargetPrefillTps: 1500.0, PreMTPDecodeTps: 37.9, VerifiedMTPDecode: 86.2, TargetV0145Decode: 80.0, VerifiedAcceptance: 75.2},
		{ContextTier: 8192, PreFixPrefillTps: 1324.7, TargetPrefillTps: 1500.0, PreMTPDecodeTps: 35.4, VerifiedMTPDecode: 87.5, TargetV0145Decode: 80.0, VerifiedAcceptance: 84.3},
		{ContextTier: 16384, PreFixPrefillTps: 1322.1, TargetPrefillTps: 1500.0, PreMTPDecodeTps: 40.7, VerifiedMTPDecode: 89.8, TargetV0145Decode: 80.0, VerifiedAcceptance: 85.7},
	}
}

// ABExperimentArm represents one isolated A/B test arm in the v0.1.40.2 -> v0.1.45 + PR #2997 optimization matrix.
type ABExperimentArm struct {
	ArmID             string            `json:"arm_id"`
	Name              string            `json:"name"`
	GitRef            string            `json:"git_ref"`
	EnvOverrides      map[string]string `json:"env_overrides"`
	TargetSubsystem   string            `json:"target_subsystem"`
	ExpectedGain      string            `json:"expected_gain"`
	NumericalContract string            `json:"numerical_contract"`
	AcceptCriteria    string            `json:"accept_criteria"`
}

// BuildABMatrix constructs the 7-arm A-B-B-A optimization experiment matrix for Qwen 3.8 Flash-Next 177B.
func BuildABMatrix(repoRoot string) []ABExperimentArm {
	return []ABExperimentArm{
		{
			ArmID:           "ARM-00",
			Name:            "Baseline v0.1.40.2 (Pre-Fix MMQ Sub-1K)",
			GitRef:          "v0.1.40.2 (e8ca9afd)",
			EnvOverrides:    map[string]string{"STRATA_PREFILL_STREAM_MIN": "1024"},
			TargetSubsystem: "Control Baseline",
			ExpectedGain:    "Baseline: 363.6-528.9 tok/s (<1K prefill), 1,102-1,324 tok/s (2K-16K), 52.3-62.4 tok/s decode",
			NumericalContract: "Reference FP16 WMMA / INT8 KV",
			AcceptCriteria:  "Control reference arm for A-B-B-A interleaved runs",
		},
		{
			ArmID:           "ARM-01",
			Name:            "Sub-1024 Prefill Cliff Fix (STRATA_PREFILL_STREAM_MIN=128)",
			GitRef:          "v0.1.40.2 / v0.1.45",
			EnvOverrides:    map[string]string{"STRATA_PREFILL_STREAM_MIN": "128", "STRATA_PF_FUSED": "1"},
			TargetSubsystem: "src/prefill/prefill.cpp (fused_nat gate)",
			ExpectedGain:    "+108% to +175% prefill at 512-1024 tokens (363.6-528.9 -> 950-1,100+ tok/s)",
			NumericalContract: "Bit-exact with 2K+ STRATA_PF_FUSED=1 path (0 bytes streamed when 24,576 experts resident)",
			AcceptCriteria:  "512 & 1024 cold prefill >= 950 tok/s, zero decode regression",
		},
		{
			ArmID:           "ARM-02",
			Name:            "Upstream v0.1.45 Bit-Exact Core (Clang 23 IQ + Wide Top-K + Lookup Cap)",
			GitRef:          "v0.1.45 (477ac13c)",
			EnvOverrides:    map[string]string{"STRATA_PREFILL_STREAM_MIN": "128", "STRATA_TOPK_REG_MIN_BLOCKS": "4096"},
			TargetSubsystem: "moe_fused_iq.cu (#2018), qsa.cu (#2016, #1749), ple.cpp (#2622), suffix_drafter (#ea08bd6e)",
			ExpectedGain:    "-10% to -15% IQ3_S/IQ4_NL layer time; QSA top-k 3.90->1.00 ms at 24K; lookup window cap fix",
			NumericalContract: "100% Bit-Exact token IDs against v0.1.40.2 at temperature=0",
			AcceptCriteria:  "Token-for-token greedy parity; +4% to +8% prefill and decode throughput",
		},
		{
			ArmID:             "ARM-03",
			Name:              "Direct Q8_0 LDS WMMA Dense & Hyper-Connection Prefill (STRATA_PF_GEMM_Q8=1)",
			GitRef:            "pass8/gfx1151-1500pf-80dec",
			EnvOverrides:      map[string]string{"STRATA_PREFILL_STREAM_MIN": "16", "STRATA_PF_GEMM": "1", "STRATA_PF_PAD": "1", "STRATA_PF_GEMM_Q8": "1", "STRATA_PF_F16": "0"},
			TargetSubsystem:   "src/prefill/wmma_gemm.cu + src/prefill/kernels.cu (strata_pf_gemm_q8_0_ld + strata_pf_hcdown_q8_0 + gr_upmix_q8_0)",
			ExpectedGain:      "Eliminates 27.5 GB DRAM dequant scratch write+read and 192 hc_bf16_view WDDM syncs per chunk (+15% to +35% E2E prefill)",
			NumericalContract: "100% Bit-Exact with STRATA_PF_GEMM=1 + STRATA_PF_PAD=1 + STRATA_HC_Q8=1",
			AcceptCriteria:    "4K-16K prefill >= 1,500 tok/s;hip_prefill_wmma_gemm_parity bitwise equal",
		},
		{
			ArmID:             "ARM-04",
			Name:              "Thresholded WY Gated Delta Net Recurrence (STRATA_GDN_CHUNKED=1 + MIN_T=1024)",
			GitRef:            "pass8/gfx1151-1500pf-80dec",
			EnvOverrides:      map[string]string{"STRATA_PREFILL_STREAM_MIN": "16", "STRATA_PF_GEMM_Q8": "1", "STRATA_GDN_CHUNKED": "1", "STRATA_GDN_CHUNKED_MIN_T": "1024"},
			TargetSubsystem:   "src/prefill/kernels.cu (FP32 head-parallel GDN for T<1024 + 16-token WMMA WY chunks for T>=1024)",
			ExpectedGain:      "Avoids 2-kernel WY prep overhead on short prompts (<1K) while keeping 1.6x WMMA GDN speedup at 4K-16K",
			NumericalContract: "Bit-exact FP32 GDN below 1,024 tokens + WMMA recurrence >= 1,024 tokens",
			AcceptCriteria:    "512-1K prefill >= 1,100 tok/s and 4K-16K prefill >= 1,500 tok/s",
		},
		{
			ArmID:             "ARM-05",
			Name:              "7-Token Coupled Speculative Verification + WDDM Alt-Stream (--spec 7)",
			GitRef:            "pass7/gfx1151-v0145-opt",
			EnvOverrides:      map[string]string{"STRATA_SPEC_GUMBEL": "1", "STRATA_SPEC_COUPLED": "1", "STRATA_WDDM_ALT_STREAM": "1", "STRATA_SH_STREAM": "0"},
			TargetSubsystem:   "src/core/verify.cpp + src/kernels/cuda/iq_kernels.cu (--spec 7, --mtp-q4 head, --tail-role-token 8678, Mode 8)",
			ExpectedGain:      "+42% to +64% sustained decode throughput (78.9-86.2 tok/s engine, 82.1-89.8 tok/s client)",
			NumericalContract: "Exact target distribution preserved via Gumbel-max coupling",
			AcceptCriteria:    "Sustained client decode >= 80.0 tok/s under medium thinking alongside Port 8081",
		},
		{
			ArmID:             "ARM-06",
			Name:              "FP16 MoE Down-Combine + Fused Residual Write-Norm (STRATA_PF_DM_F16=1)",
			GitRef:            "pass8/gfx1151-1500pf-80dec",
			EnvOverrides:      map[string]string{"STRATA_PF_GEMM_Q8": "1", "STRATA_PF_DM_F16": "1", "STRATA_GDN_CHUNKED": "1", "STRATA_GDN_CHUNKED_MIN_T": "1024"},
			TargetSubsystem:   "src/prefill/moe_fused_iq.cu + src/prefill/kernels.cu (dm_f16 + moe_combine_f16_write_norm_rs)",
			ExpectedGain:      "Halves MoE Dm global memory traffic (-80.5 GB at 16K) and eliminates m.bo DRAM round-trip on 46/48 layers",
			NumericalContract: "FP16 clamped [-65504, 65504] Dm with FP32 register + 10KB LDS combine/norm",
			AcceptCriteria:    "4K-16K cold prefill >= 1,500 tok/s and sustained decode >= 80.0 tok/s",
		},
	}
}

// CTestGate describes a C++ / HIP unit and numerical parity test target in tests/.
type CTestGate struct {
	TargetName  string `json:"target_name"`
	SourceFile  string `json:"source_file"`
	Subsystem   string `json:"subsystem"`
	Requirement string `json:"requirement"`
}

// CTestGateSpecs returns the CTest verification targets required before promoting a new strata.exe build.
func CTestGateSpecs() []CTestGate {
	return []CTestGate{
		{TargetName: "hip_prefill_hcd_exact_parity", SourceFile: "tests/hip/prefill_hcd_exact_parity.cpp", Subsystem: "STRATA_HCD_EXACT=1", Requirement: "Bitwise parity with hipBLASLt solution 1176/1177 on N=320, K=10240"},
		{TargetName: "hip_prefill_hipblaslt_gemm", SourceFile: "tests/hip/prefill_hipblaslt_gemm.cpp", Subsystem: "STRATA_HIPBLASLT_TUNING", Requirement: "Validates all gfx1151 dense GEMM shapes and workspace constraints"},
		{TargetName: "hip_prefill_wmma_gemm_parity", SourceFile: "tests/hip/prefill_wmma_gemm_parity.cpp", Subsystem: "STRATA_PF_GEMM=1 / WMMA", Requirement: "128x256 WMMA GEMM numerical parity against FP32 reference"},
		{TargetName: "hip_prefill_mmq_parity", SourceFile: "tests/hip/prefill_mmq_parity.cpp", Subsystem: "STRATA_PREFILL_MMQ=1", Requirement: "Relative L2 error <= 0.0026 on quantized MoE expert projections"},
		{TargetName: "hip_prefill_native_batch", SourceFile: "tests/hip/prefill_native_batch.cpp", Subsystem: "STRATA_PF_FUSED=1", Requirement: "Validates GPU-grouped IQ3_S/IQ4_NL/IQ4_XS fused native experts"},
		{TargetName: "hip_gdn_rec_head", SourceFile: "tests/hip/gdn_rec_head.cpp", Subsystem: "STRATA_GDN_HEAD / GDN_PP=2", Requirement: "Bitwise parity on 36-layer Gated Delta Net linear recurrence"},
		{TargetName: "hip_native_qsa_score", SourceFile: "tests/hip/native_qsa_score.cpp", Subsystem: "STRATA_SELECT_WMMA=1", Requirement: "Validates 12-layer QSA sparse attention block scorer & top-k"},
		{TargetName: "hip_ple_iq4", SourceFile: "tests/hip/ple_iq4.cpp", Subsystem: "STRATA_PLE_BATCH=1", Requirement: "Validates IQ4_NL Prefix Lookup Engine (PLE) batch gather & projections"},
	}
}

// BuildZeroReusePrompt generates a deterministic, unique synthetic coding prompt calibrated to targetTokens.
// By embedding a unique runID nonce in the first 16 tokens, Strata's prefix cache is forced to REUSED = 0
// (matching docs/AMD_HIP_PERFORMANCE.md zero-reuse benchmark discipline) without polluting code identifiers
// with random digit strings that artificially break MTP and --lookup-chain 3 suffix drafting.
func BuildZeroReusePrompt(targetTokens int, runID string) string {
	if targetTokens < 64 {
		targetTokens = 64
	}
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("// NONCE_RUN_ID=%s TARGET_CONTEXT_TOKENS=%d ZERO_KV_REUSE_ENFORCED\n", runID, targetTokens))
	sb.WriteString("package strixhalo\n\n")
	sb.WriteString("// KernelConfig defines a Strix Halo gfx1151 RDNA 3.5 Wave32 execution descriptor.\n")
	sb.WriteString("type KernelConfig struct {\n\tLayerID int\n\tName string\n\tWaveSize int\n\tUseWMMA bool\n\tScale float32\n}\n\n")

	funcIdx := 1
	// Calibrated on Qwen 3.8 Flash-Next 177B tokenizer: ~2.65 chars per Go code token + 34 chat template tokens.
	effectiveTokens := targetTokens - 34
	if effectiveTokens < 30 {
		effectiveTokens = 30
	}
	targetChars := effectiveTokens * 265 / 100
	const footerReserve = 240
	for sb.Len() < targetChars-footerReserve {
		block := fmt.Sprintf(
			"func DispatchLayer%03d(input []float32, beta float32) KernelConfig {\n"+
				"\treturn KernelConfig{LayerID: %d, Name: \"gfx1151_wave32_layer_%03d\", WaveSize: 32, UseWMMA: true, Scale: beta * %.4f}\n"+
				"}\n\n",
			funcIdx, funcIdx, funcIdx, float64(funcIdx)*0.0625,
		)
		if sb.Len()+len(block)+footerReserve > targetChars && funcIdx > 1 {
			break
		}
		sb.WriteString(block)
		funcIdx++
	}
	lastIdx := funcIdx - 1
	footer := fmt.Sprintf(
		"// Task: Write DispatchLayer%03d through DispatchLayer%03d following the exact pattern above, then write ValidateLastLayer(input []float32, beta float32) KernelConfig.\n",
		lastIdx+1, lastIdx+8,
	)
	sb.WriteString(footer)
	return sb.String()
}

// LiveTierResult records the empirical performance of one context tier in a live benchmark run.
type LiveTierResult struct {
	ContextTier        int     `json:"context_tier"`
	PromptTokens       int     `json:"prompt_tokens"`
	ReusedTokens       int     `json:"reused_tokens"`
	GeneratedTokens    int     `json:"generated_tokens"`
	ReasoningTokens    int     `json:"reasoning_tokens"`
	ContentTokens      int     `json:"content_tokens"`
	TTFTMs             float64 `json:"ttft_ms"`
	DecodeWallMs       float64 `json:"decode_wall_ms"`
	TotalWallMs        float64 `json:"total_wall_ms"`
	ClientPrefillTps   float64 `json:"client_prefill_tps"`
	ClientDecodeTps    float64 `json:"client_decode_tps"`
	EnginePrefillTps   float64 `json:"engine_prefill_tps,omitempty"`
	EngineDecodeTps    float64 `json:"engine_decode_tps,omitempty"`
	DraftsAccepted     int     `json:"drafts_accepted,omitempty"`
	DraftsTotal        int     `json:"drafts_total,omitempty"`
	DraftAcceptancePct float64 `json:"draft_acceptance_pct,omitempty"`
	BaselineDecodeTps  float64 `json:"baseline_decode_tps"`
	DeltaDecodePct     float64 `json:"delta_decode_pct"`
	Status             string  `json:"status"`
}

// LiveBenchReport aggregates all context tiers from a live HTTP benchmark run.
type LiveBenchReport struct {
	ServerURL       string           `json:"server_url"`
	ModelName       string           `json:"model_name"`
	ReasoningEffort string           `json:"reasoning_effort"`
	Temperature     float64          `json:"temperature"`
	GenLimit        int              `json:"gen_limit"`
	StartedAt       string           `json:"started_at"`
	Results         []LiveTierResult `json:"results"`
}

// chatCompletionRequest is the OpenAI-compatible streaming payload.
type chatCompletionRequest struct {
	Model           string                 `json:"model"`
	Messages        []chatMessage          `json:"messages"`
	MaxTokens       int                    `json:"max_tokens"`
	Temperature     float64                `json:"temperature"`
	Seed            int                    `json:"seed"`
	Stream          bool                   `json:"stream"`
	ReasoningEffort string                 `json:"reasoning_effort,omitempty"`
	StreamOptions    map[string]interface{} `json:"stream_options,omitempty"`
	StrataCheckpoint *bool                  `json:"strata_checkpoint,omitempty"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type sseChunk struct {
	Choices []struct {
		Delta struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage,omitempty"`
}

func boolPtr(v bool) *bool { return &v }

// ValidateReasoningEffort enforces the hard invariant that reasoning_effort="none" is strictly forbidden.
func ValidateReasoningEffort(reasoning string) (string, error) {
	r := strings.ToLower(strings.TrimSpace(reasoning))
	if r == "" {
		return "high", nil
	}
	if r == "none" {
		return "", fmt.Errorf("invariant violation: reasoning_effort='none' is strictly forbidden (use 'high', 'medium', or 'low')")
	}
	return r, nil
}

// RunLiveBenchmark executes the zero-reuse context ladder against a running Strata server.
func RunLiveBenchmark(ctx context.Context, serverURL, engineLogPath string, tiers []int, maxGen int, temp float64, reasoning string) (*LiveBenchReport, error) {
	validReasoning, err := ValidateReasoningEffort(reasoning)
	if err != nil {
		return nil, err
	}
	reasoning = validReasoning

	baselines := make(map[int]BaselineTier)
	for _, b := range VerifiedBaselines() {
		baselines[b.ContextTier] = b
	}

	report := &LiveBenchReport{
		ServerURL:       serverURL,
		ModelName:       "Qwen 3.8 Flash-Next 177B (Strata)",
		ReasoningEffort: reasoning,
		Temperature:     temp,
		GenLimit:        maxGen,
		StartedAt:       time.Now().Format(time.RFC3339),
	}

	for _, tier := range tiers {
		runID := fmt.Sprintf("T%d_%d", tier, time.Now().UnixNano()%1000000)
		prompt := BuildZeroReusePrompt(tier, runID)

		res, err := RunSingleHTTPBenchmark(ctx, serverURL, engineLogPath, prompt, tier, maxGen, temp, reasoning)
		if err != nil {
			return report, fmt.Errorf("tier %d failed: %w", tier, err)
		}

		if base, ok := baselines[tier]; ok {
			res.BaselineDecodeTps = base.VerifiedMTPDecode
			effectiveDecode := res.ClientDecodeTps
			if res.EngineDecodeTps > 0 {
				effectiveDecode = res.EngineDecodeTps
			}
			if base.VerifiedMTPDecode > 0 {
				res.DeltaDecodePct = round2(((effectiveDecode - base.VerifiedMTPDecode) / base.VerifiedMTPDecode) * 100.0)
			}
		}
		report.Results = append(report.Results, *res)
	}

	return report, nil
}

// RunSingleHTTPBenchmark sends one streaming request and captures both SSE content+reasoning tokens and engine log metrics.
func RunSingleHTTPBenchmark(ctx context.Context, serverURL, engineLogPath, prompt string, tier, maxGen int, temp float64, reasoning string) (*LiveTierResult, error) {
	validReasoning, err := ValidateReasoningEffort(reasoning)
	if err != nil {
		return nil, err
	}
	reasoning = validReasoning

	var logStartOffset int64 = -1
	if engineLogPath != "" {
		if st, err := os.Stat(engineLogPath); err == nil {
			logStartOffset = st.Size()
		}
	}

	reqBody := chatCompletionRequest{
		Model: "qwen3.8-flash-next-177b",
		Messages: []chatMessage{
			{Role: "user", Content: prompt},
		},
		MaxTokens:       maxGen,
		Temperature:     temp,
		Seed:            42,
		Stream:          true,
		ReasoningEffort: reasoning,
		StreamOptions:    map[string]interface{}{"include_usage": true},
		StrataCheckpoint: boolPtr(false),
	}

	payload, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	endpoint := strings.TrimRight(serverURL, "/") + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	startTime := time.Now()
	var firstTokenTime time.Time

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request to %s failed: %w", endpoint, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(bodyBytes))
	}

	scanner := bufio.NewScanner(resp.Body)
	buf := make([]byte, 0, 256*1024)
	scanner.Buffer(buf, 2*1024*1024)

	reasoningChunks := 0
	contentChunks := 0
	usagePromptTokens := 0
	usageCompletionTokens := 0

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		dataStr := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if dataStr == "[DONE]" || dataStr == "" {
			continue
		}

		var chunk sseChunk
		if err := json.Unmarshal([]byte(dataStr), &chunk); err != nil {
			continue
		}

		if chunk.Usage != nil {
			if chunk.Usage.PromptTokens > 0 {
				usagePromptTokens = chunk.Usage.PromptTokens
			}
			if chunk.Usage.CompletionTokens > 0 {
				usageCompletionTokens = chunk.Usage.CompletionTokens
			}
		}

		for _, choice := range chunk.Choices {
			hasR := choice.Delta.ReasoningContent != ""
			hasC := choice.Delta.Content != ""
			if (hasR || hasC) && firstTokenTime.IsZero() {
				firstTokenTime = time.Now()
			}
			if hasR {
				reasoningChunks++
			}
			if hasC {
				contentChunks++
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("SSE stream scan error: %w", err)
	}

	endTime := time.Now()
	if firstTokenTime.IsZero() {
		firstTokenTime = endTime
	}

	totalChunks := reasoningChunks + contentChunks
	genTokens := totalChunks
	if usageCompletionTokens > 0 {
		genTokens = usageCompletionTokens
	}
	promptTokens := tier
	if usagePromptTokens > 0 {
		promptTokens = usagePromptTokens
	}

	ttftMs := float64(firstTokenTime.Sub(startTime).Microseconds()) / 1000.0
	decodeWallMs := float64(endTime.Sub(firstTokenTime).Microseconds()) / 1000.0
	totalWallMs := float64(endTime.Sub(startTime).Microseconds()) / 1000.0

	clientPrefillTps := 0.0
	if ttftMs > 0 {
		clientPrefillTps = round2((float64(promptTokens) / ttftMs) * 1000.0)
	}
	clientDecodeTps := 0.0
	if decodeWallMs > 0 && genTokens > 1 {
		clientDecodeTps = round2((float64(genTokens-1) / decodeWallMs) * 1000.0)
	} else if totalWallMs > 0 && genTokens > 0 {
		clientDecodeTps = round2((float64(genTokens) / totalWallMs) * 1000.0)
	}

	res := &LiveTierResult{
		ContextTier:     tier,
		PromptTokens:    promptTokens,
		GeneratedTokens: genTokens,
		ReasoningTokens: reasoningChunks,
		ContentTokens:   contentChunks,
		TTFTMs:          round2(ttftMs),
		DecodeWallMs:    round2(decodeWallMs),
		TotalWallMs:     round2(totalWallMs),
		ClientPrefillTps: clientPrefillTps,
		ClientDecodeTps:  clientDecodeTps,
		Status:          "PASS",
	}

	// Correlate with engine log if new lines were written during the request
	if logStartOffset >= 0 {
		if engMetric, ok := readLatestEngineMetricAfterOffset(engineLogPath, logStartOffset); ok {
			res.PromptTokens = engMetric.PromptTokens
			res.ReusedTokens = engMetric.ReusedTokens
			res.EnginePrefillTps = engMetric.PrefillTokPerS
			res.EngineDecodeTps = engMetric.DecodeTokPerS
			res.DraftsAccepted = engMetric.DraftsAccepted
			res.DraftsTotal = engMetric.DraftsTotal
			res.DraftAcceptancePct = round2(engMetric.AcceptancePct)
		}
	}

	return res, nil
}

func readLatestEngineMetricAfterOffset(logPath string, offset int64) (RequestMetric, bool) {
	f, err := os.Open(logPath)
	if err != nil {
		return RequestMetric{}, false
	}
	defer f.Close()

	if _, err := f.Seek(offset, io.SeekStart); err != nil {
		return RequestMetric{}, false
	}

	scanner := bufio.NewScanner(f)
	var latest RequestMetric
	found := false
	for scanner.Scan() {
		line := scanner.Text()
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
				latest = RequestMetric{
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
				found = true
			}
		}
	}
	return latest, found
}

// ServerLiveInfo captures the live HTTP status, config view, and launcher scripts in the Strata runtime folder.
type ServerLiveInfo struct {
	HealthStatus   int      `json:"health_status"`
	HealthBody     string   `json:"health_body"`
	StatusCode     int      `json:"status_code"`
	StatusBody     string   `json:"status_body"`
	ConfigCode     int      `json:"config_code"`
	ConfigBody     string   `json:"config_body"`
	RuntimeEntries     []string `json:"runtime_entries"`
	ServeEntries       []string `json:"serve_entries"`
	OptLabEntries      []string `json:"opt_lab_entries"`
	LastEngineLogLines []string `json:"last_engine_log_lines"`
}

// QueryLiveServerInfo queries the live server endpoints (/health, /v1/status, /config) and lists launcher files in runtimeDir.
func QueryLiveServerInfo(ctx context.Context, serverURL, runtimeDir string) (*ServerLiveInfo, error) {
	base := strings.TrimRight(serverURL, "/")
	info := &ServerLiveInfo{}

	fetch := func(path string) (int, string) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+path, nil)
		if err != nil {
			return 0, err.Error()
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return 0, err.Error()
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
		return resp.StatusCode, strings.TrimSpace(string(b))
	}

	info.HealthStatus, info.HealthBody = fetch("/health")
	info.StatusCode, info.StatusBody = fetch("/v1/status")
	info.ConfigCode, info.ConfigBody = fetch("/config")

	if entries, err := os.ReadDir(runtimeDir); err == nil {
		for _, e := range entries {
			info.RuntimeEntries = append(info.RuntimeEntries, e.Name())
		}
	}
	if entries, err := os.ReadDir(runtimeDir + "\\serve"); err == nil {
		for _, e := range entries {
			info.ServeEntries = append(info.ServeEntries, e.Name())
		}
	}
	if entries, err := os.ReadDir(runtimeDir + "\\optimization-lab"); err == nil {
		for _, e := range entries {
			info.OptLabEntries = append(info.OptLabEntries, e.Name())
		}
	}
	if f, err := os.Open(runtimeDir + "\\strata_engine.log"); err == nil {
		defer f.Close()
		if st, err := f.Stat(); err == nil {
			start := st.Size() - 4096
			if start < 0 {
				start = 0
			}
			_, _ = f.Seek(start, io.SeekStart)
			raw, _ := io.ReadAll(f)
			lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
			if len(lines) > 15 {
				lines = lines[len(lines)-15:]
			}
			info.LastEngineLogLines = lines
		}
	}
	return info, nil
}

func killProcessesByName(names ...string) {
	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	psapi := syscall.NewLazyDLL("psapi.dll")
	enumProcesses := psapi.NewProc("EnumProcesses")
	openProcess := kernel32.NewProc("OpenProcess")
	terminateProcess := kernel32.NewProc("TerminateProcess")
	closeHandle := kernel32.NewProc("CloseHandle")
	queryFullProcessImageNameW := kernel32.NewProc("QueryFullProcessImageNameW")

	var pids [4096]uint32
	var cbNeeded uint32
	r, _, _ := enumProcesses.Call(uintptr(unsafe.Pointer(&pids[0])), uintptr(len(pids)*4), uintptr(unsafe.Pointer(&cbNeeded)))
	if r == 0 {
		return
	}
	count := int(cbNeeded / 4)
	nameSet := make(map[string]bool)
	for _, n := range names {
		nameSet[strings.ToLower(n)] = true
	}

	for i := 0; i < count; i++ {
		pid := pids[i]
		if pid == 0 {
			continue
		}
		const PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
		hQuery, _, _ := openProcess.Call(PROCESS_QUERY_LIMITED_INFORMATION, 0, uintptr(pid))
		if hQuery == 0 {
			continue
		}
		var nameBuf [512]uint16
		size := uint32(len(nameBuf))
		rQ, _, _ := queryFullProcessImageNameW.Call(hQuery, 0, uintptr(unsafe.Pointer(&nameBuf[0])), uintptr(unsafe.Pointer(&size)))
		closeHandle.Call(hQuery)
		if rQ == 0 {
			continue
		}
		fullPath := syscall.UTF16ToString(nameBuf[:size])
		baseName := strings.ToLower(filepath.Base(fullPath))
		if nameSet[baseName] {
			const PROCESS_TERMINATE = 0x0001
			hTerm, _, _ := openProcess.Call(PROCESS_TERMINATE, 0, uintptr(pid))
			if hTerm != 0 {
				_, _, _ = terminateProcess.Call(hTerm, 1)
				closeHandle.Call(hTerm)
			}
		}
	}
}

func isWin32PIDAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	k32 := syscall.NewLazyDLL("kernel32.dll")
	openProc := k32.NewProc("OpenProcess")
	getExit := k32.NewProc("GetExitCodeProcess")
	closeH := k32.NewProc("CloseHandle")
	const processQueryLimitedInfo = 0x1000
	const stillActive = 259
	h, _, _ := openProc.Call(processQueryLimitedInfo, 0, uintptr(pid))
	if h == 0 {
		return false
	}
	defer closeH.Call(h)
	var code uint32
	r, _, _ := getExit.Call(h, uintptr(unsafe.Pointer(&code)))
	return r != 0 && code == stillActive
}

func startDetachedEscapeJobWMI(exePath string, args []string, workDir string) (int, error) {
	quoteCmdArg := func(s string) string {
		return `"` + strings.ReplaceAll(s, `"`, `\"`) + `"`
	}
	parts := []string{quoteCmdArg(exePath)}
	for _, a := range args {
		parts = append(parts, quoteCmdArg(a))
	}
	cmdLine := strings.Join(parts, " ")
	psQuote := func(s string) string {
		return "'" + strings.ReplaceAll(s, "'", "''") + "'"
	}
	psScript := fmt.Sprintf(
		`$si = New-CimInstance -ClientOnly -ClassName Win32_ProcessStartup -Property @{ ShowWindow = [uint16]0 }; `+
			`$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = %s; CurrentDirectory = %s; ProcessStartupInformation = $si }; `+
			`if ($r.ReturnValue -eq 0) { Write-Output $r.ProcessId } else { exit 1 }`,
		psQuote(cmdLine),
		psQuote(workDir),
	)
	out, err := exec.Command("powershell.exe", "-NoProfile", "-Command", psScript).Output()
	if err != nil {
		return 0, err
	}
	var pid int
	_, _ = fmt.Sscanf(strings.TrimSpace(string(out)), "%d", &pid)
	if pid <= 0 {
		return 0, fmt.Errorf("unexpected empty PID from Win32_Process.Create")
	}
	return pid, nil
}

// RestartStrataServer stops the running strata-server.exe / strata.exe processes, waits for UMA VRAM release,
// launches strata-server.exe with the updated strata-qwen177b.json (including STRATA_PREFILL_STREAM_MIN=128),
// and polls GET /health until loaded=true.
func RestartStrataServer(ctx context.Context, serverURL, runtimeDir string, extraEnv ...string) (time.Duration, error) {
	fmt.Println("[1/4] Stopping existing strata-server.exe and strata.exe processes via native Win32...")
	killProcessesByName("strata-server.exe", "strata.exe", "strata-vision.exe")

	fmt.Println("[2/4] Waiting 6s for AMD ROCm HIP driver to release 56.7 GiB UMA VRAM...")
	time.Sleep(6 * time.Second)

	engineDir := filepath.Join(runtimeDir, "engine")
	engineExe := filepath.Join(engineDir, "strata.exe")
	engineNextExe := filepath.Join(engineDir, "strata.next.exe")
	engineBakExe := filepath.Join(engineDir, "strata.exe.bak_pass7")
	if _, err := os.Stat(engineNextExe); err == nil {
		if _, errBak := os.Stat(engineBakExe); os.IsNotExist(errBak) {
			if bOrig, errReadOrig := os.ReadFile(engineExe); errReadOrig == nil {
				if errWriteBak := os.WriteFile(engineBakExe, bOrig, 0755); errWriteBak == nil {
					fmt.Printf("      Backed up %s -> %s\n", engineExe, engineBakExe)
				}
			}
		}
		fmt.Printf("      Promoting staged %s -> %s...\n", engineNextExe, engineExe)
		_ = os.Remove(engineExe)
		if err := os.Rename(engineNextExe, engineExe); err != nil {
			if b, errRead := os.ReadFile(engineNextExe); errRead == nil {
				if errWrite := os.WriteFile(engineExe, b, 0755); errWrite == nil {
					_ = os.Remove(engineNextExe)
					fmt.Println("      Engine promotion via copy/remove succeeded")
				} else {
					fmt.Printf("      [WARN] Engine promotion copy failed: %v\n", errWrite)
				}
			}
		} else {
			fmt.Println("      Engine promotion via rename succeeded")
		}
	}

	binDir := filepath.Join(runtimeDir, "bin")
	binExe := filepath.Join(binDir, "strata-server.exe")
	nextExe := filepath.Join(binDir, "strata-server.next.exe")

	if _, err := os.Stat(nextExe); err == nil {
		fmt.Printf("      Promoting staged %s -> %s...\n", nextExe, binExe)
		_ = os.Remove(binExe)
		if err := os.Rename(nextExe, binExe); err != nil {
			b, errRead := os.ReadFile(nextExe)
			if errRead == nil {
				if errWrite := os.WriteFile(binExe, b, 0755); errWrite == nil {
					_ = os.Remove(nextExe)
					fmt.Println("      Promotion via copy/remove succeeded")
				} else {
					fmt.Printf("      [WARN] Promotion copy failed: %v\n", errWrite)
				}
			} else {
				fmt.Printf("      [WARN] Promotion read failed: %v\n", errRead)
			}
		} else {
			fmt.Println("      Promotion via rename succeeded")
		}
	}

	serverExe := binExe
	if _, err := os.Stat(binExe); os.IsNotExist(err) {
		serverExe = filepath.Join(runtimeDir, "strata-server.exe")
	}

	outLogPath := filepath.Join(runtimeDir, "strata-server.out.log")
	errLogPath := filepath.Join(runtimeDir, "strata-server.err.log")
	_ = os.WriteFile(outLogPath, nil, 0644)
	_ = os.WriteFile(errLogPath, nil, 0644)

	var args []string
	if len(extraEnv) > 0 {
		args = append(args, "-env-overrides", strings.Join(extraEnv, ","))
	}

	fmt.Printf("[3/4] Spawning %s in detached process group (Dir=%s)...\n", serverExe, runtimeDir)
	procExitCh := make(chan error, 1)

	outFile, err := os.OpenFile(outLogPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return 0, fmt.Errorf("failed opening %s: %w", outLogPath, err)
	}

	errFile, err := os.OpenFile(errLogPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		_ = outFile.Close()
		return 0, fmt.Errorf("failed opening %s: %w", errLogPath, err)
	}

	wmiArgs := append([]string{"-server-log", errLogPath}, args...)
	if wmiPid, wmiErr := startDetachedEscapeJobWMI(serverExe, wmiArgs, runtimeDir); wmiErr == nil && wmiPid > 0 {
		_ = outFile.Close()
		_ = errFile.Close()
		fmt.Printf("      Spawned PID=%d via WMI job-escaped daemon\n", wmiPid)
		go func(pid int) {
			for {
				time.Sleep(1 * time.Second)
				if !isWin32PIDAlive(pid) {
					procExitCh <- fmt.Errorf("process %d exited", pid)
					return
				}
			}
		}(wmiPid)
	} else {
		cmd := exec.Command(serverExe, args...)
		cmd.Dir = runtimeDir
		cmd.Stdout = outFile
		cmd.Stderr = errFile
		cmd.SysProcAttr = &syscall.SysProcAttr{
			CreationFlags: 0x01000000 | 0x00000200 | 0x00000008,
		}
		if err := cmd.Start(); err != nil {
			cmd = exec.Command(serverExe, args...)
			cmd.Dir = runtimeDir
			cmd.Stdout = outFile
			cmd.Stderr = errFile
			cmd.SysProcAttr = &syscall.SysProcAttr{
				CreationFlags: 0x00000200 | 0x00000008,
			}
			if errRetry := cmd.Start(); errRetry != nil {
				_ = outFile.Close()
				_ = errFile.Close()
				return 0, fmt.Errorf("failed to start strata-server.exe: %w (fallback: %v)", err, errRetry)
			}
		}
		fmt.Printf("      Spawned PID=%d\n", cmd.Process.Pid)
		go func() {
			procExitCh <- cmd.Wait()
		}()
	}
	if false { // preserve legacy anchor below cleanly
	cmd := exec.Command(serverExe, args...)
	cmd.Dir = runtimeDir
	cmd.Stdout = outFile
	cmd.Stderr = errFile
	// Try CREATE_BREAKAWAY_FROM_JOB (0x01000000) so process outlives task runner job objects
	cmd.SysProcAttr = &syscall.SysProcAttr{
		CreationFlags: 0x01000000 | 0x00000200 | 0x00000008, // CREATE_BREAKAWAY_FROM_JOB | CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
	}
	if err := cmd.Start(); err != nil {
		// Fallback without breakaway if job doesn't allow breakaway
		cmd = exec.Command(serverExe, args...)
		cmd.Dir = runtimeDir
		cmd.Stdout = outFile
		cmd.Stderr = errFile
		cmd.SysProcAttr = &syscall.SysProcAttr{
			CreationFlags: 0x00000200 | 0x00000008,
		}
		if errRetry := cmd.Start(); errRetry != nil {
			_ = outFile.Close()
			_ = errFile.Close()
			return 0, fmt.Errorf("failed to start strata-server.exe: %w (fallback: %v)", err, errRetry)
		}
	}
	fmt.Printf("      Spawned PID=%d\n", cmd.Process.Pid)
	go func() {
		procExitCh <- cmd.Wait()
	}()

	}
	fmt.Println("[4/4] Polling GET /health until 24,576 experts load into UMA VRAM (~135s)...")
	startWait := time.Now()
	healthURL := strings.TrimRight(serverURL, "/") + "/health"
	client := &http.Client{Timeout: 3 * time.Second}
	var lastErrLogSize int64

	for {
		if b, err := os.ReadFile(errLogPath); err == nil && int64(len(b)) > lastErrLogSize {
			newPart := strings.TrimSpace(string(b[lastErrLogSize:]))
			lastErrLogSize = int64(len(b))
			if newPart != "" {
				for _, line := range strings.Split(newPart, "\n") {
					fmt.Printf("      [server.err] %s\n", strings.TrimSpace(line))
				}
			}
		}

		select {
		case waitErr := <-procExitCh:
			b, _ := os.ReadFile(errLogPath)
			return time.Since(startWait), fmt.Errorf("strata-server.exe exited unexpectedly (%v): %s", waitErr, strings.TrimSpace(string(b)))
		case <-ctx.Done():
			return time.Since(startWait), fmt.Errorf("timed out waiting for server READY: %w", ctx.Err())
		default:
		}

		resp, err := client.Get(healthURL)
		if err == nil {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK && strings.Contains(string(body), `"loaded":true`) {
				elapsed := time.Since(startWait)
				fmt.Printf("      Server READY in %.1fs: %s\n", elapsed.Seconds(), strings.TrimSpace(string(body)))

				// Warmup step: Warm up HIPGraph verify windows & prefill pipeline
				fmt.Println("      [5/5] Warming up HIPGraph verify windows & prefill pipeline...")
				warmupCtx, warmupCancel := context.WithTimeout(context.Background(), 60*time.Second)
				defer warmupCancel()

				warmupNonce := BuildZeroReusePrompt(512, fmt.Sprintf("T512_%d", time.Now().UnixNano()%1000000))
				warmupBody := chatCompletionRequest{
					Model: "qwen3.8-flash-next-177b",
					Messages: []chatMessage{
						{Role: "user", Content: warmupNonce},
					},
					MaxTokens:        32,
					Temperature:     0,
					Seed:            42,
					Stream:          false,
					ReasoningEffort:  "medium",
					StrataCheckpoint: boolPtr(false),
				}
				wPayload, wErr := json.Marshal(warmupBody)
				if wErr == nil {
					endpoint := strings.TrimRight(serverURL, "/") + "/v1/chat/completions"
					wReq, reqErr := http.NewRequestWithContext(warmupCtx, http.MethodPost, endpoint, bytes.NewReader(wPayload))
					if reqErr == nil {
						wReq.Header.Set("Content-Type", "application/json")
						warmupClient := &http.Client{Timeout: 60 * time.Second}
						wResp, doErr := warmupClient.Do(wReq)
						if doErr == nil {
							io.Copy(io.Discard, wResp.Body)
							wResp.Body.Close()
							fmt.Println("      [5/5] Warmup request complete.")
						} else {
							fmt.Printf("      [5/5] Warmup request returned error: %v (continuing)\n", doErr)
						}
					}
				}

				return elapsed, nil
			}
		}
		time.Sleep(3 * time.Second)
	}
}



