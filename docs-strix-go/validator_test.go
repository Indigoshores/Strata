package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestSwitchCatalogCompleteness(t *testing.T) {
	specs := AllSwitchSpecs()
	if len(specs) < 34 {
		t.Fatalf("expected at least 34 switch specs (18 defaults + 11 active opt-ins + 6 v0.1.45 candidates), got %d", len(specs))
	}

	defaults := 0
	optins := 0
	candidates := 0
	for _, s := range specs {
		switch s.Category {
		case "default_gfx1151":
			defaults++
		case "optin_active":
			optins++
		case "candidate_v0145":
			candidates++
		}
	}
	if defaults != 18 {
		t.Errorf("expected 18 default_gfx1151 switches, got %d", defaults)
	}
	if optins < 10 {
		t.Errorf("expected at least 10 optin_active switches, got %d", optins)
	}
	if candidates < 6 {
		t.Errorf("expected at least 6 candidate_v0145 switches, got %d", candidates)
	}
}

func TestAuditConfigSyntheticAndProduction(t *testing.T) {
	// 1. Test synthetic broken config where STRATA_PREFILL_STREAM_MIN is 1024 (unfixed Sub-1024 cliff)
	tmpDir := t.TempDir()
	brokenCfg := ProductionConfig{
		Exe:       filepath.Join(tmpDir, "strata.exe"),
		ModelName: "Qwen 3.8 Flash-Next 177B Test",
		Backend:   "hip",
		Env: map[string]string{
			"STRATA_PREFILL_STREAM_MIN": "1024",
			"STRATA_PF_FUSED":           "1",
		},
		Args: []string{"--expert-cache", "auto", "--kv", "int8"},
	}
	brokenBytes, _ := json.Marshal(brokenCfg)
	brokenPath := filepath.Join(tmpDir, "broken.json")
	if err := os.WriteFile(brokenPath, brokenBytes, 0644); err != nil {
		t.Fatal(err)
	}

	repBroken, err := AuditConfig(brokenPath, DefaultRepoRoot)
	if err != nil {
		t.Fatalf("AuditConfig failed on synthetic config: %v", err)
	}
	if repBroken.Sub1024CliffFixed {
		t.Errorf("expected Sub1024CliffFixed=false when STRATA_PREFILL_STREAM_MIN=1024")
	}
	if repBroken.FailedCount == 0 {
		t.Errorf("expected failed checks on incomplete config, got 0")
	}

	// 2. Test live production config if present on disk
	if _, err := os.Stat(DefaultProdConfigPath); err == nil {
		repProd, err := AuditConfig(DefaultProdConfigPath, DefaultRepoRoot)
		if err != nil {
			t.Fatalf("AuditConfig failed on production config: %v", err)
		}
		if !repProd.Sub1024CliffFixed {
			t.Errorf("expected Sub1024CliffFixed=true in production strata-qwen177b.json")
		}
		if !repProd.TuningTableValid {
			t.Errorf("expected TuningTableValid=true in production strata-qwen177b.json")
		}
		if repProd.FailedCount != 0 {
			for _, c := range repProd.Checks {
				if c.Status == "FAIL" {
					t.Logf("FAILED CHECK: id=%s name=%s expected=%s actual=%s", c.ID, c.Name, c.Expected, c.Actual)
				}
			}
			t.Errorf("expected 0 failed checks on production config, got %d", repProd.FailedCount)
		}
	}
}

func TestAuditEngineLogSynthetic(t *testing.T) {
	tmpDir := t.TempDir()
	logFile := filepath.Join(tmpDir, "test_engine.log")
	syntheticLog := strings.Join([]string{
		`strata: gfx1151 (Strix Halo): 18 exact speed switches on by default (STRATA_GFX1151_DEFAULTS=0 turns them off; a switch you set is kept): GDN_HEAD GDN_PP GDN_CONVL2 GDN_NOY CVEC_FUSE HCD_EXACT PF_PAD Q8_PACKED Q6_PACKED MMVF_ROWS ATTN_LANECELL EXPERT_V2 TSUM LFUSE GDN_SPLIT QFUSE PLE_BATCH SH_STREAM`,
		`strata generate: R4 hit path ON - resident experts are computed on the GPU`,
		`prefill gemm: hipBLASLt tuning enabled (90 rows, gfx1151, version 100500)`,
		`strata verify: window up to 8 tokens, 100.0 MiB of device buffers (100% VRAM resident: zero-doorbell graph)`,
		`[Strata Handshake] INFO context=262144 kv=int8 kv_resident=0 expert_slots=24576 expert_cache_mib=56762 expert_slots_primary=24576 expert_cache_primary_mib=56762 spec=8 mtp_max=5 lookup=3 vram_free_mib=30016 cvec=0 arena_mib=0 pool_workers=15 pcie_frac=0.55 spec_min_p=0.05 conversation_cache_mib=0 conversation_cache_slots=4 conversation_cache_min_free_mib=2560 tail_role_token=-1 vram_elastic=0 engine=0.1.45`,
		`strata serve: prompt 512 tokens = 0 reused + 512 read in 480 ms (1066.7 tok/s), 64 generated in 1140 ms (56.1 tok/s), drafts accepted 53 of 80, 0 checkpoints`,
		`strata serve: expert tiers: GPU 0 hits this request; since the start RAM 0 blobs, files 0 blobs 59522.1 MB read (the GGUF in place)`,
		`strata serve: prompt 7636 tokens = 0 reused + 7636 read in 5764 ms (1324.7 tok/s), 256 generated in 4104 ms (62.4 tok/s), drafts accepted 209 of 248, 1 checkpoints`,
		`strata serve: prompt 186073 tokens = 185296 reused + 777 read in 710 ms (1094.4 tok/s), 128 generated in 1888 ms (67.8 tok/s), drafts accepted 103 of 122, 12 checkpoints`,
	}, "\n")

	if err := os.WriteFile(logFile, []byte(syntheticLog), 0644); err != nil {
		t.Fatal(err)
	}

	rep, err := AuditEngineLog(logFile)
	if err != nil {
		t.Fatalf("AuditEngineLog failed: %v", err)
	}
	if rep.TotalRequestsParsed != 3 {
		t.Fatalf("expected 3 parsed requests, got %d", rep.TotalRequestsParsed)
	}
	if rep.Handshake.ExpertSlots != 24576 || !rep.Handshake.R4HitPathOn || !rep.Handshake.ZeroDoorbell {
		t.Errorf("unexpected handshake state: %+v", rep.Handshake)
	}
	if rep.SustainedDecode.Max != 67.8 {
		t.Errorf("expected Max decode 67.8 tok/s, got %.1f", rep.SustainedDecode.Max)
	}
	for _, g := range rep.AcceptanceGates {
		if g.Status != "PASS" {
			t.Errorf("expected gate %s to PASS on post-fix synthetic log, got %s (%s)", g.ID, g.Status, g.Actual)
		}
	}
}

func TestZeroReusePromptGenerator(t *testing.T) {
	p1 := BuildZeroReusePrompt(512, "RUN_A")
	p2 := BuildZeroReusePrompt(512, "RUN_B")
	if p1 == p2 {
		t.Errorf("expected unique prompts across different runIDs to guarantee zero KV prefix reuse")
	}
	if !strings.HasPrefix(p1, "// NONCE_RUN_ID=RUN_A") {
		t.Errorf("expected nonce prefix at token 0 of prompt")
	}
	p8k := BuildZeroReusePrompt(8192, "RUN_8K")
	if len(p8k) < 20000 || len(p8k) > 23000 {
		t.Errorf("expected calibrated 8K prompt between 20,000 and 23,000 chars (2.65 chars/tok), got %d", len(p8k))
	}
}

func TestSSEStreamParserReasoningAndContent(t *testing.T) {
	// Verify that streaming <think> tokens in delta.reasoning_content plus delta.content
	// are both counted and never produce Gen=0 (0.0 tok/s).
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		// Emit 3 reasoning chunks and 5 content chunks with slight delay
		for i := 0; i < 3; i++ {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think_%d \"}}]}\n\n", i)
			if flusher != nil {
				flusher.Flush()
			}
			time.Sleep(5 * time.Millisecond)
		}
		for i := 0; i < 5; i++ {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"code_%d \"}}]}\n\n", i)
			if flusher != nil {
				flusher.Flush()
			}
			time.Sleep(5 * time.Millisecond)
		}
		fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":512,\"completion_tokens\":8,\"total_tokens\":520}}\n\n")
		fmt.Fprintf(w, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	}))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	res, err := RunSingleHTTPBenchmark(ctx, srv.URL, "", "test prompt", 512, 64, 0.0, "high")
	if err != nil {
		t.Fatalf("RunSingleHTTPBenchmark failed: %v", err)
	}
	if res.ReasoningTokens != 3 {
		t.Errorf("expected 3 reasoning tokens, got %d", res.ReasoningTokens)
	}
	if res.ContentTokens != 5 {
		t.Errorf("expected 5 content tokens, got %d", res.ContentTokens)
	}
	if res.GeneratedTokens != 8 {
		t.Errorf("expected 8 total generated tokens, got %d", res.GeneratedTokens)
	}
	if res.ClientDecodeTps <= 0 {
		t.Errorf("expected positive ClientDecodeTps, got %f", res.ClientDecodeTps)
	}

	// Verify hard invariant: reasoning="none" must be rejected unconditionally
	for _, banned := range []string{"none", "NONE", " None "} {
		if _, err := ValidateReasoningEffort(banned); err == nil {
			t.Errorf("expected ValidateReasoningEffort(%q) to fail, got nil", banned)
		}
		if _, err := RunSingleHTTPBenchmark(ctx, srv.URL, "", "test prompt", 512, 64, 0.0, banned); err == nil {
			t.Errorf("expected RunSingleHTTPBenchmark with reasoning=%q to fail, got nil", banned)
		}
	}
}

func TestABMatrixAndCTestGates(t *testing.T) {
	arms := BuildABMatrix(DefaultRepoRoot)
	if len(arms) != 7 {
		t.Errorf("expected 7 A-B-B-A optimization arms, got %d", len(arms))
	}
	gates := CTestGateSpecs()
	if len(gates) < 8 {
		t.Errorf("expected at least 8 CTest parity gates, got %d", len(gates))
	}
}
