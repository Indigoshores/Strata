package regressions

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

// ProductionConfigSchema represents the minimal JSON structure of strata-qwen177b.json.
type ProductionConfigSchema struct {
	Exe             string            `json:"exe"`
	Backend         string            `json:"backend"`
	ModelName       string            `json:"model_name"`
	ReasoningEffort string            `json:"reasoning_effort"`
	Env             map[string]string `json:"env"`
	Args            []string          `json:"args"`
}

// TestPass9HybridPrefillDecodeInvariants tests Pass 9.1 hybrid prefill and decode configuration invariants.
func TestPass9HybridPrefillDecodeInvariants(t *testing.T) {
	configPath := `C:\Users\Butch\Documents\10-Models\Strata\strata-qwen177b.json`
	raw, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("failed reading config at %s: %v", configPath, err)
	}

	var cfg ProductionConfigSchema
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatalf("failed parsing %s: %v", configPath, err)
	}

	// 1. Invariant: STRATA_NO_TURN_SPLIT=1 prevents turn splitting and preserves prompt single-chunk execution
	if val, ok := cfg.Env["STRATA_NO_TURN_SPLIT"]; !ok || val != "1" {
		t.Errorf("invariant violation: STRATA_NO_TURN_SPLIT must be '1', got %q (present=%v)", val, ok)
	}

	// 2. Invariant: STRATA_PF_F16=1 + STRATA_PF_F16_MIN_T=0 + STRATA_PF_MOE_SCATTER=0 unlocks full FP16 WMMA MoE prefill and Dm F16
	if val, ok := cfg.Env["STRATA_PF_F16"]; !ok || val != "1" {
		t.Errorf("invariant violation: STRATA_PF_F16 must be '1', got %q (present=%v)", val, ok)
	}
	if val, ok := cfg.Env["STRATA_PF_F16_MIN_T"]; !ok || val != "0" {
		t.Errorf("invariant violation: STRATA_PF_F16_MIN_T must be '0', got %q (present=%v)", val, ok)
	}
	if val, ok := cfg.Env["STRATA_PF_MOE_SCATTER"]; !ok || val != "0" {
		t.Errorf("invariant violation: STRATA_PF_MOE_SCATTER must be '0' (unlocks STRATA_PF_DM_F16=1 and F16 down projections), got %q (present=%v)", val, ok)
	}

	// 3. Invariant: STRATA_PREFILL_STREAM_MIN <= 128 protects against sub-1024 prefill cliff
	if val, ok := cfg.Env["STRATA_PREFILL_STREAM_MIN"]; !ok {
		t.Errorf("invariant violation: STRATA_PREFILL_STREAM_MIN must be present")
	} else {
		n, errConv := strconv.Atoi(val)
		if errConv != nil || n > 128 || n <= 0 {
			t.Errorf("invariant violation: STRATA_PREFILL_STREAM_MIN must be in [1..128], got %q", val)
		}
	}

	// 4. Invariant: --prefill must be 18432 (single-chunk covering 16K tier 17,613 tokens)
	argMap := make(map[string]string)
	for i := 0; i < len(cfg.Args); i++ {
		if strings.HasPrefix(cfg.Args[i], "--") {
			if i+1 < len(cfg.Args) && !strings.HasPrefix(cfg.Args[i+1], "--") {
				argMap[cfg.Args[i]] = cfg.Args[i+1]
				i++
			} else {
				argMap[cfg.Args[i]] = "present"
			}
		}
	}

	if val, ok := argMap["--prefill"]; !ok || val != "18432" {
		t.Errorf("invariant violation: --prefill must be '18432', got %q (present=%v)", val, ok)
	}

	// 5. Invariant: --spec must be 7 (Pass 8c coupled MTP + lookup verification)
	if val, ok := argMap["--spec"]; !ok || val != "7" {
		t.Errorf("invariant violation: --spec must be '7', got %q (present=%v)", val, ok)
	}

	// 6. Memory Safety Invariant: Vision must NOT be enabled in strata-qwen177b.json to prevent 81.6 GiB commit limit breach
	if _, ok := argMap["--vision"]; ok {
		t.Errorf("invariant violation: --vision must NOT be passed in args (exhausts Windows commit limit)")
	}
	var rawMap map[string]any
	_ = json.Unmarshal(raw, &rawMap)
	if _, ok := rawMap["vision"]; ok {
		t.Errorf("invariant violation: 'vision' block must NOT be present in strata-qwen177b.json")
	}

	// 7. Reasoning Effort Invariant: reasoning_effort='none' is strictly forbidden
	if strings.ToLower(strings.TrimSpace(cfg.ReasoningEffort)) == "none" {
		t.Errorf("invariant violation: reasoning_effort cannot be 'none'")
	}

	// 8. Binary Staging Invariant: staged strata.exe must contain compiled STRATA_NO_TURN_SPLIT symbol
	if exeBytes, errExe := os.ReadFile(cfg.Exe); errExe != nil {
		t.Errorf("invariant violation: failed reading staged binary %s: %v", cfg.Exe, errExe)
	} else if !bytes.Contains(exeBytes, []byte("STRATA_NO_TURN_SPLIT")) {
		t.Errorf("invariant violation: staged binary %s is missing STRATA_NO_TURN_SPLIT (stale pre-Pass-9 binary staged)", cfg.Exe)
	}

	// 9. Live Health Invariant (if server is active)
	client := &http.Client{Timeout: 800 * time.Millisecond}
	resp, err := client.Get("http://127.0.0.1:8080/health")
	if err == nil {
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Errorf("live server health returned status %d, want 200", resp.StatusCode)
		}
		var healthResp struct {
			Loaded bool   `json:"loaded"`
			Status string `json:"status"`
			Vision bool   `json:"vision"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&healthResp); err == nil {
			if !healthResp.Loaded {
				t.Errorf("live server report loaded=false, want true")
			}
			if healthResp.Vision {
				t.Errorf("live server report vision=true, want false for commit headroom")
			}
		}
	}
}
