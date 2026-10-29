package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestRegressionPass6PrefillSwitchesFailClosed ensures AuditConfig fails closed whenever
// any of the critical Pass 6..8c gfx1151 WMMA & Windows WDDM prefill switches are deleted
// or misconfigured in strata-qwen177b.json.
func TestRegressionPass6PrefillSwitchesFailClosed(t *testing.T) {
	raw, err := os.ReadFile(DefaultProdConfigPath)
	if err != nil {
		t.Skipf("skipping production config check: %v", err)
	}

	var baseCfg map[string]any
	if err := json.Unmarshal(raw, &baseCfg); err != nil {
		t.Fatalf("failed to parse %s: %v", DefaultProdConfigPath, err)
	}

	// 1. Restored Pass 8c production config must pass with 0 failures.
	repBase, err := AuditConfig(DefaultProdConfigPath, DefaultRepoRoot)
	if err != nil {
		t.Fatalf("AuditConfig error: %v", err)
	}
	if repBase.FailedCount != 0 {
		for _, c := range repBase.Checks {
			if c.Status == "FAIL" {
				t.Logf("FAIL check %s (%s): want=%s got=%s", c.ID, c.Name, c.Expected, c.Actual)
			}
		}
		t.Fatalf("expected FailedCount=0 on restored production config, got %d", repBase.FailedCount)
	}

	// 2. Removing any of the critical Pass 6..8c prefill switches must trigger FAIL.
	criticalKeys := []string{
		"STRATA_PF_GEMM",
		"STRATA_PF_GEMM_Q8",
		"STRATA_PF_PAD",
		"STRATA_WMMA_GEMM",
		"STRATA_PF_HCDOWN",
		"STRATA_PF_SH_STREAM",
		"STRATA_PF_Z_STREAM",
		"STRATA_PF_DM_F16",
		"STRATA_PF_F16",
		"STRATA_PF_F16_MIN_T",
		"STRATA_PF_MOE_SCATTER",
		"STRATA_PF_SWITCH_MIN_T",
		"STRATA_NO_TURN_SPLIT",
	}

	tmpDir := t.TempDir()
	for _, key := range criticalKeys {
		var mutated map[string]any
		_ = json.Unmarshal(raw, &mutated)
		envObj, _ := mutated["env"].(map[string]any)
		delete(envObj, key)

		outPath := filepath.Join(tmpDir, "missing_"+key+".json")
		b, _ := json.MarshalIndent(mutated, "", "  ")
		if err := os.WriteFile(outPath, b, 0644); err != nil {
			t.Fatal(err)
		}

		rep, err := AuditConfig(outPath, DefaultRepoRoot)
		if err != nil {
			t.Fatalf("AuditConfig failed for missing %s: %v", key, err)
		}
		if rep.FailedCount == 0 {
			t.Errorf("regression: AuditConfig must FAIL when %s is removed, got FailedCount=0", key)
		}
	}

	// 3. Setting STRATA_PF_F16=0, STRATA_PF_SH_STREAM=1, STRATA_PF_Z_STREAM=1, or reasoning_effort=none must fail closed.
	badEnvOverrides := map[string]string{
		"STRATA_PF_F16":         "0",
		"STRATA_PF_F16_MIN_T":   "20000",
		"STRATA_PF_MOE_SCATTER": "1",
		"STRATA_PF_SH_STREAM":   "1",
		"STRATA_PF_Z_STREAM":    "1",
	}
	for k, badVal := range badEnvOverrides {
		var mutatedEnv map[string]any
		_ = json.Unmarshal(raw, &mutatedEnv)
		envObj, _ := mutatedEnv["env"].(map[string]any)
		envObj[k] = badVal
		p := filepath.Join(tmpDir, "bad_"+k+".json")
		b, _ := json.MarshalIndent(mutatedEnv, "", "  ")
		_ = os.WriteFile(p, b, 0644)
		rep, err := AuditConfig(p, DefaultRepoRoot)
		if err != nil || rep.FailedCount == 0 {
			t.Errorf("regression: AuditConfig must FAIL when %s=%s (err=%v, fails=%d)", k, badVal, err, rep.FailedCount)
		}
	}
	{
		var mutatedNone map[string]any
		_ = json.Unmarshal(raw, &mutatedNone)
		mutatedNone["reasoning_effort"] = "none"
		p := filepath.Join(tmpDir, "bad_effort_none.json")
		b, _ := json.MarshalIndent(mutatedNone, "", "  ")
		_ = os.WriteFile(p, b, 0644)
		rep, err := AuditConfig(p, DefaultRepoRoot)
		if err != nil || rep.FailedCount == 0 {
			t.Errorf("regression: AuditConfig must FAIL when reasoning_effort=none (err=%v, fails=%d)", err, rep.FailedCount)
		}
	}
}
