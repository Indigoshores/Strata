package regressions

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestPass92GographIgnoreAndSmallTGemmGuard verifies:
// 1. .gographignore exists at the repo root and excludes heavy build/C++ trees to prevent 12,496-file AST OOM crashes.
// 2. src/core/arch_defaults.cpp retains STRATA_MTP_BATCH_F16=1 and STRATA_REFILL_PARTS=1 while avoiding MIN_T=1536 cublasGemmEx fallback.
func TestPass92GographIgnoreAndSmallTGemmGuard(t *testing.T) {
	repoRoot := filepath.Clean(filepath.Join("..", "..", ".."))

	ignorePath := filepath.Join(repoRoot, ".gographignore")
	rawIgnore, err := os.ReadFile(ignorePath)
	if err != nil {
		t.Fatalf("missing .gographignore at %s: %v", ignorePath, err)
	}
	ignoreStr := string(rawIgnore)
	for _, requiredPattern := range []string{"build*/", "graphify-out/", ".agents/", "src/", "include/"} {
		if !strings.Contains(ignoreStr, requiredPattern) {
			t.Errorf(".gographignore missing required exclusion pattern %q", requiredPattern)
		}
	}

	archDefPath := filepath.Join(repoRoot, "src", "core", "arch_defaults.cpp")
	rawArch, err := os.ReadFile(archDefPath)
	if err != nil {
		t.Fatalf("failed reading %s: %v", archDefPath, err)
	}
	archStr := string(rawArch)

	if strings.Contains(archStr, `"STRATA_WMMA_GEMM_MIN_T", "1536"`) {
		t.Errorf("regression: STRATA_WMMA_GEMM_MIN_T=1536 forces cublasGemmEx fallback on Windows for small-T prefill")
	}
	if strings.Contains(archStr, `"STRATA_PF_GEMM_MIN_T", "1536"`) {
		t.Errorf("regression: STRATA_PF_GEMM_MIN_T=1536 disables padded WMMA GEMM for small-T prefill")
	}
	if strings.Contains(archStr, `"STRATA_MTP_BATCH_F16", "1"`) {
		t.Errorf("regression: STRATA_MTP_BATCH_F16=1 shifts draft_kv Q8_1xQ8_0 rounding and slows T>512 prefill")
	}
	if !strings.Contains(archStr, `"STRATA_REFILL_PARTS", "1"`) {
		t.Errorf("expected STRATA_REFILL_PARTS=1 in gfx1151 defaults")
	}
}
