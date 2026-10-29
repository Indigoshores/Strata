package regressions

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
	"unsafe"
)

type memoryStatusEx struct {
	dwLength                uint32
	dwMemoryLoad            uint32
	ullTotalPhys            uint64
	ullAvailPhys            uint64
	ullTotalPageFile        uint64
	ullAvailPageFile        uint64
	ullTotalVirtual         uint64
	ullAvailVirtual         uint64
	ullAvailExtendedVirtual uint64
}

// TestHostCommitHeadroomAndGographSchemaGuard verifies Windows commit limits,
// confirms the gograph analyze_codebase tool schema uses project_path (and not path),
// and verifies no running gograph.exe process exceeds 4 GiB working set.
func TestHostCommitHeadroomAndGographSchemaGuard(t *testing.T) {
	// 1. GlobalMemoryStatusEx commit headroom check
	k32 := syscall.NewLazyDLL("kernel32.dll")
	proc := k32.NewProc("GlobalMemoryStatusEx")
	var m memoryStatusEx
	m.dwLength = uint32(unsafe.Sizeof(m))
	r, _, err := proc.Call(uintptr(unsafe.Pointer(&m)))
	if r == 0 {
		t.Fatalf("GlobalMemoryStatusEx failed: %v", err)
	}

	totalPhysGiB := float64(m.ullTotalPhys) / (1024 * 1024 * 1024)
	availPhysGiB := float64(m.ullAvailPhys) / (1024 * 1024 * 1024)
	totalPageFileGiB := float64(m.ullTotalPageFile) / (1024 * 1024 * 1024)
	availPageFileGiB := float64(m.ullAvailPageFile) / (1024 * 1024 * 1024)

	t.Logf("Host Commit State: TotalPhys=%.2f GiB, AvailPhys=%.2f GiB, TotalPageFile=%.2f GiB, AvailPageFile=%.2f GiB",
		totalPhysGiB, availPhysGiB, totalPageFileGiB, availPageFileGiB)

	if totalPageFileGiB < 32.0 {
		t.Errorf("TotalPageFileGiB %.2f < 32.0 GiB commit limit invariant", totalPageFileGiB)
	}

	// Check if strata.exe engine is actively running and holding ~61.4 GiB commit
	engineOnline := false
	cl := &http.Client{Timeout: 500 * time.Millisecond}
	if resp, err := cl.Get("http://127.0.0.1:8080/health"); err == nil {
		resp.Body.Close()
		engineOnline = true
	}

	commitThreshold := 4.0
	if engineOnline {
		// When strata.exe is resident with the 177B model, it occupies ~61.4 GiB of commit.
		// Headroom must remain positive and >= 0.5 GiB to protect system stability.
		t.Logf("Strata engine online (holding ~61.4 GiB commit); applying active-engine headroom threshold (>= 0.5 GiB)")
		commitThreshold = 0.5
	}

	if availPageFileGiB < commitThreshold {
		t.Errorf("AvailPageFileGiB %.2f < %.2f GiB commit safety threshold (engineOnline=%v)",
			availPageFileGiB, commitThreshold, engineOnline)
	}

	// 2. gograph analyze_codebase schema check
	schemaPath := `C:\Users\Butch\.gemini\antigravity\mcp\gograph\analyze_codebase.json`
	if raw, err := os.ReadFile(schemaPath); err == nil {
		var schema struct {
			Parameters struct {
				Properties map[string]any `json:"properties"`
			} `json:"parameters"`
		}
		if err := json.Unmarshal(raw, &schema); err != nil {
			t.Fatalf("failed to parse %s: %v", schemaPath, err)
		}
		if _, ok := schema.Parameters.Properties["project_path"]; !ok {
			t.Errorf("gograph analyze_codebase schema missing required property 'project_path'")
		}
		if _, ok := schema.Parameters.Properties["path"]; ok {
			t.Errorf("gograph analyze_codebase schema contains forbidden property 'path' (must use 'project_path')")
		}
	} else if !os.IsNotExist(err) {
		t.Logf("schema file read warning: %v", err)
	}

	// 3. tasklist gograph working set check (< 4 GiB = 4,194,304 KB)
	cmdCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(cmdCtx, "tasklist", "/FI", "IMAGENAME eq gograph.exe", "/FO", "CSV", "/NH")
	out, err := cmd.Output()
	if err == nil {
		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line == "" || strings.Contains(line, "No tasks are running") {
				continue
			}
			parts := strings.Split(line, "\",\"")
			if len(parts) >= 5 {
				memStr := strings.Trim(parts[4], "\" \r\n\tKkBb,")
				memStr = strings.ReplaceAll(memStr, ",", "")
				memStr = strings.ReplaceAll(memStr, " ", "")
				if kb, err := strconv.ParseInt(memStr, 10, 64); err == nil {
					t.Logf("Detected gograph.exe process with %d KB working set", kb)
					if kb > 4194304 {
						t.Errorf("gograph.exe process exceeds 4 GiB working set limit: %d KB", kb)
					}
				}
			}
		}
	}
}
