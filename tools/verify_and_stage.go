package main

import (
	"crypto/sha256"
	"debug/pe"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

func inspectPE(path string) (map[string]int, error) {
	f, err := pe.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	syms, err := f.ImportedSymbols()
	if err != nil {
		return nil, err
	}

	dllCounts := make(map[string]int)
	for _, sym := range syms {
		parts := strings.Split(sym, ":")
		if len(parts) == 2 {
			dll := strings.ToLower(parts[1])
			dllCounts[dll]++
		}
	}
	return dllCounts, nil
}

func copyFile(src, dst string) (string, int64, error) {
	in, err := os.Open(src)
	if err != nil {
		return "", 0, err
	}
	defer in.Close()

	if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
		return "", 0, err
	}

	out, err := os.Create(dst)
	if err != nil {
		return "", 0, err
	}
	defer out.Close()

	hasher := sha256.New()
	mw := io.MultiWriter(out, hasher)

	n, err := io.Copy(mw, in)
	if err != nil {
		return "", 0, err
	}

	hashStr := hex.EncodeToString(hasher.Sum(nil))
	return hashStr, n, nil
}

func main() {
	exePath := `C:\Users\Butch\Documents\02_Apps\Strata\build-pass3\strata.exe`
	fmt.Println("=== Inspecting PE Imports of", exePath, "===")
	dlls, err := inspectPE(exePath)
	if err != nil {
		fmt.Printf("Error opening PE %s: %v\n", exePath, err)
		os.Exit(1)
	}

	var names []string
	for k := range dlls {
		names = append(names, k)
	}
	sort.Strings(names)

	hasLibHipblasLt := false
	hasHipblasLt := false

	for _, name := range names {
		fmt.Printf("  %s: %d symbols\n", name, dlls[name])
		if name == "libhipblaslt.dll" {
			hasLibHipblasLt = true
		}
		if name == "hipblaslt.dll" {
			hasHipblasLt = true
		}
	}

	if hasLibHipblasLt && !hasHipblasLt {
		fmt.Println("[SUCCESS] strata.exe correctly imports libhipblaslt.dll and DOES NOT import hipblaslt.dll!")
	} else if hasHipblasLt {
		fmt.Println("[FAILURE] strata.exe imports hipblaslt.dll! (Expected libhipblaslt.dll)")
	} else {
		fmt.Println("[WARNING] strata.exe does not import libhipblaslt.dll or hipblaslt.dll!")
	}

	stageDir := `C:\Users\Butch\.gemini\antigravity\brain\2e1a2dc4-fdb7-4cfc-9fbe-940e044c14a7\scratch\pass3_engine`
	fmt.Println("\n=== Staging Binaries & Assets to", stageDir, "===")

	filesToCopy := []string{
		`C:\Users\Butch\Documents\02_Apps\Strata\build-pass3\strata.exe`,
		`C:\Users\Butch\Documents\02_Apps\Strata\build-pass3\strata-device.exe`,
		`C:\Users\Butch\Documents\02_Apps\Strata\build-pass3\native_iq_wmma_config_test.exe`,
		`C:\Users\Butch\Documents\02_Apps\Strata\build-pass3\lookahead_policy_test.exe`,
		`C:\Users\Butch\Documents\02_Apps\Strata\build-pass3\hip_prefill_hipblaslt_gemm.exe`,
		`C:\Users\Butch\Documents\02_Apps\Strata\build-pass3\tune_hipblaslt.exe`,
		`C:\Users\Butch\Documents\02_Apps\Strata\tools\hip\gfx1151-hipblaslt-100500.txt`,
	}

	for _, b := range filesToCopy {
		if _, err := os.Stat(b); os.IsNotExist(err) {
			fmt.Printf("Skipping nonexistent file: %s\n", b)
			continue
		}
		dst := filepath.Join(stageDir, filepath.Base(b))
		h, sz, err := copyFile(b, dst)
		if err != nil {
			fmt.Printf("Failed to copy %s: %v\n", b, err)
		} else {
			fmt.Printf("Copied %s (%d bytes, sha256: %s)\n", filepath.Base(b), sz, h)
		}
	}
}
