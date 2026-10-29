package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"
	"unsafe"
)

const (
	defaultMemoryLogPath = `C:\Users\Butch\.gemini\antigravity\brain\9b4d0dbe-556d-4a02-95b5-741c87e814f6\scratch\pass9_memory_log.jsonl`
	th32csSnapProcess    = 0x00000002
)

var monitoredTargetProcesses = []string{
	"strata.exe",
	"strata-server.exe",
	"strata-vision.exe",
	"llama-server.exe",
	"gograph.exe",
	"clang++.exe",
}

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

type performanceInformation struct {
	cb                uint32
	commitTotal       uintptr
	commitLimit       uintptr
	commitPeak        uintptr
	physicalTotal     uintptr
	physicalAvailable uintptr
	systemCache       uintptr
	kernelTotal       uintptr
	kernelPaged       uintptr
	kernelNonpaged    uintptr
	pageSize          uintptr
	handleCount       uint32
	processCount      uint32
	threadCount       uint32
}

type processMemoryCountersEx struct {
	cb                         uint32
	pageFaultCount             uint32
	peakWorkingSetSize         uintptr
	workingSetSize             uintptr
	quotaPeakPagedPoolUsage    uintptr
	quotaPagedPoolUsage        uintptr
	quotaPeakNonPagedPoolUsage uintptr
	quotaNonPagedPoolUsage     uintptr
	pagefileUsage              uintptr
	peakPagefileUsage          uintptr
	privateUsage               uintptr
}

type processEntry32W struct {
	dwSize              uint32
	cntUsage            uint32
	th32ProcessID       uint32
	th32DefaultHeapID   uintptr
	th32ModuleID        uint32
	cntThreads          uint32
	th32ParentProcessID uint32
	pcPriClassBase      int32
	dwFlags             uint32
	szExeFile           [260]uint16
}

type ProcessStats struct {
	Count            int      `json:"count"`
	PIDs             []uint32 `json:"pids,omitempty"`
	WorkingSetBytes  uint64   `json:"working_set_bytes"`
	PrivateBytes     uint64   `json:"private_bytes"`
	WorkingSetGiB    float64  `json:"working_set_gib"`
	PrivateCommitGiB float64  `json:"private_commit_gib"`
}

type MemorySnapshot struct {
	Timestamp       time.Time               `json:"timestamp"`
	Tag             string                  `json:"tag,omitempty"`
	TotalPhysGiB    float64                 `json:"total_phys_gib"`
	AvailPhysGiB    float64                 `json:"avail_phys_gib"`
	CommitUsedGiB   float64                 `json:"commit_used_gib"`
	CommitLimitGiB  float64                 `json:"commit_limit_gib"`
	AvailCommitGiB  float64                 `json:"avail_commit_gib"`
	Processes       map[string]ProcessStats `json:"processes"`
}

func GetMemorySnapshot(tag string) (MemorySnapshot, error) {
	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	psapi := syscall.NewLazyDLL("psapi.dll")

	procGlobalMemoryStatusEx := kernel32.NewProc("GlobalMemoryStatusEx")
	procGetPerformanceInfo := kernel32.NewProc("K32GetPerformanceInfo")
	if procGetPerformanceInfo.Find() != nil {
		procGetPerformanceInfo = psapi.NewProc("GetPerformanceInfo")
	}

	var ms memoryStatusEx
	ms.dwLength = uint32(unsafe.Sizeof(ms))
	r1, _, errGM := procGlobalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&ms)))
	if r1 == 0 {
		return MemorySnapshot{}, fmt.Errorf("GlobalMemoryStatusEx failed: %w", errGM)
	}

	totalPhysGiB := float64(ms.ullTotalPhys) / (1024 * 1024 * 1024)
	availPhysGiB := float64(ms.ullAvailPhys) / (1024 * 1024 * 1024)
	commitLimitGiB := float64(ms.ullTotalPageFile) / (1024 * 1024 * 1024)
	availCommitGiB := float64(ms.ullAvailPageFile) / (1024 * 1024 * 1024)
	commitUsedGiB := commitLimitGiB - availCommitGiB

	// Cross-verify with GetPerformanceInfo if available
	var perf performanceInformation
	perf.cb = uint32(unsafe.Sizeof(perf))
	if procGetPerformanceInfo.Find() == nil {
		rPerf, _, _ := procGetPerformanceInfo.Call(uintptr(unsafe.Pointer(&perf)), uintptr(perf.cb))
		if rPerf != 0 && perf.pageSize > 0 {
			pageSize := uint64(perf.pageSize)
			cTotal := float64(uint64(perf.commitTotal)*pageSize) / (1024 * 1024 * 1024)
			cLimit := float64(uint64(perf.commitLimit)*pageSize) / (1024 * 1024 * 1024)
			if cLimit > 0 {
				commitUsedGiB = cTotal
				commitLimitGiB = cLimit
				availCommitGiB = cLimit - cTotal
			}
		}
	}

	snap := MemorySnapshot{
		Timestamp:      time.Now().UTC(),
		Tag:            tag,
		TotalPhysGiB:   totalPhysGiB,
		AvailPhysGiB:   availPhysGiB,
		CommitUsedGiB:  commitUsedGiB,
		CommitLimitGiB: commitLimitGiB,
		AvailCommitGiB: availCommitGiB,
		Processes:      make(map[string]ProcessStats),
	}

	for _, name := range monitoredTargetProcesses {
		snap.Processes[name] = ProcessStats{Count: 0}
	}

	// Toolhelp32 process snapshot
	procCreateToolhelp32Snapshot := kernel32.NewProc("CreateToolhelp32Snapshot")
	procProcess32FirstW := kernel32.NewProc("Process32FirstW")
	procProcess32NextW := kernel32.NewProc("Process32NextW")
	procOpenProcess := kernel32.NewProc("OpenProcess")
	procCloseHandle := kernel32.NewProc("CloseHandle")

	procGetProcessMemoryInfo := kernel32.NewProc("K32GetProcessMemoryInfo")
	if procGetProcessMemoryInfo.Find() != nil {
		procGetProcessMemoryInfo = psapi.NewProc("GetProcessMemoryInfo")
	}

	hSnap, _, _ := procCreateToolhelp32Snapshot.Call(uintptr(th32csSnapProcess), 0)
	if hSnap != uintptr(syscall.InvalidHandle) && hSnap != 0 {
		defer procCloseHandle.Call(hSnap)

		var pe processEntry32W
		pe.dwSize = uint32(unsafe.Sizeof(pe))

		rFirst, _, _ := procProcess32FirstW.Call(hSnap, uintptr(unsafe.Pointer(&pe)))
		for rFirst != 0 {
			exeName := strings.ToLower(syscall.UTF16ToString(pe.szExeFile[:]))
			for _, target := range monitoredTargetProcesses {
				if exeName == target {
					stat := snap.Processes[target]
					stat.Count++
					stat.PIDs = append(stat.PIDs, pe.th32ProcessID)

					// Query process memory
					const PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
					const PROCESS_VM_READ = 0x0010
					hProc, _, _ := procOpenProcess.Call(PROCESS_QUERY_LIMITED_INFORMATION|PROCESS_VM_READ, 0, uintptr(pe.th32ProcessID))
					if hProc == 0 {
						hProc, _, _ = procOpenProcess.Call(PROCESS_QUERY_LIMITED_INFORMATION, 0, uintptr(pe.th32ProcessID))
					}
					if hProc != 0 {
						var pmc processMemoryCountersEx
						pmc.cb = uint32(unsafe.Sizeof(pmc))
						rMem, _, _ := procGetProcessMemoryInfo.Call(hProc, uintptr(unsafe.Pointer(&pmc)), uintptr(pmc.cb))
						if rMem != 0 {
							ws := uint64(pmc.workingSetSize)
							priv := uint64(pmc.privateUsage)
							if priv == 0 {
								priv = uint64(pmc.pagefileUsage)
							}
							stat.WorkingSetBytes += ws
							stat.PrivateBytes += priv
						}
						procCloseHandle.Call(hProc)
					}
					stat.WorkingSetGiB = float64(stat.WorkingSetBytes) / (1024 * 1024 * 1024)
					stat.PrivateCommitGiB = float64(stat.PrivateBytes) / (1024 * 1024 * 1024)
					snap.Processes[target] = stat
					break
				}
			}
			rFirst, _, _ = procProcess32NextW.Call(hSnap, uintptr(unsafe.Pointer(&pe)))
		}
	}

	return snap, nil
}

func AppendMemoryLog(snap MemorySnapshot, logPath string) error {
	if logPath == "" {
		logPath = defaultMemoryLogPath
	}
	dir := filepath.Dir(logPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create memory log directory: %w", err)
	}
	f, err := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("failed to open memory log file: %w", err)
	}
	defer f.Close()

	line, err := json.Marshal(snap)
	if err != nil {
		return fmt.Errorf("failed to marshal memory snapshot: %w", err)
	}
	line = append(line, '\n')
	if _, err := f.Write(line); err != nil {
		return fmt.Errorf("failed to write memory log: %w", err)
	}
	return nil
}

func isAnyProcessRunning(targets ...string) bool {
	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	procCreateToolhelp32Snapshot := kernel32.NewProc("CreateToolhelp32Snapshot")
	procProcess32FirstW := kernel32.NewProc("Process32FirstW")
	procProcess32NextW := kernel32.NewProc("Process32NextW")
	procCloseHandle := kernel32.NewProc("CloseHandle")

	hSnap, _, _ := procCreateToolhelp32Snapshot.Call(uintptr(th32csSnapProcess), 0)
	if hSnap == uintptr(syscall.InvalidHandle) || hSnap == 0 {
		return false
	}
	defer procCloseHandle.Call(hSnap)

	targetMap := make(map[string]bool)
	for _, t := range targets {
		targetMap[strings.ToLower(t)] = true
	}

	var pe processEntry32W
	pe.dwSize = uint32(unsafe.Sizeof(pe))
	r, _, _ := procProcess32FirstW.Call(hSnap, uintptr(unsafe.Pointer(&pe)))
	for r != 0 {
		exe := strings.ToLower(syscall.UTF16ToString(pe.szExeFile[:]))
		if targetMap[exe] {
			return true
		}
		r, _, _ = procProcess32NextW.Call(hSnap, uintptr(unsafe.Pointer(&pe)))
	}
	return false
}

func runMemStatusCmd(args []string) {
	fs := flag.NewFlagSet("mem-status", flag.ExitOnError)
	minAvail := fs.Float64("min-avail-commit-gib", 0.0, "Minimum available host commit required in GiB")
	tag := fs.String("tag", "", "Optional tag identifying the current phase/step")
	jsonOutput := fs.Bool("json", false, "Output snapshot as JSON")
	logPath := fs.String("log", defaultMemoryLogPath, "Path to pass9_memory_log.jsonl")
	_ = fs.Parse(args)

	snap, err := GetMemorySnapshot(*tag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error gathering memory snapshot: %v\n", err)
		os.Exit(1)
	}

	_ = AppendMemoryLog(snap, *logPath)

	if *jsonOutput {
		raw, _ := json.MarshalIndent(snap, "", "  ")
		fmt.Println(string(raw))
	} else {
		fmt.Println("================================================================================")
		fmt.Println("  WIN32 HOST COMMIT & UMA MEMORY TELEMETRY (gfx1151 / 128 GB UMA)")
		fmt.Println("================================================================================")
		if snap.Tag != "" {
			fmt.Printf("Tag:              %s\n", snap.Tag)
		}
		fmt.Printf("Timestamp:        %s\n", snap.Timestamp.Format(time.RFC3339))
		fmt.Printf("Physical RAM:     %.2f GiB Avail / %.2f GiB Total (%.1f%% available)\n",
			snap.AvailPhysGiB, snap.TotalPhysGiB, (snap.AvailPhysGiB/snap.TotalPhysGiB)*100.0)
		fmt.Printf("Windows Commit:   %.2f GiB Avail / %.2f GiB Limit (%.2f GiB in use)\n",
			snap.AvailCommitGiB, snap.CommitLimitGiB, snap.CommitUsedGiB)
		fmt.Println("Target Process Footprint:")
		for _, name := range monitoredTargetProcesses {
			stat := snap.Processes[name]
			if stat.Count > 0 {
				fmt.Printf("  - %-18s count=%-2d  pids=%v  working_set=%6.2f GiB  private_commit=%6.2f GiB\n",
					name, stat.Count, stat.PIDs, stat.WorkingSetGiB, stat.PrivateCommitGiB)
			} else {
				fmt.Printf("  - %-18s [OFFLINE]\n", name)
			}
		}
		fmt.Println("================================================================================")
	}

	if *minAvail > 0 && snap.AvailCommitGiB < *minAvail {
		fmt.Fprintf(os.Stderr, "FATAL COMMIT HEADROOM GUARD: AvailCommitGiB (%.2f GiB) < required threshold (%.2f GiB)!\n",
			snap.AvailCommitGiB, *minAvail)
		os.Exit(2)
	}
}

func runStopCmd(args []string) {
	fs := flag.NewFlagSet("stop", flag.ExitOnError)
	tag := fs.String("tag", "post-stop", "Tag for the post-stop memory snapshot")
	_ = fs.Parse(args)

	fmt.Println("[Stop] Gracefully terminating Strata engine processes: strata-server.exe, strata.exe, strata-vision.exe...")
	// Unconditionally protect llama-server.exe: killProcessesByName ONLY targets strata-server.exe, strata.exe, strata-vision.exe
	killProcessesByName("strata-server.exe", "strata.exe", "strata-vision.exe")

	fmt.Println("[Stop] Waiting up to 10s for strata.exe process exit and ROCm UMA commit release...")
	deadline := time.Now().Add(10 * time.Second)
	stopped := false
	for time.Now().Before(deadline) {
		if !isAnyProcessRunning("strata.exe", "strata-server.exe", "strata-vision.exe") {
			stopped = true
			break
		}
		time.Sleep(250 * time.Millisecond)
	}

	if !stopped {
		fmt.Println("[Stop] Warning: Some strata process(es) may still be exiting...")
	} else {
		fmt.Println("[Stop] All strata engine processes stopped successfully.")
	}

	// Print post-stop status
	runMemStatusCmd([]string{"-tag", *tag})
}
