# Go Version strata.exe - Architecture & Deployment Blueprint

This document outlines the production engineering blueprint for deploying the Strata inference optimizations into a native **Go 1.26** implementation of `strata.exe`.

---

## 1. System Architecture Overview

The Go implementation combines Go's concurrency and low-overhead runtime with native Windows kernel syscalls and AMD ROCm HIP dynamic libraries:

```
+-------------------------------------------------------------------------+
|                  Go 1.26 HTTP / SSE Server Engine                       |
|           (OpenAI / Anthropic Compatible Streaming API)                 |
+-------------------------------------------------------------------------+
                                     |
+------------------------------------+------------------------------------+
|                                    |                                    |
|   Go Memory & Buffer Manager       |   Go Speculative Decoding Tree     |
|   * VirtualAlloc / VirtualLock     |   * MTP Draft Verification Engine  |
|   * Zero-Copy sync.Pool Arenas     |   * Gumbel-Max Coupled Sampler     |
|   * DXGI 1.4 Video Budget Guard    |   * Suffix/Prompt N-gram Drafter   |
|                                    |                                    |
+------------------------------------+------------------------------------+
                                     |
+------------------------------------+------------------------------------+
|                                    |                                    |
|   Direct NVMe I/O Subsystem        |   ROCm HIP CGO / DLL Interop       |
|   * FILE_FLAG_NO_BUFFERING         |   * amdhip64_7.dll / hipblaslt.dll |
|   * Win32 Completion Ports (IOCP)  |   * gfx1151 Tuning Table Loader    |
|   * 4-Goroutine Async Worker Pool  |   * Strix Halo Environment Matrix  |
|                                    |                                    |
+------------------------------------+------------------------------------+
```

---

## 2. Core Go Subsystems

### Subsystem A: Win32 Pinned Memory Allocator (`VirtualLock`)
In Go, weights for the 24,576 experts must be pinned in physical RAM to prevent paging stalls:

```go
package platform

import (
	"fmt"
	"syscall"
	"unsafe"
)

var (
	modkernel32               = syscall.NewLazyDLL("kernel32.dll")
	procVirtualAlloc          = modkernel32.NewProc("VirtualAlloc")
	procVirtualLock           = modkernel32.NewProc("VirtualLock")
	procVirtualFree           = modkernel32.NewProc("VirtualFree")
	procSetProcessWS          = modkernel32.NewProc("SetProcessWorkingSetSizeEx")
	procGetCurrentProcess     = modkernel32.NewProc("GetCurrentProcess")
)

const (
	MEM_COMMIT      = 0x00001000
	MEM_RESERVE     = 0x00002000
	MEM_RELEASE     = 0x00008000
	PAGE_READWRITE  = 0x04
	QUOTA_LIMITS_HARDWS_MIN_DISABLE = 0x00000002
	QUOTA_LIMITS_HARDWS_MAX_DISABLE = 0x00000008
)

// AllocPinned allocates 64-byte aligned memory and pins it via VirtualLock.
func AllocPinned(bytes uintptr) (unsafe.Pointer, error) {
	ptr, _, err := procVirtualAlloc.Call(0, bytes, MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE)
	if ptr == 0 {
		return nil, fmt.Errorf("VirtualAlloc failed: %w", err)
	}

	// Expand working set limits
	hProcess, _, _ := procGetCurrentProcess.Call()
	margin := uintptr(512 << 20)
	procSetProcessWS.Call(hProcess, bytes+margin, bytes+(margin*2),
		QUOTA_LIMITS_HARDWS_MIN_DISABLE|QUOTA_LIMITS_HARDWS_MAX_DISABLE)

	// Lock in 1 GiB chunks
	chunk := uintptr(1 << 30)
	for off := uintptr(0); off < bytes; off += chunk {
		n := bytes - off
		if n > chunk {
			n = chunk
		}
		r, _, errLock := procVirtualLock.Call(ptr+off, n)
		if r == 0 {
			procVirtualFree.Call(ptr, 0, MEM_RELEASE)
			return nil, fmt.Errorf("VirtualLock failed at offset %d: %w", off, errLock)
		}
	}
	return unsafe.Pointer(ptr), nil
}
```

---

### Subsystem B: Direct Asynchronous NVMe I/O (PLE Direct File)
To eliminate Windows file cache thrashing when reading from the 28.8 GB n-gram table:

```go
package platform

import (
	"syscall"
	"unsafe"
)

const (
	FILE_FLAG_NO_BUFFERING = 0x20000000
	FILE_FLAG_OVERLAPPED   = 0x40000000
	FILE_FLAG_RANDOM_ACCESS= 0x10000000
)

type DirectFileReader struct {
	handle syscall.Handle
	iocp   syscall.Handle
}

func OpenDirectFile(filePath string) (*DirectFileReader, error) {
	pathPtr, err := syscall.UTF16PtrFromString(filePath)
	if err != nil {
		return nil, err
	}
	h, err := syscall.CreateFile(
		pathPtr,
		syscall.GENERIC_READ,
		syscall.FILE_SHARE_READ,
		nil,
		syscall.OPEN_EXISTING,
		FILE_FLAG_NO_BUFFERING|FILE_FLAG_OVERLAPPED|FILE_FLAG_RANDOM_ACCESS,
		0,
	)
	if err != nil {
		return nil, err
	}
	return &DirectFileReader{handle: h}, nil
}
```

---

### Subsystem C: Strix Halo Runtime Environment Injector
Before initializing the HIP runtime or loading ROCm DLLs, the Go runtime automatically populates the 18 default switches:

```go
package runtime

import "os"

// InitStrixHaloEnvironment applies all verified gfx1151 acceleration defaults.
func InitStrixHaloEnvironment(tuningTablePath string) {
	defaults := map[string]string{
		"STRATA_GDN_HEAD":        "1",
		"STRATA_GDN_PP":          "2",
		"STRATA_GDN_CONVL2":      "1",
		"STRATA_GDN_NOY":         "1",
		"STRATA_CVEC_FUSE":       "1",
		"STRATA_HCD_EXACT":       "1",
		"STRATA_PF_PAD":          "1",
		"STRATA_Q8_PACKED":       "1",
		"STRATA_Q6_PACKED":       "1",
		"STRATA_MMVF_ROWS":       "1",
		"STRATA_ATTN_LANECELL":   "1",
		"STRATA_EXPERT_V2":       "1",
		"STRATA_TSUM":            "1",
		"STRATA_LFUSE":           "1",
		"STRATA_GDN_SPLIT":       "1",
		"STRATA_QFUSE":           "1",
		"STRATA_PLE_BATCH":       "1",
		"STRATA_SH_STREAM":       "1",
		"ROCBLAS_USE_HIPBLASLT":  "1",
		"HSA_ENABLE_SDMA":        "0",
		"GPU_MAX_HW_QUEUES":      "8",
	}

	for k, v := range defaults {
		if os.Getenv(k) == "" {
			os.Setenv(k, v)
		}
	}

	if tuningTablePath != "" && os.Getenv("STRATA_HIPBLASLT_TUNING") == "" {
		os.Setenv("STRATA_HIPBLASLT_TUNING", tuningTablePath)
	}
}
```

---

### Subsystem D: Speculative Decoding Controller & Gumbel-Max Sampler

```go
package spec

import (
	"math"
	"math/rand/v2"
)

// GumbelSample performs coupled draft-target sampling using exponential noise.
func GumbelSample(probs []float32, seed uint64) int {
	rng := rand.New(rand.NewPCG(seed, 0x12345678))
	bestIdx := 0
	maxVal := -math.MaxFloat64

	for i, p := range probs {
		if p <= 0 {
			continue
		}
		// Generate standard exponential noise E ~ Exp(1)
		u := rng.Float64()
		if u <= 0 {
			u = 1e-12
		}
		e := -math.Log(u)

		val := float64(p) / e
		if val > maxVal {
			maxVal = val
			bestIdx = i
		}
	}
	return bestIdx
}
```

---

## 3. Resilience & Self-Healing Loop

A common failure mode in local LLM servers is process termination or connection drop (`socket hang up`). The Go implementation addresses this through:

1. **Child Process Supervisor**: Spawns inference worker with a bidirectional health channel.
2. **Crash Interception**: Captures exit codes, logs crash telemetry, and restarts the worker in sub-500ms without dropping pending HTTP connections.
3. **Graceful Connection Draining**: Holds inbound requests in an in-memory queue during transient reloads.

---

## 4. Integration with Existing `cmd/strata-server` Codebase

The local repository already houses the foundation of this architecture in `cmd/strata-server/`:

- **Main Entrypoint**: `cmd/strata-server/main.go` initializes the model configuration, tokenization engine, and HTTP server.
- **Engine Process Supervisor**: `cmd/strata-server/engine/engine.go` coordinates `strata.exe` with `CREATE_NO_WINDOW` headless flags, handles the line-based IPC protocol, and captures stderr logs.
- **Hardware Config & Environment Builder**: `cmd/strata-server/engine/config.go` maps `strata-qwen177b.json` parameters, injects ROCm library paths (`C:\Program Files\AMD\ROCm\7.2\bin`), and builds execution arguments.
- **Production Configuration**: `C:\Users\Butch\Documents\10-Models\Strata\strata-qwen177b.json` configures:
  * Model: Qwen 3.8 Flash-Next 177B UD-IQ4_XS
  * HipBLASLt Tuning: `tools/hip/gfx1151-hipblaslt-100500.txt`
  * PLE Direct I/O: `--ple-io direct --ple-inflight 1024 --ple-row-cache 4194304`
  * Native MoE Context: `--max-context 262144 --kv int8 --prefill 16384`
  * Speculative Decoding: `--spec 5 --spec-min-p 0.1 --lookup-chain 3 --mtp-q4 all`

