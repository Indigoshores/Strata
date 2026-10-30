package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"
)

func TestZeroReuseBenchmarkSendsStrataCheckpointFalse(t *testing.T) {
	var captured chatCompletionRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &captured)
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"plan \"}}]}\n\n")
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":408,\"completion_tokens\":2,\"total_tokens\":410}}\n\n")
		fmt.Fprintf(w, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	}))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	p := BuildZeroReusePrompt(512, "T512_000042")
	_, err := RunSingleHTTPBenchmark(ctx, srv.URL, "", p, 512, 256, 0.0, "medium")
	if err != nil {
		t.Fatalf("RunSingleHTTPBenchmark failed: %v", err)
	}
	if captured.StrataCheckpoint == nil || *captured.StrataCheckpoint {
		t.Fatalf("expected zero-reuse benchmark payload to set strata_checkpoint=false (ckpt=0), got %+v", captured.StrataCheckpoint)
	}
	if captured.Temperature != 0.0 {
		t.Fatalf("expected temperature=0.0, got %v", captured.Temperature)
	}
}

func TestAuditConfigFailsClosedWhenQwen38EffortPositionMismatchOrPreOpenThink(t *testing.T) {
	tmpDir := t.TempDir()
	for _, tc := range []struct {
		name         string
		effortPos    string
		preOpenThink bool
	}{
		{name: "effort_start", effortPos: "start", preOpenThink: false},
		{name: "pre_open_think_true", effortPos: "end", preOpenThink: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := ProductionConfig{
				Exe:            tmpDir + `\strata.exe`,
				ModelName:      "Qwen 3.8 Flash-Next 177B (Strata)",
				Backend:        "hip",
				EffortPosition: tc.effortPos,
				PreOpenThink:   tc.preOpenThink,
				Env:            map[string]string{"STRATA_PREFILL_STREAM_MIN": "16"},
				Args:           []string{"--expert-cache", "auto"},
			}
			raw, _ := json.Marshal(cfg)
			cfgPath := tmpDir + `\` + tc.name + `.json`
			if err := os.WriteFile(cfgPath, raw, 0644); err != nil {
				t.Fatal(err)
			}
			rep, err := AuditConfig(cfgPath, DefaultRepoRoot)
			if err != nil {
				t.Fatalf("AuditConfig returned error: %v", err)
			}
			foundFail := false
			for _, c := range rep.Checks {
				if c.ID == "CFG-10d" && c.Status == "FAIL" {
					foundFail = true
				}
			}
			if !foundFail {
				t.Fatalf("expected CFG-10d FAIL for %s, checks=%+v", tc.name, rep.Checks)
			}
		})
	}
}
