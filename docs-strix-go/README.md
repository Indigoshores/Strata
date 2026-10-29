# Strata Engine Knowledge Base & Strix Halo Optimization Catalog

Welcome to the **Strata Engine Optimization & Upstream Tracking Hub**. This repository and documentation suite provides a deep technical analysis of [Niko1221/Strata](https://github.com/Niko1221/Strata.git), specifically centered on:

1. **AMD Strix Halo (Ryzen AI Max+ 395 / Radeon 8060S gfx1151)**: Architectural deep dive, kernel execution models, ROCm 7.x, unified memory allocation (UMA/GTT), and WMMA matrix accelerators.
2. **Algorithmic & Kernel Innovations**: Disjoint MoE Expert Caching, Gated Delta Net (GDN) linear recurrence, Prefix Lookup Engine (PLE) with unbuffered async Direct NVMe I/O, Multi-Token Prediction (MTP) speculative decoding, and quantized memory paging.
3. **Go Engine Deployment Blueprint**: Concrete architectural patterns, CGO bindings, Win32 I/O completion ports, pinned memory syscalls, and concurrency models to deploy Strata optimizations into the Go implementation of `strata.exe`.
4. **Upstream Repository Synchronization & Search Tooling**: Automated Go 1.26 CLI utility (`strata-tracker.exe`) and standard operating procedures to track, search, diff, and pull upstream updates from GitHub.

---

## 🗺️ Documentation Directory & Sitemap

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
  <table style="width: 100%; min-width: 600px; border-collapse: collapse; background-color: #161b22; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 14px;">
    <thead>
      <tr style="background-color: #21262d; border-bottom: 2px solid #30363d; text-align: left;">
        <th style="padding: 12px 16px; color: #58a6ff; white-space: nowrap;">Document</th>
        <th style="padding: 12px 16px; color: #58a6ff; white-space: nowrap;">Target Area</th>
        <th style="padding: 12px 16px; color: #58a6ff; white-space: nowrap;">Description & Highlights</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><a href="STRIX_HALO_OPTIMIZATIONS.md" style="color: #58a6ff; text-decoration: none;">STRIX_HALO_OPTIMIZATIONS.md</a></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #e3b341;">AMD Strix Halo (gfx1151)</span></td>
        <td style="padding: 12px 16px;">18 default + 8 opt-in switches, ROCm 7.x, unified memory (128 GB LPDDR5X), hipBLASLt tuning tables, WMMA matrix cores.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><a href="STRATA_KERNEL_ARCHITECTURE.md" style="color: #58a6ff; text-decoration: none;">STRATA_KERNEL_ARCHITECTURE.md</a></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #3fb950;">Kernel & Model Engine</span></td>
        <td style="padding: 12px 16px;">GDN recurrence, Disjoint Expert Cache (24,576 experts), PLE unbuffered direct I/O, MTP speculative decoding, KV streaming.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><a href="GO_STRATA_DEPLOYMENT_BLUEPRINT.md" style="color: #58a6ff; text-decoration: none;">GO_STRATA_DEPLOYMENT_BLUEPRINT.md</a></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #58a6ff;">Go 1.26 Engine Port</span></td>
        <td style="padding: 12px 16px;">Go supervisor, Win32 VirtualAlloc/VirtualLock memory manager, IOCP NVMe streaming, CGO HIP runtime loading, OpenAI SSE server.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><a href="REPO_UPDATE_TRACKER.md" style="color: #58a6ff; text-decoration: none;">REPO_UPDATE_TRACKER.md</a></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #f85149;">Upstream Sync Protocol</span></td>
        <td style="padding: 12px 16px;">SOP for checking GitHub releases, diffing commits, inspecting ROCm/HIP changes, and verifying model hashes.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><a href="UPSTREAM_SEARCH_CHEATSHEET.md" style="color: #58a6ff; text-decoration: none;">UPSTREAM_SEARCH_CHEATSHEET.md</a></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #c9d1d9;">CLI &amp; Git Queries</span></td>
        <td style="padding: 12px 16px;">Pre-compiled search commands, regex filters, and file targets for rapid git and AST navigation.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><a href="../docs/GUFO.md" style="color: #58a6ff; text-decoration: none;">GUFO.md</a></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #e3b341;">Gufo Engine Dossier</span></td>
        <td style="padding: 12px 16px;">Architectural breakdown of Gufo (<code>gufo-org/gufo</code>), 8 multi-modal models, Windows TheRock 10.0.0 build, and cross-pollination with Strata.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><a href="../docs/PASS7_FINAL_VALIDATION_REPORT.md" style="color: #3fb950; text-decoration: none;">PASS7_FINAL_VALIDATION_REPORT.md</a></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #3fb950;">Pass 7 Production State</span></td>
        <td style="padding: 12px 16px;">Authoritative Pass 7 validation (29/29 checks, <code>--spec 7</code>, <code>--tail-role-token 8678</code>, 82-89.8 tok/s client decode, 1,356.7 tok/s GPU prefill).</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><a href="../docs/PASS7_GITHUB_MINING_REPORT.md" style="color: #58a6ff; text-decoration: none;">PASS7_GITHUB_MINING_REPORT.md</a></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #e3b341;">AST Mining &amp; F-01..F-10</span></td>
        <td style="padding: 12px 16px;">AST differential across Custom Pass 6, upstream Strata <code>v0.2.7</code>, Gufo <code>v0.25.0</code>, and Pixmaate; defines invariants <code>F-01</code>..<code>F-10</code>.</td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><a href="../docs/STAGE1_BENCHMARK_REPORT.md" style="color: #58a6ff; text-decoration: none;">STAGE1_BENCHMARK_REPORT.md</a></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #58a6ff;">Thinking &amp; Sweep</span></td>
        <td style="padding: 12px 16px;">4-phase <code>verify-thinking</code> suite, <code>--spec 7</code> runtime sweep, and temperature/effort matrix.</td>
      </tr>
      <tr>
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;"><a href="../docs/VALIDATION_FRAMEWORK.md" style="color: #58a6ff; text-decoration: none;">VALIDATION_FRAMEWORK.md</a></td>
        <td style="padding: 12px 16px; white-space: nowrap;"><span style="color: #3fb950;">5-Phase Framework</span></td>
        <td style="padding: 12px 16px;">Engineering spec for <code>strata-tracker.exe</code> (<code>validate</code>, <code>audit-config</code>, <code>audit-log</code>, <code>bench</code>, <code>ab-matrix</code>, <code>ctest-gate</code>, <code>verify-thinking</code>).</td>
      </tr>
    </tbody>
  </table>
</div>

---

## ⚡ Quickstart: Pure Go 1.26 Strata Tracker & Validation CLI

```powershell
# 0. Run full 5-phase Qwen 3.8 177B validation suite (Config + 1.84M-line Log + A/B Matrix + Gates)
.\docs-strix-go\strata-tracker.exe validate -out .\docs-strix-go\validation-report.json

# 1. Run automated unit & integration tests
go -C .\docs-strix-go test -v ./...

# 2. Run live Zero-Reuse Context Ladder HTTP/SSE benchmark (512..16384 tokens; 'none' reasoning is strictly forbidden)
.\docs-strix-go\strata-tracker.exe bench -url http://127.0.0.1:8080 -contexts 512,1024,2048,4096,8192,16384 -gen 256 -temp 0 -reasoning high

# 3. View 7-arm A-B-B-A optimization matrix (v0.1.40.2 -> v0.1.45 + PR #2997) & CTest parity gates
.\docs-strix-go\strata-tracker.exe ab-matrix
.\docs-strix-go\strata-tracker.exe ctest-gate

# 4. Inspect Strix Halo (gfx1151) status, commits, and files
.\docs-strix-go\strata-tracker.exe strix

# 5. View all 18 default, 11 active opt-in, and 6 v0.1.45 candidate Strix Halo switches
.\docs-strix-go\strata-tracker.exe switches

# 6. Check upstream GitHub API for new releases and main commits
.\docs-strix-go\strata-tracker.exe check-upstream

# 7. Search git history for specific kernel or hardware optimizations
.\docs-strix-go\strata-tracker.exe search hipblaslt
.\docs-strix-go\strata-tracker.exe search gdn
.\docs-strix-go\strata-tracker.exe search wmma

# 8. Display the Go port architectural blueprint
.\docs-strix-go\strata-tracker.exe blueprint

# 9. Verify end-to-end thinking effort, KV prefix reuse & Port 8080/8081 coexistence
.\docs-strix-go\strata-tracker.exe verify-thinking

# 10. Restart Strata server on Port 8080 (with optional -env "K=V,..." overrides)
.\docs-strix-go\strata-tracker.exe restart
```
