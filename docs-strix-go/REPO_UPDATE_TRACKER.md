# Upstream Strata Repository Synchronization & Update SOP

This Standard Operating Procedure (SOP) defines the workflow for monitoring, checking, diffing, and merging upstream updates from [Niko1221/Strata](https://github.com/Niko1221/Strata.git) into the local development environment.

---

## 1. Environment & Paths

- **Remote Upstream**: `https://github.com/Niko1221/Strata.git` (`upstream`)
- **Remote Origin (Fork)**: `https://github.com/Indigoshores/Strata.git` (`origin`)
- **Local Clone Path**: `C:\Users\Butch\Documents\02_Apps\Strata`
- **Active Branch**: `main` (or `pass7/gfx1151-v0145-opt`)
- **Automation Utility**: `docs-strix-go/strata-tracker.exe`

---

## 2. Upstream Verification Protocol (Step-by-Step)

### Step 1: Query GitHub API for Latest Releases & Commits
Run the compiled Go utility to query GitHub directly:

```powershell
.\docs-strix-go\strata-tracker.exe check-upstream
```

This inspects the latest published releases (e.g. `v0.1.45` / `v0.2.7`) and recent commits on `upstream/main`.

---

### Step 2: Fetch Remote Git References
Fetch remote branches and tags without modifying your working tree:

```powershell
git -C C:\Users\Butch\Documents\02_Apps\Strata fetch upstream main --tags
```

---

### Step 3: Inspect Incoming Commits
List all commits that exist upstream but have not yet been integrated locally:

```powershell
git -C C:\Users\Butch\Documents\02_Apps\Strata log HEAD..upstream/main --oneline
```

---

### Step 4: Check Specifically for Strix Halo / gfx1151 Updates
Filter incoming commits for AMD Strix Halo, ROCm, or hipBLASLt modifications:

```powershell
git -C C:\Users\Butch\Documents\02_Apps\Strata log HEAD..upstream/main --grep="strix" --grep="gfx1151" --grep="hipblaslt" -i --oneline
```

---

### Step 5: Diff Targeted Critical Subsystems
Before pulling, inspect diffs across high-priority Strix Halo architecture files:

```powershell
# 1. Architecture default switches
git -C C:\Users\Butch\Documents\02_Apps\Strata diff HEAD..upstream/main -- src/core/arch_defaults.cpp

# 2. Hardware documentation and measured throughput
git -C C:\Users\Butch\Documents\02_Apps\Strata diff HEAD..upstream/main -- docs/STRIX_HALO.md docs/AMD_HIP.md

# 3. Tuned hipBLASLt solution tables
git -C C:\Users\Butch\Documents\02_Apps\Strata diff HEAD..upstream/main -- tools/hip/

# 4. Windows packaging and build scripts
git -C C:\Users\Butch\Documents\02_Apps\Strata diff HEAD..upstream/main -- tools/hip/build_windows.bat tools/test_setup_strix_halo.py
```

---

### Step 6: Surgical Port or Fast-Forward Merge
> [!IMPORTANT]
> **Protect Pass 6/7 Custom Kernel Subsystems**: Before merging upstream commits, always cross-check [`docs/PASS7_GITHUB_MINING_REPORT.md`](../docs/PASS7_GITHUB_MINING_REPORT.md) (`F-01`..`F-10`) so custom `gfx1151` INT8 WMMA prefill (`moe_fused_iq.cu`), `STRATA_EXP_MODE=8` decode (`iq_kernels.cu`), and 32.6s bulk expert upload (`expert_cache.cpp`) are never overwritten by slower vanilla upstream paths.

```powershell
# For isolated upstream branches or cherry-picks:
git -C C:\Users\Butch\Documents\02_Apps\Strata cherry-pick <commit-sha>
```

---

### Step 7: Verify Post-Merge Health
Run the status check to ensure all required tuning tables and source files are present:

```powershell
.\docs-strix-go\strata-tracker.exe strix
```
