# Upstream Strata Search Cheat Sheet & Query Catalog

This cheat sheet provides pre-formulated search queries and file maps for quickly discovering optimizations, kernel upgrades, and bug fixes in the [Niko1221/Strata](https://github.com/Niko1221/Strata.git) repository.

---

## 🔍 Fast Search via Pure Go 1.26 CLI

Use `strata-tracker.exe` for instant local searching:

```powershell
# Search commits and file paths matching any keyword
.\docs-strix-go\strata-tracker.exe search gfx1151
.\docs-strix-go\strata-tracker.exe search hipblaslt
.\docs-strix-go\strata-tracker.exe search gdn
.\docs-strix-go\strata-tracker.exe search direct_file
.\docs-strix-go\strata-tracker.exe search wmma
```

---

## 📑 Domain-Specific Query Matrix

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
  <table style="width: 100%; min-width: 600px; border-collapse: collapse; background-color: #161b22; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 14px;">
    <thead>
      <tr style="background-color: #21262d; border-bottom: 2px solid #30363d; text-align: left;">
        <th style="padding: 12px 16px; color: #58a6ff; white-space: nowrap;">Domain</th>
        <th style="padding: 12px 16px; color: #58a6ff; white-space: nowrap;">Primary Keywords</th>
        <th style="padding: 12px 16px; color: #58a6ff; white-space: nowrap;">Target Code Files</th>
        <th style="padding: 12px 16px; color: #58a6ff; white-space: nowrap;">Git Log Command</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;">Strix Halo & APU</td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>gfx1151</code>, <code>strix</code>, <code>halo</code>, <code>8060S</code></td>
        <td style="padding: 12px 16px;"><code>src/core/arch_defaults.cpp</code><br><code>docs/STRIX_HALO.md</code></td>
        <td style="padding: 12px 16px;"><code>git log --grep="gfx1151" -i --oneline</code></td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;">Matrix Acceleration</td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>wmma</code>, <code>hipblaslt</code>, <code>100500</code></td>
        <td style="padding: 12px 16px;"><code>src/prefill/wmma_gemm.cu</code><br><code>tools/hip/gfx1151-hipblaslt-*.txt</code></td>
        <td style="padding: 12px 16px;"><code>git log --grep="hipblaslt" -i --oneline</code></td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;">Linear Attention</td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>gdn</code>, <code>delta_net</code>, <code>convl2</code></td>
        <td style="padding: 12px 16px;"><code>src/kernels/cuda/gdn.cu</code><br><code>src/kernels/cuda/native_gdn.cu</code></td>
        <td style="padding: 12px 16px;"><code>git log --grep="gdn" -i --oneline</code></td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;">Direct SSD I/O</td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>direct_file</code>, <code>iocp</code>, <code>overlapped</code></td>
        <td style="padding: 12px 16px;"><code>src/platform/direct_file.cpp</code><br><code>src/ngram/ple_reader.cpp</code></td>
        <td style="padding: 12px 16px;"><code>git log --grep="direct_file" -i --oneline</code></td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;">Unified Memory / RAM</td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>virtuallock</code>, <code>gtt</code>, <code>dxgi</code>, <code>wddm</code></td>
        <td style="padding: 12px 16px;"><code>src/platform/memory.cpp</code><br><code>src/core/expert_cache.cpp</code></td>
        <td style="padding: 12px 16px;"><code>git log --grep="memory" -i --oneline</code></td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;">Speculative Decoding</td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>mtp</code>, <code>gumbel</code>, <code>suffix_drafter</code></td>
        <td style="padding: 12px 16px;"><code>src/core/mtp.cpp</code><br><code>src/spec/controller.cpp</code></td>
        <td style="padding: 12px 16px;"><code>git log --grep="spec" -i --oneline</code></td>
      </tr>
      <tr style="border-bottom: 1px solid #30363d;">
        <td style="padding: 12px 16px; font-weight: bold; white-space: nowrap;">Windows Engine Zip</td>
        <td style="padding: 12px 16px; white-space: nowrap;"><code>build_windows</code>, <code>package_windows</code></td>
        <td style="padding: 12px 16px;"><code>tools/hip/build_windows.bat</code><br><code>tools/hip/package_windows.py</code></td>
        <td style="padding: 12px 16px;"><code>git log --grep="windows" -i --oneline</code></td>
      </tr>
    </tbody>
  </table>
</div>
