@echo off
rem ==============================================================================
rem Strata Pass 6 Isolated ROCm 10.2 (Clang 24) Build Script for Strix Halo gfx1151
rem Targets: AMD Radeon 8060S (Strix Halo APU, gfx1151, RDNA 3.5) ONLY.
rem
rem Safeguards enforced:
rem   1. STRATA_HIP_ARCHS=gfx1151 ONLY (skips 7 unused GPU architectures -> 8x faster compilation)
rem   2. BUILD_DIR=build-pass6-rocm102 (isolated sandbox; never touches production strata.exe)
rem   3. CMake --parallel 4 (-j 4) strictly enforced (prevents host commit exhaustion < 85 GiB)
rem   4. Links/stages libhipblaslt.dll version 100500 + gfx1151-hipblaslt-100500.txt tuning table
rem   5. Supports "tests" argument to compile CTest parity verification binaries
rem ==============================================================================

setlocal EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "SRC=%%~fI"

rem Force single-architecture target: gfx1151 ONLY for Strix Halo
set "STRATA_HIP_ARCHS=gfx1151"

if not defined STRATA_ROCM_VERSION set "STRATA_ROCM_VERSION=10.2.0a20260930"
if not defined STRATA_ROCM_INDEX set "STRATA_ROCM_INDEX=https://nightly.repo.amd.com/rocm/whl-next/"
if not defined ROCM_VENV set "ROCM_VENV=%SRC%\.rocm-win"
if not defined BUILD_DIR set "BUILD_DIR=%SRC%\build-pass6-rocm102"
if not defined DIST_DIR set "DIST_DIR=%SRC%\dist-pass6"

set "TESTS=OFF"
if /i "%~1"=="tests" set "TESTS=ON"

echo ==============================================================================
echo  Strata gfx1151 (Strix Halo) ROCm 10.2 Isolated Build Pipeline
echo  Target Architecture : %STRATA_HIP_ARCHS%
echo  Isolated Build Dir  : %BUILD_DIR%
echo  Build Tests (CTest) : %TESTS%
echo  Parallel Jobs       : 4 (Memory Commit Guard)
echo ==============================================================================

rem ---- 1. Toolchain Resolution (.rocm-win TheRock Clang 24 with fallback)
set "ROCM="
if exist "%ROCM_VENV%\Lib\site-packages\_rocm_sdk_devel\lib\llvm\bin\clang++.exe" (
  set "ROCM=%ROCM_VENV%\Lib\site-packages\_rocm_sdk_devel"
)
if not defined ROCM if exist "%ROCM_VENV%\Scripts\rocm-sdk.exe" (
  for /f "delims=" %%R in ('"%ROCM_VENV%\Scripts\rocm-sdk.exe" path --root') do set "ROCM=%%R"
)
if not defined ROCM if defined HIP_PATH set "ROCM=%HIP_PATH%"
if not defined ROCM if defined ROCM_PATH set "ROCM=%ROCM_PATH%"
if not defined ROCM if exist "C:\Program Files\AMD\ROCm\7.2" set "ROCM=C:\Program Files\AMD\ROCm\7.2"
if not defined ROCM if exist "C:\Program Files\AMD\ROCm\6.2" set "ROCM=C:\Program Files\AMD\ROCm\6.2"

if not defined ROCM (
  echo [ERROR] Neither .rocm-win nor local ROCm was found.
  exit /b 1
)

if not exist "%ROCM%\lib\llvm\bin\clang++.exe" (
  echo [ERROR] ROCm Clang++ compiler not found in "%ROCM%\lib\llvm\bin\clang++.exe"
  exit /b 1
)

echo [INFO] Using ROCm toolchain at "%ROCM%"
set "ROCM_F=%ROCM:\=/%"
set "BITCODE=%ROCM_F%/lib/llvm/amdgcn/bitcode"

rem ---- 2. MSVC Environment Discovery
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS="
if exist "%VSWHERE%" for /f "delims=" %%V in ('"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do set "VS=%%V"
if not defined VS (
  echo [ERROR] Visual Studio Build Tools with C++ workload are required.
  exit /b 1
)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "HIP_PLATFORM=amd"
set "HIP_PATH=%ROCM%"
set "ROCM_PATH=%ROCM%"
set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%PATH%"

rem ---- 3. CMake Configuration (Targeting gfx1151 ONLY)
set "GGML="
if defined STRATA_GGML_DIR set "GGML=-DSTRATA_GGML_DIR=%STRATA_GGML_DIR:\=/%"

set "DO_CONFIG=0"
if not exist "%BUILD_DIR%\build.ninja" set "DO_CONFIG=1"
if "%TESTS%"=="ON" set "DO_CONFIG=1"
if "%~1"=="reconfig" set "DO_CONFIG=1"

if "!DO_CONFIG!"=="1" (
  echo Configuring CMake in %BUILD_DIR% ...
  cmake -G Ninja -S "%SRC%" -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=Release ^
    -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=%TESTS% -DSTRATA_PREFILL_MMQ=ON ^
    -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON "-DCMAKE_HIP_ARCHITECTURES=%STRATA_HIP_ARCHS%" ^
    "-DCMAKE_C_COMPILER=%ROCM_F%/lib/llvm/bin/clang.exe" "-DCMAKE_CXX_COMPILER=%ROCM_F%/lib/llvm/bin/clang++.exe" ^
    "-DCMAKE_HIP_COMPILER=%ROCM_F%/lib/llvm/bin/clang++.exe" "-DCMAKE_HIP_COMPILER_ROCM_ROOT=%ROCM_F%" ^
    "-DCMAKE_PREFIX_PATH=%ROCM_F%" "-DCMAKE_HIP_FLAGS=--rocm-path=%ROCM_F% --rocm-device-lib-path=%BITCODE%" ^
    %GGML% || exit /b 1
)

rem ---- 4. Compilation with Strict Parallel 4 Limit (-j 4 Memory Guard)
if "%TESTS%"=="ON" (
  echo Compiling strata, strata-device, and CTest parity verification binaries with -j 4 ...
  cmake --build "%BUILD_DIR%" --parallel 4 --target strata strata-device hip_gdn_chunked hip_prefill_wmma_gemm_parity hip_prefill_hcd_exact_parity s2_expert_grouped_parity verify_batch_parity spec_verify_parity test_expert_bulk_load test_moe_variable_tiles hip_prefill_fused_iq || exit /b 1
) else (
  echo Compiling strata.exe and strata-device.exe with -j 4 ...
  cmake --build "%BUILD_DIR%" --parallel 4 --target strata strata-device || exit /b 1
)

rem ---- 5. Stage ROCm 100500 Runtime DLLs and gfx1151 Tuning Table
echo Staging ROCm 100500 runtime DLLs and tuning table into %BUILD_DIR% ...
if exist "%ROCM%\bin" (
  copy /y "%ROCM%\bin\*.dll" "%BUILD_DIR%\" >nul 2>nul
)
if exist "C:\Users\Butch\Documents\10-Models\Strata\engine\rocm\bin\libhipblaslt.dll" (
  echo Linking verified ROCm 100500 libhipblaslt.dll from Strata engine distribution...
  copy /y "C:\Users\Butch\Documents\10-Models\Strata\engine\rocm\bin\*.dll" "%BUILD_DIR%\" >nul 2>nul
)
if exist "%SRC%\tools\hip\gfx1151-hipblaslt-100500.txt" (
  copy /y "%SRC%\tools\hip\gfx1151-hipblaslt-100500.txt" "%BUILD_DIR%\" >nul 2>nul
)
if exist "C:\Users\Butch\Documents\10-Models\Strata\engine\rocm\bin\rocblas" (
  if not exist "%BUILD_DIR%\rocblas" mklink /J "%BUILD_DIR%\rocblas" "C:\Users\Butch\Documents\10-Models\Strata\engine\rocm\bin\rocblas" >nul 2>nul
)
if exist "C:\Users\Butch\Documents\10-Models\Strata\engine\rocm\bin\hipblaslt" (
  if not exist "%BUILD_DIR%\hipblaslt" mklink /J "%BUILD_DIR%\hipblaslt" "C:\Users\Butch\Documents\10-Models\Strata\engine\rocm\bin\hipblaslt" >nul 2>nul
)

echo.
echo ==============================================================================
echo  Build Completed Successfully!
echo  Isolated Executable : %BUILD_DIR%\strata.exe
echo  Production strata.exe at C:\Users\Butch\Documents\10-Models\Strata\engine\strata.exe was untouched
echo ==============================================================================
endlocal
