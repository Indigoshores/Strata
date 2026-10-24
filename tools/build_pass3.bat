@echo off
setlocal EnableDelayedExpansion

echo =======================================================
echo Strata Pass 3A/3B gfx1151 HIP Engine Build (Throttled -j 6)
echo =======================================================

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1

set "HIP_PLATFORM=amd"
set "HIP_PATH=C:\Program Files\AMD\ROCm\7.2"
set "ROCM_PATH=C:\Program Files\AMD\ROCm\7.2"
set "ROCM_F=C:/PROGRA~1/AMD/ROCm/7.2"
set "BITCODE=%ROCM_F%/amdgcn/bitcode"
set "PATH=C:\Program Files\AMD\ROCm\7.2\bin;C:\Users\Butch\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;C:\Program Files\CMake\bin;%PATH%"

set "SRC=C:\Users\Butch\Documents\02_Apps\Strata"
set "BUILD_DIR=C:\Users\Butch\Documents\02_Apps\Strata\build-pass3"
set "GGML_SRC=C:/Users/Butch/Documents/10-Models/Strata/optimization-lab/strata-combined/build-hip-control/_deps/strata_llamacpp-src"

echo Configuring CMake for gfx1151 in %BUILD_DIR%...
cmake -G Ninja -S "%SRC%" -B "%BUILD_DIR%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DSTRATA_ENABLE_HIP=ON ^
  -DSTRATA_ENABLE_CUDA=OFF ^
  -DSTRATA_BUILD_TESTS=ON ^
  -DSTRATA_PREFILL_MMQ=ON ^
  -DSTRATA_NATIVE_EXPERTS=ON ^
  -DSTRATA_PORTABLE=ON ^
  "-DCMAKE_HIP_ARCHITECTURES=gfx1151" ^
  "-DCMAKE_C_COMPILER=%ROCM_F%/bin/clang.exe" ^
  "-DCMAKE_CXX_COMPILER=%ROCM_F%/bin/clang++.exe" ^
  "-DCMAKE_HIP_COMPILER=%ROCM_F%/bin/clang++.exe" ^
  "-DCMAKE_HIP_COMPILER_ROCM_ROOT=%ROCM_F%" ^
  "-DCMAKE_PREFIX_PATH=%ROCM_F%" ^
  "-DCMAKE_HIP_FLAGS=--rocm-path=%ROCM_F% --rocm-device-lib-path=%BITCODE%" ^
  "-DSTRATA_GGML_DIR=%GGML_SRC%" || exit /b 1

echo Building targets strata, strata-device, native_iq_wmma_config_test, lookahead_policy_test, hip_prefill_hipblaslt_gemm, tune_hipblaslt with --parallel 6...
cmake --build "%BUILD_DIR%" --parallel 6 --target strata strata-device native_iq_wmma_config_test lookahead_policy_test hip_prefill_hipblaslt_gemm tune_hipblaslt || exit /b 1

echo Pass 3 Build finished successfully!
