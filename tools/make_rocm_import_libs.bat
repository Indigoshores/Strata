@echo off
setlocal EnableDelayedExpansion

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1

cd /d "C:\Users\Butch\Documents\02_Apps\Strata\rocm100500-libs" || exit /b 1

echo Generating libhipblaslt.lib...
lib.exe /nologo /def:libhipblaslt.def /out:libhipblaslt.lib /machine:x64 || exit /b 1

echo Generating rocblas.lib...
lib.exe /nologo /def:rocblas.def /out:rocblas.lib /machine:x64 || exit /b 1

echo Generating hipblas.lib...
lib.exe /nologo /def:hipblas.def /out:hipblas.lib /machine:x64 || exit /b 1

echo Generating amdhip64_7.lib...
lib.exe /nologo /def:amdhip64_7.def /out:amdhip64_7.lib /machine:x64 || exit /b 1

echo Copying libhipblaslt.lib to hipblaslt.lib for fallback compatibility...
copy /Y libhipblaslt.lib hipblaslt.lib

echo All import libraries created successfully!
