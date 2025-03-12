@echo off
setlocal enabledelayedexpansion

echo =====================================
echo          CUDA Version Check
echo =====================================

REM get driver's CUDA version
for /f "tokens=4" %%a in ('nvidia-smi --version ^| findstr "DRIVER version"') do (
    set "driver_version=%%a"
)
for /f "tokens=4" %%a in ('nvidia-smi --version ^| findstr "CUDA Version"') do (
    set "supported_cuda_version=%%a"
)

REM get installed CUDA version
for /f "tokens=5 delims=, " %%a in ('nvcc --version ^| findstr "release"') do (
    set "installed_cuda_version=%%a"
)

REM get CuDNN version
set cudnn_file=
for /f "delims=" %%i in ('dir /s /b "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\include\cudnn_version.h" 2^>nul') do (
    set cudnn_file=%%i
)
if defined cudnn_file (
    echo Found CuDNN header: !cudnn_file!
    for /f "tokens=3" %%a in ('type "!cudnn_file!" ^| findstr "CUDNN_MAJOR CUDNN_MINOR CUDNN_PATCHLEVEL"') do (
        set /a count+=1
        if !count! equ 1 set cudnn_major=%%a
        if !count! equ 2 set cudnn_minor=%%a
        if !count! equ 3 set cudnn_patch=%%a
    )
) else (
    echo CuDNN not found
)

REM get ONNX Runtime version
set onnx_ver=
for /f "tokens=5 delims== " %%a in ('findstr /C:"<PackageReference Include=\"Microsoft.ML.OnnxRuntime.Gpu\"\ Version=" tagmane.csproj') do (
    set ver_with_quotes=%%a
    set onnx_ver=!ver_with_quotes:"=!
)
if defined onnx_ver (
    echo:
) else (
    echo ONNX Runtime not found in tagmane.csproj
)


echo Driver Version:         %driver_version%
echo Supported CUDA Version: %supported_cuda_version%
echo Installed CUDA Version: %installed_cuda_version%
echo CuDNN Version:          !cudnn_major!.!cudnn_minor!.!cudnn_patch!
echo ONNX Runtime Version:   !onnx_ver!

echo =====================================
echo          End of Check
echo =====================================

REM pause
