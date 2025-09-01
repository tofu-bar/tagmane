@echo off
echo GLM-4.1V Environment Setup
echo ========================

:: Python実行可能ファイルのパスを取得
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found in PATH
    echo Please install Python 3.8 or later
    pause
    exit /b 1
)

:: 現在のディレクトリを保存
set SCRIPT_DIR=%~dp0
set VENV_PATH=%SCRIPT_DIR%\.venv

echo Detecting CUDA environment...

:: CUDA環境の検出
set CUDA_VERSION=""
set PYTORCH_INDEX_URL=""
set CUDA_AVAILABLE=false

:: nvidia-smiでCUDAドライバーバージョンを取得
nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected, checking CUDA version...
    set CUDA_AVAILABLE=true
    
    :: CUDAランタイムバージョンをnvidia-smiから取得
    for /f "tokens=9" %%i in ('nvidia-smi ^| findstr "CUDA Version"') do (
        set CUDA_VERSION_FULL=%%i
    )
    
    :: nvcc が利用可能かチェック（開発環境）
    nvcc --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo CUDA development tools detected
        for /f "tokens=4" %%i in ('nvcc --version ^| findstr "release"') do (
            set NVCC_VERSION=%%i
            echo NVCC version: %%i
        )
    )
    
    :: CUDAバージョンに基づいてPyTorchインデックスURLを決定
    if defined CUDA_VERSION_FULL (
        echo CUDA Version: %CUDA_VERSION_FULL%
        
        :: バージョン番号を抽出してPyTorchの対応バージョンを決定
        echo Determining PyTorch compatibility for CUDA %CUDA_VERSION_FULL%
        
        :: より確実なバージョンマッチング
        if "%CUDA_VERSION_FULL:~0,4%"=="12.6" (
            set PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
            set CUDA_VERSION=cu121
            echo Selected: PyTorch CUDA 12.1 ^(compatible with 12.6^)
        ) else if "%CUDA_VERSION_FULL:~0,4%"=="12.5" (
            set PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
            set CUDA_VERSION=cu121
            echo Selected: PyTorch CUDA 12.1 ^(compatible with 12.5^)
        ) else if "%CUDA_VERSION_FULL:~0,4%"=="12.4" (
            set PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
            set CUDA_VERSION=cu121
            echo Selected: PyTorch CUDA 12.1 ^(compatible with 12.4^)
        ) else if "%CUDA_VERSION_FULL:~0,4%"=="12.3" (
            set PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
            set CUDA_VERSION=cu121
            echo Selected: PyTorch CUDA 12.1 ^(compatible with 12.3^)
        ) else if "%CUDA_VERSION_FULL:~0,4%"=="12.2" (
            set PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
            set CUDA_VERSION=cu121
            echo Selected: PyTorch CUDA 12.1 ^(compatible with 12.2^)
        ) else if "%CUDA_VERSION_FULL:~0,4%"=="12.1" (
            set PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
            set CUDA_VERSION=cu121
            echo Selected: PyTorch CUDA 12.1
        ) else if "%CUDA_VERSION_FULL:~0,4%"=="12.0" (
            set PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
            set CUDA_VERSION=cu118
            echo Selected: PyTorch CUDA 11.8 ^(compatible with 12.0^)
        ) else if "%CUDA_VERSION_FULL:~0,4%"=="11.8" (
            set PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
            set CUDA_VERSION=cu118
            echo Selected: PyTorch CUDA 11.8
        ) else if "%CUDA_VERSION_FULL:~0,4%"=="11.7" (
            set PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu117
            set CUDA_VERSION=cu117
            echo Selected: PyTorch CUDA 11.7
        ) else if "%CUDA_VERSION_FULL:~0,4%"=="11.6" (
            set PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu116
            set CUDA_VERSION=cu116
            echo Selected: PyTorch CUDA 11.6
        ) else (
            :: 不明なバージョンの場合は最も互換性の高いものを選択
            set PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
            set CUDA_VERSION=cu118
            echo Warning: Unknown CUDA version %CUDA_VERSION_FULL%, using CUDA 11.8 PyTorch
        )
    )
) else (
    echo No NVIDIA GPU detected or nvidia-smi not available
)

:: CUDAが利用できない場合はCPU版を使用
if "%CUDA_AVAILABLE%"=="false" (
    echo Setting up CPU-only PyTorch installation
    set PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
    set CUDA_VERSION=cpu
) else if "%PYTORCH_INDEX_URL%"=="" (
    echo Warning: Could not determine CUDA version, falling back to CPU version
    set PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
    set CUDA_VERSION=cpu
)

echo PyTorch installation will use: %CUDA_VERSION%
echo Index URL: %PYTORCH_INDEX_URL%
echo.

echo Setting up Python virtual environment...

:: 仮想環境が存在するかチェック
if not exist "%VENV_PATH%" (
    echo Creating new virtual environment at %VENV_PATH%
    python -m venv "%VENV_PATH%"
    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists
)

:: 仮想環境をアクティベート
echo Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"

:: pipをアップグレード
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

:: 検出されたCUDA環境に適したPyTorchのインストール
if "%CUDA_VERSION%"=="cpu" (
    echo Installing PyTorch CPU version...
    python -m pip install torch torchvision torchaudio --index-url %PYTORCH_INDEX_URL%
) else (
    echo Installing PyTorch with %CUDA_VERSION% support...
    python -m pip install torch torchvision torchaudio --index-url %PYTORCH_INDEX_URL%
)

if %errorlevel% neq 0 (
    echo Error: Failed to install PyTorch
    pause
    exit /b 1
)

:: PyTorchの正しいインストールを確認
echo Verifying PyTorch installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

if %errorlevel% neq 0 (
    echo Error: PyTorch verification failed
    pause
    exit /b 1
)

:: その他の依存関係をインストール（PyTorch以外）
echo Installing other dependencies...
python -m pip install numpy>=1.21.0 Pillow>=9.0.0
python -m pip install huggingface-hub>=0.16.0 tokenizers>=0.13.0 safetensors>=0.3.0
python -m pip install transformers>=4.35.0 accelerate>=0.20.0 sentencepiece>=0.1.99

if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ================================
echo GLM-4.1V environment setup complete!
echo ================================
echo.
if "%CUDA_VERSION%"=="cpu" (
    echo WARNING: CPU-only PyTorch installation detected
    echo GLM-4.1V will run on CPU, which is significantly slower
    echo Consider installing CUDA drivers and toolkit for better performance
    echo.
) else (
    echo CUDA-enabled PyTorch successfully installed
    echo GLM-4.1V will utilize GPU acceleration
    echo.
)
echo To use this environment:
echo 1. Run: %VENV_PATH%\Scripts\activate.bat
echo 2. Then run: python caption_generator.py --help
echo.
echo Or use start-glm-interactive.bat to launch directly
echo.
pause