@echo off
echo GLM-4.1V Interactive Session Launcher
echo =====================================

:: 現在のディレクトリを保存
set SCRIPT_DIR=%~dp0
set VENV_PATH=%SCRIPT_DIR%\.venv

:: 仮想環境の存在をチェック
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo Error: Virtual environment not found at %VENV_PATH%
    echo Please run setup-glm-env.bat first to create the environment
    pause
    exit /b 1
)

:: 仮想環境をアクティベート
echo Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"

:: PyTorchの動作確認
echo Checking PyTorch availability...
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')" 2>nul
if %errorlevel% neq 0 (
    echo Error: PyTorch not properly installed
    echo Please run setup-glm-env.bat to fix the installation
    pause
    exit /b 1
)

:: GLM-4.1Vをインタラクティブモードで起動
echo Starting GLM-4.1V caption generator in interactive mode...
echo Use Ctrl+C to stop the process
echo.

python caption_generator.py --interactive

:: エラーコードをチェック
if %errorlevel% neq 0 (
    echo.
    echo GLM-4.1V process exited with error code: %errorlevel%
    pause
)

echo.
echo GLM-4.1V session ended.
pause