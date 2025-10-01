@echo off
setlocal

REM === Configuration ===
set "VENV_PATH=C:\PythonEnv\p311_qwen"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"

echo [1/10] Activating virtual environment at %VENV_PATH%
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo ERROR: activate.bat not found. Update VENV_PATH in setup_qwen_gpu.bat.
    exit /b 1
)
call "%VENV_PATH%\Scripts\activate.bat"

echo [2/10] Upgrading pip
python -m pip install --upgrade pip

echo [3/10] Removing CPU-only torch packages (safe to ignore if absent)
python -m pip uninstall -y torch torchvision torchaudio 

echo [4/10] Installing CUDA-enabled torch stack from %TORCH_INDEX%
python -m pip install --upgrade torch torchvision torchaudio --index-url %TORCH_INDEX%

echo [5/10] Installing/refreshing diffusers (bleeding edge for Qwen pipeline)
python -m pip install --upgrade "git+https://github.com/huggingface/diffusers.git"

echo [6/10] Installing accelerator extras (huggingface_hub, transformers, Pillow)
python -m pip install --upgrade huggingface_hub transformers pillow

echo [7/10] Installing xformers for optimized attention (optional but recommended)
python -m pip install --upgrade xformers --extra-index-url %TORCH_INDEX%

echo [8/10] Verifying CUDA availability inside Python
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'CUDA version:', torch.version.cuda)"

echo [9/10] Downloading Qwen model snapshot (resume supported)
python download.py

echo [10/10] Running image edit pipeline over images/ directory
python run_qwen_edit.py

echo Done. Check the console output above for any errors.
endlocal
