# Installation Guide

## Quick Start

### 1. Install PyTorch with CUDA Support

**Check your CUDA version first:**
```bash
nvidia-smi
```

**Install PyTorch (choose based on your CUDA version):**

```bash
# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only (not recommended for image generation)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Install Project Dependencies

```bash
# Navigate to project directory
cd D:\Testers\ai_qwen

# Install all dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA Available: True
```

---

## Installation Options

### Option A: Minimal Installation (Recommended)

For using GGUF models with ComfyUI:

```bash
pip install websocket-client Pillow requests
```

**What you can run:**
- `run_comfyui_workflow.py` ✅
- `workflow.json` (in ComfyUI GUI) ✅

**Requirements:**
- ComfyUI must be installed separately
- Your GGUF models at `D:\AIModels\Qwen-Image-FP8`

---

### Option B: Full Installation

For using Diffusers models (downloads ~20GB):

```bash
# 1. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install all dependencies
pip install -r requirements.txt
```

**What you can run:**
- `qwen-code-1.py` ✅
- `run_qwen_edit.py` ✅
- `download.py` ✅

**Note:** This will download large models from Hugging Face (~20GB)

---

### Option C: Development Installation

For development and testing:

```bash
# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install detailed requirements
pip install -r requirements-detailed.txt

# Optional: Install development tools
pip install pytest black pylint mypy
```

---

## Script Requirements Matrix

| Script | PyTorch | Diffusers | ComfyUI | WebSocket | Notes |
|--------|---------|-----------|---------|-----------|-------|
| `run_comfyui_workflow.py` | ❌ | ❌ | ✅ | ✅ | Uses GGUF models |
| `workflow.json` | ❌ | ❌ | ✅ | ❌ | Load in ComfyUI |
| `qwen-code-1.py` | ✅ | ✅ | ❌ | ❌ | Downloads 20GB |
| `run_qwen_edit.py` | ✅ | ✅ | ❌ | ❌ | Downloads 20GB |
| `download.py` | ❌ | ❌ | ❌ | ❌ | Downloads models |
| `comfyui_standalone.py` | ✅ | ✅ | ❌ | ❌ | Won't work with GGUF |

---

## Troubleshooting

### "No module named 'torch'"

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### "CUDA out of memory"

**Solution 1:** Use GGUF models (recommended)
```bash
# Your GGUF models use 60% less VRAM
python run_comfyui_workflow.py
```

**Solution 2:** Reduce batch size or image resolution

**Solution 3:** Use CPU (very slow)
```python
pipeline.to("cpu")
```

### "Cannot access gated repo"

This means you're trying to download from Hugging Face without authentication.

**Solution:** Use your local GGUF models with ComfyUI instead:
```bash
python run_comfyui_workflow.py
```

### "Connection refused" (ComfyUI API)

ComfyUI is not running.

**Solution:**
```bash
# Start ComfyUI first
cd path\to\ComfyUI
python main.py

# Then run your script
python run_comfyui_workflow.py
```

### "Model not found"

**For GGUF models:**
Make sure models are in ComfyUI directories:
```
ComfyUI/models/unet/Qwen-Image-Edit-2509-Q4_0.gguf
ComfyUI/models/clip/Qwen2.5-VL-7B-Instruct-Q3_K_S.gguf
ComfyUI/models/vae/qwen_image_vae.safetensors
```

**For Diffusers models:**
Run the download script:
```bash
python download.py
```

---

## Virtual Environment (Recommended)

### Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Deactivate

```bash
deactivate
```

---

## System Requirements

### Minimum Requirements
- **OS:** Windows 10/11, Linux, macOS
- **Python:** 3.9 or higher
- **RAM:** 16GB
- **VRAM:** 8GB (for GGUF models)
- **Disk:** 10GB free space

### Recommended Requirements
- **OS:** Windows 11, Linux
- **Python:** 3.10 or 3.11
- **RAM:** 32GB
- **VRAM:** 12GB or higher
- **Disk:** 50GB free space (if using full models)
- **GPU:** NVIDIA RTX 3060 or better

---

## Quick Reference

### For GGUF Models (Low VRAM)
```bash
pip install websocket-client Pillow requests
python run_comfyui_workflow.py
```

### For Diffusers Models (High VRAM)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python qwen-code-1.py
```

---

## Next Steps

After installation:

1. **Using GGUF models:** See `USE_YOUR_GGUF_MODELS.md`
2. **Using Diffusers:** See `README.md`
3. **ComfyUI workflow:** See `COMFYUI_WORKFLOW_README.md`
4. **Troubleshooting:** See `GGUF_MODELS_GUIDE.md`

---

## Support

- **PyTorch Installation:** https://pytorch.org/get-started/locally/
- **Hugging Face Hub:** https://huggingface.co/docs/huggingface_hub
- **ComfyUI:** https://github.com/comfyanonymous/ComfyUI
- **Qwen Models:** https://huggingface.co/Qwen
