# How to Use Your GGUF Models

## The Truth About GGUF Files

Your models at `D:\AIModels\Qwen-Image-FP8\gguf\` are:
- ✅ **Qwen-Image-Edit-2509-Q4_0.gguf** (UNet, quantized)
- ✅ **Qwen2.5-VL-7B-Instruct-Q3_K_S.gguf** (CLIP, quantized)
- ✅ **qwen_image_vae.safetensors** (VAE)

## ❌ What DOES NOT Work

**You CANNOT load GGUF files in Python standalone.**

```python
# ❌ THIS WILL NEVER WORK
from diffusers import QwenImageEditPlusPipeline
pipeline = QwenImageEditPlusPipeline.from_pretrained("D:\\AIModels\\Qwen-Image-FP8")
# Error: No model_index.json found
```

**Why?**
- GGUF = Quantized format for ComfyUI
- Python's Diffusers library expects full precision models
- No Python library can load GGUF without ComfyUI backend

## ✅ What DOES Work

### Option 1: Use ComfyUI with Python API (RECOMMENDED)

**This is the ONLY way to use your GGUF models programmatically.**

#### Step 1: Start ComfyUI
```bash
cd path\to\ComfyUI
python main.py
```

#### Step 2: Link Your Models
```bash
# Windows (as Administrator)
mklink /D "ComfyUI\models\unet" "D:\AIModels\Qwen-Image-FP8\gguf\unet"
mklink /D "ComfyUI\models\clip" "D:\AIModels\Qwen-Image-FP8\gguf\clip"
mklink /D "ComfyUI\models\vae" "D:\AIModels\Qwen-Image-FP8\vae"
```

Or copy the files manually.

#### Step 3: Use the API Script
```bash
pip install websocket-client
python run_comfyui_workflow.py
```

**This script:**
- ✅ Uses your GGUF models (low VRAM)
- ✅ Works programmatically from Python
- ✅ No additional downloads needed
- ✅ Full control over parameters

### Option 2: Use ComfyUI GUI

1. Start ComfyUI
2. Load `workflow.json`
3. Select your input image
4. Click "Queue Prompt"

**Benefits:**
- Visual interface
- Real-time parameter adjustment
- See progress in real-time

## ❌ Why Not Download "Qwen/Qwen-Image-Edit-2509"?

You mentioned **limited VRAM**. Here's why downloading from Hugging Face won't help:

| Model Type | Size | VRAM Required |
|------------|------|---------------|
| Your GGUF (Q4_0) | ~8GB | ~10GB VRAM |
| Hugging Face (FP16) | ~20GB | ~24GB VRAM |
| Hugging Face (BF16) | ~20GB | ~24GB VRAM |

**Your GGUF models are ALREADY optimized for low VRAM!**
- Q4_0 = 4-bit quantization (75% smaller)
- Q3_K_S = 3-bit quantization (even smaller)

Downloading the full models would make your VRAM situation WORSE.

## The ONLY Script You Should Use

**File: `run_comfyui_workflow.py`**

```python
from run_comfyui_workflow import ComfyUIWorkflowRunner

# Initialize (ComfyUI must be running)
runner = ComfyUIWorkflowRunner(server_address="127.0.0.1:8188")

# Run workflow with YOUR GGUF models
result = runner.run_workflow(
    input_image_path="input.png",
    prompt_text="Add the word 'Great!' into the image.",
    output_path="output.png",
    seed=748260172255095,
    steps=8,
    cfg=1.0,
    width=768,
    height=768
)
```

**This uses:**
- ✅ Your local Qwen-Image-Edit-2509-Q4_0.gguf
- ✅ Your local Qwen2.5-VL-7B-Instruct-Q3_K_S.gguf
- ✅ Your local qwen_image_vae.safetensors
- ✅ Low VRAM (quantized models)
- ✅ No downloads

## Scripts Status

| Script | Works with GGUF? | Notes |
|--------|------------------|-------|
| `run_comfyui_workflow.py` | ✅ YES | **Use this one!** |
| `workflow.json` | ✅ YES | Load in ComfyUI GUI |
| `comfyui_standalone.py` | ❌ NO | Cannot load GGUF |
| `qwen-code-1.py` | ❌ NO | Downloads 20GB models |

## Complete Working Example

```bash
# 1. Start ComfyUI (in one terminal)
cd C:\path\to\ComfyUI
python main.py

# 2. Run your workflow (in another terminal)
cd D:\Testers\ai_qwen
python run_comfyui_workflow.py
```

**That's it!** Your GGUF models will be used automatically.

## Troubleshooting

### "Connection refused"
- ComfyUI is not running
- Start ComfyUI first

### "Model not found"
- Models not in ComfyUI directories
- Create symlinks or copy files

### "Out of memory"
- Your GGUF models are already optimized
- Close other applications
- Reduce image size (width/height)

## Summary

**To use your GGUF models from the screenshot:**

1. ✅ Start ComfyUI
2. ✅ Use `run_comfyui_workflow.py`
3. ❌ Do NOT try to load GGUF in Python directly
4. ❌ Do NOT download Qwen/Qwen-Image-Edit-2509 (wastes VRAM)

Your GGUF models are perfect for low VRAM. Just use them through ComfyUI's API.
