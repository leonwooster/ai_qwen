# Using GGUF Models with Qwen Image Edit

## The Problem

Your models at `D:\AIModels\Qwen-Image-FP8` are in **GGUF format**, which is:
- ✅ **Compatible with ComfyUI**
- ❌ **NOT compatible with Python's Diffusers library**

The error you encountered happens because:
1. Diffusers expects a specific directory structure with `model_index.json`
2. GGUF files are quantized models designed for ComfyUI
3. Python cannot directly load GGUF models without ComfyUI

## Your Model Files

Located at `D:\AIModels\Qwen-Image-FP8`:

```
├── gguf/
│   ├── unet/
│   │   └── Qwen-Image-Edit-2509-Q4_0.gguf          ← UNet model (quantized)
│   └── clip/
│       └── Qwen2.5-VL-7B-Instruct-Q3_K_S.gguf      ← CLIP model (quantized)
├── vae/
│   └── qwen_image_vae.safetensors                   ← VAE model
├── text_encoders/
│   └── qwen_2.5_vl_7b_fp8_scaled.safetensors       ← Text encoder
└── loras/
    ├── Qwen-Image-Lightning-4steps-V1.0.safetensors
    └── Qwen-Image-Lightning-8steps-V2.0.safetensors
```

## Solutions

### ✅ Option 1: Use ComfyUI (RECOMMENDED)

This is the **easiest and best** option since your models are already in ComfyUI format.

#### Step 1: Start ComfyUI
```bash
# Navigate to your ComfyUI directory
cd path/to/ComfyUI

# Start ComfyUI
python main.py
```

#### Step 2: Configure Model Paths
Make sure your models are in ComfyUI's model directories:
- Copy `Qwen-Image-Edit-2509-Q4_0.gguf` → `ComfyUI/models/unet/`
- Copy `Qwen2.5-VL-7B-Instruct-Q3_K_S.gguf` → `ComfyUI/models/clip/`
- Copy `qwen_image_vae.safetensors` → `ComfyUI/models/vae/`

Or create symlinks:
```bash
# Windows (run as Administrator)
mklink /D "ComfyUI\models\unet" "D:\AIModels\Qwen-Image-FP8\gguf\unet"
mklink /D "ComfyUI\models\clip" "D:\AIModels\Qwen-Image-FP8\gguf\clip"
mklink /D "ComfyUI\models\vae" "D:\AIModels\Qwen-Image-FP8\vae"
```

#### Step 3: Load Workflow
1. Open ComfyUI in browser (usually http://127.0.0.1:8188)
2. Click "Load" button
3. Select `workflow.json` from this directory
4. Upload your input image
5. Click "Queue Prompt"

---

### ✅ Option 2: Use ComfyUI API (Programmatic Control)

Use the new `run_comfyui_workflow.py` script to control ComfyUI from Python.

#### Requirements:
```bash
pip install websocket-client
```

#### Usage:
```python
python run_comfyui_workflow.py
```

This script will:
1. Upload your image to ComfyUI
2. Queue the workflow with your parameters
3. Wait for completion
4. Download and save the result

**Customize the workflow:**
```python
from run_comfyui_workflow import ComfyUIWorkflowRunner

runner = ComfyUIWorkflowRunner(server_address="127.0.0.1:8188")

result = runner.run_workflow(
    input_image_path="my_image.png",
    prompt_text="your custom prompt here",
    output_path="output.png",
    seed=12345,
    steps=8,
    cfg=1.0,
    width=768,
    height=768
)
```

---

### ⚠️ Option 3: Download Diffusers-Compatible Models

If you want to use pure Python without ComfyUI:

```python
from diffusers import QwenImageEditPlusPipeline
import torch

# This will download ~20GB of models from Hugging Face
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    torch_dtype=torch.bfloat16,
)
pipeline.to("cuda")

# Use like qwen-code-1.py
output = pipeline(
    image=input_image,
    prompt="your prompt",
    num_inference_steps=8,
    guidance_scale=1.0
)
```

**Pros:**
- No ComfyUI needed
- Direct Python control

**Cons:**
- Downloads additional ~20GB models
- Cannot use your existing GGUF models
- Slower than quantized GGUF models

---

### ❌ Option 4: Convert GGUF to Diffusers (NOT RECOMMENDED)

Converting GGUF to Diffusers format is:
- Complex and error-prone
- May lose quantization benefits
- Not officially supported
- Requires deep knowledge of model formats

**Not recommended unless you have specific requirements.**

---

## Comparison Table

| Method | Pros | Cons | Difficulty |
|--------|------|------|-----------|
| **ComfyUI GUI** | Easy, visual, uses your models | Requires ComfyUI running | ⭐ Easy |
| **ComfyUI API** | Programmatic, uses your models | Requires ComfyUI running | ⭐⭐ Medium |
| **Diffusers** | Pure Python, no ComfyUI | Downloads new models (~20GB) | ⭐⭐ Medium |
| **Convert GGUF** | Uses existing models | Complex, unsupported | ⭐⭐⭐⭐⭐ Hard |

---

## Recommended Workflow

### For Interactive Use:
1. Start ComfyUI
2. Load `workflow.json`
3. Edit parameters in GUI
4. Run workflow

### For Automation:
1. Start ComfyUI (keep it running)
2. Use `run_comfyui_workflow.py` to control it
3. Integrate into your scripts

### For Standalone Python:
1. Use `qwen-code-1.py` approach
2. Let it download Diffusers models
3. Accept the ~20GB download

---

## Quick Start Commands

### Start ComfyUI:
```bash
cd path/to/ComfyUI
python main.py
```

### Run with API:
```bash
# Make sure ComfyUI is running first!
python run_comfyui_workflow.py
```

### Run standalone (downloads models):
```bash
python qwen-code-1.py
```

---

## Troubleshooting

### "Cannot access gated repo" error
- This means trying to download from Hugging Face without authentication
- **Solution:** Use ComfyUI with your local GGUF models instead

### "Model not found" error
- Check model paths in ComfyUI
- Verify files exist in correct directories
- Check ComfyUI console for errors

### "Connection refused" error
- ComfyUI is not running
- **Solution:** Start ComfyUI first

### Slow performance
- GGUF models (Q4_0, Q3_K_S) are quantized for speed
- Make sure you're using GPU (CUDA)
- Check VRAM usage

---

## Summary

**For your use case with GGUF models at `D:\AIModels\Qwen-Image-FP8`:**

1. ✅ **Best option:** Use ComfyUI with `run_comfyui_workflow.py`
2. ✅ **Alternative:** Use ComfyUI GUI with `workflow.json`
3. ⚠️ **Fallback:** Download Diffusers models (ignore your GGUF files)

The `comfyui_standalone.py` script **cannot work** with GGUF models directly. Use one of the recommended options above.
