# ComfyUI Workflow Conversion

This directory contains converted code from your ComfyUI workflow for Qwen Image Edit.

## Files Created

### 1. `workflow.json`
**Direct ComfyUI import file** - You can load this directly in ComfyUI:
- Open ComfyUI
- Click "Load" button
- Select `workflow.json`
- The workflow will be reconstructed exactly as shown in your screenshot

### 2. `comfyui_workflow.py`
**ComfyUI API client** - Programmatically control ComfyUI server:
```python
python comfyui_workflow.py
```

**Requirements:**
- ComfyUI server running on `127.0.0.1:8188`
- Input image placed in ComfyUI's input folder

**Key features:**
- Queue workflows via API
- Customize parameters (seed, steps, CFG, prompt)
- Retrieve generated images

### 3. `comfyui_standalone.py`
**Standalone Python implementation** - Run without ComfyUI server:
```python
python comfyui_standalone.py
```

**Requirements:**
```bash
pip install torch diffusers transformers pillow
```

**Key features:**
- Complete workflow implementation
- No ComfyUI server needed
- Uses Hugging Face Diffusers library
- Class-based architecture for easy customization

## Workflow Overview

Based on your ComfyUI screenshot, the workflow consists of:

### Step 1: Load Models
- **UnetLoader (GGUF)**: `Qwen-Image-Edit-2509-flux_unet.safetensors`
- **CLIPLoader (GGUF)**: `Qwen-Image-Lightning-8step-v1-qwen2vl_clip.safetensors`
- **Load VAE**: `qwen_image_edit_vae.safetensors`

### Step 2: Upload Image
- Load input image for editing
- Default size: 768x768

### Step 3: Prompt Configuration
- Text prompt: "make green eyes WHEN GREEN ONLY on the head to face"
- Uses `TextEncodeCLIPSimple(M)` for encoding
- `ModelSamplingFlux` with:
  - `max_shift`: 1.15
  - `base_shift`: 0.5
  - `width`: 768
  - `height`: 768

### Step 4: Sampling (KSampler)
- **Seed**: 748260172255095
- **Steps**: 8
- **CFG Scale**: 1.0
- **Sampler**: euler
- **Scheduler**: normal
- **Denoise**: 1.0

### Step 5: Decode & Save
- VAE Decode latent to image
- Save with prefix: `qwen_yiyi_edit`

## Usage Examples

### Using the Standalone Version

```python
from comfyui_standalone import QwenImageEditWorkflow

# Initialize workflow
workflow = QwenImageEditWorkflow(
    device="cuda",
    dtype=torch.bfloat16
)

# Run complete workflow
output_path = workflow.run_workflow(
    input_image_path="input.png",
    prompt="make green eyes WHEN GREEN ONLY on the head to face",
    output_path="output.png",
    seed=748260172255095,
    steps=8,
    cfg_scale=1.0
)
```

### Using the ComfyUI API Client

```python
from comfyui_workflow import run_workflow

# Run workflow (requires ComfyUI server)
prompt_id = run_workflow(
    input_image_path="input.png",
    prompt_text="make green eyes WHEN GREEN ONLY on the head to face",
    seed=748260172255095,
    steps=8,
    cfg=1.0
)
```

### Customizing Parameters

Both scripts support customization:

```python
# Custom seed for different results
workflow.run_workflow(
    input_image_path="my_image.png",
    prompt="your custom prompt here",
    seed=12345,  # Different seed
    steps=16,    # More steps for better quality
    cfg_scale=2.0  # Higher CFG for stronger prompt adherence
)
```

## Key Parameters Explained

| Parameter | Description | Default |
|-----------|-------------|---------|
| `seed` | Random seed for reproducibility | 748260172255095 |
| `steps` | Number of denoising steps (higher = better quality, slower) | 8 |
| `cfg_scale` | Classifier-free guidance (higher = stronger prompt adherence) | 1.0 |
| `denoise` | Denoising strength (1.0 = full denoise, <1.0 = partial edit) | 1.0 |
| `width/height` | Output image dimensions | 768x768 |

## Model Files Required

For standalone version, you need:
- Qwen Image Edit models from Hugging Face
- Automatically downloaded on first run

For ComfyUI version, ensure these files are in your ComfyUI directories:
- `models/unet/Qwen-Image-Edit-2509-flux_unet.safetensors`
- `models/clip/Qwen-Image-Lightning-8step-v1-qwen2vl_clip.safetensors`
- `models/vae/qwen_image_edit_vae.safetensors`

## Troubleshooting

### "Model not found" error
- Ensure models are downloaded and in correct directories
- Check model paths in the code

### "CUDA out of memory" error
- Reduce image size
- Use `dtype=torch.float16` instead of `bfloat16`
- Use CPU: `device="cpu"`

### ComfyUI API connection error
- Ensure ComfyUI server is running
- Check `COMFYUI_SERVER` address in code
- Verify port 8188 is accessible

## Notes

- The standalone version uses the Diffusers library and may have slight differences from ComfyUI's implementation
- For exact ComfyUI behavior, use `workflow.json` or `comfyui_workflow.py`
- Seed value ensures reproducible results
- CFG scale of 1.0 is typical for Flux models

## Next Steps

1. **Test with your image**: Replace `input.png` with your actual image
2. **Adjust parameters**: Experiment with different seeds, steps, and CFG values
3. **Customize prompt**: Modify the text prompt for different edits
4. **Batch processing**: Extend the code to process multiple images

## References

- [ComfyUI Documentation](https://github.com/comfyanonymous/ComfyUI)
- [Qwen Image Edit Models](https://huggingface.co/Qwen)
- [Diffusers Library](https://huggingface.co/docs/diffusers)
