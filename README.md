# Qwen Image Edit Demo

## Overview
This project demonstrates running the `QwenImageEditPlusPipeline` from Hugging Face `diffusers` on a local Windows workstation. It includes automation for downloading the large 2025-09 snapshot of the Qwen Image Edit model, generating edited images for every asset in `images/`, and capturing detailed logs for each run.

Because the official pipeline requires >50 GB of model weights, the repository also documents strategies for pausing and resuming downloads, dealing with limited VRAM, and falling back to CPU execution when an 8 GB GPU (e.g., RTX 4060) cannot host the full model in memory.

## Repository Layout
- `run_qwen_edit.py` – Entry point that loads the pipeline, logs activity to `logs/`, and iterates over `images/`.
- `download.py` – Resumable/Pausable downloader for `Qwen/Qwen-Image-Edit-2509`.
- `setup_qwen_gpu.bat` – Convenience script to activate the virtual environment, install CUDA-enabled PyTorch, refresh dependencies, download the model, and run the pipeline.
- `images/` – Input images to edit. Place your own here.
- `output/` – Generated results (one `_edit` image per input).
- `logs/` – Timestamped log files for both downloads and inference runs.
- `pause_download.flag` – Touch this file to pause the downloader; delete it to resume.

## Prerequisites
- Windows 10/11 machine.
- Python 3.11 virtual environment (example path: `C:\PythonEnv\p311_qwen`).
- Adequate disk space (~55 GB) for the full model snapshot.
- Optional but recommended: NVIDIA GPU with the matching CUDA driver. The included scripts will still run on CPU if CUDA is not available or VRAM is insufficient.

## Setup Steps
1. **Activate the environment and install dependencies**
   ```powershell
   setup_qwen_gpu.bat
   ```
   The batch script upgrades `pip`, installs CUDA-enabled PyTorch (cu121 wheel), installs the latest bleeding-edge `diffusers`, and ensures support libraries (`transformers`, `huggingface_hub`, `xformers`, etc.) are up to date.

2. **Download the model snapshot**
   If you prefer manual control (instead of the batch script’s call), run:
   ```powershell
   python download.py --only-missing
   ```
   - Progress is logged to `logs/download_<timestamp>.log`.
   - Downloads resume automatically thanks to `download_state.json` stored alongside the model (`D:\AIModels\Qwen-Image-Edit-2509\download_state.json`).
   - To pause mid-way, create the file `pause_download.flag` in the project root or press `Ctrl+C`. Remove the file and rerun `python download.py --only-missing` to resume.

3. **Verify CUDA** (optional)
   ```powershell
   python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
   ```
   If this prints `True` with a CUDA version (e.g., `12.1`), the environment has GPU support.

## Running Image Edits
```powershell
python run_qwen_edit.py
```
- Logs stream to console and `logs/run_<timestamp>.log`.
- Each processed image saves to `output/<original_name>_edit.<ext>`.
- The script automatically falls back to CPU execution if GPU initialization fails. Look for messages like `Running on CPU – generation will be slow.` in the log.

### GPU Memory Considerations
`Qwen-Image-Edit-2509` is extremely large; even with memory optimizations (attention slicing, sequential offload), 8 GB GPUs may still encounter OOM errors. If that happens:
- Ensure no other processes are using VRAM.
- Reduce the input resolution in `run_qwen_edit.py` (e.g., change the `img.resize((512, 512), ...)` line to `(384, 384)` or smaller).
- Accept CPU-only execution (slow but reliable).
- Alternatively, switch to a lighter model (see below).

### Alternative Diffusers Pipelines
If your hardware cannot handle Qwen’s model, consider editing the script to use smaller pipelines:
- `runwayml/stable-diffusion-v1-5`
- `stabilityai/stable-diffusion-2-inpainting`
- `stabilityai/sdxl-turbo`

Replace the import/pipeline class in `run_qwen_edit.py` accordingly and adjust prompts/inference parameters as needed.

## Logs & Troubleshooting
- **Download issues**: See `logs/download_<timestamp>.log`. The downloader writes detailed progress, retries, and pause/resume events.
- **Inference issues**: Check the latest `logs/run_<timestamp>.log` for stack traces, device selection, and saved outputs.
- **CUDA OOM**: Logs will capture when the pipeline falls back to CPU. Restart the Python process after an OOM to fully release memory.

## Known Limitations
- Qwen’s pipeline lacks certain helper methods (`enable_vae_slicing`) present in other diffusers pipelines; the code guards against these missing APIs.
- CPU execution is significantly slower (minutes per image). Plan accordingly if GPU memory is insufficient.
- The repository does not bundle the model weights; you must download them separately due to licensing constraints.

## Updating the Project
When dependencies or model releases change:
- Re-run `setup_qwen_gpu.bat` to upgrade packages.
- Clear or delete `download_state.json` if you want to force re-download of all files.
- Review the Hugging Face release notes for `Qwen/Qwen-Image-Edit-2509` to adjust prompts or configuration as new features become available.

## License
Refer to Hugging Face’s model card and license for usage restrictions of `Qwen/Qwen-Image-Edit-2509`. This repository only provides automation scripts and does not redistribute model weights.
