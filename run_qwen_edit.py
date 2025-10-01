import logging
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# --- Inputs ---
MODEL_ID = r"D:\\AIModels\Qwen-Image-Edit-2509"  # local download from `download.py`
# MODEL_ID = "Qwen/Qwen-Image-Edit-2509"  # or pull on-the-fly from HF
IMAGES_DIR = Path("images")
OUTPUT_DIR = Path("output")
LOGS_DIR = Path("logs")
PROMPT    = "colorise the image."


def _configure_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"run_{timestamp}.log"

    logger = logging.getLogger("qwen_image_edit")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Reset handlers to avoid duplicate logs when re-running inside the same interpreter.
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Logging to %s", log_path.resolve())
    return logger

def main() -> None:
    logger = _configure_logging()

    try:
        logger.info("Preparing pipeline components...")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        gpu_available = torch.cuda.is_available()
        execution_device = torch.device("cuda") if gpu_available else torch.device("cpu")

        if gpu_available:
            try:
                pipe.enable_model_cpu_offload()
                if hasattr(pipe, "enable_attention_slicing"):
                    pipe.enable_attention_slicing()
                if hasattr(pipe, "enable_sequential_cpu_offload"):
                    pipe.enable_sequential_cpu_offload()
                logger.info("Enabled model CPU offload for constrained VRAM GPUs.")
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Model CPU offload failed (%s). Trying full GPU load.", exc)
                try:
                    pipe.to("cuda")
                except RuntimeError as cuda_err:
                    if "out of memory" in str(cuda_err).lower():
                        logger.error("GPU out of memory during initialization. Falling back to CPU-only mode.")
                        torch.cuda.empty_cache()
                        pipe = QwenImageEditPlusPipeline.from_pretrained(
                            MODEL_ID,
                            torch_dtype=torch.float32,
                            low_cpu_mem_usage=True,
                            use_safetensors=True,
                        )
                        execution_device = torch.device("cpu")
                        gpu_available = False
                    else:
                        raise

        if not gpu_available:
            logger.warning("Running on CPU – generation will be slow.")

        execution_device = getattr(pipe, "_execution_device", execution_device)
        logger.info("Pipeline loaded. Using device: %s", execution_device)

        if not IMAGES_DIR.exists():
            raise FileNotFoundError(f"Images directory not found: {IMAGES_DIR.resolve()}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("Processing images from %s", IMAGES_DIR.resolve())
        logger.info("Saving outputs to %s", OUTPUT_DIR.resolve())

        image_paths = sorted(p for p in IMAGES_DIR.glob("*") if p.is_file())
        if not image_paths:
            logger.warning("No files found in %s. Nothing to process.", IMAGES_DIR.resolve())
            return

        for idx, image_path in enumerate(image_paths, start=1):
            try:
                img = Image.open(image_path).convert("RGB")
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Skipping %s: failed to open (%s)", image_path.name, exc)
                continue

            logger.info("[%d/%d] Processing %s", idx, len(image_paths), image_path.name)

            img = img.resize((512, 512), Image.LANCZOS)

            generator = torch.Generator(device=execution_device).manual_seed(0)
            try:
                result = pipe(
                    image=[img],
                    prompt=PROMPT,
                    true_cfg_scale=4.0,
                    num_inference_steps=24,
                    guidance_scale=1.0,
                    negative_prompt=" ",
                    generator=generator,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Failed to process %s: %s", image_path.name, exc)
                continue

            output_name = f"{image_path.stem}_edit{image_path.suffix}"
            output_path = OUTPUT_DIR / output_name
            result.images[0].save(output_path)
            logger.info("Saved: %s", output_path.resolve())

        logger.info("Processing complete.")

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Run failed due to unexpected error: %s", exc)
        raise


if __name__ == "__main__":
    main()
