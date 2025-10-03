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
PROMPT    = "Restore the photo’s original colors so it appears naturally colorized with lifelike skin tones, balanced lighting, and vibrant background details."


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
        gpu_available = torch.cuda.is_available()
        dtype = torch.bfloat16 if gpu_available else torch.float32

        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        if gpu_available:
            try:
                pipe.to("cuda")
                logger.info("Pipeline moved to CUDA with dtype %s", dtype)
            except RuntimeError as cuda_err:
                if "out of memory" in str(cuda_err).lower():
                    logger.warning("GPU out of memory during initialization. Switching to CPU float32 execution.")
                    torch.cuda.empty_cache()
                    gpu_available = False
                else:
                    raise

        if not gpu_available:
            if dtype != torch.float32:
                dtype = torch.float32
                pipe = QwenImageEditPlusPipeline.from_pretrained(
                    MODEL_ID,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                )
            pipe.to("cpu")
            logger.warning("Running on CPU – generation will be slow.")

        execution_device = torch.device("cuda") if gpu_available else torch.device("cpu")
        logger.info("Pipeline ready. Execution device: %s, dtype: %s", execution_device, dtype)

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

            generator = torch.Generator(device=execution_device).manual_seed(0)
            try:
                with torch.inference_mode():
                    result = pipe(
                        image=[img],
                        prompt=PROMPT,
                        true_cfg_scale=4.0,
                        num_inference_steps=40,
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
