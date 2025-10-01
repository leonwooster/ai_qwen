import importlib.util
import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
from urllib3.exceptions import ProtocolError, ReadTimeoutError


REPO_ID = "Qwen/Qwen-Image-Edit-2509"
TARGET_DIR = Path(r"D:\AIModels")
MAX_ATTEMPTS = 5
BASE_BACKOFF_SECONDS = 30
MAX_WORKERS = 4


def _enable_hf_transfer_if_available() -> None:
    """Enable faster transfer backend when `hf_transfer` is installed."""
    if importlib.util.find_spec("hf_transfer") and os.getenv("HF_HUB_ENABLE_HF_TRANSFER") != "1":
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("Enabled hf_transfer for improved throughput.")
    elif importlib.util.find_spec("hf_xet") and os.getenv("HF_HUB_ENABLE_HF_TRANSFER") != "1":
        # hf_xet automatically patches huggingface_hub when imported; let user know.
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("Enabled hf_xet-backed transfer for improved throughput.")
    else:
        print(
            "hf_transfer/hf_xet not installed. Consider `pip install huggingface_hub[hf_xet]` "
            "for faster and more reliable large-file downloads."
        )


def download_with_retries() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"Starting download attempt {attempt}/{MAX_ATTEMPTS}...")
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=str(TARGET_DIR),
                resume_download=True,
                max_workers=MAX_WORKERS,
                local_dir_use_symlinks=False,
            )
            print(f"Download completed successfully at {TARGET_DIR}")
            return
        except (
            ChunkedEncodingError,
            ConnectionError,
            Timeout,
            ProtocolError,
            ReadTimeoutError,
        ) as err:
            if attempt == MAX_ATTEMPTS:
                raise

            wait_seconds = BASE_BACKOFF_SECONDS * attempt
            print(
                f"Download interrupted ({err.__class__.__name__}): {err}. "
                f"Retrying in {wait_seconds} seconds..."
            )
            time.sleep(wait_seconds)


def main() -> None:
    _enable_hf_transfer_if_available()
    download_with_retries()


if __name__ == "__main__":
    main()