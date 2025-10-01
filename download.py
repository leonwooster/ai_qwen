import argparse
import importlib.util
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
from urllib3.exceptions import ProtocolError, ReadTimeoutError


REPO_ID = "Qwen/Qwen-Image-Edit-2509"
TARGET_DIR = Path(r"D:\AIModels\Qwen-Image-Edit-2509")
LOGS_DIR = Path("logs")
PAUSE_FILE = Path("pause_download.flag")
STATE_FILE = TARGET_DIR / "download_state.json"

MAX_ATTEMPTS = 5
BASE_BACKOFF_SECONDS = 30
SUPPORTED_EXCEPTIONS = (
    ChunkedEncodingError,
    ConnectionError,
    Timeout,
    ProtocolError,
    ReadTimeoutError,
    HfHubHTTPError,
)


def _configure_logging(verbose: bool) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"download_{timestamp}.log"

    logger = logging.getLogger("qwen_download")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # reset handlers for repeated runs
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


def _enable_hf_transfer_if_available(logger: logging.Logger) -> None:
    if importlib.util.find_spec("hf_transfer") and os.getenv("HF_HUB_ENABLE_HF_TRANSFER") != "1":
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        logger.info("Enabled hf_transfer for improved throughput.")
    elif importlib.util.find_spec("hf_xet") and os.getenv("HF_HUB_ENABLE_HF_TRANSFER") != "1":
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        logger.info("Enabled hf_xet-backed transfer for improved throughput.")
    else:
        logger.info(
            "hf_transfer/hf_xet not installed. Consider `pip install huggingface_hub[hf_xet]` for faster downloads."
        )


def _load_repo_manifest(logger: logging.Logger) -> List[Tuple[str, int]]:
    api = HfApi()
    logger.info("Fetching repository file list for %s", REPO_ID)
    info = api.model_info(REPO_ID, expand=["siblings"])

    entries: List[Tuple[str, int]] = []
    for sibling in info.siblings:
        size = sibling.size or 0
        if sibling.rfilename.endswith("/"):
            continue
        entries.append((sibling.rfilename, size))
    logger.info("Found %d files in repository", len(entries))
    return entries


def _human_size(num_bytes: int) -> str:
    if num_bytes == 0:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB"]
    power = min(int((num_bytes).bit_length() / 10), len(units) - 1)
    value = num_bytes / (1024**power)
    return f"{value:.2f} {units[power]}"


def _already_downloaded(target: Path, expected_size: int) -> bool:
    if not target.exists() or expected_size == 0:
        return False
    try:
        actual_size = target.stat().st_size
    except OSError:
        return False
    return actual_size == expected_size


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        with STATE_FILE.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return {}


def _should_pause(logger: logging.Logger) -> bool:
    if PAUSE_FILE.exists():
        logger.info("Pause flag detected at %s", PAUSE_FILE.resolve())
        return True
    return False


def _download_file(
    repo_file: str,
    expected_size: int,
    logger: logging.Logger,
) -> None:
    destination = TARGET_DIR / repo_file
    destination.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            logger.info(
                "Downloading %s (attempt %d/%d, size %s)",
                repo_file,
                attempt,
                MAX_ATTEMPTS,
                _human_size(expected_size),
            )
            hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_file,
                local_dir=str(TARGET_DIR),
                local_dir_use_symlinks=False,
                resume_download=True,
                force_download=False,
            )
            return
        except SUPPORTED_EXCEPTIONS as err:
            if attempt == MAX_ATTEMPTS:
                logger.error("Failed to download %s after %d attempts: %s", repo_file, attempt, err)
                raise

            wait_seconds = BASE_BACKOFF_SECONDS * attempt
            logger.warning(
                "Attempt %d for %s failed (%s). Retrying in %d seconds...",
                attempt,
                repo_file,
                err,
                wait_seconds,
            )
            time.sleep(wait_seconds)


def _download_manifest(only_missing: bool, logger: logging.Logger) -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    entries = _load_repo_manifest(logger)

    total_bytes = sum(size for _, size in entries)
    downloaded_bytes = 0
    completed_files = 0

    for repo_file, size in entries:
        local_path = TARGET_DIR / repo_file
        if _already_downloaded(local_path, size):
            downloaded_bytes += size
            completed_files += 1
            logger.info("Already have %s (%s)", repo_file, _human_size(size))
            continue

        if only_missing and state.get("completed_files", {}).get(repo_file) == size:
            logger.info("Skipping %s – state file marks it complete", repo_file)
            downloaded_bytes += size
            completed_files += 1
            continue

        if _should_pause(logger):
            logger.info("Pause requested before downloading %s. Exiting.")
            break

        try:
            _download_file(repo_file, size, logger)
        except KeyboardInterrupt:
            logger.warning("Download interrupted by user during %s", repo_file)
            break
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Aborting due to unrecoverable error on %s: %s", repo_file, err)
            raise

        actual_size = local_path.stat().st_size if local_path.exists() else 0
        downloaded_bytes += actual_size
        completed_files += 1 if actual_size == size else 0

        state.setdefault("completed_files", {})[repo_file] = actual_size
        state["downloaded_bytes"] = downloaded_bytes
        state["timestamp"] = datetime.now().isoformat()
        _save_state(state)

        logger.info(
            "Finished %s (%s). Progress: %s / %s",
            repo_file,
            _human_size(actual_size),
            _human_size(downloaded_bytes),
            _human_size(total_bytes),
        )

        if _should_pause(logger):
            logger.info("Pause requested after downloading %s. Exiting.")
            break

    else:
        logger.info("All files downloaded successfully to %s", TARGET_DIR.resolve())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable downloader for Hugging Face Qwen model")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip files already marked complete in the state file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logger = _configure_logging(verbose=args.verbose)

    def _signal_handler(signum, frame):  # pylint: disable=unused-argument
        logger.warning("Received signal %s; creating pause flag %s", signum, PAUSE_FILE)
        PAUSE_FILE.touch()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    _enable_hf_transfer_if_available(logger)

    try:
        _download_manifest(only_missing=args.only_missing, logger=logger)
    except KeyboardInterrupt:
        logger.warning("Download interrupted by user. You can resume later.")
        sys.exit(1)


if __name__ == "__main__":
    main()