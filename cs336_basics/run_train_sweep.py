import json
import logging
import os
from datetime import datetime

from cs336_basics.train import LOGGING_FOLDER, train


TOTAL_TOKEN_BUDGET = 320_000
BATCH_SIZES = [16, 32, 64]
MAX_LRS = [1e-3, 1e-4]
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LR = 1e-3
TRACKER_PATH = os.path.join(LOGGING_FOLDER, "train_sweep_tracker.json")


def format_lr(lr: float) -> str:
    return f"{lr:g}".replace(".", "p").replace("-", "m")


def load_tracker(path: str) -> dict:
    if not os.path.exists(path):
        return {"completed_jobs": []}
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if "completed_jobs" not in data or not isinstance(data["completed_jobs"], list):
        data["completed_jobs"] = []
    return data


def save_tracker(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def run_job(
    logger: logging.Logger,
    completed_jobs: set[str],
    tracker: dict,
    job_name: str,
    batch_size: int,
    max_lr: float,
) -> None:
    n_train_step = max(1, round(TOTAL_TOKEN_BUDGET / batch_size))
    min_lr = max_lr / 10
    warmup_iters = max(10, min(1000, n_train_step // 10))

    if job_name in completed_jobs:
        logger.info("skipping completed job %s", job_name)
        return

    logger.info(
        "starting job=%s, batch_size=%s, max_lr=%s, min_lr=%s, n_train_step=%s, warmup_iters=%s",
        job_name,
        batch_size,
        max_lr,
        min_lr,
        n_train_step,
        warmup_iters,
    )

    train(
        train_job_name=job_name,
        n_train_step=n_train_step,
        batch_size=batch_size,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_iters=warmup_iters,
        print_every=100,
        save_every=5000,
    )

    completed_jobs.add(job_name)
    tracker["completed_jobs"] = sorted(completed_jobs)
    tracker["last_updated"] = datetime.utcnow().isoformat() + "Z"
    save_tracker(TRACKER_PATH, tracker)
    logger.info("completed job %s", job_name)


def main() -> None:
    os.makedirs(LOGGING_FOLDER, exist_ok=True)

    logger = logging.getLogger("train_sweep")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(
            os.path.join(LOGGING_FOLDER, "train_sweep.log"), mode="a"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    tracker = load_tracker(TRACKER_PATH)
    completed_jobs = set(tracker["completed_jobs"])

    for max_lr in MAX_LRS:
        job_name = f"train_bs{DEFAULT_BATCH_SIZE}_lr{format_lr(max_lr)}"
        run_job(
            logger=logger,
            completed_jobs=completed_jobs,
            tracker=tracker,
            job_name=job_name,
            batch_size=DEFAULT_BATCH_SIZE,
            max_lr=max_lr,
        )

    for batch_size in BATCH_SIZES:
        job_name = f"train_bs{batch_size}_lr{format_lr(DEFAULT_MAX_LR)}"
        run_job(
            logger=logger,
            completed_jobs=completed_jobs,
            tracker=tracker,
            job_name=job_name,
            batch_size=batch_size,
            max_lr=DEFAULT_MAX_LR,
        )

    logger.info("sweep finished")


if __name__ == "__main__":
    main()
