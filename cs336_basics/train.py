import json
import numpy as np
import torch
import os
import logging
from einops import rearrange
from cs336_basics import data
from cs336_basics.transformer import TransformerLM
from torch.optim import AdamW
from typing import Iterable
from cs336_basics.data import DataLoader
import math
from cs336_basics.checkpointer import Checkpointer
from pathlib import Path


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(torch.stack([g.norm() ** 2 for g in grads]).sum())
    if total_norm > max_l2_norm:
        scale = max_l2_norm / total_norm
        for g in grads:
            g.mul_(scale)


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
            1 + math.cos(math.pi * progress)
        )
    else:
        return min_learning_rate


LOGGING_FOLDER = str(Path(__file__).resolve().parent.parent / "logging")


def train(
    train_job_name: str = "test_train_run",
    n_train_step: int = 100,
    batch_size: int = 30,
    max_lr: float = 0.001,
    min_lr: float = 0.0001,
    warmup_iters: int = 10,
    print_every: int = 10,
    save_every: int = 10,
):
    os.makedirs(LOGGING_FOLDER, exist_ok=True)
    logger = logging.getLogger(train_job_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(
            os.path.join(LOGGING_FOLDER, f"{train_job_name}.log"), mode="a"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    train_file_path = os.path.join(data.DATA_FOLDER, data.TRAIN_FILE_NAME)
    context_length = 256
    vocab_size = 10000
    d_model = 512
    num_layers = 4
    num_heads = 16
    d_ff = 1344
    rope_theta = 10000
    cosine_cycle_iters = n_train_step
    grad_l2_norm_clip = 1.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("using device: %s", device)

    train_data_loader = DataLoader(train_file_path)

    # initialize model & optimizer
    transformer_lm = TransformerLM(
        vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta
    ).to(device)
    optim = AdamW(transformer_lm.parameters(), lr=max_lr)

    ckpt = Checkpointer(train_job_name, transformer_lm, optim)
    latest_ckpt_iter = ckpt.load_latest_checkpoint()
    if latest_ckpt_iter is not None:
        logger.info("loaded checkpoint from iteration %s", latest_ckpt_iter)

    start_iteration = latest_ckpt_iter + 1 if latest_ckpt_iter is not None else 0

    # training loop
    for i in range(start_iteration, n_train_step):
        current_lr = get_lr_cosine_schedule(
            i, max_lr, min_lr, warmup_iters, cosine_cycle_iters
        )
        for param_group in optim.param_groups:
            param_group["lr"] = current_lr
        batch_seqs, batch_targets = train_data_loader.get_batch(
            i, batch_size, context_length, device
        )
        pred_logits = transformer_lm(batch_seqs)
        loss = torch.nn.functional.cross_entropy(
            rearrange(pred_logits, "b s v -> (b s) v"),
            rearrange(batch_targets, "b s -> (b s)"),
        )
        if i % print_every == 0:
            logger.info("step %s: loss = %.4f, lr = %.6f", i, loss.item(), current_lr)
        optim.zero_grad()
        loss.backward()
        gradient_clipping(
            [p for group in optim.param_groups for p in group["params"]],
            max_l2_norm=grad_l2_norm_clip,
        )
        optim.step()
        if i % save_every == 0:
            ckpt.save_checkpoint(i)


if __name__ == "__main__":
    train()
