from typing import IO

import torch
from pathlib import Path
import os


CHECK_POINT_FOLDER = str(Path(__file__).resolve().parent.parent / "checkpoint")


class Checkpointer:

    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.name = name
        self.model = model
        self.optimizer = optimizer

    def get_checkpoint_path(self, iteration: int) -> str:
        return os.path.join(
            CHECK_POINT_FOLDER, self.name, f"{iteration}", "checkpoint.pth"
        )

    def load_checkpoint(
        self,
        iteration: int,
    ) -> None:
        checkpoint_path = self.get_checkpoint_path(iteration)
        if not os.path.exists(checkpoint_path):
            return None
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        assert checkpoint["iteration"] == iteration

    def get_latest_checkpoint_iteration(self) -> int | None:
        name_folder = os.path.join(CHECK_POINT_FOLDER, self.name)
        if not os.path.exists(name_folder):
            return None
        iteration_folders = [
            f
            for f in os.listdir(name_folder)
            if os.path.isdir(os.path.join(name_folder, f))
        ]
        if not iteration_folders:
            return None
        latest_iteration = max(int(f) for f in iteration_folders)
        return latest_iteration

    def load_latest_checkpoint(self) -> int | None:
        latest_iteration = self.get_latest_checkpoint_iteration()
        if latest_iteration is not None:
            self.load_checkpoint(latest_iteration)
        return latest_iteration

    def save_checkpoint(
        self,
        iteration: int,
    ):
        checkpoint_path = self.get_checkpoint_path(iteration)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "iteration": iteration,
            },
            checkpoint_path,
        )
