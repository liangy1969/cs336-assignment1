import numpy as np
import torch
import json
from pathlib import Path
from einops import rearrange


DATA_FOLDER = str(Path(__file__).resolve().parent.parent / "data")
TRAIN_FILE_NAME = "TinyStoriesV2-GPT4-train-tokens.json"


class DataLoader:
    """
    Load tokenized training data from json file
    """

    def __init__(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        self._data_np = np.array(self._data, dtype=np.int64)

    @property
    def data(self) -> np.ndarray:
        return self._data_np

    @property
    def data_size(self) -> int:
        return len(self._data_np)

    def get_batch(
        self,
        batch_idx: int,
        batch_size: int,
        context_length: int,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_seqs = []
        batch_targets = []
        for j in range(batch_size):
            start_idx = (batch_idx * batch_size + j) % (self.data_size - context_length)
            batch_seqs.append(
                torch.tensor(
                    self._data_np[start_idx : start_idx + context_length],
                    device=device,
                    dtype=torch.long,
                )
            )
            batch_targets.append(
                torch.tensor(
                    self._data_np[start_idx + 1 : start_idx + context_length + 1],
                    device=device,
                    dtype=torch.long,
                )
            )
        return rearrange(batch_seqs, "batch seq -> batch seq"), rearrange(
            batch_targets, "batch seq -> batch seq"
        )
