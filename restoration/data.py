from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from .corruptions import CLASSES, apply, config, read_rgb


def tensor(array):
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


class Pets(Dataset):
    def __init__(self, root, split="train", specialist=None, augment=True):
        self.root = Path(root)
        self.split = split
        self.specialist = specialist
        self.augment = augment
        if split == "train":
            self.rows = json.loads((self.root / "manifests/pets.json").read_text())["train"]
        else:
            self.rows = json.loads((self.root / f"manifests/pets_{split}.json").read_text())
            if specialist:
                self.rows = [r for r in self.rows if r["corruption"]["kind"] == specialist]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        forced_label = None
        if isinstance(index, tuple):
            index, forced_label = index
        row = self.rows[index]
        clean = read_rgb(self.root / row["image"])
        if self.split == "train":
            if self.augment and np.random.random() < 0.5:
                clean = clean[:, ::-1].copy()
            label = forced_label if forced_label is not None else int(np.random.randint(4))
            kind = self.specialist or CLASSES[label]
            cfg = config(kind, int(np.random.randint(0, 2**31)))
        else:
            cfg = row["corruption"]
        return tensor(apply(clean, cfg)), tensor(clean), CLASSES.index(cfg["kind"])


class BalancedBatches(Sampler):
    """Every classifier/mixture batch has equal class counts, generated at runtime."""
    def __init__(self, size, batch_size, seed=42):
        if batch_size % 4:
            raise ValueError("Balanced batch size must be divisible by four")
        self.size, self.batch_size, self.seed, self.epoch = size, batch_size, seed, 0

    def __len__(self):
        return self.size // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        indices = rng.permutation(self.size)
        labels = np.tile(np.arange(4), self.batch_size // 4)
        for start in range(0, len(self) * self.batch_size, self.batch_size):
            shuffled = rng.permutation(labels)
            yield [(int(i), int(label)) for i, label in zip(indices[start:start + self.batch_size], shuffled)]


class Faces(Dataset):
    def __init__(self, root, split="train"):
        self.root, self.split = Path(root), split
        self.rows = json.loads((self.root / "manifests/fs2k.json").read_text())[split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        photo, sketch = read_rgb(self.root / row["image"]), read_rgb(self.root / row["target"])
        if self.split == "train" and np.random.random() < 0.5:
            photo, sketch = photo[:, ::-1].copy(), sketch[:, ::-1].copy()
        return tensor(photo), tensor(sketch), row["style"]


def seed_worker(_):
    np.random.seed(torch.initial_seed() % 2**32)


def loaders(task, data, batch_size, workers=0):
    specialist = task if task in CLASSES[1:] else None
    dataset = Faces(data) if task == "gan" else Pets(data, specialist=specialist)
    validation = Faces(data, "validation") if task == "gan" else Pets(data, "validation", specialist=specialist)
    common = {"num_workers": workers, "worker_init_fn": seed_worker,
              "pin_memory": torch.cuda.is_available()}
    if task in ("classifier", "soft"):
        train = DataLoader(dataset, batch_sampler=BalancedBatches(len(dataset), batch_size), **common)
    else:
        train = DataLoader(dataset, batch_size=batch_size, shuffle=True, **common)
    valid = DataLoader(validation, batch_size=batch_size, shuffle=False, **common)
    return train, valid
