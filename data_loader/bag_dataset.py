from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class BagSample:
    embeddings: torch.Tensor
    sample_id: str
    targets: torch.Tensor


def list_bag_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*.h5"))


def parse_sample_id(path: Path) -> str:
    name = path.stem.strip()
    r1_suffix = "-R1"
    if r1_suffix in name:
        name = name[: name.index(r1_suffix)]
    return name.rstrip("-")


def load_metadata(
    csv_path: Path, sample_id_column: str, targets: Iterable[str]
) -> dict[str, np.ndarray]:
    frame = pd.read_csv(csv_path)
    if sample_id_column not in frame.columns:
        raise ValueError(
            f"Missing sample_id column '{sample_id_column}' in {csv_path}"
        )

    targets = list(targets)
    missing_targets = [col for col in targets if col not in frame.columns]
    if missing_targets:
        raise ValueError(
            f"Missing target columns {missing_targets} in {csv_path}"
        )

    frame = frame.dropna(subset=targets)
    sample_ids = frame[sample_id_column].astype(str)
    target_values = frame[targets].astype(float).to_numpy()
    return dict(zip(sample_ids, target_values, strict=True))


def filter_bag_files(
    bag_files: Iterable[Path], targets_map: dict[str, np.ndarray]
) -> list[Path]:
    filtered = []
    for path in bag_files:
        sample_id = parse_sample_id(path)
        if sample_id in targets_map:
            filtered.append(path)
    return filtered


def load_embeddings(path: Path) -> tuple[np.ndarray, str]:
    with h5py.File(path, "r") as handle:
        embeddings = handle["embeddings"][()]
        sample_id = handle.attrs.get("sample_id")

    expected_sample_id = parse_sample_id(path)
    if sample_id is None:
        raise ValueError(f"Missing sample_id attribute in {path}")

    sample_id = str(sample_id).strip()

    if sample_id.endswith("-"):
        sample_id = sample_id.rstrip("-")
    if sample_id != expected_sample_id:
        raise ValueError(
            f"sample_id attribute mismatch in {path}: expected {expected_sample_id}, got {sample_id}"
        )

    return embeddings, sample_id


class BagDataset(Dataset[BagSample]):
    def __init__(
        self,
        bag_files: Iterable[Path],
        targets_map: dict[str, np.ndarray],
    ) -> None:
        self.bag_files = list(bag_files)
        self.targets_map = targets_map

    def __len__(self) -> int:
        return len(self.bag_files)

    def __getitem__(self, index: int) -> BagSample:
        path = self.bag_files[index]
        embeddings, sample_id = load_embeddings(path)
        targets = self.targets_map[sample_id]
        return BagSample(
            embeddings=torch.from_numpy(embeddings).float(),
            sample_id=sample_id,
            targets=torch.tensor(targets, dtype=torch.float32),
        )


def collate_bags(batch: list[BagSample]) -> tuple[list[torch.Tensor], list[str], torch.Tensor]:
    embeddings = [item.embeddings for item in batch]
    sample_ids = [item.sample_id for item in batch]
    targets = torch.stack([item.targets for item in batch], dim=0)
    return embeddings, sample_ids, targets


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloader(
    dataset: Dataset[BagSample],
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_bags,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
    )
