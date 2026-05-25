from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from data_loader.bag_dataset import (
    BagDataset,
    build_dataloader,
    filter_bag_files,
    list_bag_files,
    load_embeddings,
    load_metadata,
    parse_sample_id,
)
from models.mean_pool_mlp import MeanPoolMLP


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path
    metadata_csv: Path
    results_dir: Path


@dataclass(frozen=True)
class DataConfig:
    sample_id_column: str
    targets: list[str]
    stats_output: str | None


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    batch_size: int
    num_workers: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    precision: int


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int
    dropout: float


@dataclass(frozen=True)
class TrainConfig:
    paths: PathsConfig
    data: DataConfig
    split: SplitConfig
    training: TrainingConfig
    model: ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training phase for environmental regression baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the training YAML config file.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> TrainConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    return TrainConfig(
        paths=PathsConfig(
            data_dir=Path(raw["paths"]["data_dir"]).expanduser(),
            metadata_csv=Path(raw["paths"]["metadata_csv"]).expanduser(),
            results_dir=Path(raw["paths"]["results_dir"]).expanduser(),
        ),
        data=DataConfig(
            sample_id_column=str(raw["data"]["sample_id_column"]),
            targets=list(raw["data"]["targets"]),
            stats_output=(
                str(raw["data"]["stats_output"])
                if raw["data"].get("stats_output") not in (None, "")
                else None
            ),
        ),
        split=SplitConfig(
            seed=int(raw["split"]["seed"]),
            train_ratio=float(raw["split"]["train_ratio"]),
            val_ratio=float(raw["split"]["val_ratio"]),
            test_ratio=float(raw["split"]["test_ratio"]),
        ),
        training=TrainingConfig(
            seed=int(raw["training"]["seed"]),
            batch_size=int(raw["training"]["batch_size"]),
            num_workers=int(raw["training"]["num_workers"]),
            max_epochs=int(raw["training"]["max_epochs"]),
            learning_rate=float(raw["training"]["learning_rate"]),
            weight_decay=float(raw["training"]["weight_decay"]),
            precision=int(raw["training"]["precision"]),
        ),
        model=ModelConfig(
            hidden_dim=int(raw["model"]["hidden_dim"]),
            dropout=float(raw["model"]["dropout"]),
        ),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_sample_ids_to_sets(
    sample_ids: list[str], seed: int, train_ratio: float, val_ratio: float
) -> tuple[set[str], set[str], set[str]]:
    rng = random.Random(seed)
    shuffled = sample_ids[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_ids = set(shuffled[:train_end])
    val_ids = set(shuffled[train_end:val_end])
    test_ids = set(shuffled[val_end:])
    return train_ids, val_ids, test_ids


def assign_split(sample_id: str, splits: dict[str, set[str]]) -> str:
    for split_name, ids in splits.items():
        if sample_id in ids:
            return split_name
    return "unknown"


def compute_stats_rows(
    bag_files: Iterable[Path], splits: dict[str, set[str]]
) -> Iterable[dict[str, object]]:
    for path in bag_files:
        embeddings, sample_id = load_embeddings(path)
        mean = embeddings.mean(axis=0)
        var = embeddings.var(axis=0)
        split = assign_split(sample_id, splits)
        row = {
            "sample_id": sample_id,
            "bag_name": path.name,
            "split": split,
        }
        row.update({f"mean_{i}": float(value) for i, value in enumerate(mean)})
        row.update({f"var_{i}": float(value) for i, value in enumerate(var)})
        yield row


def save_stats_csv(output_path: Path, rows: Iterable[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class LightningRegressor(pl.LightningModule):
    def __init__(
        self,
        model: MeanPoolMLP,
        learning_rate: float,
        weight_decay: float,
        target_names: list[str],
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.target_names = target_names
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        return self.model(embeddings)

    def training_step(self, batch, _batch_idx: int):
        embeddings, _sample_ids, targets = batch
        preds = self(embeddings)
        loss = self.loss_fn(preds, targets)
        self.log("train_mse", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _batch_idx: int):
        embeddings, _sample_ids, targets = batch
        preds = self(embeddings)
        loss = self.loss_fn(preds, targets)
        self.log("val_mse", loss, prog_bar=True)
        self._log_per_target("val", preds, targets)

    def test_step(self, batch, _batch_idx: int):
        embeddings, _sample_ids, targets = batch
        preds = self(embeddings)
        loss = self.loss_fn(preds, targets)
        self.log("test_mse", loss)
        self._log_per_target("test", preds, targets)

    def _log_per_target(
        self, prefix: str, preds: torch.Tensor, targets: torch.Tensor
    ) -> None:
        per_target_mse = (preds - targets).pow(2).mean(dim=0)
        for idx, name in enumerate(self.target_names):
            self.log(f"{prefix}_mse_{name}", per_target_mse[idx])

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


def save_predictions(
    model: LightningRegressor,
    dataloader: torch.utils.data.DataLoader,
    output_path: Path,
    device: torch.device,
    target_names: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.to(device)
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for embeddings, sample_ids, targets in dataloader:
            embeddings = [tensor.to(device) for tensor in embeddings]
            preds = model(embeddings).cpu().numpy()
            targets_np = targets.cpu().numpy()
            for idx, sample_id in enumerate(sample_ids):
                row = {"sample_id": sample_id}
                for col_idx, name in enumerate(target_names):
                    row[f"pred_{name}"] = float(preds[idx, col_idx])
                    row[f"true_{name}"] = float(targets_np[idx, col_idx])
                rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else ["sample_id"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.training.seed)

    bag_files = list_bag_files(config.paths.data_dir)
    if not bag_files:
        raise SystemExit(f"No .h5 files found in {config.paths.data_dir}")

    targets_map = load_metadata(
        config.paths.metadata_csv,
        config.data.sample_id_column,
        config.data.targets,
    )
    # Filter bag files to those with metadata targets, and warn if any are dropped
    bag_files = filter_bag_files(bag_files, targets_map)
    if not bag_files:
        raise SystemExit("No bag files matched metadata targets.")
    print(f"Found {len(bag_files)} bag files with metadata targets.")

    sample_ids = sorted({parse_sample_id(path) for path in bag_files})
    train_ids, val_ids, test_ids = split_sample_ids_to_sets(
        sample_ids,
        config.split.seed,
        config.split.train_ratio,
        config.split.val_ratio,
    )
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}

    if config.data.stats_output:
        stats_rows = compute_stats_rows(bag_files, splits)
        stats_path = config.paths.results_dir / config.data.stats_output
        save_stats_csv(stats_path, stats_rows)

    train_files = [path for path in bag_files if parse_sample_id(path) in train_ids]
    val_files = [path for path in bag_files if parse_sample_id(path) in val_ids]
    test_files = [path for path in bag_files if parse_sample_id(path) in test_ids]

    train_dataset = BagDataset(train_files, targets_map)
    val_dataset = BagDataset(val_files, targets_map)
    test_dataset = BagDataset(test_files, targets_map)

    train_loader = build_dataloader(
        train_dataset,
        config.training.batch_size,
        config.training.num_workers,
        config.training.seed,
        shuffle=True,
    )
    val_loader = build_dataloader(
        val_dataset,
        config.training.batch_size,
        config.training.num_workers,
        config.training.seed,
        shuffle=False,
    )
    test_loader = build_dataloader(
        test_dataset,
        config.training.batch_size,
        config.training.num_workers,
        config.training.seed,
        shuffle=False,
    )

    model = MeanPoolMLP(
        input_dim=768,
        hidden_dim=config.model.hidden_dim,
        output_dim=len(config.data.targets),
        dropout=config.model.dropout,
    )
    lightning_module = LightningRegressor(
        model,
        config.training.learning_rate,
        config.training.weight_decay,
        config.data.targets,
    )

    logger = CSVLogger(
        save_dir=str(config.paths.results_dir),
        name="training",
    )
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        precision=config.training.precision,
        logger=logger,
    )

    trainer.fit(lightning_module, train_loader, val_loader)
    trainer.test(lightning_module, test_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_path = config.paths.results_dir / "predictions_test.csv"
    save_predictions(
        lightning_module,
        test_loader,
        pred_path,
        device,
        config.data.targets,
    )


if __name__ == "__main__":
    main()
