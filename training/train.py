from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import asdict, dataclass
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
from pytorch_lightning.loggers import WandbLogger

from data_loader.bag_dataset import (
    BagDataset,
    build_dataloader,
    filter_bag_files,
    list_bag_files,
    load_embeddings,
    load_metadata,
    parse_sample_id,
)
from models.mlp import MLP


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
    type: str
    hidden_dim: int
    dropout: float
    conv_channels: list[int]
    conv_kernel_sizes: list[int]
    conv_strides: list[int]
    conv_paddings: list[int]


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool
    project: str
    save_dir: Path | None
    name: str | None
    entity: str | None
    group: str | None
    job_type: str | None
    tags: list[str]


@dataclass(frozen=True)
class TrainConfig:
    paths: PathsConfig
    data: DataConfig
    split: SplitConfig
    training: TrainingConfig
    model: ModelConfig
    wandb: WandbConfig


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

    def _load_int_list(section: dict[str, object], key: str, default: list[int]) -> list[int]:
        value = section.get(key, default)
        if value in (None, ""):
            return default
        if not isinstance(value, list):
            raise ValueError(f"Expected '{key}' to be a list of integers.")
        return [int(item) for item in value]

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
            type=str(raw["model"]["type"]),
            hidden_dim=int(raw["model"]["hidden_dim"]),
            dropout=float(raw["model"]["dropout"]),
            conv_channels=_load_int_list(raw["model"], "conv_channels", [32, 64, 128, 256]),
            conv_kernel_sizes=_load_int_list(raw["model"], "conv_kernel_sizes", [5, 5, 3, 3]),
            conv_strides=_load_int_list(raw["model"], "conv_strides", [2, 2, 2, 2]),
            conv_paddings=_load_int_list(raw["model"], "conv_paddings", [2, 2, 1, 1]),
        ),
        wandb=_load_wandb_config(raw),
    )


def _load_wandb_config(raw: dict[str, object]) -> WandbConfig:
    wandb_raw = raw.get("wandb", {})
    if wandb_raw is None:
        wandb_raw = {}
    if not isinstance(wandb_raw, dict):
        raise ValueError("Expected 'wandb' config section to be a mapping.")

    save_dir_value = wandb_raw.get("save_dir")
    save_dir = (
        Path(str(save_dir_value)).expanduser()
        if save_dir_value not in (None, "")
        else None
    )

    tags_value = wandb_raw.get("tags", [])
    if not isinstance(tags_value, list):
        raise ValueError("Expected 'wandb.tags' to be a list.")

    return WandbConfig(
        enabled=bool(wandb_raw.get("enabled", True)),
        project=str(wandb_raw.get("project", "metagenomic-llm")),
        save_dir=save_dir,
        name=(
            str(wandb_raw["name"])
            if wandb_raw.get("name") not in (None, "")
            else None
        ),
        entity=(
            str(wandb_raw["entity"])
            if wandb_raw.get("entity") not in (None, "")
            else None
        ),
        group=(
            str(wandb_raw["group"])
            if wandb_raw.get("group") not in (None, "")
            else None
        ),
        job_type=(
            str(wandb_raw["job_type"])
            if wandb_raw.get("job_type") not in (None, "")
            else None
        ),
        tags=[str(tag) for tag in tags_value],
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
        tensor, sample_id = load_embeddings(path)
        split = assign_split(sample_id, splits)
        row = {
            "sample_id": sample_id,
            "bag_name": path.name,
            "split": split,
        }
        if tensor.ndim == 1:
            row.update(
                {f"mean_{i}": float(value) for i, value in enumerate(tensor)}
            )
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


def build_loggers(config: TrainConfig) -> list[object]:
    loggers: list[object] = [
        CSVLogger(
            save_dir=str(config.paths.results_dir),
            name="training",
        )
    ]

    if not config.wandb.enabled:
        return loggers

    wandb_save_dir = (
        config.wandb.save_dir
        if config.wandb.save_dir is not None
        else config.paths.results_dir / "wandb"
    )
    wandb_logger = WandbLogger(
        project=config.wandb.project,
        save_dir=str(wandb_save_dir),
        name=config.wandb.name,
        entity=config.wandb.entity,
        group=config.wandb.group,
        job_type=config.wandb.job_type,
        tags=config.wandb.tags,
        log_model=False,
    )
    wandb_logger.experiment.config.update(
        {
            "paths": {
                "data_dir": str(config.paths.data_dir),
                "metadata_csv": str(config.paths.metadata_csv),
                "results_dir": str(config.paths.results_dir),
            },
            "data": asdict(config.data),
            "split": asdict(config.split),
            "training": asdict(config.training),
            "model": asdict(config.model),
        },
        allow_val_change=True,
    )
    loggers.append(wandb_logger)
    return loggers


class LightningRegressor(pl.LightningModule):
    def __init__(
        self,
        model: MLP,
        learning_rate: float,
        weight_decay: float,
        target_names: list[str],
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.target_names = target_names
        self.loss_fn = torch.nn.L1Loss()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.model(embeddings)

    def training_step(self, batch, _batch_idx: int):
        embeddings, _sample_ids, targets = batch
        preds = self(embeddings)
        loss = self.loss_fn(preds, targets)
        self.log("train_mae", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _batch_idx: int):
        embeddings, _sample_ids, targets = batch
        preds = self(embeddings)
        loss = self.loss_fn(preds, targets)
        self.log("val_mae", loss, prog_bar=True)
        self._log_per_target("val", preds, targets)

    def test_step(self, batch, _batch_idx: int):
        embeddings, _sample_ids, targets = batch
        preds = self(embeddings)
        loss = self.loss_fn(preds, targets)
        self.log("test_mae", loss)
        self._log_per_target("test", preds, targets)

    def _log_per_target(
        self, prefix: str, preds: torch.Tensor, targets: torch.Tensor
    ) -> None:
        per_target_mae = (preds - targets).abs().mean(dim=0)
        for idx, name in enumerate(self.target_names):
            self.log(f"{prefix}_mae_{name}", per_target_mae[idx])

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
            embeddings = embeddings.to(device)
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
    print(f"Found {len(bag_files)} bag files in {config.paths.data_dir}")
    if not bag_files:
        raise SystemExit(f"No .h5 files found in {config.paths.data_dir}")

    targets_map = load_metadata(
        config.paths.metadata_csv,
        config.data.sample_id_column,
        config.data.targets,
    )
    print(f"Loaded metadata for {len(targets_map)} samples from {config.paths.metadata_csv}")
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
    print(f"Train set first 5 sample IDs: {sorted(train_ids)[:5]}")
    print(f"Validation set first 5 sample IDs: {sorted(val_ids)[:5]}")
    print(f"Test set first 5 sample IDs: {sorted(test_ids)[:5]}")

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

    if config.model.type == "mlp":
        model = MLP(
            input_dim=768,
            hidden_dim=config.model.hidden_dim,
            output_dim=len(config.data.targets),
            dropout=config.model.dropout,
        )
    elif config.model.type == "cov_cnn":
        from models.cnn import CovCNNRegressor
        model = CovCNNRegressor(
            latent_dim=config.model.hidden_dim,
            num_targets=len(config.data.targets),
            conv_channels=config.model.conv_channels,
            conv_kernel_sizes=config.model.conv_kernel_sizes,
            conv_strides=config.model.conv_strides,
            conv_paddings=config.model.conv_paddings,
            dropout=config.model.dropout,
        )
    else:
        raise ValueError(f"Unsupported model type: {config.model.type}")
        
    lightning_module = LightningRegressor(
        model,
        config.training.learning_rate,
        config.training.weight_decay,
        config.data.targets,
    )

    loggers = build_loggers(config)
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        precision=config.training.precision,
        logger=loggers,
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
