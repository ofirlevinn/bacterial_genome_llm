from __future__ import annotations

import argparse
import itertools
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SweepPaths:
    output_root: Path
    python_bin: Path
    train_script: Path


@dataclass(frozen=True)
class ClusterConfig:
    queue: str
    gpu_mem_gb: int
    host_mem_gb: int
    cpu_threads: int


@dataclass(frozen=True)
class SweepWandb:
    group: str | None
    job_type: str | None
    tags: list[str]


@dataclass(frozen=True)
class SweepSpec:
    base_config: Path
    sweep_name: str
    paths: SweepPaths
    cluster: ClusterConfig
    wandb: SweepWandb
    fixed_overrides: dict[str, Any]
    grid: dict[str, list[object]]
    include: list[dict[str, object]] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate training configs and LSF jobs for a hyperparameter sweep."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "training_sweep.yaml",
        help="Path to the sweep YAML config.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the generated jobs with bsub after writing them.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def load_sweep_spec(path: Path) -> SweepSpec:
    raw = load_yaml(path)
    paths_raw = raw["paths"]
    cluster_raw = raw["cluster"]
    wandb_raw = raw.get("wandb", {})
    constraints_raw = raw.get("constraints", {})
    include_raw = constraints_raw.get("include")
    fixed_overrides_raw = raw.get("fixed_overrides", {})

    if include_raw is not None and not isinstance(include_raw, list):
        raise ValueError("Expected constraints.include to be a list of parameter maps.")
    if not isinstance(fixed_overrides_raw, dict):
        raise ValueError("Expected fixed_overrides to be a mapping.")

    grid = raw["grid"]
    if not isinstance(grid, dict) or not grid:
        raise ValueError("Expected a non-empty grid mapping in sweep config.")

    normalized_grid: dict[str, list[object]] = {}
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Expected grid.{key} to be a non-empty list.")
        normalized_grid[str(key)] = values

    return SweepSpec(
        base_config=Path(raw["base_config"]).expanduser(),
        sweep_name=str(raw["sweep_name"]),
        paths=SweepPaths(
            output_root=Path(paths_raw["output_root"]).expanduser(),
            python_bin=Path(paths_raw["python_bin"]).expanduser(),
            train_script=Path(paths_raw["train_script"]).expanduser(),
        ),
        cluster=ClusterConfig(
            queue=str(cluster_raw["queue"]),
            gpu_mem_gb=int(cluster_raw["gpu_mem_gb"]),
            host_mem_gb=int(cluster_raw["host_mem_gb"]),
            cpu_threads=int(cluster_raw["cpu_threads"]),
        ),
        wandb=SweepWandb(
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
            tags=[str(tag) for tag in wandb_raw.get("tags", [])],
        ),
        fixed_overrides=fixed_overrides_raw,
        grid=normalized_grid,
        include=include_raw,
    )


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).replace("/", "_")


def build_run_name(combo: dict[str, object]) -> str:
    parts = []
    for key in sorted(combo):
        key_alias = {
            "learning_rate": "lr",
            "weight_decay": "wd",
            "dropout": "do",
            "hidden_dim": "hd",
        }.get(key, key)
        parts.append(f"{key_alias}{format_value(combo[key])}")
    return "_".join(parts)


def iter_combinations(spec: SweepSpec) -> list[dict[str, object]]:
    keys = list(spec.grid.keys())
    product = itertools.product(*(spec.grid[key] for key in keys))
    combos = [dict(zip(keys, values, strict=True)) for values in product]
    if spec.include is None:
        return combos

    allowed = []
    for combo in combos:
        for include_entry in spec.include:
            if all(combo.get(key) == value for key, value in include_entry.items()):
                allowed.append(combo)
                break
    return allowed


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if (
            isinstance(value, dict)
            and key in base
            and isinstance(base[key], dict)
        ):
            deep_update(base[key], value)
        else:
            base[key] = value


def apply_overrides(base_config: dict[str, Any], combo: dict[str, object], spec: SweepSpec) -> dict[str, Any]:
    config = yaml.safe_load(yaml.safe_dump(base_config))
    run_name = build_run_name(combo)

    deep_update(config, spec.fixed_overrides)

    training_section = config.setdefault("training", {})
    model_section = config.setdefault("model", {})
    wandb_section = config.setdefault("wandb", {})

    for key, value in combo.items():
        if key in {"learning_rate", "weight_decay", "batch_size", "max_epochs"}:
            training_section[key] = value
        elif key in {"dropout", "hidden_dim"}:
            model_section[key] = value
        else:
            raise ValueError(f"Unsupported sweep override key: {key}")

    wandb_section["enabled"] = True
    wandb_section["name"] = run_name
    wandb_section["group"] = spec.wandb.group or spec.sweep_name
    if spec.wandb.job_type is not None:
        wandb_section["job_type"] = spec.wandb.job_type

    existing_tags = wandb_section.get("tags", [])
    if not isinstance(existing_tags, list):
        raise ValueError("Expected wandb.tags in base config to be a list.")
    merged_tags = [str(tag) for tag in existing_tags]
    for tag in spec.wandb.tags:
        if tag not in merged_tags:
            merged_tags.append(tag)
    for key, value in combo.items():
        combo_tag = f"{key}={format_value(value)}"
        if combo_tag not in merged_tags:
            merged_tags.append(combo_tag)
    wandb_section["tags"] = merged_tags

    return config


def build_job_script(
    spec: SweepSpec,
    config_path: Path,
    logs_dir: Path,
    run_name: str,
) -> str:
    out_log = logs_dir / f"{run_name}.out.log"
    err_log = logs_dir / f"{run_name}.err.log"
    working_dir = spec.paths.train_script.parent.parent
    return f"""#!/bin/bash
#BSUB -J "{spec.sweep_name}_{run_name}"
#BSUB -q {spec.cluster.queue}
#BSUB -R "span[hosts=1]"
#BSUB -gpu num=1:j_exclusive=yes:gmem={spec.cluster.gpu_mem_gb}GB
#BSUB -R "rusage[mem={spec.cluster.host_mem_gb}GB]"
#BSUB -R "affinity[thread*{spec.cluster.cpu_threads}]"
#BSUB -oo {out_log}
#BSUB -eo {err_log}

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/home/projects/zeevid/ofirlev/.conda/envs/training/lib:${{LD_LIBRARY_PATH}}

cd {working_dir}

{spec.paths.python_bin} -u {spec.paths.train_script} \\
--config {config_path}
"""


def main() -> None:
    args = parse_args()
    spec = load_sweep_spec(args.config.expanduser())
    base_config = load_yaml(spec.base_config)

    sweep_root = spec.paths.output_root / spec.sweep_name
    configs_dir = sweep_root / "configs"
    jobs_dir = sweep_root / "jobs"
    logs_dir = sweep_root / "logs"
    for directory in (configs_dir, jobs_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    combinations = iter_combinations(spec)
    if not combinations:
        raise SystemExit("No sweep combinations were generated.")

    print(
        f"Generating {len(combinations)} run(s) for sweep '{spec.sweep_name}' in {sweep_root}",
        flush=True,
    )

    for combo in combinations:
        run_name = build_run_name(combo)
        config = apply_overrides(base_config, combo, spec)
        config_path = configs_dir / f"{run_name}.yaml"
        job_path = jobs_dir / f"{run_name}.sh"

        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        job_script = build_job_script(spec, config_path, logs_dir, run_name)
        with job_path.open("w", encoding="utf-8") as handle:
            handle.write(job_script)

        print(f"Wrote config: {config_path}", flush=True)
        print(f"Wrote job: {job_path}", flush=True)

        if args.submit:
            with job_path.open("r", encoding="utf-8") as handle:
                subprocess.run(["bsub"], stdin=handle, check=True)
            print(f"Submitted job: {job_path.name}", flush=True)

    print("Sweep generation completed.", flush=True)


if __name__ == "__main__":
    main()
