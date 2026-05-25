from __future__ import annotations

import argparse
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import torch
from transformers import AutoModel, AutoTokenizer
import yaml

SANITY_PASSED_JOB_PREFIX = "__"
EXPECTED_BAGS_PER_SAMPLE = 100
EXPECTED_READS_PER_BAG = 10000
EXPECTED_EMBEDDING_DIM = 768


@dataclass(frozen=True)
class PathsConfig:
    output_dir: Path
    jobs_dir: Path


@dataclass(frozen=True)
class DataConfig:
    sample_dir_suffix: str


@dataclass(frozen=True)
class ModelConfig:
    pretrained_name: str
    batch_size: int
    device: str


@dataclass(frozen=True)
class RuntimeConfig:
    overwrite: bool


@dataclass(frozen=True)
class StepZeroConfig:
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    runtime: RuntimeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute DNABERT-S embeddings for FASTQ bags."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--paths-file",
        type=Path,
        required=True,
        help="Text file containing one FASTQ path per line.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> StepZeroConfig:
    print(f"Loading config from {config_path}", flush=True)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    config = StepZeroConfig(
        paths=PathsConfig(
            output_dir=Path(raw_config["paths"]["output_dir"]).expanduser(),
            jobs_dir=Path(raw_config["paths"]["jobs_dir"]).expanduser(),
        ),
        data=DataConfig(
            sample_dir_suffix=str(raw_config["data"]["sample_dir_suffix"]),
        ),
        model=ModelConfig(
            pretrained_name=str(raw_config["model"]["pretrained_name"]),
            batch_size=int(raw_config["model"]["batch_size"]),
            device=str(raw_config["model"]["device"]),
        ),
        runtime=RuntimeConfig(
            overwrite=bool(raw_config["runtime"]["overwrite"])
        ),
    )
    print(
        "Loaded config: "
        f"output_dir={config.paths.output_dir}, "
        f"jobs_dir={config.paths.jobs_dir}, "
        f"sample_dir_suffix={config.data.sample_dir_suffix}, "
        f"batch_size={config.model.batch_size}, "
        f"device={config.model.device}, "
        f"overwrite={config.runtime.overwrite}",
        flush=True,
    )
    return config


def iter_fastq_sequences(fastq_path: Path) -> Iterator[str]:
    with gzip.open(fastq_path, "rt", encoding="utf-8") as handle:
        while True:
            header = handle.readline()
            if not header:
                return

            sequence = handle.readline().strip()
            plus_line = handle.readline()
            quality = handle.readline()

            if not plus_line or not quality:
                raise ValueError(f"Incomplete FASTQ record in {fastq_path}")

            yield sequence


def load_all_reads(fastq_path: Path) -> list[str]:
    reads = list(iter_fastq_sequences(fastq_path))
    if not reads:
        raise ValueError(f"No reads found in {fastq_path}")
    return reads


def load_backbone(model_config: ModelConfig) -> tuple[AutoTokenizer, AutoModel]:
    print(
        f"Loading tokenizer and model: {model_config.pretrained_name} "
        f"on device={model_config.device}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.pretrained_name,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        model_config.pretrained_name,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    model.eval()
    print(f"Model device: {model_config.device}", flush=True)
    model.to(model_config.device)
    print("Model loaded and moved to device.", flush=True)
    return tokenizer, model


def embed_reads(
    reads: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    batches: list[torch.Tensor] = []
    total_batches = (len(reads) + batch_size - 1) // batch_size
    print(
        f"Embedding {len(reads)} reads in {total_batches} batch(es) "
        f"with batch_size={batch_size}",
        flush=True,
    )

    with torch.inference_mode():
        for start in range(0, len(reads), batch_size):
            batch_reads = reads[start : start + batch_size]
            batch_index = (start // batch_size) + 1
            # print(
            #     f"  Running batch {batch_index}/{total_batches} "
            #     f"({len(batch_reads)} reads)",
            #     flush=True,
            # )
            tokens = tokenizer(
                batch_reads,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            outputs = model(**tokens)
            batch_embeddings = outputs[0][:, 0, :].detach().cpu()
            batches.append(batch_embeddings)

    print("Finished embedding reads.", flush=True)
    return torch.cat(batches, dim=0)


def parse_sample_id(fastq_path: Path, sample_dir_suffix: str) -> str:
    bag_name = fastq_path.name.replace(".fastq.gz", "")
    normalized_suffix = sample_dir_suffix.lstrip("-")
    part_marker = f"-{normalized_suffix}.part_"
    sample_id, separator, _part_suffix = bag_name.partition(part_marker)
    if not separator or not sample_id:
        raise ValueError(
            "Expected bag name like '<SampleName>-{sample_dir_suffix}.part_<PartIndex>.fastq.gz', "
            f"got {fastq_path.name}"
        )
    return sample_id


def build_output_path(
    output_dir: Path,
    fastq_path: Path,
    sample_dir_suffix: str,
) -> Path:
    sample_id = parse_sample_id(fastq_path, sample_dir_suffix)
    bag_name = fastq_path.name.replace(".fastq.gz", "")
    return output_dir / f"{bag_name}.h5"


def save_bag_embeddings(
    output_path: Path,
    embeddings: torch.Tensor,
    sample_id: str,
    source_fastq: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("embeddings", data=embeddings.numpy())
        handle.attrs["sample_id"] = sample_id
        handle.attrs["source_fastq"] = str(source_fastq)
        handle.attrs["num_reads"] = embeddings.shape[0]
        handle.attrs["embedding_dim"] = embeddings.shape[1]


def validate_embeddings(embeddings: torch.Tensor, fastq_path: Path) -> None:
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embeddings for {fastq_path}, got shape {tuple(embeddings.shape)}"
        )
    if embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            f"Expected embedding dimension {EXPECTED_EMBEDDING_DIM} for {fastq_path}, got {embeddings.shape[1]}"
        )
    if torch.isnan(embeddings).any():
        raise ValueError(f"Found NaNs in embeddings for {fastq_path}")


def validate_bag_output(output_path: Path, expected_num_reads: int) -> None:
    if not output_path.exists():
        raise FileNotFoundError(f"Expected output file was not created: {output_path}")

    with h5py.File(output_path, "r") as handle:
        if "embeddings" not in handle:
            raise ValueError(f"Saved output is missing 'embeddings': {output_path}")

        embeddings = handle["embeddings"]
        if embeddings.ndim != 2:
            raise ValueError(
                f"Saved output {output_path} is not a 2D matrix: {embeddings.shape}"
            )
        if embeddings.size == 0:
            raise ValueError(f"Saved output is empty: {output_path}")
        if embeddings.shape[0] != expected_num_reads:
            raise ValueError(
                f"Saved output {output_path} has {embeddings.shape[0]} reads, "
                f"expected {expected_num_reads}"
            )
        if embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
            raise ValueError(
                f"Saved output {output_path} has embedding_dim={embeddings.shape[1]}, expected {EXPECTED_EMBEDDING_DIM}"
            )
        saved_num_reads = handle.attrs.get("num_reads")
        saved_embedding_dim = handle.attrs.get("embedding_dim")
        if saved_num_reads is not None and int(saved_num_reads) != expected_num_reads:
            raise ValueError(
                f"Saved output {output_path} has num_reads attribute={saved_num_reads}, "
                f"expected {expected_num_reads}"
            )
        if (
            saved_embedding_dim is not None
            and int(saved_embedding_dim) != EXPECTED_EMBEDDING_DIM
        ):
            raise ValueError(
                f"Saved output {output_path} has embedding_dim attribute={saved_embedding_dim}, "
                f"expected {EXPECTED_EMBEDDING_DIM}"
            )


def load_fastq_paths(paths_file: Path) -> list[Path]:
    print(f"Loading FASTQ paths from {paths_file}", flush=True)
    if not paths_file.exists():
        raise FileNotFoundError(f"Paths file does not exist: {paths_file}")

    with paths_file.open("r", encoding="utf-8") as handle:
        fastq_paths = [
            Path(line.strip()).expanduser()
            for line in handle
            if line.strip()
        ]

    if not fastq_paths:
        raise FileNotFoundError(f"No FASTQ paths were found in {paths_file}")

    first_fastq_path = fastq_paths[0]
    if not first_fastq_path.exists():
        raise FileNotFoundError(f"First FASTQ path does not exist: {first_fastq_path}")

    print(f"Loaded {len(fastq_paths)} FASTQ path(s).", flush=True)
    return fastq_paths


def run_step_zero(config: StepZeroConfig, fastq_paths: list[Path]) -> None:
    print(f"Starting Step Zero for {len(fastq_paths)} bag(s).", flush=True)

    bags_to_process: list[tuple[int, Path, Path]] = []
    for bag_index, fastq_path in enumerate(fastq_paths, start=1):
        output_path = build_output_path(
            config.paths.output_dir,
            fastq_path,
            config.data.sample_dir_suffix,
        )
        if output_path.exists() and not config.runtime.overwrite:
            try:
                validate_bag_output(
                    output_path, expected_num_reads=EXPECTED_READS_PER_BAG
                )
                print(
                    f"[{bag_index}/{len(fastq_paths)}] Skipping validated output: {output_path}",
                    flush=True,
                )
                continue
            except (OSError, ValueError, KeyError) as exc:
                print(
                    f"[{bag_index}/{len(fastq_paths)}] Recomputing invalid output {output_path}: {exc}",
                    flush=True,
                )

        bags_to_process.append((bag_index, fastq_path, output_path))

    if not bags_to_process:
        print("All bag outputs are already present and validated.", flush=True)
        return

    tokenizer, model = load_backbone(config.model)

    for bag_index, fastq_path, output_path in bags_to_process:
        print(f"[{bag_index}/{len(fastq_paths)}] Processing {fastq_path}", flush=True)
        reads = load_all_reads(fastq_path)
        print(f"Loaded {len(reads)} read(s) from {fastq_path.name}", flush=True)
        embeddings = embed_reads(
            reads=reads,
            tokenizer=tokenizer,
            model=model,
            batch_size=config.model.batch_size,
            device=config.model.device,
        )
        validate_embeddings(embeddings, fastq_path)
        save_bag_embeddings(
            output_path=output_path,
            embeddings=embeddings,
            sample_id=parse_sample_id(fastq_path, config.data.sample_dir_suffix),
            source_fastq=fastq_path,
        )
        validate_bag_output(output_path, expected_num_reads=len(reads))
        print(f"Saved embeddings to {output_path}", flush=True)

    print("Step Zero completed successfully.", flush=True)


def run_sanity_checks(config: StepZeroConfig, fastq_paths: list[Path]) -> bool:
    expected_bags = len(fastq_paths)
    print(
        f"Running sanity checks for {expected_bags} expected bag output(s).",
        flush=True,
    )

    if expected_bags != EXPECTED_BAGS_PER_SAMPLE:
        print(
            "Sanity check failed: "
            f"expected exactly {EXPECTED_BAGS_PER_SAMPLE} bags, found {expected_bags}",
            flush=True,
        )
        return False

    seen_output_paths: set[Path] = set()
    for fastq_path in fastq_paths:
        output_path = build_output_path(
            config.paths.output_dir,
            fastq_path,
            config.data.sample_dir_suffix,
        )
        if output_path in seen_output_paths:
            print(
                f"Sanity check failed: duplicate output path resolved for {output_path}",
                flush=True,
            )
            return False
        seen_output_paths.add(output_path)

        try:
            validate_bag_output(
                output_path, expected_num_reads=EXPECTED_READS_PER_BAG
            )
        except (OSError, ValueError, KeyError) as exc:
            print(
                f"Sanity check failed for {output_path}: {exc}",
                flush=True,
            )
            return False

    print("Sanity checks passed for all bag outputs.", flush=True)
    return True


def prefix_corresponding_job_script(config: StepZeroConfig, paths_file: Path) -> None:
    sample_name = paths_file.stem
    source_job_path = config.paths.jobs_dir / f"{sample_name}.sh"
    prefixed_job_path = config.paths.jobs_dir / f"{SANITY_PASSED_JOB_PREFIX}{sample_name}.sh"

    if prefixed_job_path.exists():
        print(
            f"Sanity-check prefix already applied to job script: {prefixed_job_path}",
            flush=True,
        )
        return

    if not source_job_path.exists():
        raise FileNotFoundError(
            f"Could not find the corresponding job script for {sample_name}: "
            f"{source_job_path}"
        )

    source_job_path.rename(prefixed_job_path)
    print(f"Renamed job script to {prefixed_job_path}", flush=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    fastq_paths = load_fastq_paths(args.paths_file)

    run_step_zero(config, fastq_paths)

    if run_sanity_checks(config, fastq_paths):
        prefix_corresponding_job_script(config, args.paths_file)


if __name__ == "__main__":
    main()
