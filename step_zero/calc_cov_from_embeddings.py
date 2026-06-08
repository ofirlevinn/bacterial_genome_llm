from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

EXPECTED_EMBEDDING_DIM = 768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute covariance matrices from full read-embedding H5 files."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing full-embedding .h5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to sibling <input_dir>_cov.",
    )
    return parser.parse_args()


def resolve_output_dir(input_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return input_dir.parent / f"{input_dir.name}_cov"


def list_embedding_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    h5_files = sorted(path for path in input_dir.glob("*.h5") if path.is_file())
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {input_dir}")
    return h5_files


def load_read_embeddings(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as handle:
        if "read_embeddings" not in handle:
            raise ValueError(f"{h5_path} does not contain a read_embeddings dataset")

        read_embeddings = handle["read_embeddings"][()]

    if read_embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D read_embeddings in {h5_path}, got shape {read_embeddings.shape}"
        )

    if read_embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            f"Expected embedding dimension {EXPECTED_EMBEDDING_DIM} in {h5_path}, "
            f"got {read_embeddings.shape[1]}"
        )

    return read_embeddings.astype(np.float64, copy=False)


def compute_covariance(read_embeddings: np.ndarray) -> np.ndarray:
    if read_embeddings.shape[0] < 2:
        raise ValueError("At least two reads are required to compute covariance.")

    centered = read_embeddings - read_embeddings.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / (read_embeddings.shape[0] - 1)
    return covariance


def save_covariance(
    output_path: Path,
    covariance: np.ndarray,
    source_path: Path,
    num_reads: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("covariance", data=covariance)
        handle.attrs["source_h5"] = str(source_path)
        handle.attrs["num_reads"] = int(num_reads)
        handle.attrs["embedding_dim"] = EXPECTED_EMBEDDING_DIM


def process_file(input_path: Path, output_dir: Path) -> Path:
    read_embeddings = load_read_embeddings(input_path)
    covariance = compute_covariance(read_embeddings)

    output_path = output_dir / input_path.name
    save_covariance(
        output_path=output_path,
        covariance=covariance,
        source_path=input_path,
        num_reads=read_embeddings.shape[0],
    )
    print(
        f"Saved covariance matrix {covariance.shape} for {input_path.name} to {output_path}",
        flush=True,
    )
    return output_path


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser()
    output_dir = resolve_output_dir(input_dir, args.output_dir)

    h5_files = list_embedding_files(input_dir)
    print(
        f"Found {len(h5_files)} embedding file(s) in {input_dir}. Writing to {output_dir}",
        flush=True,
    )

    for h5_path in h5_files:
        process_file(h5_path, output_dir)

    print("Finished computing covariance matrices.", flush=True)


if __name__ == "__main__":
    main()
