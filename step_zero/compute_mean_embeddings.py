from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

EXPECTED_EMBEDDING_DIM = 768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute one mean embedding vector per bag HDF5 file."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/projects/zeevid/ofirlev/llm/data"),
        help="Directory containing source bag .h5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/projects/zeevid/ofirlev/llm/data/mean"),
        help="Directory where mean-vector .h5 files will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files that already exist.",
    )
    return parser.parse_args()


def list_bag_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("*ABBY*.h5") if path.is_file())


def load_mean_embedding(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with h5py.File(path, "r") as handle:
        if "embeddings" not in handle:
            raise ValueError(f"Missing 'embeddings' dataset in {path}")

        embeddings = handle["embeddings"][()]
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D embeddings in {path}, got shape {embeddings.shape}"
            )
        if embeddings.shape[0] == 0:
            raise ValueError(f"Embeddings dataset is empty in {path}")
        if embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
            raise ValueError(
                f"Expected embedding dim {EXPECTED_EMBEDDING_DIM} in {path}, got {embeddings.shape[1]}"
            )

        mean_embedding = embeddings.mean(axis=0, dtype=np.float32)
        if mean_embedding.shape != (EXPECTED_EMBEDDING_DIM,):
            raise ValueError(
                f"Expected pooled shape ({EXPECTED_EMBEDDING_DIM},) in {path}, got {mean_embedding.shape}"
            )
        if np.isnan(mean_embedding).any():
            raise ValueError(f"Found NaNs in pooled embedding for {path}")

        attrs = {key: handle.attrs[key] for key in handle.attrs.keys()}

    return mean_embedding.astype(np.float32, copy=False), attrs


def save_mean_embedding(
    output_path: Path,
    mean_embedding: np.ndarray,
    attrs: dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("mean_embedding", data=mean_embedding)
        for key, value in attrs.items():
            handle.attrs[key] = value
        handle.attrs["num_reads_pooled"] = int(attrs.get("num_reads", 0))
        handle.attrs["embedding_dim"] = int(mean_embedding.shape[0])


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    bag_files = list_bag_files(input_dir)
    print(f"Found {len(bag_files)} bag files in {input_dir}", flush=True)
    if not bag_files:
        raise SystemExit(f"No .h5 files found in {input_dir}")

    written = 0
    skipped = 0
    for bag_path in bag_files:
        output_path = output_dir / bag_path.name
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        mean_embedding, attrs = load_mean_embedding(bag_path)
        save_mean_embedding(output_path, mean_embedding, attrs)
        written += 1

        if written <= 3 or written % 1000 == 0:
            print(f"Wrote {output_path}", flush=True)

    print(
        f"Finished mean pooling. wrote={written}, skipped={skipped}, output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
