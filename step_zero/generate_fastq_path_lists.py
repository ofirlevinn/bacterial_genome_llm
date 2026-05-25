from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one text file per R1 sample with FASTQ bag paths."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path(
            "/home/projects/zeevid/Analyses/2023-EbG/EnvDNABERT/10kSubSamples"
        ),
        help="Root directory containing project folders and sample subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/projects/zeevid/ofirlev/llm/step_zero/fastq_paths"),
        help="Directory where per-sample FASTQ path lists will be written.",
    )
    parser.add_argument(
        "--sample-dir-suffix",
        default="R1",
        help="Only sample directories ending with this suffix are included.",
    )
    parser.add_argument(
        "--fastq-pattern",
        default="*-R1.part_*.fastq.gz",
        help="Glob pattern used to select FASTQ bag files within each sample directory.",
    )
    parser.add_argument(
        "--max-files-per-sample",
        type=int,
        default=100,
        help="Maximum number of FASTQ paths to write per sample.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sample path files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per sample instead of periodic progress updates.",
    )
    return parser.parse_args()


def discover_r1_sample_dirs(raw_data_dir: Path, sample_dir_suffix: str) -> list[Path]:
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_data_dir}")

    project_roots = sorted(path for path in raw_data_dir.iterdir() if path.is_dir())
    if not project_roots:
        raise FileNotFoundError(f"No project directories found under {raw_data_dir}")

    sample_dirs: list[Path] = []
    suffix = sample_dir_suffix
    for project_root in project_roots:
        matching_dirs = sorted(
            path
            for path in project_root.iterdir()
            if path.is_dir() and path.name.endswith(suffix)
        )
        sample_dirs.extend(matching_dirs)

    if not sample_dirs:
        raise FileNotFoundError(
            f"No sample directories ending with '{sample_dir_suffix}' were found under "
            f"{raw_data_dir}"
        )

    return sample_dirs


def write_sample_paths_file(
    sample_dir: Path,
    output_dir: Path,
    fastq_pattern: str,
    max_files_per_sample: int,
    overwrite: bool,
) -> tuple[Path, int] | None:
    sample_fastqs = sorted(sample_dir.glob(fastq_pattern))
    if not sample_fastqs:
        print(f"Skipping {sample_dir.name}: no FASTQs matched {fastq_pattern}", flush=True)
        return None

    chosen_fastqs = sample_fastqs[:max_files_per_sample]
    output_path = output_dir / f"{sample_dir.name}.txt"

    if output_path.exists() and not overwrite:
        print(f"Skipping existing file: {output_path}", flush=True)
        return output_path, len(chosen_fastqs)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(str(path.resolve()) for path in chosen_fastqs) + "\n",
        encoding="utf-8",
    )
    return output_path, len(chosen_fastqs)


def main() -> None:
    args = parse_args()
    if args.max_files_per_sample <= 0:
        raise ValueError("--max-files-per-sample must be greater than 0")

    print(
        "Generating per-sample FASTQ path lists "
        f"from {args.raw_data_dir} into {args.output_dir}",
        flush=True,
    )
    sample_dirs = discover_r1_sample_dirs(args.raw_data_dir, args.sample_dir_suffix)
    print(f"Found {len(sample_dirs)} matching sample directories.", flush=True)

    written_count = 0
    total_fastqs = 0
    progress_interval = 100
    for sample_index, sample_dir in enumerate(sample_dirs, start=1):
        result = write_sample_paths_file(
            sample_dir=sample_dir,
            output_dir=args.output_dir,
            fastq_pattern=args.fastq_pattern,
            max_files_per_sample=args.max_files_per_sample,
            overwrite=args.overwrite,
        )
        if result is None:
            continue

        output_path, num_fastqs = result
        written_count += 1
        total_fastqs += num_fastqs
        if args.verbose:
            print(
                f"Wrote {num_fastqs} FASTQ path(s) for {sample_dir.name} to {output_path}",
                flush=True,
            )
        elif sample_index % progress_interval == 0 or sample_index == len(sample_dirs):
            print(
                f"Processed {sample_index}/{len(sample_dirs)} sample directories "
                f"and created {written_count} path files so far.",
                flush=True,
            )

    if written_count == 0:
        raise RuntimeError("No sample FASTQ path files were created.")

    print(
        f"Created {written_count} sample path file(s) covering {total_fastqs} FASTQ bag(s).",
        flush=True,
    )


if __name__ == "__main__":
    main()
