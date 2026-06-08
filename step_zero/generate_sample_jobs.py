from __future__ import annotations

import argparse
from pathlib import Path

LOGS_DIR = Path("/home/projects/zeevid/ofirlev/llm/step_zero/logs")
QUEUE = "short-gpu"
GPU_SPEC = "num=1:j_exclusive=yes:gmem=5GB"
MEMORY = "8GB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one LSF shell script per sample path list."
    )
    parser.add_argument(
        "--paths-dir",
        type=Path,
        default=Path("/home/projects/zeevid/ofirlev/llm/step_zero/fastq_paths"),
        help="Directory containing one .txt FASTQ path list per sample.",
    )

    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("/home/projects/zeevid/ofirlev/llm/configs/step_zero_config.yaml"),
        help="Path to YAML config file passed to each generated job.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/projects/zeevid/ofirlev/llm/step_zero/jobs"),
        help="Directory where per-sample job scripts will be written.",
    )
    parser.add_argument(
        "--python-script",
        type=Path,
        default=Path("/home/projects/zeevid/ofirlev/llm/step_zero/precompute_embeddings.py"),
        help="Python script invoked by each generated job.",
    )
    parser.add_argument(
        "--conda-env-prefix",
        type=Path,
        default=Path("/home/projects/zeevid/ofirlev/miniconda3/envs/dnabert_s"),
        help="Conda environment prefix passed to conda run -p.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=20,
        help="Thread count used in #BSUB affinity.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated job files.",
    )
    return parser.parse_args()


def validate_paths_dir(paths_dir: Path) -> list[Path]:
    if not paths_dir.exists():
        raise FileNotFoundError(f"Paths directory does not exist: {paths_dir}")

    path_files = sorted(path for path in paths_dir.glob("*.txt") if path.is_file())
    if not path_files:
        raise FileNotFoundError(f"No .txt sample path files found in {paths_dir}")

    return path_files


def build_job_script(
    sample_name: str,
    logs_dir: Path,
    conda_env_prefix: Path,
    python_script: Path,
    config_path: Path,
    paths_file: Path,
    queue: str,
    gpu_spec: str,
    memory: str,
    threads: int,
) -> str:
    return f"""#!/bin/bash
#BSUB -J "run {sample_name}"
#BSUB -q {queue}
#BSUB -R "span[hosts=1]"
#BSUB -gpu {gpu_spec}
#BSUB -R "rusage[mem={memory}]"
#BSUB -R "affinity[thread*{threads}]"
#BSUB -oo {logs_dir}/{sample_name}.out.log
#BSUB -eo {logs_dir}/{sample_name}.err.log

module load miniconda/4.10.3_environmentally
conda activate {conda_env_prefix}

export PYTHONUNBUFFERED=1

python -u {python_script} \\
--config {config_path} \\
--paths-file {paths_file}
"""

def main() -> None:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be greater than 0")

    path_files = validate_paths_dir(args.paths_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    written_count = 0
    for path_file in path_files:
        sample_name = path_file.stem
        job_path = args.output_dir / f"{sample_name}.sh"
        if job_path.exists() and not args.overwrite:
            print(f"Skipping existing job script: {job_path}", flush=True)
            continue

        job_content = build_job_script(
            sample_name=sample_name,
            logs_dir=LOGS_DIR,
            conda_env_prefix=args.conda_env_prefix,
            python_script=args.python_script,
            config_path=args.config_path,
            paths_file=path_file,
            queue=QUEUE,
            gpu_spec=GPU_SPEC,
            memory=MEMORY,
            threads=args.threads,
        )
        job_path.write_text(job_content, encoding="utf-8")
        job_path.chmod(0o755)
        written_count += 1
        print(f"Wrote job script for {sample_name} to {job_path}", flush=True)

    if written_count == 0:
        raise RuntimeError("No job scripts were created.")

    print(f"Created {written_count} per-sample job script(s).", flush=True)


if __name__ == "__main__":
    main()
