from __future__ import annotations

import argparse
from pathlib import Path


PREFIX = "__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove the '__' prefix from Step Zero job filenames."
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=Path("/home/projects/zeevid/ofirlev/llm/step_zero/jobs_dnabert_2"),
        help="Directory containing Step Zero job scripts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs_dir = args.jobs_dir.expanduser().resolve()

    if not jobs_dir.exists():
        raise SystemExit(f"Jobs directory does not exist: {jobs_dir}")

    renamed = 0
    for path in sorted(jobs_dir.iterdir()):
        if not path.is_file() or not path.name.startswith(PREFIX):
            continue

        new_path = path.with_name(path.name[len(PREFIX) :])
        if new_path.exists():
            raise FileExistsError(
                f"Cannot rename {path.name} to {new_path.name}: target already exists"
            )

        path.rename(new_path)
        renamed += 1
        print(f"Renamed {path.name} -> {new_path.name}", flush=True)

    print(f"Finished. Renamed {renamed} file(s).", flush=True)


if __name__ == "__main__":
    main()
