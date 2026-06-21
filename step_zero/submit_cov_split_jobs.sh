#!/usr/bin/env bash
set -euo pipefail

base_dir="/home/projects/zeevid/ofirlev/llm/step_zero"

# Missing non-SA prefixes (not currently running in your set G, L, M, O, SA, T, U, W, Y)
single_prefixes=(B C D H J K N R S)

# SA split into smaller shards for parallel processing
sa_shards=(
  SAMN071365
  SAMN071366
  SAMN071367
  SAMN071368
  SAMN071369
  SAMN071370
  SAMN048
  SAMN074
  SAMN3
  SAMN4
  SAMEA26
  SAMEA4
  SAMEA5
  SAMEA25
)

submit_one() {
  local prefix="$1"
  local job_path="$base_dir/jobs_calc_cov/calc_cov_from_embeddings_${prefix}.sh"

  if [[ ! -f "$job_path" ]]; then
    echo "Missing job file: $job_path" >&2
    return 1
  fi

  echo "Submitting $job_path"
  bsub < "$job_path"
}

for p in "${single_prefixes[@]}"; do
  submit_one "$p"
done

for p in "${sa_shards[@]}"; do
  submit_one "$p"
done
