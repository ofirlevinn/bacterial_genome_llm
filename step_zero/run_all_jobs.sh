find /home/projects/zeevid/ofirlev/llm/step_zero/jobs_dnabert_2/ -maxdepth 1 -type f -name '*.sh' ! -name '__*.sh' -print0 |
while IFS= read -r -d '' job_path; do
  bsub < "$job_path"
done

