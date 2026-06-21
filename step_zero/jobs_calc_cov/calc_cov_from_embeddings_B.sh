#!/bin/bash
#BSUB -J "compute mean embeddings"
#BSUB -q short-gpu
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "affinity[thread*20]"
#BSUB -oo /home/projects/zeevid/ofirlev/llm/step_zero/logs/calc_cov_from_embeddings_B.out.log
#BSUB -eo /home/projects/zeevid/ofirlev/llm/step_zero/logs/calc_cov_from_embeddings_B.err.log

module load miniconda/4.10.3_environmentally
conda activate /home/projects/zeevid/ofirlev/miniconda3/envs/dnabert_s

export PYTHONUNBUFFERED=1

python -u /home/projects/zeevid/ofirlev/llm/step_zero/calc_cov_from_embeddings.py /home/projects/zeevid/ofirlev/llm/data/full_dnabert_2 --output_dir /home/projects/zeevid/ofirlev/llm/data/cov_dnabert_2 \
--h5_prefix B
