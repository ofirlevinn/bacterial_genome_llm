#!/bin/bash
#BSUB -J "compute mean embeddings"
#BSUB -q short-gpu
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "affinity[thread*20]"
#BSUB -oo /home/projects/zeevid/ofirlev/llm/step_zero/logs/compute_mean_embeddings.out.log
#BSUB -eo /home/projects/zeevid/ofirlev/llm/step_zero/logs/compute_mean_embeddings.err.log

module load miniconda/4.10.3_environmentally
conda activate /home/projects/zeevid/ofirlev/miniconda3/envs/dnabert_s

export PYTHONUNBUFFERED=1

python -u /home/projects/zeevid/ofirlev/llm/step_zero/compute_mean_embeddings.py 