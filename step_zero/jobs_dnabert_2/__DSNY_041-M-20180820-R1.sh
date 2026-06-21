#!/bin/bash
#BSUB -J "run DSNY_041-M-20180820-R1"
#BSUB -q short-gpu
#BSUB -R "span[hosts=1]"
#BSUB -gpu num=1:j_exclusive=yes:gmem=5GB
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "affinity[thread*20]"
#BSUB -oo /home/projects/zeevid/ofirlev/llm/step_zero/logs/DSNY_041-M-20180820-R1.out.log
#BSUB -eo /home/projects/zeevid/ofirlev/llm/step_zero/logs/DSNY_041-M-20180820-R1.err.log

module load miniconda/4.10.3_environmentally
conda activate /home/projects/zeevid/ofirlev/miniconda3/envs/dnabert_s

export PYTHONUNBUFFERED=1

python -u /home/projects/zeevid/ofirlev/llm/step_zero/precompute_embeddings.py \
--config /home/projects/zeevid/ofirlev/llm/configs/step_zero_config_dnabert_2.yaml \
--paths-file /home/projects/zeevid/ofirlev/llm/step_zero/fastq_paths/DSNY_041-M-20180820-R1.txt
