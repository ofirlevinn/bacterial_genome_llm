#!/bin/bash
#BSUB -J "paths lists generation"
#BSUB -q short-gpu
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "affinity[thread*20]"
#BSUB -oo /home/projects/zeevid/ofirlev/llm/step_zero/logs/paths_lists_generation.out.log
#BSUB -eo /home/projects/zeevid/ofirlev/llm/step_zero/logs/paths_lists_generation.err.log

module load miniconda/4.10.3_environmentally

conda run -p /home/projects/zeevid/ofirlev/miniconda3/envs/dnabert_s \
python /home/projects/zeevid/ofirlev/llm/step_zero/generate_fastq_path_lists.py  --overwrite
