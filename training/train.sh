#!/bin/bash
#BSUB -J "training baseline"
#BSUB -q long-gpu
#BSUB -R "span[hosts=1]"
#BSUB -gpu num=1:j_exclusive=yes:gmem=180GB
#BSUB -R "rusage[mem=64GB]"
#BSUB -R "affinity[thread*20]"
#BSUB -oo /home/projects/zeevid/ofirlev/llm/logs/training.out.log
#BSUB -eo /home/projects/zeevid/ofirlev/llm/logs/training.err.log

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/home/projects/zeevid/ofirlev/.conda/envs/training/lib:${LD_LIBRARY_PATH}

/home/projects/zeevid/ofirlev/.conda/envs/training/bin/python -u /home/projects/zeevid/ofirlev/llm/training/train.py \
--config /home/projects/zeevid/ofirlev/llm/configs/training_config_dnabert_2.yaml
