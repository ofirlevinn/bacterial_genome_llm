#!/bin/bash
#BSUB -J "do0.1_lr0.0003_wd0.0001_dnabert_2_cov_cnn_kernels_ks3-3-3-3_pd1-1-1-1_st1-2-2-2_do0.1_lr0.0003_wd0.0001"
#BSUB -q long-gpu
#BSUB -R "span[hosts=1]"
#BSUB -gpu num=1:j_exclusive=yes:gmem=128GB
#BSUB -R "rusage[mem=64GB]"
#BSUB -R "affinity[thread*20]"
#BSUB -oo /home/projects/zeevid/ofirlev/llm/training/sweeps/do0.1_lr0.0003_wd0.0001_dnabert_2_cov_cnn_kernels/logs/ks3-3-3-3_pd1-1-1-1_st1-2-2-2_do0.1_lr0.0003_wd0.0001.out.log
#BSUB -eo /home/projects/zeevid/ofirlev/llm/training/sweeps/do0.1_lr0.0003_wd0.0001_dnabert_2_cov_cnn_kernels/logs/ks3-3-3-3_pd1-1-1-1_st1-2-2-2_do0.1_lr0.0003_wd0.0001.err.log

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/home/projects/zeevid/ofirlev/.conda/envs/training/lib:${LD_LIBRARY_PATH}

cd /home/projects/zeevid/ofirlev/llm

/home/projects/zeevid/ofirlev/.conda/envs/training/bin/python -u /home/projects/zeevid/ofirlev/llm/training/train.py \
--config /home/projects/zeevid/ofirlev/llm/training/sweeps/do0.1_lr0.0003_wd0.0001_dnabert_2_cov_cnn_kernels/configs/ks3-3-3-3_pd1-1-1-1_st1-2-2-2_do0.1_lr0.0003_wd0.0001.yaml
