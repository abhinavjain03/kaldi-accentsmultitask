#!/bin/bash

#SBATCH --gres=gpu:3,gpu_mem:12000M # number of GPUs (keep it at 3) and memory limit
#SBATCH --cpus-per-task=20         # number of CPU cores
#SBATCH --output=slurm-%j.out       # output file

./multitask_run_2_base.sh --affix "acc300btn_tdnn1024_15shrink_0.9_0.1_withcontext" --train-stage 63
# ./run_multitask_with_separate_accents.sh --affix "separate_accents_variable_nz_4_accents" --train-stage -10