#!/bin/bash
#SBATCH --gres=gpu:3,gpu_mem:16000M  # number of GPUs (keep it at 3) and memory limit
#SBATCH --cpus-per-task=20         # number of CPU cores
#SBATCH --output=slurm-%j.out       # output file

./multitask_run_2_base_1.sh --affix "acc300btn_tdnn1024_extraacclayer_0.5_0.5_withcontext" --train-stage 44


# ./multitask_run_2_base_1.sh --affix "acc300btn_tdnn1024_0.5_0.5_withdifferentcontext" --train-stage 44
# ./run_multitask_with_separate_accents.sh --affix "separate_accents_variable_nz_4_accents" --train-stage -10
