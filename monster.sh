#!/bin/bash

#SBATCH --gres=gpu:3,gpu_mem:2000M # number of GPUs (keep it at 3) and memory limit
#SBATCH --cpus-per-task=20         # number of CPU cores
#SBATCH --output=slurm-%j.out       # output file

./multitask_monster_with_inputembeddings.sh --affix "a300_t024_10shrink_0.9_0.1_withsaloneembedinput_utt_diffinputstage" --train-stage 59
# ./run_multitask_with_separate_accents.sh --affix "separate_accents_variable_nz_4_accents" --train-stage -10