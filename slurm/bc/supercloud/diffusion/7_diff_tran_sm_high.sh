#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=7_diff_tran_sm_high
#SBATCH -c 20

python -m src.train.bc +experiment=state/diff_tran \
    randomness='[high]' \
    data.data_subset=50 \
    rollout.randomness=high \
    wandb.mode=offline \
    wandb.project=ol-state-dr-high-1 \
    dryrun=false