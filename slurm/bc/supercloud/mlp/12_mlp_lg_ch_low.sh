#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 0-12:00
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
#SBATCH --job-name=12_mlp_lg_ch_low

python -m src.train.bc +experiment=state/mlp_lg_ch \
    randomness='[low,low_perturb]' \
    rollout.randomness=low \
    wandb.project=ol-state-dr-low-1 \
    wandb.mode=offline \
    dryrun=false