#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 4-00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --account=parkes_low_priority
#SBATCH --job-name=2_rt_10k_low_state

python -m src.train.bc +experiment=state/scaling/round_table/10k \
    dryrun=false