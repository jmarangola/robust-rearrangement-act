#!/bin/bash

#SBATCH -p vision-pulkitag-a100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=2_fph_10k_low_state

python -m src.train.bc +experiment=state/scaling/peg_hole/10k \
    wandb.continue_run_id=fkf3cvom \
    dryrun=false
