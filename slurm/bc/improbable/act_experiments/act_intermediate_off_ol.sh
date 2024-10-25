#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --job-name=ol_state_trans_diff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1

python -m src.train.bc +experiment=state/act \
    randomness='[low,low_perturb]' \
    rollout.randomness=low \
    task=one_leg \
    dryrun=false actor.ConditionalVAE.TransformerDecoder.return_intermediate=false \
    wandb.project=act-retint-low \
    rollout.max_steps=700