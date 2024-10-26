#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=2_ol_state_act

python -m src.train.bc +experiment=state/act \
    randomness=low \
    task=one_leg \
    actor.ConditionalVAE.ActionTransformer.TransformerDecoder.return_intermediate=false \
    rollout.max_steps=700 \
    wandb.project=act-retint-low \
    dryrun=false