#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-v100,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=8_rl_100050_low_state

python -m src.train.bc +experiment=state/scaling_100k \
    dryrun=false
