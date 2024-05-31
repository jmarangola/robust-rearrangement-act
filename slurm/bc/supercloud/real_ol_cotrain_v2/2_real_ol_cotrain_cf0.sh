#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=real_ol_cotrain_trans
#SBATCH -c 20

python -m src.train.bc +experiment=image/real_ol_cotrain \
    training.actor_lr=1e-4 \
    demo_source='[teleop,rollout]' \
    training.num_epochs=5000 \
    furniture='[one_leg_render_rppo,one_leg_simple,one_leg_full]' \
    randomness='[low,med,med_perturb]' \
    actor.confusion_loss_beta=0.0 \
    environment='[real,sim]' \
    +data.max_episode_count.one_leg_simple.teleop.low.success=25 \
    wandb.mode=offline \
    dryrun=false
