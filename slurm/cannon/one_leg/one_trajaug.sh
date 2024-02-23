#!/bin/bash

#SBATCH -p gpu                
#SBATCH -t 0-08:00            
#SBATCH --mem=256G            
#SBATCH --gres=gpu:1          
#SBATCH -c 32                 
#SBATCH -o wandb_output_%j.log  
#SBATCH -e wandb_error_%j.log   

python -m src.train.bc_no_rollout +experiment=image_traj_aug furniture=one_leg
