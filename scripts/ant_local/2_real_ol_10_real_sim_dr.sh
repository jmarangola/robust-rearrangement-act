python -m src.train.bc +experiment=image/real_ol_cotrain \
    actor/diffusion_model=transformer \
    training.actor_lr=1e-4 \
    training.num_epochs=2000 \
    training.batch_size=64 \
    actor.confusion_loss_beta=0.1 \
    demo_source=teleop \
    furniture='[one_leg_render_dr_low,one_leg_simple]' \
    randomness='[low,med,med_perturb]' \
    environment='[real,sim]' \
    +data.max_episode_count.one_leg_simple.teleop.low.success=10 \
    wandb.mode=offline \
    dryrun=false