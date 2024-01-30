import os
from pathlib import Path
from src.common.context import suppress_all_output, suppress_stdout


with suppress_stdout():
    import furniture_bench

import numpy as np
import torch
import wandb
from diffusers.optimization import get_scheduler
from src.dataset.dataset import (
    FurnitureImageDataset,
    FurnitureFeatureDataset,
)
from src.dataset.normalizer import StateActionNormalizer
from src.eval.rollout import do_rollout_evaluation
from src.common.tasks import furniture2idx
from src.gym import get_env
from tqdm import tqdm, trange
from ipdb import set_trace as bp
from src.behavior import get_actor
from src.dataset.dataloader import FixedStepsDataloader
from src.common.pytorch_util import dict_apply
import argparse
from torch.utils.data import random_split, DataLoader
from src.common.earlystop import EarlyStopper
from src.common.files import get_processed_paths


from ml_collections import ConfigDict

from gym import logger

import hydra
from omegaconf import DictConfig, OmegaConf

logger.set_level(logger.DISABLED)


def set_dryrun_params(config: ConfigDict):
    if config.dryrun:
        OmegaConf.set_struct(config, False)
        config.training.steps_per_epoch = 10
        config.training.data_subset = 2

        if config.rollout.rollouts:
            config.rollout.every = 1
            config.rollout.num_rollouts = 1

        config.wandb.mode = "disabled"

        OmegaConf.set_struct(config, True)


@hydra.main(config_path="../config", config_name="base")
def main(config: DictConfig):
    set_dryrun_params(config)
    print(config)
    env = None
    device = torch.device(
        f"cuda:{config.training.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    data_path = get_processed_paths(
        environment="sim",
        task=None,
        # task=config.furniture,
        demo_source=["teleop", "scripted"],
        randomness=None,
        demo_outcome="success",
    )

    print(f"Using data from {data_path}")

    if config.training.observation_type == "image":
        dataset = FurnitureImageDataset(
            dataset_paths=data_path,
            pred_horizon=config.actor.pred_horizon,
            obs_horizon=config.actor.obs_horizon,
            action_horizon=config.actor.action_horizon,
            augment_image=config.regularization.augment_image,
            data_subset=config.training.data_subset,
            first_action_idx=config.actor.first_action_index,
        )
    elif config.training.observation_type == "feature":
        dataset = FurnitureFeatureDataset(
            dataset_paths=data_path,
            pred_horizon=config.actor.pred_horizon,
            obs_horizon=config.actor.obs_horizon,
            action_horizon=config.actor.action_horizon,
            encoder_name=config.vision_encoder.model,
            data_subset=config.training.data_subset,
            first_action_idx=config.actor.first_action_index,
        )
    else:
        raise ValueError(f"Unknown observation type: {config.observation_type}")

    # Split the dataset into train and test
    train_size = int(len(dataset) * (1 - config.training.test_split))
    test_size = len(dataset) - train_size
    print(f"Splitting dataset into {train_size} train and {test_size} test samples.")
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    OmegaConf.set_struct(config, False)
    config.robot_state_dim = dataset.robot_state_dim

    # Create the policy network
    actor = get_actor(config, device)

    # Set the data path in the config object
    config.data_path = [str(f) for f in data_path]
    # Update the config object with the action dimension
    config.action_dim = dataset.action_dim
    config.n_episodes = len(dataset.episode_ends)
    # Update the config object with the observation dimension
    config.timestep_obs_dim = actor.timestep_obs_dim
    OmegaConf.set_struct(config, True)

    if config.training.load_checkpoint_path is not None:
        print(f"Loading checkpoint from {config.training.load_checkpoint_path}")
        actor.load_state_dict(torch.load(config.training.load_checkpoint_path))

    # Create dataloaders
    trainloader = FixedStepsDataloader(
        dataset=train_dataset,
        n_batches=config.training.steps_per_epoch,
        batch_size=config.training.batch_size,
        num_workers=config.training.dataloader_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )

    testloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.dataloader_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )

    # AdamW optimizer for noise_net
    opt_noise = torch.optim.AdamW(
        params=actor.parameters(),
        lr=config.training.actor_lr,
        weight_decay=config.regularization.weight_decay,
    )

    n_batches = len(trainloader)

    lr_scheduler = get_scheduler(
        name=config.lr_scheduler.name,
        optimizer=opt_noise,
        num_warmup_steps=config.lr_scheduler.warmup_steps,
        num_training_steps=len(trainloader) * config.training.num_epochs,
    )

    early_stopper = EarlyStopper(
        patience=config.early_stopper.patience,
        smooth_factor=config.early_stopper.smooth_factor,
    )
    config_dict = OmegaConf.to_container(config, resolve=True)
    # Init wandb
    wandb.init(
        id=config.wandb.continue_run_id,
        resume=config.wandb.continue_run_id is not None,
        project=config.wandb.project,
        entity="robot-rearrangement",
        config=config_dict,
        mode=config.wandb.mode,
        notes=config.wandb.notes,
    )

    # save stats to wandb and update the config object
    wandb.log(
        {
            "num_samples": len(train_dataset),
            "num_samples_test": len(test_dataset),
            "num_episodes": int(
                len(dataset.episode_ends) * (1 - config.training.test_split)
            ),
            "num_episodes_test": int(
                len(dataset.episode_ends) * config.training.test_split
            ),
            "stats": StateActionNormalizer().stats_dict,
        }
    )

    # Create model save dir
    model_save_dir = Path(config.training.model_save_dir) / wandb.run.name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Train loop
    best_test_loss = float("inf")
    test_loss_mean = float("inf")
    best_success_rate = 0

    tglobal = trange(
        config.training.start_epoch,
        config.training.num_epochs,
        initial=config.training.start_epoch,
        total=config.training.num_epochs,
        desc=f"Epoch ({config.rollout.furniture if config.rollout.rollouts else 'multitask'}, {config.training.observation_type}, {config.vision_encoder.model})",
    )
    for epoch_idx in tglobal:
        epoch_loss = list()
        test_loss = list()

        # batch loop
        actor.train_mode()
        dataset.train()
        tepoch = tqdm(trainloader, desc="Training", leave=False, total=n_batches)
        for batch in tepoch:
            opt_noise.zero_grad()

            # device transfer
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

            # Get loss
            loss = actor.compute_loss(batch)

            # backward pass
            loss.backward()

            # optimizer step
            opt_noise.step()
            lr_scheduler.step()

            # logging
            loss_cpu = loss.item()
            epoch_loss.append(loss_cpu)
            lr = lr_scheduler.get_last_lr()[0]
            wandb.log(
                dict(
                    lr=lr,
                    batch_loss=loss_cpu,
                )
            )

            tepoch.set_postfix(loss=loss_cpu, lr=lr)

        tepoch.close()

        train_loss_mean = np.mean(epoch_loss)
        tglobal.set_postfix(
            loss=train_loss_mean,
            test_loss=test_loss_mean,
            best_success_rate=best_success_rate,
        )
        wandb.log({"epoch_loss": np.mean(epoch_loss), "epoch": epoch_idx})

        # Evaluation loop
        actor.eval_mode()
        dataset.eval()
        test_tepoch = tqdm(testloader, desc="Validation", leave=False)
        for test_batch in test_tepoch:
            with torch.no_grad():
                # device transfer for test_batch
                test_batch = dict_apply(
                    test_batch, lambda x: x.to(device, non_blocking=True)
                )

                # Get test loss
                test_loss_val = actor.compute_loss(test_batch)

                # logging
                test_loss_cpu = test_loss_val.item()
                test_loss.append(test_loss_cpu)
                test_tepoch.set_postfix(loss=test_loss_cpu)

        test_tepoch.close()

        test_loss_mean = np.mean(test_loss)
        tglobal.set_postfix(
            loss=train_loss_mean,
            test_loss=test_loss_mean,
            best_success_rate=best_success_rate,
        )

        wandb.log({"test_epoch_loss": test_loss_mean, "epoch": epoch_idx})

        # Save the model if the test loss is the best so far
        if config.training.checkpoint_model and test_loss_mean < best_test_loss:
            best_test_loss = test_loss_mean
            save_path = str(model_save_dir / f"actor_chkpt_best_test_loss.pt")
            torch.save(
                actor.state_dict(),
                save_path,
            )
            wandb.save(save_path)

        # Early stopping
        if early_stopper.update(test_loss_mean):
            print(
                f"Early stopping at epoch {epoch_idx} as test loss did not improve for {early_stopper.patience} epochs."
            )
            break

        # Log the early stopping stats
        wandb.log(
            {
                "early_stopper/counter": early_stopper.counter,
                "early_stopper/best_loss": early_stopper.best_loss,
                "early_stopper/ema_loss": early_stopper.ema_loss,
            }
        )

        if (
            config.rollout.rollouts
            and (epoch_idx + 1) % config.rollout.every == 0
            and np.mean(test_loss_mean) < config.rollout.loss_threshold
        ):
            # Do not load the environment until we successfuly made it this far
            if env is None:
                from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv

                env: FurnitureSimEnv = get_env(
                    config.training.gpu_id,
                    furniture=config.rollout.furniture,
                    num_envs=config.rollout.num_envs,
                    randomness=config.rollout.randomness,
                    # Now using full size images in sim and resizing to be consistent
                    resize_img=False,
                    act_rot_repr=config.training.act_rot_repr,
                    ctrl_mode="osc",
                )

            best_success_rate = do_rollout_evaluation(
                config,
                env,
                config.rollout.save_rollouts,
                actor,
                best_success_rate,
                epoch_idx,
            )

    tglobal.close()
    wandb.finish()


if __name__ == "__main__":
    main()
