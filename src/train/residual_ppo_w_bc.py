import os
from pathlib import Path
import furniture_bench  # noqa

from ipdb import set_trace as bp


from tqdm import tqdm, trange
import random
import time
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig, OmegaConf

from src.behavior.diffusion import DiffusionPolicy
from src.behavior.residual_diffusion import (
    ResidualDiffusionPolicy,
    ResidualTrainingValues,
)
from src.dataset.dataset import (
    FurnitureStateDataset,
)
from torch.utils.data import DataLoader
from src.dataset.dataloader import FixedStepsDataloader
from src.common.pytorch_util import dict_to_device
from src.common.files import get_processed_paths, path_override
from src.eval.eval_utils import get_model_from_api_or_cached
from diffusers.optimization import get_scheduler


from src.gym.env_rl_wrapper import ResidualPolicyEnvWrapper
from src.common.config_util import merge_base_bc_config_with_root_config
from src.models.ema import SwitchEMA


from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv

from furniture_bench.envs.observation import DEFAULT_STATE_OBS
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from typing import Dict, List

import wandb

from src.gym import turn_off_april_tags

# Register the eval resolver for omegaconf
OmegaConf.register_new_resolver("eval", eval)


def to_native(obj):
    try:
        return OmegaConf.to_object(obj)
    except ValueError:
        return obj


@torch.no_grad()
def calculate_advantage(
    values: torch.Tensor,
    next_value: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_done: torch.Tensor,
    steps_per_iteration: int,
    discount: float,
    gae_lambda: float,
):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(steps_per_iteration)):
        if t == steps_per_iteration - 1:
            nextnonterminal = 1.0 - next_done.to(torch.float)
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1].to(torch.float)
            nextvalues = values[t + 1]

        delta = rewards[t] + discount * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = (
            delta + discount * gae_lambda * nextnonterminal * lastgaelam
        )
    returns = advantages + values
    return advantages, returns


class TorchReplayBuffer:
# class TorchReplayBuffer(torch.utils.data.Dataset):
    def __init__(
        self,
        max_size: int,
        state_dim: int,
        action_dim: int,
        pred_horizon: int = 8,
        obs_horizon: int = 1,
        action_horizon: int = 32,
        device: str="cuda:0"
    ):
        
        self.device = device

        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

        self.sequence_length = obs_horizon + pred_horizon - 1

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states = torch.empty(
            (max_size, state_dim), dtype=torch.float32)
        self.actions = torch.empty(
            (max_size, action_dim), dtype=torch.float32)
        self.rewards = torch.empty(max_size, dtype=torch.float32)
        self.dones = torch.zeros(max_size, dtype=torch.bool)

        self.train_data = {
            "obs": self.states,
            "action": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
        }

        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.indices = None

    def create_sample_indices(
        self,
        episode_ends: np.ndarray,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
    ):
        indices = list()
        for i in range(len(episode_ends)):
            start_idx = 0
            if i > 0:
                start_idx = episode_ends[i - 1]
            end_idx = episode_ends[i]
            episode_length = end_idx - start_idx

            min_start = -pad_before
            max_start = episode_length - sequence_length + pad_after

            # range stops one idx before end
            for idx in range(min_start, max_start + 1):
                buffer_start_idx = max(idx, 0) + start_idx
                buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
                start_offset = buffer_start_idx - (idx + start_idx)
                end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
                sample_start_idx = 0 + start_offset
                sample_end_idx = sequence_length - end_offset
                indices.append(
                    [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx, i]
                )
        indices = np.array(indices)
        return indices

    def sample_sequence(
        self,
        train_data: Dict[str, torch.Tensor],
        sequence_length: int,
        buffer_start_idx: int,
        buffer_end_idx: int,
        sample_start_idx: int,
        sample_end_idx: int,
    ) -> Dict[str, torch.Tensor]:
        result = dict()
        # TODO: Implement the performance improvement (particularly for image-based training):
        # https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/common/sampler.py#L130-L138
        for key, input_arr in train_data.items():
            sample = input_arr[buffer_start_idx:buffer_end_idx]
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
                data = torch.zeros(
                    size=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
                )
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result

    def add_trajectory(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor):
        # Get the indices corresponding to the end of each episode
        episode_ends = torch.where(dones)[0]
        episode_idxs = torch.where(dones)[1]

        # Only add the timesteps that are part of the episode
        for ep_idx, end_idx in zip(episode_idxs, episode_ends):
            # print(f'Index: {ep_idx}, End: {end_idx}')

            # Decide what slice of the buffer to use - if the episode is too long, just cut it off
            restart = False
            if self.ptr + end_idx + 1 > self.max_size:
                end_idx = self.max_size - self.ptr - 1
                restart = True
            
            # Add the data to the buffer
            self.states[self.ptr : self.ptr + end_idx + 1] = states[:end_idx + 1, ep_idx] 
            self.actions[self.ptr : self.ptr + end_idx + 1] = actions[:end_idx + 1, ep_idx]    
            self.rewards[self.ptr : self.ptr + end_idx + 1] = rewards[:end_idx + 1, ep_idx]
            self.dones[self.ptr : self.ptr + end_idx + 1] = dones[:end_idx + 1, ep_idx]

            # Increment the start_idx (go to the next full episode)
            self.ptr = self.ptr + end_idx + 1 if not restart else 0
            self.size = min(self.size + end_idx + 1, self.max_size)
    
    def form_batch(self, nsample_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        out_batch = dict()
        for key in nsample_list[0].keys():
            out_batch[key] = torch.stack([nsample[key] for nsample in nsample_list], dim=0)
        return out_batch
    
    def rebuild_seq_indices(self):
        # First, get the valid indices depending on our episode ends and sequence length
        episode_ends = torch.where(self.dones[:self.size])[0].cpu().numpy()
        self.indices = self.create_sample_indices(
            episode_ends, 
            sequence_length=self.sequence_length, 
            pad_before=self.obs_horizon - 1,
            pad_after=self.action_horizon - 1)
    
    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
            demo_idx,
        ) = self.indices[idx]

        # get normalized data using these indices
        nsample = self.sample_sequence(
            train_data=self.train_data,
            sequence_length=self.sequence_length,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        return nsample
    
    def __len__(self):
        return len(self.indices)

    def sample_batch(self, batch_size: int):
        from IPython import embed; embed()

        # Loop over the batch size and sample sequences
        nsample_list = list()
        for _ in range(batch_size):

            idx = np.random.randint(0, len(self.indices))
            nsample = self[idx]
            nsample_list.append(nsample)

        # Collate into a batch 
        out_batch = self.form_batch(nsample_list)
    
        return out_batch


@hydra.main(
    config_path="../config",
    config_name="base_residual_rl",
    version_base="1.2",
)
def main(cfg: DictConfig):

    OmegaConf.set_struct(cfg, False)

    # TRY NOT TO MODIFY: seeding
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)

    if "task" not in cfg.env:
        cfg.env.task = "one_leg"

    # assert not (cfg.anneal_lr and cfg.adaptive_lr)

    run_name = f"{int(time.time())}__residual_ppo__{cfg.actor.residual_policy._target_.split('.')[-1]}__{cfg.seed}"

    run_directory = f"runs/debug-residual_ppo-residual-8"
    run_directory += "-delete" if cfg.debug else ""
    print(f"Run directory: {run_directory}")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    gpu_id = 0
    device = torch.device(f"cuda:{gpu_id}")

    turn_off_april_tags()

    env: FurnitureRLSimEnv = FurnitureRLSimEnv(
        act_rot_repr=cfg.control.act_rot_repr,
        action_type=cfg.control.control_mode,
        april_tags=False,
        concat_robot_state=True,
        ctrl_mode=cfg.control.controller,
        obs_keys=DEFAULT_STATE_OBS,
        furniture=cfg.env.task,
        # gpu_id=1,
        compute_device_id=gpu_id,
        graphics_device_id=gpu_id,
        headless=cfg.headless,
        num_envs=cfg.num_envs,
        observation_space="state",
        randomness=cfg.env.randomness,
        max_env_steps=100_000_000,
    )

    n_parts_to_assemble = len(env.pairs_to_assemble)

    # Load the behavior cloning actor
    base_cfg, base_wts = get_model_from_api_or_cached(
        cfg.base_policy.wandb_id,
        wt_type=cfg.base_policy.wt_type,
        wandb_mode=cfg.wandb.mode,
    )

    merge_base_bc_config_with_root_config(cfg, base_cfg)
    cfg.actor_name = f"residual_{cfg.base_policy.actor.name}"

    agent = ResidualDiffusionPolicy(device, base_cfg)
    agent.load_base_state_dict(base_wts)
    agent.to(device)
    agent.eval()

    residual_policy = agent.residual_policy

    # Set the inference steps of the actor
    if isinstance(agent, DiffusionPolicy):
        agent.inference_steps = 4

    env: ResidualPolicyEnvWrapper = ResidualPolicyEnvWrapper(
        env,
        max_env_steps=cfg.num_env_steps,
        normalize_reward=cfg.normalize_reward,
        reset_on_success=cfg.reset_on_success,
        reset_on_failure=cfg.reset_on_failure,
        reward_clip=cfg.clip_reward,
        device=device,
    )

    optimizer_actor = optim.AdamW(
        agent.actor_parameters,
        lr=cfg.learning_rate_actor,
        eps=1e-5,
        weight_decay=1e-6,
    )

    lr_scheduler_actor = get_scheduler(
        name=cfg.lr_scheduler.name,
        optimizer=optimizer_actor,
        num_warmup_steps=cfg.lr_scheduler.warmup_steps,
        num_training_steps=cfg.num_iterations,
    )

    optimizer_critic = optim.AdamW(
        agent.critic_parameters,
        lr=cfg.learning_rate_critic,
        eps=1e-5,
        weight_decay=1e-6,
    )

    lr_scheduler_critic = get_scheduler(
        name=cfg.lr_scheduler.name,
        optimizer=optimizer_critic,
        num_warmup_steps=cfg.lr_scheduler.warmup_steps,
        num_training_steps=cfg.num_iterations,
    )

    optimizer_base = optim.AdamW(
        agent.base_actor_parameters,
        lr=cfg.base_bc.learning_rate,
        eps=1e-5,
        weight_decay=1e-6
    )

    lr_scheduler_base = get_scheduler(
        name=cfg.base_bc.lr_scheduler.name,
        optimizer=optimizer_base,
        num_warmup_steps=cfg.base_bc.lr_scheduler.warmup_steps,
        num_training_steps=cfg.base_bc.num_iterations
    )

    if cfg.base_bc.ema.use:
        base_ema = SwitchEMA(agent.model, cfg.base_bc.ema.decay)
        base_ema.register()

    if cfg.data.data_paths_override is None:
        data_path = get_processed_paths(
            controller=to_native(cfg.control.controller),
            domain=to_native(cfg.data.environment),
            task=to_native(cfg.data.furniture),
            demo_source=to_native(cfg.data.demo_source),
            randomness=to_native(cfg.data.randomness),
            demo_outcome=to_native(cfg.data.demo_outcome),
            suffix=to_native(cfg.data.suffix),
        )
    else:
        data_path = path_override(cfg.data.data_paths_override)

    base_bc_dataset = FurnitureStateDataset(
        dataset_paths=data_path,
        pred_horizon=cfg.data.pred_horizon,
        obs_horizon=cfg.data.obs_horizon,
        action_horizon=cfg.data.action_horizon,
        data_subset=cfg.data.data_subset,
        control_mode=cfg.control.control_mode,
        predict_past_actions=cfg.data.predict_past_actions,
        pad_after=cfg.data.get("pad_after", True),
        max_episode_count=cfg.data.get("max_episode_count", None),
        include_future_obs=cfg.data.include_future_obs,
    )

    # Create dataloaders
    base_bc_trainload_kwargs = dict(
        dataset=base_bc_dataset,
        batch_size=cfg.base_bc.batch_size,
        num_workers=0,
        # num_workers=cfg.data.dataloader_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )
    # base_bc_trainloader = (
    #     FixedStepsDataloader(**base_bc_trainload_kwargs, n_batches=cfg.training.steps_per_epoch)
    #     if cfg.training.steps_per_epoch != -1
    #     else DataLoader(**base_bc_trainload_kwargs)
    # )
    base_bc_trainloader = DataLoader(**base_bc_trainload_kwargs)

    if (
        "pretrained_wts" in cfg.actor.residual_policy
        and cfg.actor.residual_policy.pretrained_wts
    ):
        print(
            f"Loading pretrained weights from {cfg.actor.residual_policy.pretrained_wts}"
        )
        run_state_dict = torch.load(cfg.actor.residual_policy.pretrained_wts)

        if "actor_logstd" in run_state_dict["model_state_dict"]:
            agent.residual_policy.load_state_dict(run_state_dict["model_state_dict"])
        else:
            agent.load_state_dict(run_state_dict["model_state_dict"])
        optimizer_actor.load_state_dict(run_state_dict["optimizer_actor_state_dict"])
        optimizer_critic.load_state_dict(run_state_dict["optimizer_critic_state_dict"])
        lr_scheduler_actor.load_state_dict(run_state_dict["scheduler_actor_state_dict"])
        lr_scheduler_critic.load_state_dict(
            run_state_dict["scheduler_critic_state_dict"]
        )

    steps_per_iteration = cfg.data_collection_steps

    print(f"Total timesteps: {cfg.total_timesteps}, batch size: {cfg.batch_size}")
    print(
        f"Mini-batch size: {cfg.minibatch_size}, num iterations: {cfg.num_iterations}"
    )

    print(OmegaConf.to_yaml(cfg, resolve=True))

    run = wandb.init(
        id=cfg.wandb.continue_run_id,
        resume=None if cfg.wandb.continue_run_id is None else "must",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name,
        save_code=True,
        mode=cfg.wandb.mode if not cfg.debug else "disabled",
    )

    if cfg.wandb.continue_run_id is not None:
        print(f"Continuing run {cfg.wandb.continue_run_id}, {run.name}")

        run_id = f"{cfg.wandb.project}/{cfg.wandb.continue_run_id}"

        # Load the weights from the run
        _, wts = get_model_from_api_or_cached(
            run_id, "latest", wandb_mode=cfg.wandb.mode
        )

        print(f"Loading weights from {wts}")

        run_state_dict = torch.load(wts)

        if "actor_logstd" in run_state_dict["model_state_dict"]:
            agent.residual_policy.load_state_dict(run_state_dict["model_state_dict"])
        else:
            agent.load_state_dict(run_state_dict["model_state_dict"])

        optimizer_actor.load_state_dict(run_state_dict["optimizer_actor_state_dict"])
        optimizer_critic.load_state_dict(run_state_dict["optimizer_critic_state_dict"])
        lr_scheduler_actor.load_state_dict(run_state_dict["scheduler_actor_state_dict"])
        lr_scheduler_critic.load_state_dict(
            run_state_dict["scheduler_critic_state_dict"]
        )

        # Set the best test loss and success rate to the one from the run
        try:
            best_eval_success_rate = run.summary["eval/best_eval_success_rate"]
        except KeyError:
            best_eval_success_rate = run.summary["eval/success_rate"]

        iteration = run.summary["iteration"]
        global_step = run.step

    else:
        global_step = 0
        iteration = 0
        best_eval_success_rate = 0.0

    obs: torch.Tensor = torch.zeros(
        (
            steps_per_iteration,
            cfg.num_envs,
            residual_policy.obs_dim,
        )
    )
    actions = torch.zeros((steps_per_iteration, cfg.num_envs) + env.action_space.shape)
    full_nactions = torch.zeros(
        (steps_per_iteration, cfg.num_envs) + env.action_space.shape
    )
    logprobs = torch.zeros((steps_per_iteration, cfg.num_envs))
    rewards = torch.zeros((steps_per_iteration, cfg.num_envs))
    dones = torch.zeros((steps_per_iteration, cfg.num_envs))
    values = torch.zeros((steps_per_iteration, cfg.num_envs))

    start_time = time.time()
    training_cum_time = 0
    running_mean_success_rate = 0.0

    next_done = torch.zeros(cfg.num_envs)
    next_obs = env.reset()
    agent.reset()

    # Create model save dir
    model_save_dir: Path = Path("models") / wandb.run.name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # create replay buffer
    buffer = TorchReplayBuffer(
        max_size=cfg.base_bc.replay_buffer_size,
        state_dim=agent.obs_dim,
        action_dim=agent.action_dim,
        pred_horizon=agent.pred_horizon,
        obs_horizon=agent.obs_horizon,
        action_horizon=agent.action_horizon,
        device=device,
    )

    while global_step < cfg.total_timesteps:
        iteration += 1
        print(f"Iteration: {iteration}/{cfg.num_iterations}")
        print(f"Run name: {run_name}")
        iteration_start_time = time.time()

        # If eval first flag is set, we will evaluate the model before doing any training
        eval_mode = (iteration - int(cfg.eval_first)) % cfg.eval_interval == 0

        # Also reset the env to have more consistent results
        if eval_mode or cfg.reset_every_iteration:
            next_obs = env.reset()
            agent.reset()

        print(f"Eval mode: {eval_mode}")

        for step in range(0, steps_per_iteration):
            if not eval_mode:
                # Only count environment steps during training
                global_step += cfg.num_envs

            # Get the base normalized action
            base_naction = agent.base_action_normalized(next_obs)

            # Process the obs for the residual policy
            next_nobs = agent.process_obs(next_obs)
            next_residual_nobs = torch.cat([next_nobs, base_naction], dim=-1)

            dones[step] = next_done
            obs[step] = next_residual_nobs

            with torch.no_grad():
                residual_naction_samp, logprob, _, value, naction_mean = (
                    residual_policy.get_action_and_value(next_residual_nobs)
                )

            residual_naction = residual_naction_samp if not eval_mode else naction_mean
            naction = base_naction + residual_naction * residual_policy.action_scale

            action = agent.normalizer(naction, "action", forward=False)
            next_obs, reward, next_done, truncated, info = env.step(action)

            if cfg.truncation_as_done:
                next_done = next_done | truncated

            values[step] = value.flatten().cpu()
            actions[step] = residual_naction.cpu()
            logprobs[step] = logprob.cpu()
            rewards[step] = reward.view(-1).cpu()
            next_done = next_done.view(-1).cpu()
            full_nactions[step] = naction.cpu()

            if step > 0 and (env_step := step * 1) % 100 == 0:
                print(
                    f"env_step={env_step}, global_step={global_step}, mean_reward={rewards[:step+1].sum(dim=0).mean().item()} fps={env_step * cfg.num_envs / (time.time() - iteration_start_time):.2f}"
                )

        # Find which environments are successful, and fetch these trajectories
        success_idxs = rewards.sum(dim=0) >= n_parts_to_assemble
        success_obs = obs[:, success_idxs, :-10]
        success_actions = full_nactions[:, success_idxs]
        success_rewards = rewards[:, success_idxs]

        # This has all timesteps including and after episode is done
        success_dones = rewards.cumsum(dim=0)[:, success_idxs] >= n_parts_to_assemble

        # Let's mask out the ones that come after the first "done" was received 
        first_done_mask = success_dones.cumsum(dim=0) > 1
        success_dones[first_done_mask] = False  

        # Add the successful trajectories to the replay buffer
        buffer.add_trajectory(
            success_obs, success_actions, success_rewards, success_dones
        )

        # replay_batch = buffer.sample(batch_size=4)

        # Calculate the success rate
        # Find the rewards that are not zero
        # Env is successful if it received a reward more than or equal to n_parts_to_assemble
        env_success = (rewards > 0).sum(dim=0) >= n_parts_to_assemble
        success_rate = env_success.float().mean().item()

        # Calculate the share of timesteps that come from successful trajectories that account for the success rate and the varying number of timesteps per trajectory
        # Count total timesteps in successful trajectories
        timesteps_in_success = rewards[:, env_success]

        # Find index of last reward in each trajectory
        last_reward_idx = torch.argmax(timesteps_in_success, dim=0)

        # Calculate the total number of timesteps in successful trajectories
        total_timesteps_in_success = last_reward_idx.sum().item()

        # Calculate the share of successful timesteps
        success_timesteps_share = total_timesteps_in_success / rewards.numel()

        running_mean_success_rate = 0.5 * running_mean_success_rate + 0.5 * success_rate

        print(
            f"SR: {success_rate:.4%}, SR mean: {running_mean_success_rate:.4%}, SPS: {steps_per_iteration * cfg.num_envs / (time.time() - iteration_start_time):.2f}"
        )

        if eval_mode:
            # If we are in eval mode, we don't need to do any training, so log the result and continue

            # Save the model if the evaluation success rate improves
            if success_rate > best_eval_success_rate:
                best_eval_success_rate = success_rate
                model_path = str(model_save_dir / f"actor_chkpt_best_success_rate.pt")
                torch.save(
                    {
                        # Save the weights of the residual policy (base + residual)
                        "model_state_dict": agent.state_dict(),
                        "optimizer_actor_state_dict": optimizer_actor.state_dict(),
                        "optimizer_critic_state_dict": optimizer_critic.state_dict(),
                        "scheduler_actor_state_dict": lr_scheduler_actor.state_dict(),
                        "scheduler_critic_state_dict": lr_scheduler_critic.state_dict(),
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "success_rate": success_rate,
                        "success_timesteps_share": success_timesteps_share,
                        "iteration": iteration,
                    },
                    model_path,
                )

                wandb.save(model_path)
                print(f"Evaluation success rate improved. Model saved to {model_path}")

            wandb.log(
                {
                    "eval/success_rate": success_rate,
                    "eval/best_eval_success_rate": best_eval_success_rate,
                    "iteration": iteration,
                },
                step=global_step,
            )
            # Start the data collection again
            # NOTE: We're not resetting here now, that happens before the next
            # iteration only if the reset_every_iteration flag is set
            continue

        b_obs = obs.reshape((-1, residual_policy.obs_dim))
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_values = values.reshape(-1)

        # Get the base normalized action
        # Process the obs for the residual policy
        base_naction = agent.base_action_normalized(next_obs)
        next_nobs = agent.process_obs(next_obs)
        next_residual_nobs = torch.cat([next_nobs, base_naction], dim=-1)
        next_value = residual_policy.get_value(next_residual_nobs).reshape(1, -1).cpu()

        # bootstrap value if not done
        advantages, returns = calculate_advantage(
            values,
            next_value,
            rewards,
            dones,
            next_done,
            steps_per_iteration,
            cfg.discount,
            cfg.gae_lambda,
        )

        b_advantages = advantages.reshape(-1).cpu()
        b_returns = returns.reshape(-1).cpu()

        # Optimizing the policy and value network
        b_inds = np.arange(cfg.batch_size)
        clipfracs = []
        for epoch in trange(cfg.update_epochs, desc="Policy update"):
            early_stop = False

            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                # Get the minibatch and place it on the device
                mb_obs = b_obs[mb_inds].to(device)
                mb_actions = b_actions[mb_inds].to(device)
                mb_logprobs = b_logprobs[mb_inds].to(device)
                mb_advantages = b_advantages[mb_inds].to(device)
                mb_returns = b_returns[mb_inds].to(device)
                mb_values = b_values[mb_inds].to(device)

                # Calculate the loss
                _, newlogprob, entropy, newvalue, action_mean = (
                    residual_policy.get_action_and_value(mb_obs, mb_actions)
                )
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    ]

                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                policy_loss = 0

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        newvalue - mb_values,
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean() * cfg.ent_coef

                ppo_loss = pg_loss - entropy_loss

                # Add the auxiliary regularization loss
                residual_l1_loss = torch.mean(torch.abs(action_mean))
                residual_l2_loss = torch.mean(torch.square(action_mean))

                # Normalize the losses so that each term has the same scale
                if iteration > cfg.n_iterations_train_only_value:

                    # Scale the losses using the calculated scaling factors
                    policy_loss += ppo_loss
                    policy_loss += cfg.residual_l1 * residual_l1_loss
                    policy_loss += cfg.residual_l2 * residual_l2_loss

                # Total loss
                loss: torch.Tensor = policy_loss + v_loss * cfg.vf_coef

                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()

                loss.backward()
                nn.utils.clip_grad_norm_(
                    residual_policy.parameters(), cfg.max_grad_norm
                )

                optimizer_actor.step()
                optimizer_critic.step()

                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    print(
                        f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl:.4f} > {cfg.target_kl:.4f}"
                    )
                    early_stop = True
                    break

            if early_stop:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        action_norms = torch.norm(b_actions[:, :3], dim=-1).cpu()

        training_cum_time += time.time() - iteration_start_time
        sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0

        wandb.log(
            {
                "charts/learning_rate_actor": optimizer_actor.param_groups[0]["lr"],
                "charts/learning_rate_critic": optimizer_critic.param_groups[0]["lr"],
                "charts/SPS": sps,
                "charts/rewards": rewards.sum().item(),
                "charts/success_rate": success_rate,
                "charts/success_timesteps_share": success_timesteps_share,
                "charts/action_norm_mean": action_norms.mean(),
                "charts/action_norm_std": action_norms.std(),
                "values/advantages": b_advantages.mean().item(),
                "values/returns": b_returns.mean().item(),
                "values/values": b_values.mean().item(),
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/total_loss": loss.item(),
                "losses/entropy_loss": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
                "losses/residual_l1": residual_l1_loss.item(),
                "losses/residual_l2": residual_l2_loss.item(),
                "histograms/values": wandb.Histogram(values),
                "histograms/returns": wandb.Histogram(b_returns),
                "histograms/advantages": wandb.Histogram(b_advantages),
                "histograms/logprobs": wandb.Histogram(logprobs),
                "histograms/rewards": wandb.Histogram(rewards),
                "histograms/action_norms": wandb.Histogram(action_norms),
            },
            step=global_step,
        )

        # Step the learning rate scheduler
        lr_scheduler_actor.step()
        lr_scheduler_critic.step()

        # Checkpoint every cfg.checkpoint_interval steps
        if iteration % cfg.checkpoint_interval == 0:
            model_path = str(model_save_dir / f"actor_chkpt_{iteration}.pt")
            torch.save(
                {
                    "model_state_dict": agent.state_dict(),
                    "optimizer_actor_state_dict": optimizer_actor.state_dict(),
                    "optimizer_critic_state_dict": optimizer_critic.state_dict(),
                    "scheduler_actor_state_dict": lr_scheduler_actor.state_dict(),
                    "scheduler_critic_state_dict": lr_scheduler_critic.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "success_rate": success_rate,
                    "iteration": iteration,
                },
                model_path,
            )

            wandb.save(model_path)
            print(f"Model saved to {model_path}")

        # Print some stats at the end of the iteration
        print(
            f"Iteration {iteration}/{cfg.num_iterations}, global step {global_step}, SPS {sps}"
        )

        # Prepare the replay buffer and data loader for this epoch
        buffer.rebuild_seq_indices()
        buffer_trainloader = DataLoader(
            buffer,
            batch_size=cfg.base_bc.batch_size,
            # num_workers=cfg.data.dataloader_workers,
            num_workers=0,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            persistent_workers=False,
        )
        
        if cfg.base_bc.train_bc and iteration % cfg.base_bc.train_with_bc_every == 0:

            base_bc_epoch_loss = list()
            buffer_epoch_loss = list()
            tepoch = tqdm(
                zip(base_bc_trainloader, buffer_trainloader), 
                desc="Training", 
                leave=False, 
                total=min(len(base_bc_trainloader), len(buffer_trainloader)))

            # Train the base policy with BC for a few iterations
            for base_batch, buffer_batch in tepoch:

                # Zero the gradients in all optimizers
                optimizer_base.zero_grad()

                # Make predictions with agent
                base_batch = dict_to_device(base_batch, device)
                base_bc_loss, base_bc_losses_log = agent.compute_loss(base_batch)
                base_bc_loss.backward()

                # Make predictions with agent
                buffer_batch = dict_to_device(buffer_batch, device)
                buffer_loss, buffer_losses_log = agent.compute_loss(buffer_batch)
                buffer_loss.backward()

                # Step the optimizers and schedulers
                optimizer_base.step()
                lr_scheduler_base.step()

                if cfg.base_bc.ema.use:
                    base_ema.update()

                # Log losses
                base_bc_loss_cpu = base_bc_loss.item()
                buffer_loss_cpu = buffer_loss.item()
                base_bc_epoch_loss.append(base_bc_loss_cpu)
                buffer_epoch_loss.append(buffer_loss_cpu)
                bc_step_log = {
                    "base_batch_loss": base_bc_loss_cpu,
                    "buffer_batch_loss": buffer_loss_cpu,
                    **base_bc_losses_log,
                    **buffer_losses_log,
                }

                wandb.log(bc_step_log)
                tepoch.set_postfix(base_loss=base_bc_loss_cpu, buffer_loss=buffer_loss_cpu)

    print(f"Training finished in {(time.time() - start_time):.2f}s")


if __name__ == "__main__":
    main()
