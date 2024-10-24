import torch
import hydra

from collections import deque
from src.behavior.act import ACTPolicy
from omegaconf import DictConfig
from src.common.geometry import proprioceptive_quat_to_6d_rotation
from src.models.residual import ResidualPolicy
from typing import Dict


# TODO: A perhaps better approach (but WIP) has been started as RefinementActor
# which could be a bit cleaner, however it is a bit more complex since
# an Actor is passed into the constructor and it dynamically infers the policy from the weights
# instead of requiring the end user to specify it themselves.
class ResidualActPolicy(ACTPolicy):
    def __init__(self, device: torch.device,
                 cfg: DictConfig):
        super().__init__(device, cfg)

        # TODO: Reconsider the way we deal with this
        # E.g., can we separate out this so that it's not in the base class to be overwritten like this?
        # Also, is there a way that's (a) more efficient and (b) allows us to reset just a subset of environments?
        self.actions = None
        self.observations = deque(maxlen=self.obs_horizon)
        self.base_nactions = deque(maxlen=self.action_horizon)

        # Make the residual layers:
        # This is an MLP that takes in the state and predicted action
        # and outputs the residual to be added to the predicted action
        # NOTE: What about having a ensemble of residual layers?
        # They're cheap to compute and we can use them to both improve the
        # performance of the policy and to estimate the uncertainty of the
        # policy.
        self.residual_policy: ResidualPolicy = hydra.utils.instantiate(
            cfg.actor.residual_policy,
            obs_shape=(self.timestep_obs_dim,),
            action_shape=(self.action_dim,),
        )

    def load_base_state_dict(self, path: str):
        base_state_dict = torch.load(path)
        if "model_state_dict" in base_state_dict:
            base_state_dict = base_state_dict["model_state_dict"]

        # Load the model weights
        base_model_state_dict = {
            key[len("model.") :]: value
            for key, value in base_state_dict.items()
            if key.startswith("model.")
        }
        self.model.load_state_dict(base_model_state_dict)

        # Load normalizer parameters
        base_normalizer_state_dict = {
            key[len("normalizer.") :]: value
            for key, value in base_state_dict.items()
            if key.startswith("normalizer.")
        }
        self.normalizer.load_state_dict(base_normalizer_state_dict)

    @torch.no_grad()
    def action(self, obs: Dict[str, torch.Tensor]):
        """
        Predict the action given the batch of observations
        """
        self.observations.append(obs)

        # Normalize observations
        nobs = self._normalized_obs(self.observations, flatten=self.flatten_obs)

        if not self.base_nactions:
            # If there are no base actions, predict the action
            base_nactioon_pred = self._normalized_action(nobs)

            # Add self.action_horizon base actions
            start = self.obs_horizon - 1 if self.predict_past_actions else 0
            end = start + self.action_horizon
            for i in range(start, end):
                self.base_nactions.append(base_nactioon_pred[:, i, :])

        # Pop off the next base action
        base_naction = self.base_nactions.popleft()

        # Concatenate the state and base action
        nobs = nobs.flatten(start_dim=1)
        residual_nobs = torch.cat([nobs, base_naction], dim=-1)

        # Predict the residual (already scaled)
        residual = self.residual_policy.get_action(residual_nobs)

        # Add the residual to the base action
        naction = base_naction + residual

        # Denormalize and return the action
        return self.normalizer(naction, "action", forward=False)

    @torch.no_grad()
    def action_pred(self, batch):
        nobs = self._training_obs(batch, flatten=self.flatten_obs)
        naction = self._normalized_action(nobs)

        residual_nobs = torch.cat([batch["obs"], naction], dim=-1)
        residual = self.residual_policy.get_action(residual_nobs)

        return self.normalizer(naction + residual, "action", forward=False)

    @torch.no_grad()
    def base_action_normalized(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        action = super().action(obs)
        return self.normalizer(action, "action", forward=True)

    def process_obs(self, obs: Dict[str, torch.Tensor]):
        # Robot state is [pos, ori_quat, pos_vel, ori_vel, gripper]
        robot_state = obs["robot_state"]

        # Parts poses is [pos, ori_quat] for each part
        parts_poses = obs["parts_poses"]

        # Make the robot state have 6D proprioception
        robot_state = proprioceptive_quat_to_6d_rotation(robot_state)

        robot_state = self.normalizer(robot_state, "robot_state", forward=True)
        parts_poses = self.normalizer(parts_poses, "parts_poses", forward=True)

        obs = torch.cat([robot_state, parts_poses], dim=-1)

        # Clamp the observation to be bounded to [-5, 5]
        obs = torch.clamp(obs, -3, 3)

        return obs

    def get_value(self, residual_nobs) -> torch.Tensor:
        return self.residual_policy.get_value(residual_nobs)

    def action_normalized(self, obs: Dict[str, torch.Tensor]):
        raise NotImplementedError

    def correct_action(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict the correction to the action given the observation and the action
        """
        nobs = self.process_obs(obs)
        naction = self.normalizer(action, "action", forward=True)

        # Concatenate normalized observation with normalized action from the BC policy
        residual_nobs = torch.cat([nobs, naction], dim=-1)

        # Apply correction
        naction_corrected = (
            naction
            + self.residual_policy.actor_mean(residual_nobs)
            * self.residual_policy.action_scale
        )

        # Return the corrected action
        return self.normalizer(naction_corrected, "action", forward=False)

    def reset(self):
        """
        Reset the actor
        """
        self.base_nactions.clear()
        self.observations.clear()
        if self.actions is not None:
            self.actions.clear()

    @property
    def actor_parameters(self):
        return [
            p for n, p in self.residual_policy.named_parameters() if "critic" not in n
        ]

    @property
    def critic_parameters(self):
        return [p for n, p in self.residual_policy.named_parameters() if "critic" in n]

    @property
    def base_actor_parameters(self):
        """
        Return the parameters of the base model (actor only)
        """
        return [
            p
            for n, p in self.model.named_parameters()
            # TODO: generalize to encoderk.* doesn't do anything without images yet because these don't exist at the moment
            if not (n.startswith("encoder1.") or n.startswith("encoder2."))
        ]
