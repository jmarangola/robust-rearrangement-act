from omegaconf import OmegaConf
import torch
import torch.nn as nn
from typing import Dict, Tuple, Union
from collections import deque
from ipdb import set_trace as bp  # noqa

from src.behavior.base import Actor
from src.models.mlp import MLP
from src.models import get_encoder
from src.common.control import RotationMode

from src.common.geometry import proprioceptive_quat_to_6d_rotation


import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class MLPActor(Actor):
    def __init__(
        self,
        device: Union[str, torch.device],
        config,
    ) -> None:
        super().__init__()
        self.device = device

        actor_cfg = config.actor
        self.obs_horizon = actor_cfg.obs_horizon
        self.action_dim = (
            10 if config.control.act_rot_repr == RotationMode.rot_6d else 8
        )
        self.pred_horizon = actor_cfg.pred_horizon
        self.action_horizon = actor_cfg.action_horizon
        self.observation_type = config.observation_type

        # A queue of the next actions to be executed in the current horizon
        self.actions = deque(maxlen=self.action_horizon)

        # Regularization
        self.feature_noise = config.regularization.feature_noise
        self.feature_dropout = config.regularization.feature_dropout
        self.feature_layernorm = config.regularization.feature_layernorm
        self.state_noise = config.regularization.get("state_noise", False)

        encoder_kwargs = OmegaConf.to_container(config.vision_encoder, resolve=True)
        encoder_name = config.vision_encoder.model
        freeze_encoder = config.vision_encoder.freeze

        self.encoder1 = get_encoder(
            encoder_name,
            device=device,
            **encoder_kwargs,
        )
        self.encoder2 = (
            self.encoder1
            if freeze_encoder
            else get_encoder(
                encoder_name,
                device=device,
                **encoder_kwargs,
            )
        )

        self.encoding_dim = self.encoder1.encoding_dim

        if actor_cfg.get("projection_dim") is not None:
            self.encoder1_proj = nn.Linear(
                self.encoding_dim, actor_cfg.projection_dim
            ).to(device)
            self.encoder2_proj = nn.Linear(
                self.encoding_dim, actor_cfg.projection_dim
            ).to(device)
            self.encoding_dim = actor_cfg.projection_dim
        else:
            self.encoder1_proj = nn.Identity()
            self.encoder2_proj = nn.Identity()

        self.timestep_obs_dim = config.robot_state_dim + 2 * self.encoding_dim

        self.model = MLP(
            input_dim=self.timestep_obs_dim * self.obs_horizon,
            output_dim=self.action_dim * self.pred_horizon,
            hidden_dims=actor_cfg.hidden_dims,
            dropout=actor_cfg.dropout,
            residual=actor_cfg.residual,
        ).to(device)

        loss_fn_name = actor_cfg.loss_fn if hasattr(actor_cfg, "loss_fn") else "MSELoss"
        self.loss_fn = getattr(nn, loss_fn_name)()

    # === Inference ===
    @torch.no_grad()
    def action(self, obs: deque):
        # Normalize observations
        nobs = self._normalized_obs(obs)

        # If the queue is empty, fill it with the predicted actions
        if not self.actions:
            # Predict normalized action
            naction = self.model(nobs).reshape(
                nobs.shape[0], self.pred_horizon, self.action_dim
            )

            # unnormalize action
            # (B, pred_horizon, action_dim)
            action_pred = self.normalizer(naction, "action", forward=False)

            # Add the actions to the queue
            # only take action_horizon number of actions
            start = self.obs_horizon - 1
            end = start + self.action_horizon
            for i in range(start, end):
                self.actions.append(action_pred[:, i, :])

        # Return the first action in the queue
        return self.actions.popleft()

    # === Training ===
    def compute_loss(self, batch):
        # State already normalized in the dataset
        obs_cond = self._training_obs(batch, flatten=True)

        # Action already normalized in the dataset
        naction = batch["action"]

        # forward pass
        naction_pred = self.model(obs_cond).reshape(
            naction.shape[0], self.pred_horizon, self.action_dim
        )

        loss = self.loss_fn(naction_pred, naction)

        return loss


class MLPStateActor(Actor):
    def __init__(
        self,
        device: Union[str, torch.device],
        config,
    ) -> None:
        super().__init__()
        self.device = device

        actor_cfg = config.actor
        self.obs_horizon = actor_cfg.obs_horizon
        self.action_dim = (
            10 if config.control.act_rot_repr == RotationMode.rot_6d else 8
        )
        self.pred_horizon = actor_cfg.pred_horizon
        self.action_horizon = actor_cfg.action_horizon
        self.observation_type = config.observation_type

        # A queue of the next actions to be executed in the current horizon
        self.actions = deque(maxlen=self.action_horizon)

        # Get the dimension of the parts poses
        self.parts_poses_dim = 35

        self.timestep_obs_dim = config.robot_state_dim + self.parts_poses_dim

        self.model = MLP(
            input_dim=self.timestep_obs_dim * self.obs_horizon,
            output_dim=self.action_dim * self.pred_horizon,
            hidden_dims=actor_cfg.hidden_dims,
            dropout=actor_cfg.dropout,
            residual=actor_cfg.residual,
        ).to(device)

        loss_fn_name = actor_cfg.loss_fn if hasattr(actor_cfg, "loss_fn") else "MSELoss"
        self.loss_fn = getattr(nn, loss_fn_name)()

    # === Inference ===
    def _normalized_obs(self, obs: deque, flatten: bool = True):
        """
        Normalize the observations

        Takes in a deque of observations and normalizes them
        And concatenates them into a single tensor of shape (n_envs, obs_horizon * obs_dim)
        """
        # Convert robot_state from obs_horizon x (n_envs, 14) -> (n_envs, obs_horizon, 14)
        robot_state = torch.cat([o["robot_state"].unsqueeze(1) for o in obs], dim=1)

        # Convert the robot_state to use rot_6d instead of quaternion
        robot_state = proprioceptive_quat_to_6d_rotation(robot_state)

        # Normalize the robot_state
        nrobot_state = self.normalizer(robot_state, "robot_state", forward=True)

        # Convert parts_poses from obs_horizon x (n_envs, 14) -> (n_envs, obs_horizon, 14)
        parts_poses = torch.cat([o["parts_poses"].unsqueeze(1) for o in obs], dim=1)

        # Normalize the parts_poses
        nparts_poses = self.normalizer(parts_poses, "parts_poses", forward=True)

        # Reshape concatenate the features
        nobs = torch.cat([nrobot_state, nparts_poses], dim=-1)

        if flatten:
            # (n_envs, obs_horizon, obs_dim) --> (n_envs, obs_horizon * obs_dim)
            nobs = nobs.flatten(start_dim=1)

        return nobs

    @torch.no_grad()
    def action(self, obs: deque):
        # Normalize observations
        nobs = self._normalized_obs(obs)

        # If the queue is empty, fill it with the predicted actions
        if not self.actions:
            # Predict normalized action
            naction = self.model(nobs).reshape(
                nobs.shape[0], self.pred_horizon, self.action_dim
            )

            # unnormalize action
            # (B, pred_horizon, action_dim)
            action_pred = self.normalizer(naction, "action", forward=False)

            # Add the actions to the queue
            # only take action_horizon number of actions
            start = self.obs_horizon - 1
            end = start + self.action_horizon
            for i in range(start, end):
                self.actions.append(action_pred[:, i, :])

        # Return the first action in the queue
        return self.actions.popleft()

    # === Training ===
    def _training_obs(self, batch, flatten: bool = True) -> torch.Tensor:
        nobs = batch["obs"]

        if flatten:
            # (n_envs, obs_horizon, obs_dim) --> (n_envs, obs_horizon * obs_dim)
            nobs = nobs.flatten(start_dim=1)

        return nobs

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # State already normalized in the dataset
        obs_cond = self._training_obs(batch, flatten=True)

        # Action already normalized in the dataset
        naction: torch.Tensor = batch["action"]

        # forward pass
        naction_pred: torch.Tensor = self.model(obs_cond).reshape(
            naction.shape[0], self.pred_horizon, self.action_dim
        )

        loss: torch.Tensor = self.loss_fn(naction_pred, naction)

        if loss > 1:
            print("Loss is greater than 1", loss.item())

        return loss

    def train_mode(self):
        """
        Set models to train mode
        """
        pass

    def eval_mode(self):
        """
        Set models to eval mode
        """
        pass

    def set_task(self, *args, **kwargs):
        """
        Set the task for the actor
        """
        pass


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SmallMLPAgent(nn.Module):
    def __init__(self, obs_shape: tuple, action_shape: tuple, init_logstd=0):
        super().__init__()

        assert (
            len(action_shape) == 2
        ), "Actions must be of shape (action_horizon, action_dim)"

        self.action_horizon, self.action_dim = action_shape

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_shape)), std=0.01),
            nn.Unflatten(1, action_shape),
        )
        self.actor_logstd = nn.Parameter(
            torch.ones(1, 1, self.action_dim) * init_logstd
        )

    def get_value(self, nobs: torch.Tensor) -> torch.Tensor:
        return self.critic(nobs)

    def get_action_and_value(self, nobs: torch.Tensor, action=None):
        action_mean: torch.Tensor = self.actor_mean(nobs)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action).sum(dim=(1, 2)),
            probs.entropy().sum(dim=(1, 2)),
            self.critic(nobs),
        )


class BigMLPAgent(SmallMLPAgent):
    """
    A bigger agent with more hidden layers than the SmallMLPAgent
    """

    def __init__(self, obs_shape, action_shape, init_logstd=0):
        super().__init__(obs_shape, action_shape, init_logstd)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, np.prod(action_shape)), std=0.01),
            nn.Unflatten(1, action_shape),
        )


class ResidualMLPAgent(nn.Module):

    def __init__(
        self, obs_shape: tuple, action_shape: tuple, init_logstd=0, dropout=0.1
    ):
        super().__init__()

        assert (
            len(action_shape) == 2
        ), "Actions must be of shape (action_horizon, action_dim)"

        self.action_horizon, self.action_dim = action_shape

        self.backbone_emb_dim = 1024

        self.backbone = MLP(
            input_dim=np.array(obs_shape).prod(),
            output_dim=self.backbone_emb_dim,
            hidden_dims=[1024] * 5,
            dropout=0.0,
            residual=True,
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.backbone_emb_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.value_head.apply(self.init_weights)

        self.action_head = nn.Sequential(
            nn.Linear(self.backbone_emb_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, np.prod(action_shape)),
            nn.Unflatten(1, action_shape),
        )

        self.action_head.apply(self.init_weights)

        self.actor_logstd = nn.Parameter(
            torch.ones(1, 1, self.action_dim) * init_logstd
        )

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def get_value(self, nobs: torch.Tensor) -> torch.Tensor:
        representation = self.backbone(nobs)
        return self.value_head(representation)

    def actor_mean(self, nobs: torch.Tensor) -> torch.Tensor:
        representation = self.backbone(nobs)
        return self.action_head(representation)

    def get_action_and_value(
        self, nobs: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        representation: torch.Tensor = self.backbone(nobs)
        action_mean: torch.Tensor = self.action_head(representation)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return (
            action,
            # The probs are calculated over a whole chunk as a single action
            # (batch, action_horizon, action_dim) -> (batch, )
            probs.log_prob(action).sum(dim=(1, 2)),
            probs.entropy().sum(dim=(1, 2)),
            self.value_head(representation),
        )


class ResidualMLPAgentSeparate(SmallMLPAgent):
    def __init__(self, obs_shape: tuple, action_shape: tuple, init_logstd=0):
        super().__init__(obs_shape, action_shape, init_logstd)

        self.actor_mean = nn.Sequential(
            MLP(
                input_dim=np.array(obs_shape).prod(),
                output_dim=np.prod(action_shape),
                hidden_dims=[1024] * 5,
                dropout=0.0,
                residual=True,
            ),
            nn.Unflatten(1, action_shape),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=0.01),
        )
