import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import numpy as np

from queue import Queue
from src.behavior.base import Actor
from typing import Tuple, Union
from omegaconf import DictConfig
from typing import Optional, Any, Callable, List, Iterable, Set, Type, Dict
from torch import Tensor
from copy import deepcopy


def _get_module_clones(module: object, n: int) -> nn.Module:
    """Utility function to clone a module n times by deepcopy."""
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


def _get_nnf_fn(activation: str) -> Callable[[Tensor], Tensor]:
    """
    Returns the specified function from torch.nn.functional or raises a RuntimeError for invalid input.
    Supports all functions implemented by torch.nn.functional.
    """
    if hasattr(F, activation):
        return getattr(F, activation)
    else:
        raise RuntimeError(f"Activation '{activation}' not implemented by torch.nn.functional.")


class CModule(nn.Module):
    """
    A utility class to dynamically initialize the network parameters from a config.
    """
    def __init__(self, module: object, cfg: DictConfig):
        super().__init__()
        if getattr(module, 'params', None) is not None:
            self.set_parameters(module, cfg)
            assert all([hasattr(module, attr) for attr in module.params]), \
                f"Failed to construct '{module.__class__.__name__}'."

    def set_parameters(self, module: object, cfg: DictConfig):
        """
        Sets module configuration parameters from the configuration from an iterable
        of acceptable parameters, each given by self.params.

        Args:
            module (object): Target module.
            cfg (DictConfig): Config containing parameter values.
        """
        for k, v in cfg.items():
            if not isinstance(v, (DictConfig, dict)) and k in module.params:
                setattr(module, k, v)


def build(module_type: Type, cfg: DictConfig) -> nn.Module:
    """
    Builds a nn.Module of type module_type automatically from a config.
    """
    return module_type(cfg[module_type.__name__])


# TODO: Refactor. Ugly implementation.
class TransformerEncoderLayer(CModule):
    def __init__(self,
                 cfg: DictConfig):
        self.params = (
            "dim_model",
            "num_heads",
            "dim_feedforward",
            "dropout",
            "activation",
            "normalize_before"
        )
        super().__init__(self, cfg)

        self.self_attn = nn.MultiheadAttention(self.dim_model, self.num_heads,
                                               dropout=self.dropout)

        self.linear1 = nn.Linear(self.dim_model, self.dim_feedforward)
        self.dropout0 = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.dim_feedforward, self.dim_model)

        self.norm1 = nn.LayerNorm(self.dim_model)
        self.norm2 = nn.LayerNorm(self.dim_model)
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

        self.activation = _get_nnf_fn(self.activation)
        self.normalize_before = self.normalize_before

    def with_pos_embed(self, input: Tensor, pos: Optional[Tensor] = None):
        return input if pos is None else input + pos

    def forward_post(self,
                     src,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout0(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    def forward_pre(self, src, pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout0(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src

    def forward(self,
                src,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, pos)
        return self.forward_post(src, pos)


class TransformerEncoder(CModule):
    def __init__(self, cfg: DictConfig):
        self.params = ("num_layers", "layer_norm")
        super().__init__(self, cfg)

        encoder_layer = build(TransformerEncoderLayer, cfg)
        self.layers = _get_module_clones(encoder_layer, self.num_layers)
        self.norm = nn.LayerNorm(encoder_layer.dim_model) if self.layer_norm else False

    def forward(self, src, pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)

        if self.norm:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(CModule):
    def __init__(self, cfg: DictConfig):
        self.params = (
            'dim_model',
            'num_heads',
            'dim_feedforward',
            'dropout',
            'normalize_before',
            'activation'
        )
        super().__init__(self, cfg)

        self.self_attn = nn.MultiheadAttention(
            self.dim_model, self.num_heads, dropout=self.dropout
        )
        self.multihead_attn = nn.MultiheadAttention(
            self.dim_model, self.num_heads, dropout=self.dropout
        )

        self.linear1 = nn.Linear(self.dim_model, self.dim_feedforward)
        self.dropout0 = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.dim_feedforward, self.dim_model)

        self.layernorm1 = nn.LayerNorm(self.dim_model)
        self.layernorm2 = nn.LayerNorm(self.dim_model)
        self.layernorm3 = nn.LayerNorm(self.dim_model)

        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        self.dropout3 = nn.Dropout(self.dropout)

        self.activation = _get_nnf_fn(self.activation)
        self.normalize_before = self.normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.layernorm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.layernorm2(tgt)
        tgt2 = self.linear2(self.dropout0(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.layernorm3(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.layernorm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.layernorm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.layernorm3(tgt)
        tgt2 = self.linear2(self.dropout0(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

    def forward(self, tgt, memory,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, pos, query_pos)
        return self.forward_post(tgt, memory, pos, query_pos)


class TransformerDecoder(CModule):
    def __init__(self, cfg: DictConfig):
        self.params = ('num_layers', 'return_intermediate')
        super().__init__(self, cfg)

        transformer_decoder_layer = build(TransformerDecoderLayer, cfg)
        self.layers = _get_module_clones(transformer_decoder_layer, self.num_layers)
        self.layer_norm = nn.LayerNorm(transformer_decoder_layer.dim_model)

    def forward(self, tgt: Tensor, memory: Tensor, pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.layer_norm(output))

        output = self.layer_norm(output)

        if self.return_intermediate:
            intermediate.pop()
            intermediate.append(output)
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class ActionTransformer(CModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(self, cfg)

        self.encoder = build(TransformerEncoder, cfg)
        self.decoder = build(TransformerDecoder, cfg)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query_embed: Tensor,
                object_state_input: Tensor,
                latent_input: Tensor,
                proprio_input: Tensor,
                additional_modal_pos_embedding: Tensor) -> Tensor:
        B = proprio_input.shape[0]

        src = torch.stack([latent_input, proprio_input, object_state_input], axis=0)

        # Modal pos embed shape: torch.Size([2, batch_size, hidden_dim])
        modal_pose_embed = additional_modal_pos_embedding.unsqueeze(1).repeat(1, B, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, pos=modal_pose_embed)
        hs = self.decoder(tgt, memory, pos=modal_pose_embed, query_pos=query_embed)
        hs = hs.transpose(1, 2)

        return hs


class ConditionalVAE(CModule):
    def __init__(self,
                 cfg: DictConfig,
                 dim_action: int,
                 dim_object_state: int,
                 dim_robot_state: int) -> None:
        """Conditional variational autoencoder for training an action chunking policy.

        Args:
            cfg (DictConfig):
        """
        self.params = ('dim_latent',)

        module_conf = cfg[self.__class__.__name__]
        super().__init__(self, module_conf)

        self.pred_horizon = cfg.pred_horizon
        self.obs_horizon = cfg.obs_horizon

        assert self.obs_horizon == 1, "Markov property is assumed by ACT CVAE."
        self.action_transformer = build(ActionTransformer, module_conf)

        # CVAE encoder, the word encoder is a bit overloaded here. There is an encoder
        # here and subsequently, another transformer encoder in the action transformer.
        # this encoder is just used for training and serves no other purpose than to train
        # the ACT as a CVAE.
        self.encoder = build(TransformerEncoder, module_conf)

        self.dim_hidden = self.action_transformer.encoder.layers[0].dim_model
        self.action_head = nn.Linear(self.dim_hidden, dim_action)

        # Learnable embeddings for action chunking
        self.query_embedding = nn.Embedding(self.pred_horizon, self.dim_hidden)

        # Additional CVAE encoder parameters
        self.global_context_embedding = nn.Embedding(1, self.dim_hidden)
        self.encoder_action_proj = nn.Linear(dim_action, self.dim_hidden)
        self.encoder_state_proj = nn.Linear(dim_robot_state, self.dim_hidden)
        self.latent_proj_hidden = nn.Linear(self.dim_hidden, self.dim_latent * 2)  # Project to latent space represented by a mean and a variance ( * 2 )
        self.input_proj_robot_state = nn.Linear(dim_robot_state, self.dim_hidden)
        # Projection from object state to decoder input space
        self.input_proj_obj_state = nn.Linear(dim_object_state, self.dim_hidden)

        # Register a table for positional encodings which is instantiated as a buffer so it does not get updated during backprop
        # We can leverage the same positional encoding scheme for actions as we do for observations (non-Markovian case)
        self.register_positional_enc_table(
            'pos_enc_table', 2 + self.pred_horizon, self.dim_hidden
        )

        # Additional CVAE decoder parameters
        self.latent_out_proj = nn.Linear(self.dim_latent, self.dim_hidden)

        # Additional positional embeddings for proprio, latent and object state
        # Each corresponding corresponds to a different type of input feature (proprio, latent or object state)
        # and the model can leverage these positional embeddings to learn to distinguish between these features and learn higher level relationships between them
        self.positional_embeddings = nn.Embedding(3, self.dim_hidden)

    def forward(self, robot_state: Tensor, obj_state: Tensor,
                actions: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass for the ACT as a CVAE.
        
        Args: 
            robot_state (Tensor): Flattened robot state (B, R).
            obj_state (Tensor): Flattened object state (B, O).
            actions (Tensor): Expert actions (B, pred_horizon, A).
            
        Returns: 
             Tuple[Tensor, Tensor, Tensor]: Output is a tuple containing: 
                [0] Sequence of predicted actions, ahat (B, pred_horizon, A).
                [1] Mean (mu) of the latent representation outputted by the encoder during training, or None otherwise.
                [2] Log-variance (logvar) of the latent distribution outputted by the encoder during training, or None otherwise.
        """
        is_training = actions is not None

        if is_training:
            B, _, _ = actions.shape

            action_embedding = self.encoder_action_proj(actions)
            robot_state_embedding = self.encoder_state_proj(robot_state).unsqueeze(1)

            global_context = self.global_context_embedding.weight
            global_context = torch.unsqueeze(global_context, axis=0).repeat(B, 1, 1)

            # Construct the input to the encoder
            # Note: here the object state is not encoded into the input. Instead, object state is treated as a conditional
            # variable that informs the decoder about contextual information in the environment without cluttering the
            # learned latent space.
            encoder_input = torch.cat(
                [global_context, robot_state_embedding, action_embedding], axis=1
            ).permute(1, 0, 2)

            # Pass through the encoder, take [global_context] output only
            encoder_output = self.encoder(encoder_input, pos=self.pos_enc_table)
            encoder_output = encoder_output[0]

            latent_embed = self.latent_proj_hidden(encoder_output)
            mu, logvar = torch.split(latent_embed, self.dim_latent, dim=1)

            # Use reparamatrization trick to efficiently sample from the posterior
            latent_sample = self.reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            # Inference
            mu = logvar = None
            B = robot_state.shape[0]
            latent_sample = torch.zeros([B, self.dim_latent],
                                        dtype=torch.float32).to(robot_state.device)
            latent_input = self.latent_out_proj(latent_sample)

        obj_state_embedding = self.input_proj_obj_state(obj_state)
        robot_state_decoder_input_embedding = self.input_proj_robot_state(robot_state)

        # Forward pass through CVAE decoder
        hs = self.action_transformer(self.query_embedding.weight,
                                     obj_state_embedding,
                                     latent_input,
                                     robot_state_decoder_input_embedding,
                                     self.positional_embeddings.weight)[0]

        a_hat = self.action_head(hs)
        return a_hat, mu, logvar

    def register_positional_enc_table(self, name: str, n_position: int, d_hid: int):
        """
        Generates a sinusoidal positional encoding table.

        Args:
            n_position (int): Number of positions (sequence length)
            d_hid (int): Dimension of the hidden representations.

        Returns:
            Tensor: Positional encoding table (seq + 1, 1, dim_hidden)
        """
        position = torch.arange(n_position).unsqueeze(1)
        div_term = torch.pow(10000, -torch.arange(0, d_hid, 2).float() / d_hid)

        # Compute the sinusoid values (even indices get sin, odd indices get cos)
        sinusoid_table = torch.zeros(n_position, d_hid)
        sinusoid_table[:, 0::2] = torch.sin(position.float() * div_term)
        sinusoid_table[:, 1::2] = torch.cos(position.float() * div_term)

        self.register_buffer(name, sinusoid_table.unsqueeze(0).permute(1, 0, 2))

    def reparametrize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample from p(z) using the reparameterization trick.

        Args:
            mu (Tensor): Mean of the distribution.
            logvar (Tensor): Log-variance of the distribution.

        Returns:
            Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps


def kl_divergence(mu: Tensor, logvar: Tensor):
    """Computes the kl-divergence between the approximate posterior q(z|x)
        and the standard normal prior N(0, 1).

    Args:
        mu (Tensor): Mean of approximate posterior distribution, shape (B, latent_dim)
        logvar (Tensor): Log-variance of approximate posterior distribution, shape (B, latent_dim)

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
          - total_kld: The total kl divergence, summed over latent dims and averaged over the batch.
          - dimension_wise_kld: KL divergence per latent dimension, averaged across the batch.
          - mean_kld: Mean KL divergence, averaged over both batch and latent dimension.
    """
    batch_size = mu.size(0)
    assert batch_size != 0, "Batch size must be non-zero to compute KL-divergence"

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class ACTPolicy(Actor):
    def __init__(
        self,
        device: Union[str, torch.device],
        cfg: DictConfig,
    ) -> None:
        super().__init__(device, cfg)
        self.actor_cfg = cfg.actor

        # TODO: experiment with beta scheduling
        self.beta_kl = cfg.actor.beta_kl
        self.ewise_reconstruction_loss_fn = _get_nnf_fn(
            cfg.actor.ewise_reconstruction_loss_fn
        )

        self.model = ConditionalVAE(
            cfg.actor,
            self.action_dim,
            self.parts_poses_dim,
            self.robot_state_dim
        ).to(device)

    # === Inference ===
    def _normalized_action(self, nobs: Tensor) -> Tensor:
        """
        Perform action chunking with temporal ensembling to generate a chunk of actions given the observations.
        """
        B, OH = nobs.shape[:2]

        robot_state = nobs[..., :self.robot_state_dim]
        object_state = nobs[..., self.robot_state_dim:]

        a_hat, _, _ = self.model(robot_state, object_state, None)

        return a_hat

    def compute_loss(self,
                     batch: Dict[str, Tensor]) -> Tuple[Tensor, dict]:
        """Compute the loss during training.

        Args:
            batch (Dict[str, Tensor]): Dictionary of tensors containing batch of normalized actions
              and observations.

        Returns:
            Tuple[torch.Tensor, dict]: Scalar loss and loss dictionary.
        """
        # Note: State already normalized in the dataset
        obs_cond = self._training_obs(batch, flatten=True)
        naction = batch["action"]
        B, AH, _ = naction.shape

        # Obtain robot state and flatten for now
        # TODO: generalize for non-markovian case where observation horizon \neq 1
        robot_state = obs_cond[..., :self.robot_state_dim]
        object_state = obs_cond[..., self.robot_state_dim:]

        a_hat, mu, logvar = self.model(robot_state, object_state, naction)

        # Enforce the prior
        total_kld, _, _ = kl_divergence(mu, logvar)

        # Compute the action-expert action reconstruction loss
        mean_recons_loss = self.compute_reconstruction_loss(a_hat, naction)

        loss = mean_recons_loss + self.beta_kl * total_kld

        return loss, {
            'bc_loss': loss.item(),
            'total_kld_loss': total_kld.item(),
            'reconstruction_loss': mean_recons_loss.item()
        }

    def compute_reconstruction_loss(self, a_hat: Tensor, action_gt: Tensor) -> Tensor:
        reconstruction_loss = self.ewise_reconstruction_loss_fn(a_hat, action_gt)
        return reconstruction_loss
