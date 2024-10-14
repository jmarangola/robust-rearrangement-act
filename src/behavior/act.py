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
        # TODO: add docstring
        for k, v in cfg.items():
            if not isinstance(v, (DictConfig, dict)) and k in module.params:
                setattr(module, k, v)


def build(module_type: Type, cfg: DictConfig) -> nn.Module:
    return module_type(cfg[module_type.__name__])


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

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout0(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    def forward_pre(self,
                    src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout0(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(CModule):
    def __init__(self, cfg: DictConfig):
        self.params = ("num_layers", "layer_norm")
        super().__init__(self, cfg)

        encoder_layer = build(TransformerEncoderLayer, cfg)
        self.layers = _get_module_clones(encoder_layer, self.num_layers)
        self.norm = nn.LayerNorm(encoder_layer.dim_model) if self.layer_norm else False

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

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

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.layernorm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.layernorm2(tgt)
        tgt2 = self.linear2(self.dropout0(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.layernorm3(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.layernorm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.layernorm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.layernorm3(tgt)
        tgt2 = self.linear2(self.dropout0(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoder(CModule):
    def __init__(self, cfg: DictConfig):
        self.params = ("num_layers",)
        super().__init__(self, cfg)

        transformer_decoder_layer = build(TransformerDecoderLayer, cfg)
        self.layers = _get_module_clones(transformer_decoder_layer, self.num_layers)
        self.layer_norm = nn.LayerNorm(transformer_decoder_layer.dim_model)
        self.return_intermediate = False  # TODO parametrize

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
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

    def forward(self, mask, query_embed, pos_embed, object_state_input,
                latent_input=None, proprio_input=None,
                additional_pos_embedding=None):
        B = proprio_input.shape[0]

        src = torch.stack([latent_input, proprio_input, object_state_input], axis=0)

        # pos embed shape: torch.Size([2, batch_size, hidden_dim])
        pos_embed = additional_pos_embedding.unsqueeze(1).repeat(1, B, 1)
        # query embed shape: torch.Size([num_queries, batch_size, hidden_dim])
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        hs = hs.transpose(1, 2)

        return hs


class ConditionalVAE(CModule):
    def __init__(self, cfg: DictConfig) -> None:
        """Conditional variational autoencoder for training an action chunking policy.

        Args:
            cfg (DictConfig):
        """
        self.params = (
            'dim_latent', 'dim_action', 'dim_object_state', 'dim_robot_state'
        )
        module_conf = cfg[self.__class__.__name__]
        super().__init__(self, module_conf)

        self.pred_horizon = cfg.pred_horizon
        self.action_transformer = build(ActionTransformer, module_conf)
        self.encoder = build(TransformerEncoder, module_conf)

        self.dim_hidden = self.action_transformer.encoder.layers[0].dim_model
        self.action_head = nn.Linear(self.dim_hidden, self.dim_action)
        self.is_pad_head = nn.Linear(self.dim_hidden, 1)

        # Learnable embeddings for action chunking
        self.query_embedding = nn.Embedding(self.pred_horizon, self.dim_hidden)

        # Additional CVAE encoder parameters
        self.cls_embedding = nn.Embedding(1, self.dim_hidden)
        self.encoder_action_proj = nn.Linear(self.dim_action, self.dim_hidden)
        self.encoder_state_proj = nn.Linear(self.dim_robot_state, self.dim_hidden)
        self.latent_proj_hidden = nn.Linear(self.dim_hidden, self.dim_latent * 2)  # Project to latent space represented by a mean and a variance ( * 2 )
        self.input_proj_robot_state = nn.Linear(self.dim_robot_state, self.dim_hidden)
        # Projection from object state to decoder input space
        self.input_proj_obj_state = nn.Linear(self.dim_object_state, self.dim_hidden)

        # TODO: cleanup -- this is only called once so not a big deal, but still really ugly
        self.register_buffer('pos_table', self.get_sinusoid_encoding_table(2 + self.pred_horizon, self.dim_hidden))  # [CLS], qpos, a_seq

        # Additional CVAE decoder parameters
        self.latent_out_proj = nn.Linear(self.dim_latent, self.dim_hidden)
        self.positional_embeddings = nn.Embedding(3, self.dim_hidden)  # additional embeddings for proprio, latent and object state

    def forward(self,
                robot_state: Tensor,
                obj_state: Tensor,
                actions: Optional[Tensor] = None,
                is_pad: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:

        # Training
        # Actions: (B, OH, A)
        is_training = actions is not None
        if is_training:
            B, _, _ = actions.shape

            action_embedding = self.encoder_action_proj(actions)  # (B, seq, hidden)

            robot_state_embedding = self.encoder_state_proj(robot_state)  # (B, hidden)
            robot_state_embedding = torch.unsqueeze(robot_state_embedding, axis=1)  # (B, 1, hidden)

            cls_embedding = self.cls_embedding.weight  # (1, hidden)
            cls_embedding = torch.unsqueeze(cls_embedding, axis=0).repeat(B, 1, 1)  #  (B, 1, hidden)

            # Construct the input to the encoder
            encoder_input = torch.cat([cls_embedding, robot_state_embedding, action_embedding], axis=1)  #  (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim) THIS shape is wrong?

            # Do not mask [CLS]
            cls_joint_is_pad = torch.full((B, 2), False).to(robot_state.device)  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)   #  (B, seq + 1)

            # Get positional encoding
            # TODO: cleanup
            # this is really ugly
            pos_enc = self.pos_table.clone().detach()
            pos_enc = pos_enc.permute(1, 0, 2)  # (seq+1, 1, hidden)

            # Pass through the encoder
            encoder_output = self.encoder(encoder_input, pos=pos_enc,
                                          src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]  # Take [CLS] output only

            latent_embed = self.latent_proj_hidden(encoder_output)
            mu, logvar = torch.split(latent_embed, self.dim_latent, dim=1)

            # Use reparamatrization trick to efficiently sample from the posterior
            # for the decoder
            # (B, hidden)
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

        # Forward pass (latent_embedding, decoder_state_embedding) though the CVAE decoder
        robot_state_decoder_input_embedding = self.input_proj_robot_state(robot_state)
        hs = self.action_transformer(None,
                                     self.query_embedding.weight,
                                     None,
                                     obj_state_embedding,
                                     latent_input=latent_input,
                                     proprio_input=robot_state_decoder_input_embedding,
                                     additional_pos_embedding=self.positional_embeddings.weight)[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        return a_hat, is_pad_hat, mu, logvar

    def get_sinusoid_encoding_table(self, n_position: int, d_hid: int):
        # TODO: this will break torchscript!
        def get_position_angle_vec(position: np.ndarray):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        # TODO: cleanup -- this is ugly
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

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

    Note: This function  WILL need modifications in the form of flattening/reshaping to extend to latent representations of images
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
        # TODO: implement beta scheduling
        self.beta_kl = cfg.actor.beta_kl
        self.ewise_reconstruction_loss_fn = _get_nnf_fn(
            cfg.actor.ewise_reconstruction_loss_fn
        )
        self.model = ConditionalVAE(cfg.actor).to(device)

    # === Inference ===
    def _normalized_action(self, nobs: Tensor) -> Tensor:
        """
        Perform action chunking with temporal ensembling to generate a chunk of actions given the observations.
        """
        B, AH = nobs.shape[:2]

        # If the observation is not flattened, we need to reshape it to (B, obs_horizon, obs_dim)
        if not self.flatten_obs and len(nobs.shape) == 2:
            nobs = nobs.reshape(B, self.obs_horizon, self.obs_dim)

        # MVP for now
        robot_state = nobs[:, :self.robot_state_dim]
        object_state = nobs[:, self.robot_state_dim:]

        is_pad = torch.zeros((B, AH), device=self.device)
        is_pad[:, self.model.pred_horizon:] = 1
        is_pad = is_pad.type(torch.bool)

        a_hat, _, _, _ = self.model(robot_state, object_state, None, is_pad)

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

        is_pad = torch.zeros((B, AH), device=self.device)
        is_pad[:, self.model.pred_horizon:] = 1
        is_pad = is_pad.type(torch.bool)

        # Obtain robot state and object state
        robot_state = obs_cond[:, :self.robot_state_dim]
        object_state = obs_cond[:, self.robot_state_dim:]

        a_hat, _, mu, logvar = self.model(robot_state, object_state,
                                          naction[:, :self.pred_horizon],
                                          is_pad[:, :self.pred_horizon])

        # Enforce the prior
        total_kld, _, _ = kl_divergence(mu, logvar)

        # Compute the action reconstruction loss, discounting padded actions
        mean_recons_loss = self.compute_reconstruction_loss(a_hat, naction, is_pad)

        loss = mean_recons_loss + self.beta_kl * total_kld

        return loss, {
            'bc_loss': loss.item(),
            'total_kld_loss': total_kld.item(),
            'reconstruction_loss': mean_recons_loss.item()
        }

    def compute_reconstruction_loss(self, a_hat: Tensor, action_gt: Tensor,
                                    is_pad: Tensor) -> Tensor:
        action_gt = action_gt[:, :a_hat.size(1)]
        is_pad = is_pad[:, :a_hat.size(1)]

        ignore_mask = ~is_pad.unsqueeze(-1)
        reconstruction_loss = self.ewise_reconstruction_loss_fn(a_hat * ignore_mask, action_gt * ignore_mask)

        return reconstruction_loss




