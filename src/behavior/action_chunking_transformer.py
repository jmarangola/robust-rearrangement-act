import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra

from queue import Queue
from src.behavior.base import Actor
from typing import Tuple, Union
from omegaconf import DictConfig
from typing import Optional, Any, Callable, List, Iterable, Set, Type
from torch import Tensor
from copy import deepcopy


def _get_module_clones(module: object, n: int) -> nn.Module:
    """Utility function to clone a module n times by deepcopy."""
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns the specified activation function from torch.nn.functional or raises a RuntimeError for invalid input.
    Supports all activation functions implemented by torch.nn.functional.
    """
    if hasattr(F, activation):
        return getattr(F, activation)
    else:
        raise RuntimeError(f"Activation '{activation}' not implemented by torch.nn.functional.")


class ModuleConfigurator(nn.Module):
    """
    A utility class to dynamically initialize the network parameters from a config.
    """
    def __init__(self, module: object, cfg: DictConfig, args: Iterable[str], recursive: bool = True):
        super().__init__()
        self.set_parameters(module, cfg, set(args), recursive)
        assert all([hasattr(module, attr) for attr in args]), \
            f"Failed to construct '{module.__class__.__name__}'."

    def set_parameters(self, module: object, cfg: DictConfig, args: Set[str], find_recursively: bool):
        """
        Recursively set all parent non-dict keys as member variables in the module,
        and allows child configurations to override parent values.

        Args:
            module (nn.Module): The module to set the attributes for.
            cfg (DictConfig): The configuration to pull values from, if the root of the config is
                              not the module name, the module will be recursively found.
            args (Set[str]): A set of allowed arguments to enforce as attributes in the module.
        """
        module_name = module.__class__.__name__

        if find_recursively:
            for key, val in cfg.items():
                if isinstance(val, (DictConfig, dict)) and module_name not in cfg:
                    self.set_parameters(module, val, args, find_recursively)
                elif key in args and not isinstance(val, (DictConfig, dict)):
                    setattr(module, key, val)

        if module_name in cfg:
            module_cfg = cfg[module_name]
            for key, val in module_cfg.items():
                if key in args and not isinstance(val, (DictConfig, dict)):
                    setattr(module, key, val)

    def build(self, cfg: DictConfig, parent_type: Type, module_type: Type) -> nn.Module:
        return module_type(cfg[parent_type.__name__]) if parent_type is not None else module_type(cfg)


class TransformerEncoderLayer(ModuleConfigurator):
    def __init__(self,
                 cfg: DictConfig):
        args = (
            "dim_model",
            "num_heads",
            "dim_feedforward",
            "dropout",
            "activation",
            "normalize_before"
        )
        super().__init__(self, cfg, args)

        self.self_attn = nn.MultiheadAttention(self.dim_model, self.num_heads,
                                               dropout=self.dropout)

        self.linear1 = nn.Linear(self.dim_model, self.dim_feedforward)
        self.dropout0 = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.dim_feedforward, self.dim_model)

        self.norm1 = nn.LayerNorm(self.dim_model)
        self.norm2 = nn.LayerNorm(self.dim_model)
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

        self.activation = _get_activation_fn(self.activation)
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


class TransformerEncoder(ModuleConfigurator):
    def __init__(self, cfg: DictConfig):
        super().__init__(self, cfg, ("num_layers", "layer_norm"))

        encoder_layer = self.build(cfg, None, TransformerEncoderLayer)
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


class TransformerDecoderLayer(ModuleConfigurator):
    def __init__(self, cfg: DictConfig):
        args = (
            'dim_model',
            'num_heads',
            'dim_feedforward',
            'dropout',
            'normalize_before',
            'activation'
        )
        super().__init__(self, cfg, args)

        self.self_attn = nn.MultiheadAttention(self.dim_model, self.num_heads, dropout=self.dropout)
        self.multihead_attn = nn.MultiheadAttention(self.dim_model, self.num_heads, dropout=self.dropout)

        self.linear1 = nn.Linear(self.dim_model, self.dim_feedforward)
        self.dropout0 = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.dim_feedforward, self.dim_model)

        self.layernorm1 = nn.LayerNorm(self.dim_model)
        self.layernorm2 = nn.LayerNorm(self.dim_model)
        self.layernorm3 = nn.LayerNorm(self.dim_model)

        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        self.dropout3 = nn.Dropout(self.dropout)

        self.activation = _get_activation_fn(self.activation)
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


class TransformerDecoder(ModuleConfigurator):
    def __init__(self, cfg: DictConfig):
        super().__init__(self, cfg, ("num_layers",))

        transformer_decoder_layer = self.build(cfg, None, TransformerDecoderLayer)
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


class ActionTransformer(ModuleConfigurator):
    def __init__(self, cfg: DictConfig):
        super().__init__(self, cfg, ('dim_model',))

        self.encoder = self.build(cfg, type(self), TransformerEncoder)
        self.decoder = self.build(cfg, type(self), TransformerDecoder)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, mask, query_embed, pos_embed, latent_input=None, proprio_input=None, additional_pos_embedding=None):
        B = proprio_input.shape[0]
        src = torch.stack([latent_input, proprio_input], axis=0)

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


class ConditionalVAE(ModuleConfigurator):
    def __init__(self, cfg: DictConfig) -> None:
        """Action chunking with transformers policy.

        Args:
            cfg (DictConfig):
        """
        super().__init__(self, cfg, ('dim_state', 'dim_latent', 'dim_action'), recursive=False)

        self.action_transformer = self.build(cfg, type(self), ActionTransformer)
        self.encoder = self.build(cfg, type(self), TransformerEncoder)

        self.dim_hidden = self.action_transformer.dim_model
        self.action_head = nn.Linear(self.dim_hidden, self.dim_action)
        self.is_pad_head = nn.Linear(self.dim_hidden, 1)

        # Embeddings
        self.cls_embedding = nn.Embedding(1, self.dim_hidden)

        # Projections
        self.encoder_action_proj = nn.Linear(self.dim_action, self.dim_hidden)
        self.encoder_state_proj = nn.Linear(self.dim_state, self.dim_hidden)

        # Project to latent space represented by a mean and a variance ( * 2 )
        self.latent_proj = nn.Linear(self.dim_hidden, self.dim_latent * 2)

        self.input_proj_robot_state = nn.Linear(self.dim_state, self.dim_hidden)

        # CVAE decoder parameters
        self.latent_out_proj = nn.Linear(self.dim_latent, self.dim_hidden)
        # Additional embeddings for proprio and latent
        self.positional_embeddings = nn.Embedding(2, self.dim_hidden)

    def forward(self,
                normalized_obs: torch.Tensor,
                actions: Optional[torch.Tensor] = None,
                is_pad: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if actions is not None:
            B, _, _ = actions
            # Training
            # Actions: (B, OH, A)

            # Take first OH actions
            actions = actions[:, :self.obs_horizon]
            assert is_pad is not None, 'is pad cannot be None'
            is_pad = is_pad[:, :self.obs_horizon]

            action_embedding = self.encoder_action_proj(actions)
            obs_embedding = self.encoder_state_proj(normalized_obs)
            cls_embedding = self.cls_embedding.weight
            cls_embedding = torch.unsqueeze(cls_embedding, axis=0).repeat(B, 1, 1)
            encoder_input = torch.cat([cls_embedding, obs_embedding, action_embedding], axis=1)

            # Permute to (1, 0, 2)
            encoder_input = encoder_input.permute(1, 0, 2)

            # TODO: cleanup
            # this is really ugly
            cls_joint_is_pad = torch.full(B(B, 2), False).to(actions.device)
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)

            pos_embedding = self.pos_table.clone().detach()
            pos_embedding = pos_embedding.permute(1, 0, 2)

            enc_output = self.encoder(encoder_input) # TODO: fix
            cls_output = enc_output[0]

            latent_projection = self.latent_proj(enc_output)
            mu, logvar = torch.split(latent_projection, self.dim_latent, dim=1)

            # Sample from the latent space
            latent_sample = self.reparametrize(mu, logvar)

            # Re-project the sampled latent variable back to the desired embedding space
            latent_input = self.latent_out_proj(latent_sample)


        else:
            # Inference
            pass

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
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


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor):
    """Computes the kl-divergence between the approximate posterior q(z|x)
        and the standard normal prior N(0, 1).

    Args:
        mu (torch.Tensor): Mean of approximate posterior distribution, shape (B, latent_dim)
        logvar (torch.Tensor): Log-variance of approximate posterior distribution, shape (B, latent_dim)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
          - total_kld: The total kl divergence, summed over latent dims and averaged over the batch (minimizing this guy)
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


@hydra.main(config_path="../config/actor", config_name="act")
def load_cfg(cfg: DictConfig):
    # print(cfg.model)
    tk = ConditionalVAE(cfg)
    print(tk)
# print(tk)
load_cfg()
