import torch
import torch.nn as nn
import torch.nn.functional as F

from src.behavior.base import Actor
from typing import Dict, Tuple, Union
from omegaconf import DictConfig, OmegaConf
from torch.autograd import Variable
from typing import Optional, Any, Callable
from torch import Tensor
from copy import deepcopy


def reparametrize(mu, logvar):
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

# TODO: Add positional encodings for (T, J, V) temporal settings


def _get_module_clones(module: nn.Module, n: int) -> nn.Module:
    """Utility function to clone a module n times by deepcopy."""
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns the specified activation function from torch.nn.functional or raises a RuntimeError for invalid input.
    Supports all activation functions implemented by torch.nn.functional.

    Args:
        activation (str): The name of the activation function to retrieve (e.g., 'relu', 'gelu', 'glu').

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: The corresponding activation function.

    Raises:
        RuntimeError: If the provided activation string is not implemented in torch.nn.functional.
    """
    if hasattr(F, activation):
        return getattr(F, activation)
    else:
        raise RuntimeError(f"Activation '{activation}' not implemented by torch.nn.functional.")


class ModuleConfigurator():
    """
    A utility class to initialize attributes from a config.
    """
    def __init__(self, cltype: type, cfg: DictConfig, args: List[str]):
        clname = cltype.__name__
        if (module_cfg := self.find_recursive(cfg, clname)) is None:
            raise ValueError(f"ModuleConfigurator failed to find {clname} in input cfg '{cfg}'.")
        for arg in args:
            if arg in module_cfg:
                setattr(self, arg, module_cfg[arg])
            else:
                raise ValueError(f"Argument {arg} not found in configuration.")

    def find_recursive(self, cfg: DictConfig, module_name: str) -> DictConfig:
        """
        Recursively searches through a DictConfig for a specific module config.

        Args:
            cfg (DictConfig): The configuration to search.
            module_name (str): The module name that is being searched for.

        Returns:
            DictConfig: The config associated with the key if found, otherwise None.
        """
        if module_name in cfg:
            return cfg[module_name]
        for key in set(cfg.keys()):
            value = cfg[key]
            if isinstance(value, (DictConfig, dict)):
                result = self.find_recursive(value, module_name)
                if result is not None:
                    return result
        return None


class TransformerEncoderLayer(nn.Module, ModuleConfigurator):
    def __init__(self,
                 cfg: DictConfig):
        args = [
            "dim_model",
            "num_heads",
            "dim_feedforward",
            "dropout",
            "activation",
            "normalize_before"
        ]
        super().__init__()
        ModuleConfigurator.__init__(self, self.__class__, cfg, args)

        self.self_attn = nn.MultiheadAttention(self.dim_model, self.num_heads, dropout=self.dropout)

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


class TransformerEncoder(nn.Module, ModuleConfigurator):
    def __init__(self, cfg: DictConfig):
        # TODO: Allow encoder to dynamically take different types of transformer blocks !
        args = ["num_layers", "layer_norm"]
        super().__init__()
        ModuleConfigurator.__init__(self, self.__class__, cfg, args)

        encoder_layer = TransformerEncoderLayer(cfg)
        self.layers = _get_module_clones(encoder_layer, self.num_layers)
        self.num_layers = self.num_layers
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

# TODO: Move network to a base implementation separate from this class derived from the Actor.
# This is an ugly MVP to get things working intially.
class ActionChunkingTransformerPolicy(Actor):
    def __init__(self,
                 device: Union[str, torch.device],
                 cfg: DictConfig,
                 ) -> None:
        """Action chunking with transformers policy.

        Args:
            device (Union[str, torch.device]): Target device for model.
            cfg (DictConfig):
        """
        super().__init__(device, cfg)
        # TODO: impl

    def _normalized_action(self, nobs: torch.Tensor) -> torch.Tensor:
        """Perform inference

        Args:
            nobs (torch.Tensor): Tensor of normalized observations.

        Returns:
            torch.Tensor: A tensor of action_horizon normalized actions.
        """
        pass
        # TODO: impl

    def compute_loss(self, batch) -> Tuple[torch.Tensor, dict]:
        """Training"""
        # TODO: impl
        pass

        # return loss, loss_dict


mu_var = torch.rand(3, 2)
logvar = torch.ones_like(mu_var)

print(kl_divergence(mu_var, logvar))
