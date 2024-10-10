import torch

from src.behavior.base import Actor
from typing import Dict, Tuple, Union
from omegaconf import DictConfig, OmegaConf
from torch.autograd import Variable


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

        self.model = None
        self.loss_fn = None

    def _normalized_action(self, nobs: torch.Tensor) -> torch.Tensor:
        """Perform inference

        Args:
            nobs (torch.Tensor): Tensor of normalized observations.

        Returns:
            torch.Tensor: A tensor of action_horizon normalized actions.
        """
        pass

    def compute_loss(self, batch) -> Tuple[torch.Tensor, dict]:
        """Training"""
        loss_dict = {}
        loss = 0

        return loss, loss_dict


mu_var = torch.rand(3, 2)
logvar = torch.ones_like(mu_var)

print(kl_divergence(mu_var, logvar))
