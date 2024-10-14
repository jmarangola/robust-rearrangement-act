import torch

from act import ACTPolicy
from omegaconf import DictConfig


class ResidualACT(ACTPolicy):
   def __init__(self, device: torch.device,
                cfg: DictConfig):
         super().__init__(device, cfg)

