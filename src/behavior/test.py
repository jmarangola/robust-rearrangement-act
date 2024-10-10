import torch
import torch.optim as optim
import torch.nn.functional as F
import hydra

from omegaconf import DictConfig
from torch import nn, Tensor
from typing import Optional, Callable, List
from copy import deepcopy


def _get_module_clones(module: nn.Module, n: int) -> nn.Module:
    """Utility function to clone a module n times by deepcopy."""
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns the activation function ('relu', 'gelu', or 'glu') or raises a RuntimeError for invalid input.

    Raises:
        RuntimeError: If the provided activation string is not 'relu', 'gelu', or 'glu'.
    """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "glu":
        return F.glu
    else:
        raise RuntimeError(f"Activation should be 'relu', 'gelu', or 'glu', not {activation}.")


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



@hydra.main(config_path="../config/actor", config_name="act")
def load_cfg(cfg: DictConfig):
    # print(cfg.model)
    tk = TransformerEncoder(cfg.model)
    print(tk)
# print(tk)
load_cfg()
