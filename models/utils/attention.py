# attention.py

import torch
import torch.nn as nn
from typing import Callable, Optional, Union, Any
from .hook import HookContextManager
from ...visualisation.plotting import image_grid

def patch_attn_to_return_weights(
        m : nn.MultiheadAttention,
        average_attn_weights : Optional[bool] = None,
        ) -> None:
    """
    Force the attention module to return the attention weights.

    Parameters:
    ----------
    m (nn.MultiheadAttention): 
        The attention module.
    average_attn_weights (bool | None):
        Whether to average the attention weights across the heads. If None,
        then the default setting of the attention module is used. The reason 
        to use None is that some attention modules do not have this setting.
    """
    forward_orig = m.forward

    def patched_forward(*args, **kwargs):
        kwargs["need_weights"] = True
        if average_attn_weights is not None:
            kwargs["average_attn_weights"] = average_attn_weights
        return forward_orig(*args, **kwargs)

    m.forward = patched_forward


class AttentionWeightsHook(HookContextManager):
    """
    A hook that saves the attention weights of a transformer or attention layer.
    The weights are saved in a list, which can be retrieved using the
    get_weights method.

    Parameters:
    ----------
    model (nn.Module):
        The model to register the hook on.
    layer_type (tuple[type[nn.Module]]):
        The type of layer to register the hook on.

    Attributes:
    ----------
    outputs (list[torch.Tensor]):
        A list to store the attention weights.
    """
    def __init__(self, model : nn.Module, layer_type : tuple[type[nn.Module]]):
        super().__init__(
            module=model,
            layer_type=layer_type,
            patch=patch_attn_to_return_weights,
            )
        # a list to store the attention weights
        self.outputs = []


    def __call__(
            self, 
            module : nn.Module, 
            module_in : Any, 
            module_out : Any,
            ) -> None:
        """
        Save the attention weights of the given module.
        
        Parameters:
        ----------
        module (nn.Module):
            The module whose attention weights are to be saved.
        module_in (Any):
            The input to the module.
        module_out (Any):
            The output of the module.
        """
        self.outputs.append(
            module_out[1].detach()
            )


    def get_weights(self) -> torch.Tensor:
        """
        Get the attention weights of the model, as saved by the hook. The model
        must be run before calling this method.

        Returns:
        --------
        torch.Tensor: 
            The attention weights of the model, of shape 
            (*, n_layers, n_heads, seq_len, seq_len).
        """
        return torch.stack(self.outputs, dim=1)
    

    def attn_weights(self, x : torch.Tensor) -> torch.Tensor:
        """
        Get the attention weights of the model, as saved by the hook, when the
        given input is passed through the model.

        Parameters:
        ----------
        x (torch.Tensor):
            The input to the model.

        Returns:
        --------
        torch.Tensor: 
            The attention weights of the model, of shape 
            (*, n_layers, n_heads, seq_len, seq_len).
        """
        self.module(x)
        return self.get_weights()
    


def get_attn_layer(model : type[nn.Module]) -> nn.Module:
    """
    Get the attention layer of the given model, which should be an instance of
    one of the following classes: nn.MultiheadAttention, 
    nn.TransformerEncoderLayer, nn.TransformerDecoderLayer.

    Parameters:
    ----------
    model (AttentiveModule): 
        The model to get the attention layer from.

    Returns:
    --------
    nn.Module: 
        The attention layer of the model. If the model is an instance of
        nn.MultiheadAttention, then the model itself is returned. Else, 
        the self-attention module of the model is returned.
    """
    return model if isinstance(model, nn.MultiheadAttention) else model.self_attn



def plot_attention_weights(
        model : nn.Module,
        x : torch.Tensor,
        attn_layer_type : type[nn.Module] = nn.MultiheadAttention,
        ) -> None:
    """
    Plot the attention weights of the given model when the given input is passed
    through the model.

    Parameters:
    ----------
    model (nn.Module):
        The model to plot the attention weights of.
    x (torch.Tensor):
        The input to the model.
    attn_layer_type (type[nn.Module]):
        The type of attention layer to plot the attention weights of.
    """
    with AttentionWeightsHook(model, attn_layer_type) as hook:
       attn_weights = hook.attn_weights(x).squeeze()
    image_grid(attn_weights.numpy())