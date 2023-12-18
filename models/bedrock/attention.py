# attention.py

import torch
import torch.nn as nn
from typing import Callable, Optional, Union, Any
from .abstract_models import BaseModel
from .hook import HookContextManager
import fm

# will this return the self-attention weights twice?
AttentiveModule = (
    nn.MultiheadAttention, 
    # nn.TransformerEncoderLayer,
    # nn.TransformerDecoderLayer,
    fm.multihead_attention.MultiheadAttention,
)
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

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        if average_attn_weights is not None:
            kwargs["average_attn_weights"] = average_attn_weights
        return forward_orig(*args, **kwargs)

    m.forward = wrap


class AttentionWeightsHook(HookContextManager):
    """
    A hook that saves the attention weights of a transformer or attention layer.
    """
    def __init__(self, model):
        super().__init__(
            module=model,
            layer_type=AttentiveModule,
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
        Get the attention weights of the model, as saved by the hook.

        Returns:
        --------
        torch.Tensor: 
            The attention weights of the model, of shape 
            (*, n_layers, n_heads, seq_len, seq_len).
        """
        return torch.stack(self.outputs, dim=1)
    


def get_attn_layer(model : AttentiveModule) -> nn.Module:
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
    


def patch_forward_to_return_attn_weights(m : nn.Module):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        f = forward_orig(*args, **kwargs)
        return f, m.attention_weights.pop_weights()

    m.forward = wrap



# class AttentionWeightMetaClass(WeightInitialisationMetaClass):
#     """
#     A metaclass that automatically calls the _weight_init() method of a class
#     after all child classes are initialized. This is useful for initializing
#     the weights of a model after it is initialized.

#     Any class that inherits from this metaclass must implement the _weight_init()
#     method, and have a save_attention_weights attribute.

#     Inherits:
#     --------
#     type: 
#         A metaclass for creating classes.
#     """
#     def __call__(cls, *args, **kwargs):
#         # create an instance of the class using the __call__ method of the class
#         assert issubclass(cls, BaseAttentionModel), 'The class must inherit from BaseAttentionModel'
#         obj = WeightInitialisationMetaClass.__call__(cls, *args, **kwargs)

#         if hasattr(obj, 'attention_weights'):
#             # patch the model so that the attention layers return the attention weights
#             # and register a hook to save the attention weights
#             obj.patch_and_register_layer_hooks(
#                 layer_type=AttentiveModule,
#                 hook=obj.attention_weights,
#                 transform=None,
#                 patch=patch_attn_to_return_weights,
#             )

#             # patch the model so that the forward pass returns the attention weights
#             # patch_forward_to_return_attn_weights(obj)

#         return obj


class BaseAttentionModel(BaseModel):
    def __init__(self):
        super().__init__()

    
    def get_attn_weights(self, x : torch.Tensor) -> torch.Tensor:
        """
        Get the attention weights of each layer of the model when the given 
        input is passed through the model.

        Parameters:
        ----------
        x (torch.Tensor): 
            The input to the model, of shape (*, seq_len, d_model)

        Returns:
        --------
        torch.Tensor: 
            The attention weights of the model, of shape 
            (*, n_layers, n_heads, seq_len, seq_len)
        """
        with AttentionWeightsHook(self) as hook:
            self(x)
        return hook.get_weights()

