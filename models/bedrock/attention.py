# attention.py

import torch.nn as nn
from typing import Callable, Optional, Union
from .abstract_models import BaseModel, WeightInitialisationMetaClass
import fm


def patch_and_register_layer_hooks(
        model : nn.Module, 
        layer_type : type[nn.Module], 
        hook : Callable,
        transform : Optional[Callable] = None,
        patch : Optional[Callable] = None,
        ) -> None:
    """
    Register a hook on all layers of the given model that are of the given type.
    Along the way, optionally patch the layers with the given patch function.

    Parameters:
    ----------
    model (nn.Module): 
        The model to register the hook on.
    layer_type (type[nn.Module]):
        The type of layer to register the hook on.
    hook (Callable): 
        The hook to register.
    transform (Callable):
        A function to transform the layer before registering the hook.
        This is useful, for example, for registering hooks on the attention
        modules of a transformer layer. Defaults to None.
    patch (Optional[Callable]):
        A function to patch the layer before registering the hook.
        This is useful, for example, for guaranteeing that the attention
        modules of a transformer layer return the attention weights.
        Defaults to None.
    """
    for m in model.modules():
        if isinstance(m, layer_type):
            if transform is not None:
                m  = transform(m)
            if patch is not None:
                patch(m)
            m.register_forward_hook(hook)



# will this return the self-attention weights twice?
AttentiveModule = Union[
    nn.MultiheadAttention, 
    # nn.TransformerEncoderLayer,
    # nn.TransformerDecoderLayer,
    fm.multihead_attention.MultiheadAttention,
    ]
def patch_attn_to_return_weights(m : nn.MultiheadAttention) -> None:
    """
    Force the attention module to return the attention weights.

    Parameters:
    ----------
    m (nn.MultiheadAttention): 
        The attention module.
    """
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return forward_orig(*args, **kwargs)

    m.forward = wrap


class SaveAttentionWeights:
    """
    A hook that saves the attention weights of a transformer layer.
    """
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []

    def pop_weights(self):
        out = self.outputs
        self.clear()
        return out
    


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



class AttentionWeightMetaClass(WeightInitialisationMetaClass):
    """
    A metaclass that automatically calls the _weight_init() method of a class
    after all child classes are initialized. This is useful for initializing
    the weights of a model after it is initialized.

    Any class that inherits from this metaclass must implement the _weight_init()
    method, and have a save_attention_weights attribute.

    Inherits:
    --------
    type: 
        A metaclass for creating classes.
    """
    def __call__(cls, *args, **kwargs):
        # create an instance of the class using the __call__ method of the class
        assert issubclass(cls, BaseAttentionModel), 'The class must inherit from BaseAttentionModel'
        obj = WeightInitialisationMetaClass.__call__(cls, *args, **kwargs)

        if hasattr(obj, 'attention_weights'):
            # patch the model so that the attention layers return the attention weights
            # and register a hook to save the attention weights
            patch_and_register_layer_hooks(
                model=obj,
                layer_type=AttentiveModule,
                hook=obj.attention_weights,
                transform=None,
                patch=patch_attn_to_return_weights,
            )

            # patch the model so that the forward pass returns the attention weights
            patch_forward_to_return_attn_weights(obj)

            return obj


class BaseAttentionModel(BaseModel, metaclass=AttentionWeightMetaClass):
    def __init__(self, save_attn_weights : bool = False):
        super().__init__()
        if save_attn_weights:
            self.attention_weights = SaveAttentionWeights()

