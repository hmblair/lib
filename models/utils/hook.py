# hook.py

import torch.nn as nn
from typing import Any, Callable, Optional
from pytorch_lightning.utilities import rank_zero_warn

class HookList(list):
    """
    A list that can be used to store hooks. It has an additional method, 
    remove_hooks(), that can be used to remove all hooks from the list.

    Inherits:
    --------
    list: 
        A list.
    """
    def remove_hooks(self) -> None:
        """
        Removes all hooks from the list.
        """
        for hook in self:
            hook.remove()
        self.clear()



def patch_and_register_layer_hooks(
        model : nn.Module, 
        layer_type : type[nn.Module], 
        hook : Callable,
        transform : Optional[Callable] = None,
        patch : Optional[Callable] = None,
        ) -> HookList:
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

    Returns:
    --------
    HookList:
        A list of handles returned by the register_forward_hook() method of
        the layers of the given model that are of the given type. This list 
        can be used to remove the hooks later, via the remove_hooks() method.
    """
    handles = []
    for m in model.modules():
        if isinstance(m, layer_type):
            if transform is not None:
                m  = transform(m)
            if patch is not None:
                patch(m)
            handles.append(
                m.register_forward_hook(hook)
                )
    if not handles:
        rank_zero_warn(
            f'No {layer_type} layers were found in the model. The hook will not be registered.'
            )
    return HookList(handles)



from abc import ABCMeta, abstractmethod
class HookContextManager(metaclass=ABCMeta):
    """
    A context manager for registering a hook on a model. This class is abstract
    and must be subclassed. The subclass must implement the __call__ method,
    which is the hook that will be registered on the model.

    Parameters:
    ----------
    module (nn.Module):
        The model to register the hook on.
    layer_type (type[nn.Module]):
        The type of layer to register the hook on.
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
    def __init__(
            self, 
            module : nn.Module,
            layer_type : type[nn.Module],
            transform : Optional[Callable] = None,
            patch : Optional[Callable] = None,
            ) -> None:
        # save the parameters
        self.module = module
        self.layer_type = layer_type
        self.transform = transform
        self.patch = patch
        # create a list to store the hooks
        self.hook = HookList()


    @abstractmethod
    def __call__(
            self, 
            module : nn.Module, 
            module_in : Any, 
            module_out : Any,
            ) -> None:
        """
        The hook that will be registered on the model. This method must be
        implemented in the subclass.

        Parameters:
        ----------
        module (nn.Module):
            The model.
        module_in (Any):
            The input to the layer.
        module_out (Any):
            The output from the layer.
        """
        return


    def __enter__(self) -> 'HookContextManager':
        """
        Enter the context manager, registering the hook on the model.

        Returns:
        --------
        HookContextManager: 
            This same instance of the class.
        """
        self.hook = patch_and_register_layer_hooks(
            model=self.module,
            layer_type=self.layer_type,
            hook=self,
            transform=self.transform,
            patch=self.patch,
        )
        return self


    def __exit__(
            self, 
            type : Optional[type], 
            value : Optional[Exception], 
            traceback : Optional[Any],
            ) -> None:
        """
        Exit the context manager, removing the hook from the model.

        Parameters:
        ----------
        type (Optional[type]): 
            The type of the exception.
        value (Optional[Exception]):
            The exception.
        traceback (Optional[Any]):
            The traceback.
        """
        if self.hook:
            self.hook.remove_hooks()