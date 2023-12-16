from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn, rank_zero_info
import torch
import torch.nn as nn
from typing import Optional
from models.abstract_models import BaseModel
from typing import Union, Iterable


class LoRALayerWrapper(nn.Module):
    """
    Wraps a pre-trained nn.Module with a LoRA layer.

    Parameters:
    -----------
    base_module (nn.Module): 
        The base module to be wrapped.
    lora_rank (int): 
        The rank of the LoRA layer.
    device (str | torch.device): 
        The device to initialise the LoRA weights on.

    Attributes:
    -----------
    base_module (nn.Module): 
        The base module being wrapped.
    lora_A (nn.Parameter): 
        LoRA weight A.
    lora_B (nn.Parameter): 
        LoRA weight B.
    """
    def __init__(
            self, 
            base_module: nn.Module, 
            lora_rank: int, 
            device: Union[str, torch.device]
            ) -> None:
        super().__init__()
        self.base_module = base_module
        weight_shape = self.base_module.weight.shape

        self.lora_A = nn.Parameter(
            torch.zeros(weight_shape[0], lora_rank, device=device)
        )
                
        self.lora_B = nn.Parameter(
            torch.randn(weight_shape[1], lora_rank, device=device)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LoRALayerWrapper.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the base module and 
            adding on the LoRA perturbation.
        """
        # compute the output of the base module
        base_out = self.base_module(x)  

        # add on the LoRA perturbation
        return base_out + (x @ self.lora_A) @ self.lora_B.T 


from pytorch_lightning.utilities.model_summary import summarize
from abc import ABCMeta, abstractmethod
class BaseFineTuningModel(BaseModel, metaclass = ABCMeta):
    """
    Base class for fine-tuning pre-trained models. Adds logic for freezing and
    unfreezing the weights of the pre-trained model, as well as logic for 
    wrapping the pre-trained model with a LoRA layer.

    Parameters:
    ----------
    unfreeze_epoch (int, optional): 
        The epoch to unfreeze the pre-trained model. Defaults to None.
    lora_p (int, optional): 
        The rank of LoRA. Defaults to None.
    *args: 
        Variable length argument list.
    **kwargs: 
        Arbitrary keyword arguments.

    Attributes:
    ----------
    pt_model (nn.Module): 
        The pre-trained model.
    unfreeze_epoch (int | None): 
        The epoch to unfreeze the pre-trained model. If using LoRA, this is the
        epoch to unfreeze the LoRA parameters.
    lora_p (int | None): 
        The rank of LoRA.
    """
    def __init__(
            self, 
            unfreeze_epoch: Optional[int] = None, 
            lora_p: Optional[int] = None, 
            *args, **kwargs,
            ) -> None:
        super().__init__(*args, **kwargs)

         # load the pre-trained model
        self.pt_model = self.load_model() 
         # freeze the model to begin with
        self.freeze()

        # store the epoch to unfreeze the pre-trained model and the rank of LoRA
        self.unfreeze_epoch = unfreeze_epoch  
        self.lora_p = lora_p

        # wrap the LoRA layer around the pre-trained model    
        if lora_p is not None:
            self._lora_wrapper() 


    @abstractmethod
    def load_model(self) -> nn.Module:
        """
        Loads a pre-trained model. An abstract method which must be implemented
        by the child class.
        """
        return


    def on_train_epoch_start(self) -> None:
        """
        Callback function called at the start of each training epoch.
        If the `unfreeze_epoch` parameter is set and the current epoch matches the 
        `unfreeze_epoch` value, unfreezes the RNA-FM model by setting `requires_grad` 
        to True.
        """
        if self.unfreeze_epoch is not None and self.current_epoch == self.unfreeze_epoch:
            rank_zero_info('We have reached the unfreeze epoch.')
            self.unfreeze() # unfreeze the model

    
    def freeze(self) -> None:
        """
        Freezes the pre-trained model.
        """
        self.pt_model.eval() # set the model to eval mode
        self.pt_model.requires_grad_(False) # freeze the model
        rank_zero_info('Freezing the pre-trained model...')
        

    def unfreeze(self) -> None:
        """
        Unfreezes the pre-trained model, or the LoRA weights if using LoRA.
        """
        if self.lora_p is None:
            rank_zero_info('Unfreezing the pre-trained model...')
            for layer in self._layers_to_unfreeze():
                layer.train()
                layer.requires_grad_(True)
        else:
            for param in self.lora_params:
                param.requires_grad_(True) # unfreeze the LoRA parameters
            rank_zero_info('Unfreezing the LoRA parameters...')
        
        rank_zero_info(summarize(self)) # print a model summary showing the updated number of trainable parameters
    

    def _lora_wrapper(self) -> None:
        """
        Wraps the LoRALayerWrapper around the non-trainable parameters.
        The LoRA weights are initialised as untrainable, and are only unfrozen
        when 'unfreeze' is called.
        """
        lora_params = []
        for module in self.pt_model.modules():
            if not module.children(): 
                # wrap the LoRA layer around the module
                module = LoRALayerWrapper(module, self.lora_p, self.device) 
                # store the LoRA parameters
                lora_params += [module.lora_A, module.lora_B] 
        # set the LoRA parameters to untrainable
        self.lora_params = nn.ParameterList(lora_params).requires_grad_(False) 


    def _layers_to_unfreeze(self) -> Iterable:
        """
        Get the layers of the pre-trained model to unfreeze. 
        Override this method if only a subset of the layers should be unfrozen.
        This is only called if LoRA is not being used.

        Returns:
        --------
        Iterable:
            The layers to unfreeze. Defaults to the entire pre-trained model.
        """
        # yield the pre-trained model by default
        yield self.pt_model 
    