# lora.py

from typing import Union
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.model_summary import summarize

class LoRALayerWrapper(nn.Module):
    """
    An nn.Module wrapper which adds a LoRA layer to the output of the base
    module. By replacing the base module with this wrapper, a single LoRA layer
    can be added to a pre-trained model.

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
    


def wrap_with_lora(
        base_module: nn.Module, 
        lora_rank: int, 
        device: Union[str, torch.device]
        ) -> nn.ParameterList:
    """
    Wraps a pre-trained nn.Module with a LoRA layer. Unlike the LoraLayerWrapper
    class, this wraps all leaf modules in the base module. The base module is 
    modified in-place, and the LoRA parameters are returned.
    
    The LoRA weights are initialised as untrainable, and should be unfrozen 
    manually when fine-tuning.

    Parameters:
    -----------
    base_module (nn.Module): 
        The base module to be wrapped.
    lora_rank (int): 
        The rank of the LoRA layer.
    device (str | torch.device): 
        The device to initialise the LoRA weights on.

    Returns:
    --------
    nn.ParameterList:
        The LoRA parameters.
    """
    lora_params = []
    for module in base_module.modules():
        if not module.children(): 
            # wrap the LoRA layer around the module
            module = LoRALayerWrapper(module, lora_rank, device) 
            # store the LoRA parameters
            lora_params += [module.lora_A, module.lora_B] 
    # collect the LoRA parameters and set them to be untrainable
    return nn.ParameterList(lora_params).requires_grad_(False)



from pytorch_lightning.callbacks import BaseFinetuning
class LoRACallback(BaseFinetuning):
    def __init__(
            self, 
            lora_rank : int, 
            unfreeze_epoch : int,
            pt_model : str = 'pt_model',
            ) -> None:
        self.lora_rank = lora_rank
        self.unfreeze_epoch = unfreeze_epoch
        self.pt_model = pt_model
        self.lora_params = None
    

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        """
        Wraps the pre-trained model with LoRA layers, and freezes the LoRA
        parameters.

        Parameters:
        -----------
        pl_module (pl.LightningModule):
            The LightningModule to be fine-tuned.
        """
        if hasattr(pl_module, self.pt_model):
            self.lora_params = wrap_with_lora(
                getattr(pl_module, self.pt_model),
                self.lora_rank, 
                pl_module.device,
                )
        else:
            raise ValueError(
                f'No attribute {self.pt_model} found in the LightningModule.'
                )
        

    def finetune_function(
            self, 
            pl_module: pl.LightningModule, 
            epoch: int, 
            optimizer: torch.optim.Optimizer,
            ) -> None:
        """
        Unfreezes the LoRA parameters at the specified epoch.

        Parameters:
        -----------
        pl_module (pl.LightningModule):
            The LightningModule to be fine-tuned.
        epoch (int):
            The current epoch.
        optimizer (torch.optim.Optimizer):
            The optimizer being used to train the model.
        """
        if epoch == self.unfreeze_epoch:
            rank_zero_info(
                f'We have reached the unfreeze epoch of {self.unfreeze_epoch}.' \
                ' Unfreezing the LoRA parameters...'
                )
            if self.lora_params is None:
                raise ValueError(
                    'The LoRA parameters have not been initialised, but the ' \
                    'unfreeze epoch has been reached. Please check that the ' \
                    'LoRACallback is being used correctly.'
                )
            # unfreeze the LoRA parameters
            for param in self.lora_params:
                param.requires_grad_(True) 
            # add the LoRA parameters to the optimizer
            optimizer.add_param_group(
                {'lora_params': self.lora_params, 'lr': optimizer.defaults['lr']}
            )
            # print a model summary showing the updated number of trainable 
            # parameters
            rank_zero_info(summarize(pl_module))