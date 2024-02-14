# lora.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.model_summary import summarize

class LoRALayerWrapper(nn.Linear):
    """
    A wrapper class for nn.Linear modules which adds a LoRA perturbation to the
    output of the base module. By replacing the base module with this wrapper, 
    a single LoRA layer can be added to a pre-trained model.

    Parameters:
    -----------
    base_module (nn.Module): 
        The base module to be wrapped.
    lora_rank (int): 
        The rank of the LoRA layer.
    frozen (bool):
        Whether the LoRA parameters should be frozen. If True, the LoRA 
        parameters will be initialised as untrainable.

    Attributes:
    -----------
    base_module (nn.Linear): 
        The base module being wrapped. It should be an instance of nn.Linear.
    lora_A (nn.Parameter): 
        LoRA weight A.
    lora_B (nn.Parameter): 
        LoRA weight B.

    Inherits:
    ---------
    nn.Linear: 
        Pytorch linear layer base class.
    """
    def __init__(
            self, 
            base_module: nn.Linear, 
            lora_rank: int, 
            frozen : bool = True,
            ) -> None:
        if not isinstance(base_module, nn.Linear):
            raise ValueError(
                'The base module must be an instance of nn.Linear.'
                )
        super().__init__(
            base_module.in_features, 
            base_module.out_features, 
            bias=base_module.bias is not None,
            device=base_module.weight.device,
            dtype=base_module.weight.dtype,
        )
        self.weight = base_module.weight
        self.bias = base_module.bias
        
        weight_shape = self.weight.shape

        self.lora_A = nn.Parameter(
            torch.randn(lora_rank, weight_shape[1], device=self.weight.device)
        )
                
        self.lora_B = nn.Parameter(
            torch.zeros(weight_shape[0], lora_rank, device=self.weight.device)
        )

        if frozen:
            self.lora_A.requires_grad_(False)
            self.lora_B.requires_grad_(False)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LoRALayerWrapper.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the base module and 
            adding on the LoRA perturbation.
        """
        return super().forward(x) + (x @ self.lora_A.T) @ self.lora_B.T



def get_lora_params(module : nn.Module, get_names : bool = False):
    lora_params = []
    for name, module in module.named_modules():
        if isinstance(module, LoRALayerWrapper):
            if get_names:
                lora_params += (name, module.lora_A, module.lora_B)
            else:
                lora_params += [module.lora_A, module.lora_B]
    return lora_params



def unfreeze_lora_params(module: nn.Module, optimizer : torch.optim.Optimizer = None) -> None:
    lora_params = nn.ParameterList(
        get_lora_params(module)
    )
    for param in lora_params:
        param.requires_grad_(True)
    if optimizer is not None:
        optimizer.add_param_group(
            {'params': lora_params, 'lr': optimizer.defaults['lr']}
        )



def wrap_with_lora(
        module: nn.Module, 
        lora_rank: int, 
        frozen : bool = True,
        ) -> None:
    """
    Wraps a pre-trained nn.Module with a LoRA layer. Unlike the LoraLayerWrapper
    class, this wraps all leaf modules in the base module. The base module is 
    modified in-place, and the LoRA parameters are returned.
    
    The LoRA weights are initialised as untrainable, and should be unfrozen 
    manually when fine-tuning.

    Parameters:
    -----------
    module (nn.Module): 
        The base module to be wrapped.
    lora_rank (int): 
        The rank of the LoRA layer.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Wrap the linear layer
            wrapped = LoRALayerWrapper(child, lora_rank, frozen)
            setattr(module, name, wrapped)
        else:
            # Recursively replace in child modules
            wrap_with_lora(child, lora_rank, frozen)   



from pytorch_lightning.callbacks import BaseFinetuning
class LoRACallback(BaseFinetuning):
    def __init__(
            self, 
            lora_rank : int, 
            unfreeze_epoch : int,
            pt_model : str = 'pt_model',
            ) -> None:
        super().__init__()
        self.lora_rank = lora_rank
        self.unfreeze_epoch = unfreeze_epoch
        self.pt_model = pt_model
    

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        """
        Wraps the pre-trained model with LoRA layers, which are initialised as
        untrainable.

        Parameters:
        -----------
        pl_module (pl.LightningModule):
            The LightningModule to be fine-tuned.
        """
        if hasattr(pl_module, self.pt_model):
            wrap_with_lora(
                getattr(pl_module, self.pt_model),
                self.lora_rank, 
                frozen=True,
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
            # unfreeze the LoRA parameters
            unfreeze_lora_params(getattr(pl_module, self.pt_model), optimizer)
            # summarize the model with the LoRA parameters
            rank_zero_info(summarize(pl_module))