# unfreeze_scheduler.py

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.utilities.model_summary import summarize
from pytorch_lightning.utilities import rank_zero_info

class FineTuningScheduler(BaseFinetuning):
    """
    A finetuning scheduler that unfreezes layers of a model at a specified rate.
    In order to use this scheduler, the pretrained model must implement the
    __getitem__ method, so that the layers can be accessed by index.

    Parameters:
    ----------
    layers_to_unfreeze (list[int]):
        The layers to unfreeze.
    unfreeze_rate (int):
        The rate at which layers are unfrozen.
    pt_model (str):
        The name of the pretrained model in the LightningModule. Defaults to
        "model".
    """
    def __init__(
            self, 
            layers_to_unfreeze : list[int],
            unfreeze_rate : int, 
            pt_model : str = 'model',
            ) -> None:
        super().__init__()
        self.unfreeze_rate = unfreeze_rate
        self.pt_model = pt_model
        unfreeze_epochs = range(
            unfreeze_rate, 
            (len(layers_to_unfreeze) + 1) * unfreeze_rate, 
            unfreeze_rate,
        )
        self._unfreeze_dict = dict(
            zip(unfreeze_epochs, layers_to_unfreeze)
            )
    

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        """
        Freezes the pretrained model before training.
        """
        if self.unfreeze_rate == 0:
            return
        try:
            getattr(pl_module, self.pt_model).requires_grad_(False)
            getattr(pl_module, self.pt_model).eval()
        except AttributeError as e:
            raise AttributeError(
                f'Cannot find {self.pt_model} in the LightningModule.'
                ' Please check the name of the attribute.'
            ) from e
    

    def finetune_function(
            self, 
            pl_module: pl.LightningModule, 
            epoch: int, 
            optimizer: torch.optim.Optimizer,
            ) -> None:
        """
        Unfreezes a layer of the model if the current epoch is divisible by the
        unfreeze rate. The layer to be unfrozen is determined by the unfreeze
        dictionary. The model is then summarized on rank 0.
        """
        if epoch in self._unfreeze_dict:
            getattr(pl_module, self.pt_model)[
                self._unfreeze_dict[epoch]
            ].requires_grad_(True)
            rank_zero_info(summarize(pl_module))