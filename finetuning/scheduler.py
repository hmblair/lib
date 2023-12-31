# scheduler.py

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.utilities.model_summary import summarize

class FineTuningScheduler(BaseFinetuning):
    def __init__(
            self, 
            layers_to_unfreeze : list[int],
            unfreeze_rate : int, 
            pt_model : str = 'pt_model',
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
        if self.unfreeze_rate == 0:
            return
        try:
            getattr(pl_module, self.pt_model).requires_grad_(False)
            getattr(pl_module, self.pt_model).eval()
        except AttributeError as e:
            raise AttributeError(
                f'Cannot find {self.pt_model} in the LightningModule. '
                'Please check the name of the attribute.'
            ) from e
    

    def finetune_function(
            self, 
            pl_module: pl.LightningModule, 
            epoch: int, 
            optimizer: torch.optim.Optimizer,
            ) -> None:
        if epoch in self._unfreeze_dict:
            getattr(pl_module, self.pt_model)[
                self._unfreeze_dict[epoch]
            ].requires_grad_(True)
            print(summarize(pl_module))