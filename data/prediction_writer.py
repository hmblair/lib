from typing import Any, Optional, Sequence, Union
import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from abc import ABCMeta, abstractmethod
import numpy as np

class DistributedPredictionWriter(BasePredictionWriter, metaclass=ABCMeta):
    """
    Abstract writer class for saving predictions to file when making predictions
    on a distributed system.

    Parameters:
    -----------
    write_interval (int): 
        Interval at which to write the predictions, either 'batch' or 'epoch'. 
        Defaults to 'batch'.

    Attributes:
    ----------
    write_interval (str): 
        Interval at which to write the predictions.
    """ 
    def __init__(self, write_interval : str = 'batch'):
        super().__init__(write_interval)


    @abstractmethod
    def _write(self, prediction : tuple[np.ndarray, np.ndarray]) -> None:
        """
        Write the prediction to a file or other output medium. An abstract method
        that must be implemented by a subclass.

        Parameters:
        -----------
        prediction (tuple[np.ndarray, np.ndarray]): 
            The prediction to be written.
        """
        return
    

    def _gather_tensor(
            self, 
            tensor : torch.Tensor
            ) -> torch.Tensor:
        if torch.distributed.is_initialized():
            gathered_predictions = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gathered_predictions, tensor)
        else:
            gathered_predictions = [tensor]

        return gathered_predictions


    def _gather(
            self, 
            trainer : pl.Trainer, 
            pl_module : pl.LightningModule, 
            prediction : tuple[torch.Tensor, torch.Tensor], 
            batch_indices : Sequence[int], 
            batch : Any, 
            batch_idx : int, 
            dataloader_idx : int, 
            ) -> torch.Tensor:
        """
        Gathers predictions from all distributed processes, concatenates them, 
        and returns them on the root process.

        Parameters:
        -----------
        trainer (pl.Trainer): 
            The PyTorch Lightning Trainer object.
        pl_module (pl.LightningModule): 
            The PyTorch Lightning module.
        prediction (tuple[torch.Tensor, torch.Tensor]): 
            The input and corresponding prediction.
        batch_indices (Sequence[int]): 
            The indices of the batch.
        batch (Any): 
            The batch data which is fed to the model.
        batch_idx (int): 
            The index of the current batch.
        dataloader_idx (int): 
            The index of the current dataloader.

        Returns:
        --------
        torch.Tensor: 
            The gathered predictions.
        """
        gathered_x, gathered_y = self._gather_tensor(prediction[0]), self._gather_tensor(prediction[1])

        if trainer.global_rank == 0:
            return torch.cat(gathered_x, dim=0).cpu().numpy(), torch.cat(gathered_y, dim=0).cpu().numpy()
    

    def _gather_and_write(self, trainer, *args, **kwargs) -> None:
        """
        Gathers the predictions from all distributed processes and writes them 
        to a file on the root process.
        """
        prediction = self._gather(trainer, *args, **kwargs)
        if trainer.global_rank == 0:
            self._write(prediction)


    def write_on_batch_end(self, trainer, *args, **kwargs) -> None:
        """
        Writes the output to a file at the end of a batch.
        """
        self._gather_and_write(trainer, *args, **kwargs)


    def write_on_epoch_end(self, trainer, *args, **kwargs) -> None:
        """
        Writes the output to a file at the end of the epoch.
        """
        self._gather_and_write(trainer, *args, **kwargs)



from data.h5tools import HDF5File
class DistributedPredictionWriterToH5(DistributedPredictionWriter):
    """
    A writer class for saving predictions to an HDF5 file when making 
    predictions on a distributed system.

    Parameters:
    -----------
    path (str | os.PathLike): 
        The directory to save the predictions to.

    Attributes:
    ----------
    table_group (GroupedTables): 
        The GroupedTables object used to save the predictions.
    """
    def __init__(
            self, 
            path : Union[str, os.PathLike], 
            root_uep : str = '/',
            overwrite : bool = False
            ) -> None:
        super().__init__()
        # check that the output file does not already exist, and create a new
        # file at the specified path.
        if os.path.exists(path):
            if overwrite:
                os.remove(path)
            else:
                raise ValueError(f'The output file {path} already exists.')
        self.table_group = HDF5File(path, root_uep=root_uep)


    def _write(self, prediction : tuple[np.ndarray, np.ndarray]) -> None:
        """
        Writes the output to an HDF5 file, with sequences of different lengths
        stored in different files.

        Parameters:
        -----------
        prediction (tuple[np.ndarray, np.ndarray]): 
            The model output.
        """
        # get the length of the sequences, in order to save them to the correct
        # table in the HDF5 file.

        x, y = prediction

        seq_len = x.shape[-1]
        dt=[('input', x.dtype, (seq_len,)), ('output', y.dtype)]
        data = np.zeros(y.shape, dtype=dt)
        data['input'] = x
        data['output'] = y

        seq_len = str(seq_len)
        if seq_len not in self.table_group:
            self.table_group.table_from_struct(
                table_name=seq_len, 
                data=data,
                )
        else:
            self.table_group[seq_len].append(data)
