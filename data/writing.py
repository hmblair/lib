# writing.py

from typing import Any, Optional, Sequence, Callable
import os
from abc import ABCMeta, abstractmethod
import numpy as np
import xarray as xr
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter

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
    def write(self, prediction : np.ndarray) -> None:
        """
        Write the prediction to a file or other output medium. An abstract method
        that must be implemented by a subclass.

        Parameters:
        -----------
        prediction (np.ndarray): 
            The prediction to be written.
        """
        return
    

    def _gather_tensor(self, tensor : torch.Tensor) -> torch.Tensor:
        if torch.distributed.is_initialized():
            gathered_predictions = [
                torch.zeros_like(tensor).contiguous() 
                for _ in range(torch.distributed.get_world_size())
                ]
            torch.distributed.all_gather(
                gathered_predictions, 
                tensor.contiguous()
                )
        else:
            gathered_predictions = [tensor]

        return gathered_predictions


    def _gather_and_write(
            self, 
            trainer : pl.Trainer, 
            pl_module : pl.LightningModule, 
            prediction : torch.Tensor, 
            batch_indices : Sequence[int], 
            batch : Any, 
            batch_idx : int, 
            dataloader_idx : int,
            ) -> None:
        """
        Gathers the predictions from all distributed processes and writes them 
        to a file on the root process.
        """
        # gather the predictions from all distributed processes
        gathered_predictions = self._gather_tensor(prediction)
        if trainer.global_rank == 0:
            # get the data name and model name
            data_names = trainer.datamodule.data_names['predict']
            data_name = data_names[dataloader_idx]
            model_name = pl_module.__class__.__name__
            # write the predictions to a file
            prediction = torch.cat(gathered_predictions, dim=0).cpu().numpy()
            self.write(prediction, data_name, model_name)


    def write_on_batch_end(self, *args, **kwargs) -> None:
        """
        Writes the output to a file at the end of a batch.
        """
        self._gather_and_write(*args, **kwargs)


    def write_on_epoch_end(self, *args, **kwargs) -> None:
        """
        Writes the output to a file at the end of the epoch.
        """
        self._gather_and_write(*args, **kwargs)



class netCDFDistributedPredictionWriter(DistributedPredictionWriter):
    """
    A writer class for saving predictions to a netCDF file when making
    predictions on a distributed system.

    Parameters:
    -----------
    path (str):
        The directory to save the predictions to.
    variable_name (str):
        The name of the netCDF variable to save the predictions to.
    dimension_names (list[str]):
        The names of the dimensions of the netCDF variable. If None, the 
        dimensions will be named 'batch' and then 'dim_0', 'dim_1', etc.
    """
    def __init__(
            self, 
            path : str, 
            variable_name : str = 'predictions',
            dimension_names : Optional[list[str]] = None,
            transforms : list[Callable] = [],
            *args, **kwargs,
            ) -> None:
        super().__init__(*args, **kwargs)
        if not os.path.exists(path):
            os.mkdir(path)
        self.path = path
        self.variable_name = variable_name   
        self.dimension_names = dimension_names    


    def write(
            self, 
            prediction : np.ndarray, 
            data_name : str, 
            model_name : str,
            ) -> None:
        """
        Writes the output to a netCDF file.

        Parameters:
        -----------
        prediction (np.ndarray): 
            The model output.
        data_name (str):
            The name of the data, which is used to name the netCDF file.
        """
        # get the directory to save the netCDF file to, and create it if it
        # does not exist.
        dir = os.path.join(self.path, model_name)
        if not os.path.exists(dir):
            os.mkdir(dir)
        
        # get the path to the netCDF file
        file = os.path.join(dir, f'{data_name}.nc')

        # get the dimensions of the prediction
        if self.dimension_names is not None:
            if not len(self.dimension_names) == prediction.ndim:
                raise ValueError(
                    'The number of dimension names must match the number of dimensions in the prediction.'
                    )
            dims = self.dimension_names
        else:
            dims = ['batch'] + [f'dim_{i}' for i in range(prediction.ndim - 1)]

        # load the netCDF file if it exists, and prepare the dataset to append
        if not os.path.exists(file):
            dataset = xr.Dataset(
                {self.variable_name : (dims, prediction)}
                )      
        else:
            dataset = xr.load_dataset(file, mode='a')            
            dataset_to_append = xr.Dataset(
                {self.variable_name : (dims, prediction)}
                )
            
            dataset = xr.concat([dataset, dataset_to_append], dim='batch')

        # save the predictions to the netCDF file       
        dataset.to_netcdf(file, engine='h5netcdf')