# datamodules.py

import warnings
import os
from typing import Any, Union, Iterable, Sequence, Optional, Callable
from abc import ABCMeta, abstractmethod
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .datasets import netCDFIterableDataset, netCDFIterableDatasetBase
from .utils import get_filename, xarray_to_dict
import numpy as np
import xarray as xr

class BarebonesDataModule(pl.LightningDataModule, metaclass=ABCMeta):
    """
    An abstract base class for Pytorch Lightning DataModules. It provides a
    simple interface for creating datasets and dataloaders, and for extracting
    inputs and targets from a batch. 

    Parameters:
    ----------
    batch_size (int): 
        The batch size to use for the dataloaders.
    num_workers (int):
        The number of workers to use for the dataloaders. If set to -1, the 
        number of workers is set to the number of available CPUs. Defaults to 1.
    """
    def __init__(
            self, 
            batch_size : int, 
            num_workers : int = 1,
            ) -> None:
        super().__init__()

        # initialise the data dictionary
        self.data = {
            'train': None,
            'validate': None,
            'test': None,
            'predict': None,
            }

        # save batch size
        self.batch_size = batch_size

        # determine the number of workers from the number of available CPUs if 
        # num_workers is set to -1, otherwise use the provided value
        if num_workers == -1:
            num_workers = os.cpu_count()
        self.num_workers = num_workers


    def distributed_info(self) -> tuple[int, int]:
        """
        Returns the rank and world size of the current process. If this is 
        called too early, before the trainer object has been created, the rank
        is set to 0 and the world size is set to 1, which may cause issues with
        distributed training.

        Returns:
        -------
        tuple[int, int]: 
            The rank and world size of the current process.
        """
        try:
            rank = self.trainer.global_rank
            world_size = self.trainer.world_size
        except Exception:
            warnings.warn(
                message = 'No trainer object found. Setting rank to 0 and world' \
                    ' size to 1. To use distributed training, please pass this' \
                    ' DataModule to a trainer object.', 
                category = UserWarning, 
                stacklevel = 2
                )
            rank = 0
            world_size = 1

        return rank, world_size


    @abstractmethod
    def _create_datasets(
        self, 
        phase : str, 
        rank : int, 
        world_size : int,
        ) -> Union[Sequence, Iterable]:
        """
        Create a dataset for the specified phase. An abstract method that must
        be implemented by a subclass.

        Parameters:
        ----------
        phase (str): 
            The phase for which to create the datasets. Can be one of 'train', 
            'val', 'test', or 'predict'.
        rank (int):
            The rank of the current process.
        world_size (int):
            The total number of processes.

        Returns:
        -------
        (Sequence | Iterable): 
            The dataset for the specified phase.
        """
        return   
    

    def _create_dataloaders(self, phase: str) -> DataLoader:
        """
        Create a dataloader for the specified phase. Overwrite this method if 
        you want to use a custom dataloader construction, such as if batching
        is handled by the dataset itself.

        The default implementation creates a dataloader with the following 
        parameters:
        - num_workers = self.num_workers
        - batch_size = self.batch_size
        - shuffle = (phase == 'train')

        Parameters:
        ----------
        phase (str): 
            The phase for which to create the dataloaders. Can be one of 
            'train', 'validate', 'test', or 'predict'.

        Returns:
        -------
        torch.utils.data.DataLoader: 
            The dataloader for the specified phase.
        """
        if phase not in ['train', 'validate', 'test', 'predict']:
            raise ValueError(
                f'Unknown phase {phase}. Please specify one of "train", "validate", "test", or "predict".'
                )
        
        if self.data[phase] is None:
            raise ValueError(
                f'There is no {phase} dataset. Please call the setup method with the appropriate stage first, and ensure your _create_datasets method returns a dataset for the {phase} phase.'
                )

        return DataLoader(
            dataset = self.data[phase],
            num_workers = self.num_workers,
            batch_size = self.batch_size,
            shuffle = (phase == 'train'),
            multiprocessing_context = 'fork' if torch.mps.is_available() else None,
        )
    

    def setup(self, stage: str) -> None:
        """
        Creates datasets for the specified stage, and stores them in the 
        'self.data' dictionary.

        Parameters:
        ----------
        stage (str): 
            The stage of the data setup. Must be either 'fit', 'validate', 
            'test', or 'predict'.

        Raises:
        -------
        ValueError: 
            If the stage is not one of 'fit', 'validate', 'test', or 'predict'.
        """
        rank, world_size = self.distributed_info()

        if stage == 'fit':
            self.data['train'] = self._create_datasets(
                'train', rank, world_size,
                )
            self.data['validate'] = self._create_datasets(
                'validate', rank, world_size,
                )
        elif stage in ['test', 'validate', 'predict']:
            self.data[stage] = self._create_datasets(
                stage, rank, world_size,
                )
        else:
            raise ValueError(
                f'Invalid stage {stage}. The stage must be either "fit", "validate", "test" or "predict".'
                )


    def train_dataloader(self) -> DataLoader:
        """
        Returns the train dataloader.

        Returns:
        -------
        torch.utils.data.DataLoader: 
            The train dataloader.
        """
        return self._create_dataloaders('train')


    def val_dataloader(self) -> DataLoader:
        """
        Returns the validaiton dataloader, if a validation dataset exists. Else,
        raises a NotImplementedError.

        Returns:
        -------
        torch.utils.data.DataLoader: 
            The validation dataloader.
        """
        if self.data['validate'] is None:
            return super().val_dataloader()
        else:
            return self._create_dataloaders('validate')


    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader.

        Returns:
        -------
        torch.utils.data.DataLoader: 
            The test dataloader.
        """
        if self.data['test'] is None:
            raise ValueError(
                'No test dataset found. Please ensure there is a test dataset when initialising the data module.'
                )
        return self._create_dataloaders('test')


    def predict_dataloader(self) -> DataLoader:
        """
        Returns the prediction dataloader.

        Returns:
        -------
        torch.utils.data.DataLoader: 
            The prediction dataloader.
        """
        if self.data['predict'] is None:
            raise ValueError(
                'No prediction dataset found. Please ensure there is a prediction dataset when initialising the data module.'
                )
        return self._create_dataloaders('predict')


    @abstractmethod
    def get_inputs(self, batch : Any) -> torch.Tensor:
        """
        Extracts the inputs from the batch. This is useful as the batch may be
        of varying types, such as a tuple or a dictionary, and so this method
        allows you to provide an interface for the model to retrieve the inputs.

        Parameters:
        ----------
        batch (Any): 
            The batch to extract the inputs from.

        Returns:
        -------
        torch.Tensor: 
            The inputs.
        """
        return
    

    @abstractmethod
    def get_targets(self, batch : Any) -> torch.Tensor:
        """
        Extracts the targets from the batch. This is useful as the batch may be
        of varying types, such as a tuple or a dictionary, and so this method
        allows you to provide an interface for the model to retrieve the targets.

        Parameters:
        ----------
        batch (Any): 
            The batch to extract the targets from.

        Returns:
        -------
        torch.Tensor: 
            The targets.
        """
        return
    

    def on_before_batch_transfer(
            self, 
            batch: Any, 
            dataloader_idx: int,
            ) -> tuple[Any, Any]:
        """
        Called before a batch is transferred to the device. This method extracts
        the inputs and, if the phase not 'predict', the targets from the batch.
        In the latter case, the targets are returned as None.

        Parameters:
        ----------
        batch (Any): 
            The batch to extract the inputs and targets from.
        dataloader_idx (int):
            The index of the dataloader.

        Returns:
        -------
        tuple[Any, Any]:
            The inputs and, if the phase is not 'predict', targets, else None.
        """
        if self.trainer.predicting:
            return self.get_inputs(batch), None
        else:
            return self.get_inputs(batch), self.get_targets(batch)



class netCDFDataModule(BarebonesDataModule):
    """
    A DataModule for netCDF data, providing functionality for loading, 
    transforming, and batching the data.

    Parameters:
    ----------
    input_variables (list[str]):
        The names of the input variables.
    target_variables (list[str], optional):
        The names of the target variables. Defaults to an empty list.
    stack_dim (int, optional):
        The dimension to stack the input and target variables along. Defaults
        to -1.
    train_paths (list[str], optional):
        The paths to the training data. Defaults to None.
    validate_paths (list[str], optional):
        The paths to the validation data. Defaults to None.
    test_paths (list[str], optional)):
        The paths to the testing data. Defaults to None.
    predict_paths (list[str], optional)):
        The paths to the prediction data. Defaults to None.
    """
    def __init__(
            self, 
            input_variables : list[str],
            target_variables : list[str] = [],
            stack_dim : int = -1,
            train_paths : Optional[list[str]] = None,
            validate_paths : Optional[list[str]] = None,
            test_paths : Optional[list[str]] = None,
            predict_paths : Optional[list[str]] = None,
            train_transforms : list[Callable[[xr.Dataset], xr.Dataset]] = [],
            validate_transforms : list[Callable[[xr.Dataset], xr.Dataset]] = [],
            test_transforms : list[Callable[[xr.Dataset], xr.Dataset]] = [],
            predict_transforms : list[Callable[[xr.Dataset], xr.Dataset]] = [],
            *args, **kwargs,
            ) -> None:
        super().__init__(*args, **kwargs)

        # store the variables
        self.input_variables = input_variables
        self.target_variables = target_variables

        # store the stack dimension
        self.stack_dim = stack_dim

        # store the paths to each dataset
        self.data_paths = {
            'train': train_paths,
            'validate': validate_paths,
            'test': test_paths,
            'predict': predict_paths,
            }
        
        # store the names of each dataset
        get_data_names = lambda paths: [get_filename(path) for path in paths] if paths is not None else None
        self.data_names = {
            'train': get_data_names(train_paths),
            'validate': get_data_names(validate_paths),
            'test': get_data_names(test_paths),
            'predict': get_data_names(predict_paths),
            }
        
        # store the transform
        self.transforms = {
            'train': train_transforms,
            'validate': validate_transforms,
            'test': test_transforms,
            'predict': predict_transforms,
            }

        # raise an error if the number of workers is greater than 1
        if self.num_workers > 1:
            raise ValueError(
                'The number of workers cannot exceed 1 for netCDF datasets.' \
                ' Exactly one is preferable.'
                )


    def _create_datasets(
            self, 
            phase: str, 
            rank: int, 
            world_size: int,
            ) -> Union[Sequence, Iterable]:
        """
        Create a dataset for the specified phase, if a path to the data is
        specified.
        """
        if self.data_paths[phase] is not None:
            if phase == 'train' and not self.target_variables:
                raise ValueError(
                    'The target variables must be specified if the phase is "train".'
                    )
            
            if phase == 'train':
                return netCDFIterableDataset(
                    paths = self.data_paths[phase],
                    batch_size = self.batch_size,
                    rank = rank,
                    world_size = world_size,
                    should_shuffle = phase == 'train',
                    transforms = self.transforms[phase]
                    )
            else:
                return [netCDFIterableDatasetBase(
                    path = path,
                    batch_size = self.batch_size,
                    rank = rank,
                    world_size = world_size,
                    should_shuffle = phase == 'train',
                    transforms = self.transforms[phase]
                    ) for path in self.data_paths[phase]]
    

    def _create_dataloaders(self, phase: str) -> DataLoader:
        """
        Create a dataloader for the specified phase.

        Parameters:
        ----------
        phase (str): 
            The phase for which to create the dataloaders. Can be one of 
            'train', 'val', 'test', or 'predict'.

        Returns:
        -------
        torch.utils.data.DataLoader: 
            The dataloader for the specified phase.
        """        
        if phase not in ['train', 'validate', 'test', 'predict']:
            raise ValueError(
                f'Unknown phase {phase}. Please specify one of "train", "val", "test", or "predict".'
                )

        if self.data[phase] is not None:
            if phase == 'train':
                return DataLoader(
                    dataset = self.data[phase],
                    num_workers = self.num_workers,
                    batch_size = (None if self.num_workers <= 1 else self.num_workers),
                    collate_fn = xarray_to_dict,
                    multiprocessing_context = 'fork' if torch.backends.mps.is_available() and self.num_workers > 0 else None,
                    )
            else:
                return [DataLoader(
                    dataset = data,
                    num_workers = self.num_workers,
                    batch_size = (None if self.num_workers <= 1 else self.num_workers),
                    collate_fn = xarray_to_dict,
                    multiprocessing_context = 'fork' if torch.backends.mps.is_available() and self.num_workers > 0 else None,
                ) for data in self.data[phase]]
    
    
    def get_inputs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the inputs from the batch.

        Parameters:
        ----------
        batch (dict[str, torch.Tensor]): 
            The batch of data.

        Returns:
        -------
        torch.Tensor:
            The inputs from the batch.
        """
        return torch.stack(
            [batch[name] for name in self.input_variables], 
            dim = self.stack_dim
            ).squeeze(-1)

    
    def get_targets(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the targets from the batch.

        Parameters:
        ----------
        batch (dict[str, torch.Tensor]): 
            The batch of data.

        Returns:
        -------
        torch.Tensor:
            The targets from the batch.
        """
        return torch.stack(
            [batch[name] for name in self.target_variables], 
            dim = self.stack_dim
            ).squeeze(-1) if self.target_variables is not None else None