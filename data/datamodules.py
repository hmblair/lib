# datamodules.py

import warnings
import os
from typing import Iterable, Sequence, Callable
from abc import ABCMeta, abstractmethod
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_warn
from .datasets import XarrayDataset
from .utils import get_filename
import xarray as xr
from .constants import NC_EXTENSION

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
    def create_datasets(
        self, 
        phase : str, 
        rank : int, 
        world_size : int,
        ) -> Sequence | Iterable:
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
    

    def create_dataloaders(self, phase: str) -> DataLoader:
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
            self.data['train'] = self.create_datasets(
                'train', rank, world_size,
                )
            self.data['validate'] = self.create_datasets(
                'validate', rank, world_size,
                )
        elif stage in ['test', 'validate', 'predict']:
            self.data[stage] = self.create_datasets(
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
        return self.create_dataloaders('train')


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
            return self.create_dataloaders('validate')


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
        return self.create_dataloaders('test')


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
        return self.create_dataloaders('predict')



def recursively_find_files(dir : str, extension : str) -> list[str]:
    """
    Recursively find all files with a given extension in a directory.

    Parameters:
    -----------
    dir (str):
        The directory to search.
    extension (str):
        The file extension to search for.

    Returns:
    --------
    list[str]:
        A list of file paths.
    """
    files = []
    for root, _, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files


class XarrayDataModule(BarebonesDataModule):
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
    paths (dict[str, list[str]]):
        A dictionary containing the paths to the data for each phase. The 
        phases can be 'train', 'validate', 'test', or 'predict'. The paths can
        be directories containing netCDF files, or the netCDF files themselves.
    transforms (dict[str, list[Callable[[xr.Dataset], xr.Dataset]]]):
        A dictionary containing a list of transforms for each phase. The 
        transforms are applied to the dataset before it is returned.
    """
    def __init__(
            self, 
            input_variables : list[tuple[str, str]] = [],
            target_variables : list[tuple[str, str]] = [],
            stack_dim : int = -1,
            paths : dict[str, list[str]] = {},
            transforms : dict[str, list[Callable[[xr.Dataset], xr.Dataset]]] = {},
            *args, **kwargs,
            ) -> None:
        super().__init__(*args, **kwargs)

        # specify the valid phases, and ensure the provided paths are valid
        VALID_PHASES = ['train', 'validate', 'test', 'predict']
        for phase in paths.keys():
            if phase not in VALID_PHASES:
                raise ValueError(
                    f'Invalid phase {phase}. The phase must be one of {VALID_PHASES}.'
                    )

        # store the variables
        self.input_variables = input_variables
        self.target_variables = target_variables

        # store the stack dimension
        self.stack_dim = stack_dim

        # recursively find files in the specified directories
        self.data_paths = {}
        for phase, phase_paths in paths.items():
            self.data_paths[phase] = []
            for path in phase_paths:
                if os.path.isdir(path):
                    self.data_paths[phase].extend(
                        recursively_find_files(path, NC_EXTENSION)
                        )
                else:
                    self.data_paths[phase].append(path)
        
        # store the names of each dataset
        get_data_names = lambda paths: [get_filename(path) for path in paths] if paths is not None else None
        self.data_names = {
            phase : get_data_names(paths) for phase, paths in self.data_paths.items()
            }
        
        # store the transforms
        self.transforms = {phase : transforms.get(phase, []) for phase in VALID_PHASES}

        # raise an error if the number of workers is greater than 1
        if self.num_workers > 1:
            raise ValueError(
                'The number of workers cannot exceed 1 for netCDF datasets.' \
                ' Exactly one is preferable.'
                )


    def create_datasets(
            self, 
            phase: str, 
            rank: int, 
            world_size: int,
            ) -> Iterable:
        """
        Create a dataset for the specified phase, if a path to the data is
        specified.
        """
        if self.data_paths[phase] is not None:
            if phase != 'predict' and not self.target_variables:
                rank_zero_warn(
                    f'No target variables were specified for the {phase} dataset. ' \
                    'If this was intended, please ignore this warning. Otherwise, ' \
                    'your model may fail to train or validate correctly.'
                    )
            
            if phase != 'predict':
                return XarrayDataset(
                    paths = self.data_paths[phase],
                    batch_size = self.batch_size,
                    input_variables = self.input_variables,
                    target_variables = self.target_variables if phase != 'predict' else [],
                    rank = rank,
                    world_size = world_size,
                    should_shuffle = phase == 'train',
                    transforms = self.transforms[phase]
                    )
            else:
                return [XarrayDataset(
                    paths = [path],
                    batch_size = self.batch_size,
                    input_variables = self.input_variables,
                    target_variables = self.target_variables if phase != 'predict' else [],
                    rank = rank,
                    world_size = world_size,
                    should_shuffle = phase == 'train',
                    transforms = self.transforms[phase]
                    ) for path in self.data_paths[phase]]
    

    def create_dataloaders(self, phase: str) -> DataLoader:
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
            if phase != 'predict':
                return DataLoader(
                    dataset = self.data[phase],
                    num_workers = self.num_workers,
                    batch_size = (None if self.num_workers <= 1 else self.num_workers),
                    multiprocessing_context = 'fork' if torch.backends.mps.is_available() and self.num_workers > 0 else None,
                    )
            else:
                return [DataLoader(
                    dataset = data,
                    num_workers = self.num_workers,
                    batch_size = (None if self.num_workers <= 1 else self.num_workers),
                    multiprocessing_context = 'fork' if torch.backends.mps.is_available() and self.num_workers > 0 else None,
                ) for data in self.data[phase]]