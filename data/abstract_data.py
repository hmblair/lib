# base_data.py

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Union, Optional, Any
from collections.abc import Iterable, Sequence, Mapping
from abc import ABCMeta, abstractmethod
import warnings
import os

class BaseDataModule(pl.LightningDataModule, metaclass=ABCMeta):
    """
    Base class for Pytorch Lightning data modules that provides a standard
    interface for creating datasets and dataloaders. Abstracts away some of the
    boilerplate code for creating dataloaders that exists in the bare PL
    implementation.

    Parameters:
    ----------
    batch_size (int): 
        The batch size for the dataloaders.
    num_workers (int, optional): 
        The number of workers for data loading. If set to -1, the number of
        workers is set to the number of available CPUs. Defaults to -1.
    
    Attributes:
    ----------
    data (dict[str, Sequence | Iterable]): 
        A dictionary containing the datasets for each phase. This is initialised
        empty and is populated by the setup method.
    batch_size (int): 
        The batch size for the dataloaders.
    rank (int):
        The rank of the current process, as determined by the trainer object.
        If no trainer object is found, the rank is set to 0.
    world_size (int):
        The total number of processes, as determined by the trainer object.
        If no trainer object is found, the world size is set to 1.
    num_workers (int): 
        The number of workers to be used for data loading. 

    Methods:
    -------
    _create_datasets: 
        Creates the dataset for the specified phase. Must be implemented by the 
        subclass.
    _create_dataloaders: 
        Creates the dataloader for the specified phase. Can be overwritten by 
        a subclass.
    setup: 
        Calls the _create_datasets method for the specified stage.
    train_dataloader: 
        Calls the _create_dataloaders method for the 'train' phase.
    val_dataloader: 
        Calls the _create_dataloaders method for the 'validate' phase.
    test_dataloader: 
        Calls the _create_dataloaders method for the 'test' phase.
    pred_dataloader: 
        Calls the _create_dataloaders method for the 'predict' phase.
    get_inputs:
        Extracts the inputs from the batch. Must be implemented by the subclass.
    get_targets:
        Extracts the targets from the batch. Must be implemented by the subclass.
    on_before_batch_transfer:
        Called before a batch is transferred to the device. This method extracts
        the inputs and, if the phase not 'predict', the targets from the batch.
    compute_losses:
        Computes the losses given inputs and outputs. Must be implemented by the
        subclass.
    """
    def __init__(
            self, 
            batch_size : int, 
            num_workers : int = -1,
            ) -> None:
        super().__init__()

        # initialise the data dictionary
        self.data = {}

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
        if self.trainer is not None:
            rank = self.trainer.global_rank
            world_size = self.trainer.world_size
        else:
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
        Create a dataset for the specified phase.

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
        
        if phase not in self.data:
            raise ValueError(
                f'There is no {phase} dataset. Please call the setup method with the appropriate stage first.'
                )

        return DataLoader(
            dataset = self.data[phase],
            num_workers = self.num_workers,
            batch_size = self.batch_size,
            shuffle = (phase == 'train'),
        )
    

    def prepare_data(self) -> None:
        """
        Prepares the data for the data module. This method is called only on 
        the root process. If you need to perform operations on the data that
        should only be done once, such as downloading, this is the place to do
        so. The default implementation does nothing.
        """
        return super().prepare_data()
    

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
            self.data['train'] = self._create_datasets('train', rank, world_size)
            self.data['validate'] = self._create_datasets('validate', rank, world_size)
        elif stage in ['test', 'validate', 'predict']:
            self.data[stage] = self._create_datasets(stage, rank, world_size)
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
        if self.data['validate'] is not None:
            return self._create_dataloaders('validate')
        else:
            return super().val_dataloader()


    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader.

        Returns:
        -------
        torch.utils.data.DataLoader: 
            The test dataloader.
        """
        return self._create_dataloaders('test')


    def predict_dataloader(self) -> DataLoader:
        """
        Returns the prediction dataloader.

        Returns:
        -------
        torch.utils.data.DataLoader: 
            The prediction dataloader.
        """
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
        

    @abstractmethod
    def compute_losses(
            self, 
            model_outputs: torch.Tensor, 
            targets: torch.Tensor, 
            ) -> dict[str, torch.Tensor]:
        """
        Compute the losses given inputs and outputs. The loss named 'loss' will 
        be the one which is used to train the model.

        Parameters:
        ----------
        model_outputs (torch.Tensor): 
            The outputs of the model.
        targets (torch.Tensor):
            The targets.

        Returns:
        -------
        dict[str, torch.Tensor]: 
            A dictionary containing the computed losses and their respective 
            names.
        """
        return