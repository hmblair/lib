# datasets.py

import os
from typing import Union, Sequence, Iterable, Callable
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, IterableDataset
from .utils import construct_slices_for_iterable_dataset

class SimpleDataset(Dataset):
    """
    A simple PyTorch dataset, that allows for indexing a sequence of tensors.

    Parameters
    ----------
    data : Sequence
        The data to be loaded.
    """
    def __init__(self, data : Sequence) -> None:
        super().__init__()
        if not all(len(array) == len(data[0]) for array in data):
            raise ValueError(
                'All tensors must have the same length.'
                )
        self.data = data


    def __len__(self) -> int:
        """
        Return the length of the dataset.
        """
        return len(self.data[0])


    def __getitem__(
            self, 
            idx : Union[int, list, slice],
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return all tensors at the given index.

        Parameters
        ----------
        idx : int | list | slice
            The index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The input and target tensors.
        """
        return tuple(array[idx] for array in self.data)
    


class SimpleIterableDataset(IterableDataset):
    """
    A simple PyTorch iterable dataset, that allows for iterating over a sequence
    of tensors in batches, optionally shuffling the data.

    Parameters
    ----------
    data : Sequence
        The tensors to iterate over.
    batch_size : int
        The batch size.
    rank : int
        The rank of the current process.
    world_size : int
        The total number of processes.
    should_shuffle : bool
        Whether to shuffle the data.
    """
    def __init__(
            self, 
            data : tuple[torch.Tensor], 
            batch_size : int,
            rank : int = 0,
            world_size : int = 1,
            should_shuffle : bool = True,
            ) -> None:
        super().__init__()
        if not all(len(array) == len(data[0]) for array in data):
            raise ValueError(
                'All tensors must have the same length.'
                )
        self.data = data
        self.batch_size = batch_size
        self.slices = [
            slice(i, i+batch_size)
            for i in range(0, len(self.data[0]), batch_size)
            ]
        self.slices = self.slices[rank::world_size]
        self.should_shuffle = should_shuffle


    def __iter__(self) -> Iterable[tuple[torch.Tensor]]:
        """
        Return an iterator over the dataset.
        """
        while True:
            for slice in self.slices:
                yield tuple(array[slice] for array in self.data)
            if self.should_shuffle:
                ix = torch.randperm(len(self.data[0]))
                self.tensors = tuple(array[ix] for array in self.data)


    def __len__(self) -> int:
        """
        Return the length of the dataset.
        """
        return len(self.slices)
    


def stack_xarray(ds : xr.Dataset, variables : list[str]) -> np.ndarray | None:
    """
    Stack the variables in an xarray dataset into a numpy array. If the list of
    variables is empty, return None.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to stack.

    Returns
    -------
    np.ndarray | None
        The stacked dataset, or None if the list of variables is empty.
    """

    # if the list of variables is empty, return None
    if not variables:
        return None
    
    # stack the variables into a numpy array
    stack = np.stack([ds[name].values for name in variables], axis=-1)

    # if the stack has a single dimension, remove it
    if stack.shape[-1] == 1:
        stack = stack[..., 0]
    return stack



class netCDFIterableDatasetBase(IterableDataset):
    """
    A PyTorch iterable dataset, that allows for iterating over batches taken 
    from a netCDF dataset, optionally shuffling the data. The dataset makes use
    of Xarray to interface with the netCDF dataset.

    Parameters:
    ----------
    path (str):
        The path to the netCDF dataset.
    batch_size (int):
        The batch size.
    input_variables (list[str]):
        The names of the input variables.
    target_variables (list[str], optional):
        The names of the target variables. Defaults to None.
    rank (int):
        The rank of the current device. Defaults to 0.
    world_size (int):
        The number of devices. Defaults to 1.
    should_shuffle (bool):
        Whether the dataset should be shuffled. Defaults to True.
    batch_dimension (str):
        The name of the batch dimension. Defaults to 'batch'.
    transforms (list[Callable[[xr.Dataset], xr.Dataset]], optional):
        A list of transforms to apply to the dataset. Defaults to an empty list.
    """
    def __init__(
            self, 
            path : str,
            batch_size : int, 
            input_variables : list[tuple[str, str]] = [],
            target_variables : list[tuple[str, str]] = [],
            rank : int = 0,
            world_size : int = 1,
            should_shuffle : bool = True,
            batch_dimension : str = 'batch',
            transforms : list[Callable[[xr.Dataset], xr.Dataset]] = [],
            ) -> None:
        # verify that the path exists, and open the dataset
        if not os.path.exists(path):    
            raise ValueError(f'The path "{path}" does not exist.')
        self.ds = xr.open_dataset(path, engine='h5netcdf')

        # verify that the batch dimension exists in the dataset, and store the
        # number of datapoints and the batch dimension
        if not batch_dimension in self.ds.dims:
            raise ValueError(
                f'The specified batch dimension "{batch_dimension}" does not exist in the dataset.'
                )
        self.num_datapoints = self.ds.sizes[batch_dimension]
        self.batch_dimension = getattr(self.ds, batch_dimension)
        
        # construct a list of slices that will be used to iterate over the dataset
        self.slices = construct_slices_for_iterable_dataset(
            num_datapoints=self.num_datapoints,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            )
        
        # verify that the input and target variables exist in the dataset, and
        # store the input and target variables
        # for variable in input_variables + target_variables:
        #     if not variable in self.ds.data_vars:
        #         raise ValueError(
        #             f'The variable "{variable}" does not exist in the dataset.'
        #             )
        # if not input_variables:
        #     raise ValueError('At least one input variable must be specified.')
        self.input_variables = input_variables
        self.target_variables = target_variables

        # store whether the dataset should be shuffled
        self.should_shuffle = should_shuffle

        # store the transforms
        self.transforms = transforms


    def __len__(self) -> int:
        """
        Return the number of batches in the dataset.
        """
        return len(self.slices)
    

    def shuffle(self) -> None:
        """
        Shuffle the dataset along the batch dimension.
        """
        ix = np.random.permutation(
            self.batch_dimension.values
            )
        self.ds = self.ds.isel(batch=ix)
    

    def __iter__(self) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """
        Iterate over the dataset in batches, shuffling the dataset along the
        batch dimension if specified. The input and target variables of each 
        batch are stacked into numpy arrays and yielded. If the list of target
        variables is empty, the targets are set to None.
        """

        # shuffle the dataset if specified
        if self.should_shuffle:
            self.shuffle()    

        # iterate over the slices, transforming the batches as necessary, and
        # yielding the slice at the input and target variables
        for s in self.slices:
            batch = self.ds.isel(batch=s)
            for transform in self.transforms:
                batch = transform(batch)
            
            # get the input and target variables
            x = {name : batch[var].values for var, name in self.input_variables} if self.input_variables else None
            y = {name : batch[var].values for var, name in self.target_variables} if self.target_variables else None
            yield x, y



class netCDFIterableDataset(IterableDataset):
    """
    A wrapper for multiple netCDF datasets, allowing for iterating over batches
    taken from each dataset, optionally shuffling the data.
    """
    def __init__(
            self, 
            paths : list[str], 
            batch_size : int,
            input_variables : list[tuple[str, str]] = [],
            target_variables : list[tuple[str, str]] = [],
            rank : int = 0,
            world_size : int = 1,
            should_shuffle : bool = True,
            transforms : list[Callable[[xr.Dataset], xr.Dataset]] = [],
            ) -> None:
        # verify that at least one path is specified
        if not paths:
            raise ValueError('At least one path must be specified.')
        
        # initialize the datasets
        self.data = [
            netCDFIterableDatasetBase(
                path = path,
                batch_size = batch_size,
                input_variables = input_variables,
                target_variables = target_variables,
                rank = rank,
                world_size = world_size,
                should_shuffle = should_shuffle,
                transforms = transforms,
                )
            for path in paths
            ]
    

    def __iter__(self) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """
        Iterate over the dataset in batches, shuffling the dataset along the
        batch dimension if specified.
        """
        while True:
            for datum in self.data:
                for batch in datum:
                    yield batch

    
    def __len__(self) -> int:
        """
        Return the number of batches across all datasets.
        """
        return sum([len(d) for d in self.data])
    


import dgl

class DeepGraphLibraryIterableDataset(IterableDataset):
    """
    A PyTorch iterable dataset, that allows for iterating over batches taken
    from a list of DGL graphs, optionally shuffling the data.

    Parameters
    ----------
    graphs : list[DGLGraph]
        The graphs to iterate over.
    batch_size : int
        The batch size.
    rank : int
        The rank of the current device. Defaults to 0.
    world_size : int
        The number of devices. Defaults to 1.
    should_shuffle : bool
        Whether the dataset should be shuffled. Defaults to True.
    """
    def __init__(
            self, 
            graphs : list[dgl.DGLGraph],
            batch_size : int,
            rank : int = 0,
            world_size : int = 1,
            should_shuffle : bool = True,
            ) -> None:
        if not graphs:
            raise ValueError('At least one graph must be specified.')
        self.graphs = graphs

        self.slices = construct_slices_for_iterable_dataset(
            num_datapoints=len(self.graphs),
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            )
        
        self.should_shuffle = should_shuffle


    def __len__(self) -> int:
        """
        Return the number of batches in the dataset.
        """
        return len(self.slices)
    

    def shuffle(self) -> None:
        """
        Shuffle the graphs in the dataset.
        """
        ix = np.random.permutation(len(self.graphs))
        self.graphs = [self.graphs[i] for i in ix]


    def __iter__(self) -> Iterable[dgl.DGLGraph]:
        """
        Iterate over the dataset in batches, shuffling the dataset if specified.
        The graphs are yielded in DGL batches.
        """
        if self.should_shuffle:
            self.shuffle()
        for s in self.slices:
            yield dgl.batch(self.graphs[s])