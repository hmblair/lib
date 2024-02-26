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
    


def stack_xarray(ds : xr.Dataset) -> np.ndarray:
    """
    Stack the variables in an xarray dataset into a numpy array.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to stack.

    Returns
    -------
    np.ndarray
        The stacked dataset.
    """
    return np.stack([ds[name].values for name in ds.data_vars], axis=0)



class netCDFIterableDatasetBase(IterableDataset):
    """
    A PyTorch iterable dataset, that allows for iterating over batches taken 
    from a netCDF dataset, optionally shuffling the data. The dataset makes use
    of Xarray to interface with the netCDF dataset.

    Parameters:
    ----------
    path (str):
        The path to the netCDF dataset.
    variables (list[str]):
        The variables to load from the netCDF dataset.
    batch_size (int):
        The batch size.
    rank (int):
        The rank of the current device. Defaults to 0.
    world_size (int):
        The number of devices. Defaults to 1.
    should_shuffle (bool):
        Whether the dataset should be shuffled. Defaults to True.
    batch_dimension (str):
        The name of the batch dimension. Defaults to 'batch'.
    transform (Callable[[xr.Dataset], np.ndarray], optional):
        A transform to apply to the dataset. If None, the dataset is simply
        converted to a numpy array by stacking the variables. Defaults to None.
    """
    def __init__(
            self, 
            path : str,
            batch_size : int, 
            rank : int = 0,
            world_size : int = 1,
            should_shuffle : bool = True,
            batch_dimension : str = 'batch',
            transform : Callable[[xr.Dataset], np.ndarray] = None,
            ) -> None:
        if not os.path.exists(path):    
            raise ValueError(f'The path "{path}" does not exist.')
        self.ds = xr.open_dataset(path, engine='h5netcdf')
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

        # store whether the dataset should be shuffled
        self.should_shuffle = should_shuffle

        # store the transform
        self.transform = transform if transform is not None else stack_xarray
        if not callable(self.transform):
            raise ValueError('The transform must be callable.')


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
    

    def __iter__(self) -> Iterable:
        """
        Iterate over the dataset in batches, shuffling the dataset along the
        batch dimension if specified.
        """
        # shuffle the dataset if specified
        if self.should_shuffle:
            self.shuffle()    
        # iterate over the slices, transforming the batches as necessary
        for s in self.slices:
            batch = self.ds.isel(batch=s)
            yield self.transform(batch)



class netCDFIterableDataset(IterableDataset):
    """
    A wrapper for multiple netCDF datasets, allowing for iterating over batches
    taken from each dataset, optionally shuffling the data.
    """
    def __init__(
            self, 
            paths : list[str], 
            batch_size : int,
            rank : int = 0,
            world_size : int = 1,
            should_shuffle : bool = True,
            transform : Callable[[xr.Dataset], np.ndarray] = None,
            ) -> None:
        self.data = [
            netCDFIterableDatasetBase(
                path = path,
                batch_size = batch_size,
                rank = rank,
                world_size = world_size,
                should_shuffle = should_shuffle,
                transform = transform,
                )
            for path in paths
            ]
    

    def __iter__(self) -> Iterable:
        """
        Iterate over the dataset in batches, shuffling the dataset along the
        batch dimension if specified.
        """
        while True:
            for d in self.data:
                for batch in d:
                    yield batch

    
    def __len__(self) -> int:
        """
        Return the number of batches across all datasets.
        """
        return sum([len(d) for d in self.data])