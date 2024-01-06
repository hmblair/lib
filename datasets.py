# datasets.py

import torch
from torch.utils.data import Dataset
from typing import Any, Iterable, Sequence, Union
from torch.utils.data import IterableDataset

class DistributedIterableDataset(IterableDataset):
    def __init__(self, rank : int = 0, world_size : int = 1) -> None:
        self.rank = rank
        self.world_size = world_size
        if self.rank >= self.world_size:
            raise ValueError(
                'The rank must be less than the world size.'
                )
        
        
    def split(self, tensor : torch.Tensor) -> Iterable[torch.Tensor]:
        """
        Split a tensor into chunks of equal size, one for each process.
        """
        return tensor[self.rank::self.world_size]
    

def slice_output(func):
    def wrapper(self, *args, **kwargs):
        generator = func(self, *args, **kwargs)
        for value in generator:
            yield value[self.rank::self.world_size]
    return wrapper


class SimpleDataset(Dataset):
    """
    A simple PyTorch dataset, that allows for indexing the input and target
    tensors.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor.
    y : torch.Tensor
        The target tensor.
    """
    def __init__(
            self, x : torch.Tensor, y : torch.Tensor, 
            ) -> None:
        super().__init__()
        if not len(x) == len(y):
            raise ValueError(
                'The input and target tensors must have the same length.'
                )
        self.x = x
        self.y = y


    def __len__(self) -> int:
        """
        Return the length of the dataset.
        """
        return len(self.x)


    def __getitem__(
            self, 
            idx : Union[int, list, slice],
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the input and target tensors for a given index.

        Parameters
        ----------
        idx : int | list | slice
            The index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The input and target tensors.
        """
        return self.x[idx], self.y[idx]
    


from typing import Sequence
def group(seq : Sequence, group_size : int) -> list[Sequence]:
    """
    Group a sequence into groups of a given size.
    """
    return [seq[i:i+group_size] for i in range(0, len(seq), group_size)]


def ceildiv(a : int, b : int) -> int:
    """
    Divide a by b and round up to the nearest integer.
    """
    return -(-a // b)


from typing import Iterable
import random
class VariableLengthIterableDataset(DistributedIterableDataset):
    def __init__(
            self, 
            data : list[torch.Tensor],
            batch_size : int,
            shuffle : bool = True,
            *args, **kwargs,
            ) -> None:
        super().__init__(*args, **kwargs)
        self.data = [self.split(tensor) for tensor in data]
        self.batch_size = batch_size
        self.shuffle = shuffle


    def get_ix(self) -> Iterable[list[int]]:
        indices = [
            list(range(len(tensor)))
            for tensor in self.data
            ]
        while indices:
            for ix in indices:
                if self.shuffle:
                    random.shuffle(ix)
                yield group(ix, self.batch_size)
    

    def __iter__(self) -> Iterable[torch.Tensor]:
        ix_generator = self.get_ix()
        for tensor in self.data:
            grouped_ix = next(ix_generator)
            for ix in grouped_ix:
                yield tensor[ix]


    def __len__(self) -> int:
        return sum(
            ceildiv(len(tensor), self.batch_size)
            for tensor in self.data
            )
    

from .data.h5tools import HDF5File
class VariableLengthIterableHDF5Dataset(VariableLengthIterableDataset):
    def __init__(
            self, 
            path : str,
            root_uep : str,
            batch_size : int,
            *args, **kwargs,
            ) -> None:
        file = HDF5File(
            path=path,
            root_uep=root_uep
        )
        data = []
        for table in file.values():
            data.append(table.read())
        super().__init__(
            batch_size=batch_size,
            data=data,
            *args, **kwargs,
            )
        
# bs = 256
# dataset = VariableLengthIterableHDF5Dataset(
#     path='../Data/data.h5',
#     table='table_train',
#     root_uep='/train',
#     batch_size=bs,
#     rank=0,
#     world_size=1,
#     )

# from time import time
# start = time()
# i=0
# for batch in dataset:
#     i+=1
#     print(batch.shape)
# end = time()

# print(f'Time taken: {end-start:.2f} seconds')

# print(len(dataset))
# print(i)


# from data.h5tools import MappingIterableDataset
# file = HDF5File(
#     path='../Data/data.h5',
#     root_uep='/train', 
#     read_only=True,
#     )
# data = MappingIterableDataset(
#     mapping = file,
#     batch_size = bs,
#     rank = 0,
#     world_size = 1,
# )

# i=0
# start = time()
# for batch in data:
#     i+=1
#     if i == len(data):
#         break
# end = time()

# print(f'Time taken: {end-start:.2f} seconds')

# print(len(data)) 


