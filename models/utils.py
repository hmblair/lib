# utils.py

import psutil
from typing import Callable
from itertools import tee
import torch
import torch.nn as nn

def pairwise(iterable):
    """
    Iterate over an iterable in consecutive pairs. For example, if the iterable 
    is [1, 2, 3, 4], then the pairs are (1, 2), (2, 3), (3, 4). 

    Parameters:
    -----------
    iterable (iterable): 
        The iterable to iterate over.

    Returns:
    --------
    iterable: 
        An iterable over consecutive pairs of elements.
    """
    # create two copies of the iterable
    a, b = tee(iterable)

    # advance the second copy by one
    next(b, None)

    # return the zipped iterable
    return zip(a, b)


def is_leaf_module(module: nn.Module) -> bool:
    """
    Returns True if the module has no children.

    Parameters:
    -----------
    module (nn.Module): 
        The module to be checked.

    Returns:
    --------
    bool:
        True if the module has no children.
    """
    return not list(module.children())


def get_mem_usage() -> float:
    """
    Retrieves the current memory usage of the entire system.

    Returns:
    --------
    float: 
        The current memory usage.
    """
    process = psutil.Process()
    return float(process.memory_info().rss)


def get_gpu_mem_usage() -> tuple[float, float]:
    """
    Retrieves the current absolute and relative memory usage of the GPU.

    Returns:
    --------
    tuple[float, float]: 
        The current absolute and relative GPU memory usage.
    """
    available, total = torch.cuda.mem_get_info()
    return float(available), float(available) / float(total)


def log_mem_usage(f : Callable) -> Callable:
    """
    A decorator to log the memory usage of the system after the function is
    called.

    Parameters:
    ----------
    f (Callable): 
        The function to be decorated.

    Returns:
    --------
    Callable: 
        The decorated function.
    """
    def wrapper(self, *args, **kwargs):
        result = f(self, *args, **kwargs)
        mem_usage = get_mem_usage()
        self._log('mem_usage', mem_usage, on_epoch=False)
        if self.device.type == 'cuda':
            abs_gpu_mem_usage, rel_gpu_mem_usage = get_gpu_mem_usage()
            self._log('abs_gpu_mem_usage', abs_gpu_mem_usage, on_epoch=False)
            self._log('rel_gpu_mem_usage', rel_gpu_mem_usage, on_epoch=False)
        return result
    return wrapper