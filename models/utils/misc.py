# misc.py

from itertools import tee
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