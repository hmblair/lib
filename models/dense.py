# dense.py

import torch
import torch.nn as nn
from .utils import pairwise
from collections.abc import Sequence
from typing import Optional

class Pooling(nn.Module):
    """
    A layer which performs pooling on a specified dimension of the input tensor.

    Parameters:
    -----------
    pool_type (str):
        The type of pooling to perform. Must be one of None, 'mean', 'max', 
        'min', or 'sum'.
    dim (int):
        The dimension to perform the pooling on.
    nonlinearity (nn.Module, optional):
        A nonlinearity to apply after the pooling operation. Defaults to None.
    """
    def __init__(
            self, 
            pool_type : str, 
            dim : int, 
            nonlinearity : Optional[str] = None,
            ) -> None:
        super().__init__()
        pools = {
            None : lambda x : x,
            'mean' : lambda x : torch.mean(x, dim=dim),
            'max' : lambda x : torch.max(x, dim=dim).values,
            'min' : lambda x : torch.min(x, dim=dim).values,
            'sum' : lambda x : torch.sum(x, dim=dim),
        }
        if pool_type not in pools:
            raise ValueError(
                f'Invalid pool type {pool_type}. Must be one of {list(pools.keys())}.'
                )
        self.pool_fn = pools[pool_type]

        nonlinearities = {
            None : None,
            'relu' : nn.ReLU(),
            'exp' : torch.exp,
            'softplus' : nn.Softplus(),
            'sigmoid' : nn.Sigmoid(),
        }
        if nonlinearity is not None and nonlinearity not in nonlinearities:
            raise ValueError(
                f'Invalid nonlinearity {nonlinearity}. Must be one of {list(nonlinearities.keys())}.'
                )
        self.nonlinearity = nonlinearities[nonlinearity]
        self.dim = dim


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Perform the pooling operation on the input tensor.
        """
        x = self.pool_fn(x)
        if self.nonlinearity is not None:
            x = self.nonlinearity(x)
        return x
        



class DenseNetwork(nn.Module, Sequence):
    """
    A dense neural network with an arbitrary number of hidden layers.

    Parameters:
    -----------
    in_size (int): 
        The size of the input features.
    out_size (int): 
        The size of the output features.
    hidden_sizes (list): 
        A list of hidden layer sizes. Defaults to an empty list.
    bias (bool): 
        Whether to use bias in the linear layers. Defaults to True.
    dropout (float):
        The dropout probability. Defaults to 0.0.
    activation (nn.Module): 
        The activation function to use. Defaults to nn.ReLU().
    pool (dict, optional):
        A dictionary specifying the type of pooling to use. Defaults to None.
        The keys should be 'type' and 'dim', and the values should be the type
        of pooling to apply and the dimension to apply it to, respectively.
        A third key, 'nonlinearity', can be included to specify a nonlinearity
        to apply after the pooling operation. If not included, no nonlinearity
        will be applied.
    """
    def __init__(
            self,
            in_size : int,
            out_size : int,
            hidden_sizes: list = [],
            bias : bool = True,
            dropout : float = 0.0,
            activation : nn.Module = nn.ReLU(),
            pooling : Optional[dict] = None,
            ) -> None:
        super().__init__()
        # define the pooling layer if specified
        if pooling is not None:
            if not pooling.keys() >= {'type', 'dim'}:
                raise ValueError(
                    'Pooling dictionary must contain keys "type", "dim".'
                    )
            self.pooling = Pooling(
                pool_type=pooling['type'], 
                dim=pooling['dim'],
                nonlinearity=pooling.get('nonlinearity', None),
                )
        else:
            self.pooling = None

        # define the hidden sizes of the network
        features = [in_size] + hidden_sizes + [out_size]

        # construct the layers
        layers = []
        for l1, l2 in pairwise(features):
            layers.append(
                nn.Linear(l1, l2, bias)
                    )

        # store the layers and other relevant attributes
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # initialize the weights
        try:
            gain = nn.init.calculate_gain(activation.__class__.__name__.lower())
        except AttributeError:
            gain = 1.0
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the model. 

        Parameters:
        -----------
        x (torch.Tensor): 
            The input tensor, of shape (*b, in_size).

        Returns:
        --------
        torch.Tensor: 
            The output tensor, of shape (*b, out_size).
        """
        # apply each layer, save for the last, and corresponding dropout and 
        # activation
        for layer in self.layers[:-1]:
            x = self.dropout(
                self.activation(layer(x))
                )

        # apply the final layer, with no activation or dropout, and squeeze
        # the last dimension
        x = self.layers[-1](x).squeeze(-1)

        # apply pooling if specified
        if self.pooling is not None:
            x = self.pooling(x)
        
        return x
    

    def __getitem__(self, index : int) -> nn.Module:
        """
        Returns the layer at the given index.

        Parameters:
        -----------
        index (int): 
            The index of the layer to return.

        Returns:
        --------
        nn.Module: 
            The layer at the given index.
        """
        return self.layers[index]
    

    def __len__(self) -> int:
        """
        Returns the number of layers in the network.

        Returns:
        --------
        int: 
            The number of layers in the network.
        """
        return len(self.layers)