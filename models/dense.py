# dense.py

import torch
import torch.nn as nn
from .pairwise import pairwise

class DenseNetwork(nn.Module):
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
    activation (nn.Module): 
        The activation function to use. Defaults to nn.ReLU().

    Attributes:
    ----------
    layers (nn.ModuleList): 
        The layers of the network.
    activation (nn.Module): 
        The activation function to use.

    Inherits:
    ---------
    nn.Module: 
        The base PyTorch module class.

    Methods:
    --------
    forward(): 
        The forward pass of the model.
    """
    def __init__(
            self,
            in_size : int,
            out_size : int,
            hidden_sizes: list = [],
            bias : bool = True,
            activation : nn.Module = nn.ReLU()
            ) -> None:
        super().__init__()

        # the hidden sizes of the network
        features = [in_size] + hidden_sizes + [out_size]

        # construct the layers
        layers = []
        for l1, l2 in pairwise(features):
            layers.append(
                nn.Linear(l1, l2, bias)
                    )

        # store the layers and activation
        self.layers = nn.ModuleList(layers)
        self.activation = activation


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the model.

        Parameters:
        -----------
        x (torch.Tensor): 
            The input tensor.

        Returns:
        --------
        torch.Tensor: 
            The output tensor.
        """
        # apply each layer, save for the last, and activation
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        # apply the final layer, with no activation
        x = self.layers[-1](x)

        # squeeze the final dimension if applicable
        return x.squeeze(-1)