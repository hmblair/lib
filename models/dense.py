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
    dropout (float):
        The dropout probability. Defaults to 0.0.
    activation (nn.Module): 
        The activation function to use. Defaults to nn.ReLU().

    Attributes:
    ----------
    layers (nn.ModuleList): 
        The layers of the network.
    dropout (nn.Dropout):
        The dropout layer.
    activation (nn.Module): 
        The activation function to use.
    """
    def __init__(
            self,
            in_size : int,
            out_size : int,
            hidden_sizes: list = [],
            bias : bool = True,
            dropout : float = 0.0,
            activation : nn.Module = nn.ReLU(),
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

        # store the layers, dropout, and activation
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the model. The input has shape (*b, in_size), and
        the output has shape (*b, out_size).

        Parameters:
        -----------
        x (torch.Tensor): 
            The input tensor.

        Returns:
        --------
        torch.Tensor: 
            The output tensor.
        """
        # apply each layer, save for the last, and corresponding dropout and 
        # activation
        for layer in self.layers[:-1]:
            x = self.dropout(
                self.activation(layer(x))
                )

        # apply the final layer, with no activation or dropout
        x = self.layers[-1](x)

        # squeeze the final dimension if applicable
        return x.squeeze(-1)