# conv.py

import torch
import torch.nn as nn
from .utils import pairwise
from .dense import DenseNetwork

def construct_cnn(
        features : list[int],
        kernel_size : int,
        ) -> None:
    layers = []
    for l1, l2 in pairwise(features):
        layers.append(
            nn.Conv1d(
                in_channels = l1,
                out_channels = l2,
                kernel_size = kernel_size,
                padding = 'same'
                )
            )
    return nn.ModuleList(layers)



class TemporalConvolutionalNetwork(nn.Module):
    """
    A simple implementation of a Temporal Convolutional Network (TCN), including
    skip connections and a final dense layer.

    Parameters:
    ----------
    in_size (int):
        The size of the input tensor.
    out_size (int):
        The size of the output tensor.
    kernel_size (int):
        The size of the kernel.
    hidden_sizes (list[int]):
        The hidden sizes of the network. Defaults to an empty list.
    dropout (float):
        The dropout rate. Defaults to 0.0.
    activation (nn.Module):
        The activation function. Defaults to nn.ReLU().
    """
    def __init__(
            self,
            in_size : int,
            out_size : int,
            kernel_size : int,
            hidden_sizes : list[int] = [],
            dropout : float = 0.0,
            activation : nn.Module = nn.ReLU(),
            ) -> None:
        super().__init__()
        
        # the hidden sizes of the network
        features = [in_size] + hidden_sizes + [out_size]

        # construct the layers
        self.layers = construct_cnn(features, kernel_size)

        # store the activation
        self.activation = activation

        # create the dropout layer
        self.dropout = nn.Dropout1d(dropout)

        # create the final layer
        self.dense = DenseNetwork(
            in_size = out_size,
            out_size = out_size,
            dropout = dropout,
            activation = activation,
            )


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # permute, since conv1d likes that
        x = torch.transpose(x, -2, -1)

        # apply each sublayer
        for layer in self.layers:
            x = self.activation(
                self.dropout(layer(x))
                ) + x
        
        # permute back
        x = torch.transpose(x, -2, -1)

        # apply the final dense layer without the activation or dropout
        return self.dense(x)