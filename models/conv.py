# conv.py

import torch
import torch.nn as nn
from typing import Optional

from .pairwise import pairwise
from .dense import DenseNetwork


class ConvLayer(nn.Module):
    """
    A single multiscale-convolution layer.

    Args:
        in_size (int): The number of input features.
        out_size (int): The number of output features.
        kernel_sizes (list): A list of kernel sizes corresponding to the multiple scales of the convolution.
        conv_bias (bool): Whether or not to include a bias term in the convolution.
        pooling_kernel_size (int, optional): The size of the pooling kernel. Defaults to None.
        dilation (int, optional): The dilation factor. Defaults to 1.

    Attributes:
        layers (nn.ModuleList): The layers of the network.
        activation (nn.Module): The activation function to use.

    Inherits:
        nn.Module: The base PyTorch module class.

    Methods:
        forward(): The forward pass of the model.
    """
    def __init__(
            self,
            in_size : int,
            out_size : int,
            kernel_sizes : list,
            conv_bias : bool,
            pooling_kernel_size : Optional[int] = None,
            dilation : int = 1,
            ) -> None:
        super().__init__()

        if pooling_kernel_size is not None:
            self.pooling = nn.MaxPool1d(kernel_size = pooling_kernel_size,
                                        stride = 1)


        # if there is more than one convolution kernel, then the multiple kernels
        # will need to be combined using a linear layer
        if len(kernel_sizes) > 1:
            self.dense = DenseNetwork(
                in_size = len(kernel_sizes),
                out_size = out_size,
                bias = False
                )

        # consruct the layers
        layers = []
        for kernel_size in kernel_sizes:
            if kernel_size % 2 == 0:
                print('The kernel sizes must be odd (for now)')
                kernel_size +=1
            layers.append(
                nn.Conv1d(
                    in_channels = in_size,
                    out_channels = out_size,
                    kernel_size = kernel_size,
                    bias = conv_bias,
                    padding = 'same',
                    dilation = dilation
                    )
                    )

        self.layers = nn.ModuleList(layers)


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
        convs = []
        for sublayer in self.layers:
            convs.append(sublayer(x))

        # bring together the multiscale convolutions
        x = torch.stack(convs, dim = -1)

        # apply the dense layer if applicable
        if hasattr(self, 'dense'):
            x = self.dense(x)

        # squeeze
        x = x.squeeze(-1)

        # apply the pooling layer if applicable
        if hasattr(self, 'pooling'):
            x = self.pooling(x)

        return torch.transpose(x, -2, -1)
    



class CNN(nn.Module):
    def __init__(
            self,
            num_embeddings : int,
            out_size : int,
            kernel_sizes : list[int],
            hidden_sizes : list[int] = [],
            conv_bias : bool = True,
            pooling_kernel_size : Optional[int] = None,
            dilation : int = 1,
            activation : nn.Module = nn.ReLU()
            ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings,
            embedding_dim = hidden_sizes[0]
            )

        # the hidden sizes of the network
        features = [num_embeddings] + hidden_sizes + [out_size]

        # construct the layers
        layers = []
        for l1, l2 in pairwise(features):
            layers.append(
                ConvLayer(
                    in_size = l1,
                    out_size = l2,
                    kernel_sizes = kernel_sizes,
                    conv_bias = conv_bias,
                    dilation = dilation,
                    pooling_kernel_size = pooling_kernel_size
                    )
                    )

        # store the layers and activation
        self.layers = nn.ModuleList(layers)
        self.activation = activation


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)





class BareCNN(nn.Module):
    def __init__(
            self,
            num_embeddings : int,
            hidden_size : int,
            kernel_size : int,
            out_size : int,
            num_layers : int,       
            dropout : float = 0.0,
            conv_bias : bool = True,
            ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings, hidden_size)
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels = hidden_size,
                    out_channels = hidden_size,
                    kernel_size = kernel_size,
                    bias = conv_bias,
                    padding = 'same'
                    )
                    )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_size, out_size)


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x.long()
        x = self.embedding(x)
        x = torch.transpose(x, -2, -1)
        x = self.layers(x)
        x = torch.transpose(x, -2, -1)
        x = self.linear(x)
        return x.squeeze(-1)