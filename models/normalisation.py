# normalisation.py

import torch
import torch.nn as nn

class BatchNorm(nn.BatchNorm1d):
    """
    Batch normalisation layer for 1D data with the batch dimension first.
    """
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the batch normalisation layer.

        Parameters:
        -----------
        x (torch.Tensor):
            The input tensor.

        Returns:
        --------
        torch.Tensor:
            The output tensor.
        """
        x = x.transpose(1, -1)
        x = super().forward(x)
        return x.transpose(1, -1)