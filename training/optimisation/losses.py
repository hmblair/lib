# losses.py

import torch
import torch.nn as nn

class LogMSELoss(nn.Module):
    """
    The logarithmic mean squared error loss function.
    """
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        """
        Compute the logarithmic mean squared error between the given tensors.

        Parameters:
        ----------
        x (torch.Tensor):
            The data tensor.
        y (torch.Tensor):
            The target tensor.

        Returns:
        -------
        torch.Tensor:
            The logarithmic mean squared error between the given tensors.
        """
        return self.mse(torch.log(x), torch.log(y))
    

class LogNormalLoss(nn.Module):
    def __init__(self, sigma : float = 1) -> None:
        super().__init__()
        self.logmse = LogMSELoss()
        self.sigma = sigma
    
    def forward(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        """
        Compute the NLL for a log-normal distribution with means given by y and
        standard deviations given by sigma.

        Parameters:
        ----------
        x (torch.Tensor):
            The data tensor.
        y (torch.Tensor):
            The target tensor.

        Returns:
        -------
        torch.Tensor:
            The logarithmic mean squared error between the given tensors.
        """
        return self.logmse(x, y) + self.sigma * torch.log(x)
