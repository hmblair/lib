# losses.py

import torch
import torch.nn as nn

class CrossEntropy(nn.Module):
    """
    The cross-entropy loss function.
    """
    def __init__(self) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-entropy between the given tensors.

        Parameters:
        ----------
        x (torch.Tensor):
            The data tensor, in logit form.
        y (torch.Tensor):
            The target tensor, containing the class labels as integers. This
            will be cast to long.

        Returns:
        -------
        torch.Tensor:
            The cross-entropy between the given tensors.
        """
        print(x)
        print(y.long())
        return self.ce(x, y.long())



class Accuracy(nn.Module):
    """
    The accuracy loss function.
    """
    def __init__(self) -> None:
        super().__init__()
    

    def forward(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        """
        Compute the accuracy between the given tensors.

        Parameters:
        ----------
        x (torch.Tensor):
            The data tensor, in logit form.
        y (torch.Tensor):
            The target tensor, containing the class labels as integers.

        Returns:
        -------
        torch.Tensor:
            The accuracy between most likely class labels and the target labels.
        """
        p = torch.argmax(x, dim=1)
        return (p == y).float().mean()
    


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



class MetricLoss(nn.Module):
    """
    Computes the difference between the pairwise distances of the inputs, 
    and the pairwise distances of the targets. 
    """
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def compute_pairwise_distances(self, x : torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return (x[:, None] - x[None, :]) ** 2
        return torch.sum(
            (x[:, None] - x[None, :]) ** 2, 
            dim=-1,
            )

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
        # compute the pairwise distances
        pairwise_distances_x = self.compute_pairwise_distances(x)
        pairwise_distances_y = self.compute_pairwise_distances(torch.log(y))
        # compute the difference between the pairwise distances
        return self.mse(pairwise_distances_x, pairwise_distances_y)