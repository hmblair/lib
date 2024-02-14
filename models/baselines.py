# baselines.py

import numpy as np
import json
from sklearn.base import BaseEstimator
import torch
import torch.nn as nn

class BayesEstimator(BaseEstimator):
    """
    An empirical Bayes estimator (lookup table) for the mean and variance of a 
    target variable given an input variable. The estimates are based on the 
    empirical mean and variance of the target variable for each unique value of 
    the input variable.

    Parameters
    ----------
    nearest_neighboor : bool
        Whether to use the nearest neighboor in the train set to predict the mean 
        and variance. If False, and the input x to self.predict is not in 
        self.x_unique, then an error will be raised. Default is True.
    """
    def __init__(
            self,
            nearest_neighboor : bool = True,
            ) -> None:
        # whether to use the nearest neighboor to predict the mean and variance
        self.nearest_neighboor = nearest_neighboor
        # initialize the attributes
        self.x_unique = None
        self.y_by_x = None
        self.mean_by_x = None
        self.var_by_x = None


    def fit(self, x : np.ndarray, y : np.ndarray) -> None:
        """
        Fit the empirical Bayes model.

        Parameters
        ----------
        x : np.ndarray
            The input values, of shape (n,).
        y : np.ndarray
            The target values, of shape (n,).
        """
        # get the unique values of x
        self.x_unique = np.unique(x)
        # split the data by x
        self.y_by_x = {
            x_val: y[x == x_val]
            for x_val in self.x_unique
        }
        # get the mean and variance of each y
        self.mean_by_x = {
            x_val: y_vals.mean()
            for x_val, y_vals in self.y_by_x.items()
        }
        self.var_by_x = {
            x_val: y_vals.var()
            for x_val, y_vals in self.y_by_x.items()
        }


    def predict(self, x : np.ndarray) -> np.ndarray:
        """
        Predict the mean and variance of the y values for the given x values.

        Parameters
        ----------
        x : np.ndarray
            The input values, of shape (n,).

        Returns
        -------
        mean : np.ndarray
            The predicted mean of the y values, of shape (n,).
        var : np.ndarray
            The predicted variance of the y values, of shape (n,).
        """
        if self.nearest_neighboor:
            # project the x values to the nearest neighbor in self.x_unique
            x = np.array([
                self.x_unique[np.abs(self.x_unique - x_val).argmin()]
                for x_val in x
            ])

        mean = np.array([
            self.mean_by_x[x_val]
            for x_val in x
        ])
        var = np.array([
            self.var_by_x[x_val]
            for x_val in x
        ])
        return mean, var
    

    def load(self, path : str) -> None:
        """
        Load the empirical Bayes model from a JSON file.

        Parameters
        ----------
        path : str
            The path to the file.
        """
        with open(path, 'r') as file:
            data = json.load(file)
        if not all(
            key in data for key in ['x_unique', 'mean_by_x', 'var_by_x']
            ):
            raise ValueError(
                f'Invalid data. The file must contain the keys'
                f'" x_unique", "mean_by_x", and "var_by_x".'
                )
        self.x_unique = np.array(data['x_unique'])
        self.mean_by_x = {
            float(x_val): mean
            for x_val, mean in data['mean_by_x'].items()
        }
        self.var_by_x = {
            float(x_val): var
            for x_val, var in data['var_by_x'].items()
        }
    

    def save(self, path : str) -> None:
        """
        Save the empirical Bayes model to a JSON file.

        Parameters
        ----------
        path : str
            The path to the file.
        """
        data = {
            'x_unique': self.x_unique.tolist(),
            'mean_by_x': {
                str(x_val): mean
                for x_val, mean in self.mean_by_x.items()
            },
            'var_by_x': {
                str(x_val): var
                for x_val, var in self.var_by_x.items()
            }
        }
        with open(path, 'w') as file:
            json.dump(data, file)



class Constant(nn.Module):
    """
    A module that returns a constant tensor of the same shape as the input.
    The constant value is a learnable parameter.
    """
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(
            torch.randn(1),
            requires_grad=True,
            )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Returns a tensor of the same shape as the input, with all values set to
        the value of the constant.

        Parameters:
        ----------
        x (torch.Tensor):
            The input tensor.

        Returns:
        -------
        torch.Tensor:
            The output constant tensor.
        """
        return torch.ones_like(x) * self.value