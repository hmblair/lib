# linear_regression.py

import numpy as np
from data.h5tools import TableSequence, HDF5File
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

class SKLearnBase:
    """
    A base class for all scikit-learn models.
    """
    def fit(self, x, y):
        """
        Fit the model to the data.

        Parameters:
        ----------
        x (numpy.ndarray):
            The input data.
        y (numpy.ndarray):
            The output data.
        """
        self.regressor.fit(x, y)


    def predict(self, x):
        """
        Make predictions using the model.

        Parameters:
        ----------
        x (numpy.ndarray):
            The input data.

        Returns:
        -------
        numpy.ndarray:
            The predictions.
        """
        return self.regressor.predict(x)
    

    def compute_mae(self, x, y):
        """
        Make predictions using the model and compute the mean absolute error.

        Parameters:
        ----------
        x (numpy.ndarray):
            The input data.
        y (numpy.ndarray):
            The output data.

        Returns:
        -------
        float:
            The mean absolute error.
        """
        preds = self.predict(x).astype(int)
        return mean_absolute_error(y, preds)


class LinearRegression(SKLearnBase):
    """
    A simple linear regression model.
    """
    def __init__(self):
        """
        Initialize the model.
        """
        self.regressor = LinearRegression()
    

class XGBoost(SKLearnBase):
    """
    A gradient boosted random forest model.
    """
    def __init__(
            self, 
            colsample_bytree : int,
            learning_rate : float,
            max_depth : int,
            alpha : int,
            n_estimators : int,
            ):
        """
        Initialize the model.
        """
        self.regressor = xgb.XGBRegressor(
            objective ='reg:absoluteerror', 
            colsample_bytree = colsample_bytree,
            learning_rate = learning_rate,
            max_depth = max_depth,
            alpha = alpha,
            n_estimators = n_estimators,
            )