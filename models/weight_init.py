# weight_init.py

import torch
import torch.nn as nn
import warnings
from typing import Union
import numpy as np
from pytorch_lightning.utilities import rank_zero_warn

def xavier_init(
        m : nn.Module, 
        gain : Union[str, float] = 'relu', 
        verbose : bool = False,
        ) -> None:
    """
    Initializes the given module using Xavier/Glorot initialization, as 
    described in 'Understanding the difficulty of training deep feedforward 
    neural networks'. 

    Parameters:
    -----------
    m (nn.Module): 
        The module to be initialized.
    gain (Union[str, float]): 
        The gain factor for the given activation function. Defaults to 'relu'.
        If a string is given, the gain factor is calculated using the given 
        string via nn.init.calculate_gain.
    verbose (bool, optional):
        Whether to print information about the initialization. Defaults to 
        False.
    """
    # if gain is a string, calculate the gain factor for the given activation function
    if isinstance(gain, str):
            gain = nn.init.calculate_gain(gain)

    # initialize the weights of the module
    if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight, gain)
        if m.padding_idx is not None:
            nn.init.constant_(m.weight[m.padding_idx], 0)
    elif isinstance(m, (nn.RNN, nn.GRU, nn.LSTM, nn.LSTMCell, nn.GRUCell, nn.RNNCell, nn.Transformer, nn.TransformerEncoder, nn.TransformerDecoder)):
        for name, param in m.named_parameters():
            if 'weight' in name and param.data.dim() == 2:
                nn.init.xavier_uniform_(param, gain)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    else:
        rank_zero_warn(
            f'No initialization implemented for {m.__class__.__name__}.'
            )
        pass

