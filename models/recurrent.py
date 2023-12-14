# recurrent.py

import torch
import torch.nn as nn
from models.abstract_models import BaseModel

class RNN(BaseModel):
    """
    A bidirectional encoder-decoder style recurrent neural network with an 
    arbitrary number of hidden layers.

    Parameters:
    -----------
    embedding_dim (int):
        The size of the embedding dimension.
    hidden_size (int):
        The number of features in the hidden state.
    output_dim (int):
        The size of the output dimension.
    num_layers (int):
        Number of recurrent layers.
    variety (str):
        The type of RNN to use. Can be one of 'rnn', 'gru', or 'lstm'.
    dropout (float):
        The dropout probability. Defaults to 0.0.
    num_embeddings (int):
        The number of embeddings. Defaults to 8.

    Attributes:
    -----------
    module (nn.Module):
        The RNN module, which is either nn.RNN, nn.GRU, or nn.LSTM.
    embedding (nn.Module):
        The embedding layer.    
    encoder (nn.Module):
        The encoder RNN.
    decoder (nn.Module):
        The decoder RNN.
    linear (nn.Module):
        The final linear layer.

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
            embedding_dim : int,
            hidden_size : int,
            output_dim : int, 
            num_layers : int,
            variety : str,
            dropout : float = 0.0,
            num_embeddings : int = 8,
            *args, **kwargs,
            ):
        super().__init__(*args, **kwargs)

        variety = variety.lower()
        varieties = {
            'rnn' : nn.RNN,
            'gru' : nn.GRU,
            'lstm' : nn.LSTM
            }
        if variety not in varieties.keys():
            raise ValueError(f'{variety} is not a valid variety. The valid varieties are {list(varieties.keys())}')
        self.module = varieties[variety]

        # an embedding layer to embed the input
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        # the encoder and decoder RNNs
        self.encoder = self._construct_rnn(
            in_size = embedding_dim,
            hidden_size = hidden_size,
            num_layers = num_layers,
            dropout = dropout
            )
        
        self.decoder = self._construct_rnn(
            in_size = hidden_size * 2,
            hidden_size = hidden_size,
            num_layers = num_layers,
            dropout = dropout
            )

        # a linear layer to map to the output dimension
        self.linear =nn.Linear(hidden_size * 2, output_dim)


    def _construct_rnn(
            self, 
            in_size : int, 
            hidden_size : int, 
            num_layers : int, 
            dropout : float
            ) -> nn.Module:
        """
        Constructs an RNN module with the given parameters.
        
        Parameters:
        -----------
        in_size (int): 
            The size of the input features.
        hidden_size (int): 
            The number of features in the hidden state.
        num_layers (int): 
            Number of recurrent layers.
        dropout (float): 
            The dropout probability.
            
        Returns:
        --------
        nn.Module: 
            The constructed RNN module.
        """
        return self.module(
            input_size = in_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bias = True,
            dropout = dropout,
            batch_first = True,
            bidirectional = True
            )


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters:
        -----------
        x (torch.Tensor): 
            The input tensor.

        Returns:
        --------
        torch.Tensor: 
            The output tensor.
        """
        # convert to long tensor for nn.Embedding and embed the input
        x = x.long() 
        x = self.embedding(x)

        # pass through the encoder and decoder
        x, h = self.encoder(x)
        x, _ = self.decoder(x, h)

        # pass through the final linear layer
        x = self.linear(x)

        # squeeze the final dimension if applicable
        return x.squeeze(-1) 