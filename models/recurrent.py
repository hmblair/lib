# recurrent.py

import torch
import torch.nn as nn
from .transformers import MultiHeadSelfAttention

class BaseRNN(nn.Module):
    """
    A bidirectional encoder-decoder style recurrent neural network with an 
    arbitrary number of hidden layers. This class prepares the base structure
    for the RNN, but the actual forward pass is not defined.

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
    encode():
        Encodes the input tensor.
    decode():
        Decodes the input tensor.
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
            'lstm' : nn.LSTM,
            }
        if variety not in varieties.keys():
            raise ValueError(
                f'{variety} is not a valid variety. The valid varieties are {list(varieties.keys())}.'
                ) 
        self.module = varieties[variety]

        # an embedding layer to embed the input
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        # store the hidden size
        self.hidden_size = hidden_size

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

    
    def encode(self, x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input tensor.

        Parameters:
        -----------
        x (torch.Tensor): 
            The input tensor.

        Returns:
        --------
        tuple[torch.Tensor, torch.Tensor]: 
            The encoded tensor and the hidden state.
        """
        # convert to long tensor for nn.Embedding and embed the input
        x = x.long() 
        x = self.embedding(x)

        # pass through the encoder
        x, h = self.encoder(x)
        return x, h
    

    def decode(self, x : torch.Tensor, h : torch.Tensor) -> torch.Tensor:
        """
        Decodes the input tensor.

        Parameters:
        -----------
        x (torch.Tensor): 
            The input tensor.
        h (torch.Tensor): 
            The hidden state.

        Returns:
        --------
        torch.Tensor: 
            The decoded tensor.
        """
        # pass through the decoder
        x, _ = self.decoder(x, h)

        # pass through the final linear layer
        x = self.linear(x)

        # squeeze the final dimension if applicable
        return x.squeeze(-1)


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



class RNN(BaseRNN):
    """
    A vanilla bidirectional RNN, as described in the BaseRNN class.
    """
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
        # encode the input
        x, h = self.encode(x)

        # decode the input
        return self.decode(x, h)
    


class RNNWithAttention(BaseRNN):
    """
    A bidirectional RNN with an attention layer.

    Parameters:
    -----------
    num_heads (int):
        The number of attention heads.
    attention_dropout (float):
        The dropout probability for the attention layer. Defaults to 0.0.
    """
    def __init__(
            self,
            num_heads : int, 
            attention_dropout : float = 0.0,
            *args, **kwargs,
            ):
        super().__init__(*args, **kwargs)
        self.attention = MultiHeadSelfAttention(
            embed_dim = self.hidden_size * 2,
            num_heads = num_heads,
            dropout = attention_dropout,
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
        # encode the input
        x, h = self.encode(x)

        # pass through the attention layer
        x = self.attention(x)

        # decode the input
        return self.decode(x, h)