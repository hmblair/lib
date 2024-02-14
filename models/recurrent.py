# recurrent.py

import torch
import torch.nn as nn
from .embedding import IntegerEmbedding
from .attention import MultiHeadSelfAttention
from .dense import DenseNetwork
from typing import Optional

class BareBonesRecurrentNetwork(nn.Module):
    """
    A simple birdirectional LSTM, with Xavier initialization.

    Parameters:
    -----------
    in_size (int): 
        The number of input features.
    hidden_size (int):
        The hidden size.
    num_layers (int):
        The number of layers.
    dropout (float):
        The dropout rate.
    """
    def __init__(
            self,
            in_size : int,
            hidden_size : int,
            num_layers : int,
            dropout : float = 0.0,
            *args, **kwargs,
            ) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.LSTM(
            input_size = in_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bias = True,
            dropout = dropout,
            batch_first = True,
            bidirectional = True,
            )
        
        # initialize the RNN weights
        gain = nn.init.calculate_gain('tanh')
        for name, param in self.named_parameters():
            print(name)
            if 'weight' in name and param.data.dim() == 2:
                nn.init.xavier_uniform_(param, gain)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    
    def forward(
            self, 
            x : torch.Tensor, 
            h : Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Foward pass of the recurrent network.

        Parameters:
        -----------
        x (torch.Tensor): 
            The input tensor, of shape (batch, seq_len, in_size).
        h (tuple[torch.Tensor, torch.Tensor], optional):
            A tuple containing the hidden and cell states, of shape 
            (num_layers * num_directions, batch, hidden_size). Defaults to None.

        Returns:
        --------
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: 
            The output tensor and a tuple containing the hidden and cell states.
        """
        return self.model(x, h)
    


class RecurrentEncoder(BareBonesRecurrentNetwork):
    """
    A simple recurrent network, preceded by an embedding layer.

    Parameters:
    -----------
    num_embeddings (int):
        The number of embeddings.
    embedding_dim (int):
        The embedding dimension.
    *args:
        Additional positional arguments to pass to the recurrent network.
    **kwargs:
        Additional keyword arguments to pass to the recurrent network.
    """
    def __init__(
            self,
            num_embeddings : int,
            embedding_dim : int,
            *args, **kwargs,
            ) -> None:
        super().__init__(in_size = embedding_dim, *args, **kwargs)
        self.embedding = IntegerEmbedding(
            num_embeddings = num_embeddings, 
            embedding_dim = embedding_dim,
            )
        

    def forward(
            self, 
            x : torch.Tensor, 
            ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Foward pass of the recurrent network.

        Parameters:
        -----------
        x (torch.Tensor): 
            The input tensor, of shape (batch, seq_len).

        Returns:
        --------
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: 
            The output tensor and a tuple containing the hidden and cell states.
        """
        x = self.embedding(x)
        return super().forward(x)



class RecurrentDecoder(BareBonesRecurrentNetwork):
    """
    A simple recurrent network, followed by a linear layer.

    Parameters:
    -----------
    out_size (int):
        The number of output features.
    hidden_size (int):
        The hidden size.
    dropout (float):
        The dropout rate.
    pooling (dict, optional):
        The pooling layer to use. Defaults to None.
    *args:
        Additional positional arguments to pass to the recurrent network.
    **kwargs:
        Additional keyword arguments to pass to the recurrent network.
    """
    def __init__(
            self, 
            out_size : int,
            hidden_size : int, 
            dropout : float = 0.0,
            pooling : Optional[dict] = None,
            *args, **kwargs,
            ) -> None:
        super().__init__(
            hidden_size = hidden_size,
            dropout = dropout,
            *args, **kwargs,
            )

        # a linear layer to map to the output dimension
        self.linear = DenseNetwork(
            in_size = hidden_size * 2, 
            out_size = out_size, 
            dropout = dropout,
            pooling = pooling,
            )
    

    def forward(
            self, 
            x : torch.Tensor, 
            h : Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Foward pass of the recurrent network.

        Parameters:
        -----------
        x (torch.Tensor): 
            The input tensor, of shape (batch, seq_len, in_size).
        h (tuple[torch.Tensor, torch.Tensor], optional):
            A tuple containing the hidden and cell states, of shape 
            (num_layers * num_directions, batch, hidden_size). Defaults to None.

        Returns:
        --------
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: 
            The output tensor and a tuple containing the hidden and cell states.
        """
        x, _ = super().forward(x, h)
        return self.linear(x)



class RecurrentEncoderDecoderWithAttention(nn.Module):
    def __init__(
            self,
            num_embeddings : int,
            embedding_dim : int,
            hidden_size : int, 
            out_size : int,
            num_encoder_layers : int,
            num_decoder_layers : int,
            num_heads : int, 
            dropout : float = 0.0,
            attention_dropout : float = 0.0,
            pooling : Optional[dict] = None,
            *args, **kwargs,
            ):
        super().__init__(*args, **kwargs)
        # initialize the encoder
        self.encoder = RecurrentEncoder(
            num_embeddings = num_embeddings,
            embedding_dim = embedding_dim,
            hidden_size = hidden_size,
            num_layers = num_encoder_layers,
            dropout = dropout,
            )
        # initialize the attention layer
        self.x_attention = MultiHeadSelfAttention(
            embed_dim = hidden_size * 2,
            num_heads = num_heads,
            dropout = attention_dropout,
            )
        # initialize the decoder
        self.decoder = RecurrentDecoder(
            in_size = hidden_size * 2,
            hidden_size = hidden_size,
            out_size = out_size,
            num_layers = num_decoder_layers,
            dropout = dropout,
            pooling = pooling,
            )
    

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM with attention.

        Parameters:
        -----------
        x (torch.Tensor): 
            The input tensor, of shape (batch, seq_len, in_size).
        
        Returns:
        --------
        torch.Tensor: 
            The output tensor, of shape (batch, seq_len, out_size).
        """
        # pass through the encoder
        x, h = self.encoder(x)
        # pass through the attention layer
        x = self.x_attention(x)
        # pass through the decoder
        return self.decoder(x, h)