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
    bidirectional (bool):
        Whether to use a bidirectional LSTM. Defaults to True.
    """
    def __init__(
            self,
            in_size : int,
            hidden_size : int,
            num_layers : int,
            dropout : float = 0.0,
            bidirectional : bool = True,
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
            bidirectional = bidirectional,
            )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.d = 2 if bidirectional else 1
        
        # initialize the RNN weights
        gain = nn.init.calculate_gain('tanh')
        for name, param in self.named_parameters():
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
            num_concat_dims : int = 1,
            *args, **kwargs,
            ) -> None:
        super().__init__(in_size = embedding_dim, *args, **kwargs)
        self.embedding = IntegerEmbedding(
            num_embeddings = num_embeddings, 
            embedding_dim = embedding_dim,
            num_concat_dims = num_concat_dims,
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
            pooling : Optional[dict] = None,
            *args, **kwargs,
            ) -> None:
        super().__init__(*args, **kwargs)

        # a linear layer to map to the output dimension
        self.linear = DenseNetwork(
            in_size = self.d * self.hidden_size, 
            out_size = out_size, 
            dropout = self.dropout,
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
    


class RecurrentClassiferDecoder(BareBonesRecurrentNetwork):
    """
    A simple recurrent network, followed by a linear layer applied to the
    final hidden state.

    Parameters:
    -----------
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
            *args, **kwargs,
            ) -> None:
        super().__init__(*args, **kwargs)

        # a linear layer to map to the output dimension
        self.linear = DenseNetwork(
            in_size = self.d * self.hidden_size,
            out_size = out_size, 
            dropout = self.dropout,
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
        # pass through the recurrent network
        _, (h, _) = super().forward(x, h)
        # reshape h
        if self.d == 2:
            h = h.view(self.num_layers, 2, -1, self.hidden_size)
        # get the last hidden state of each direction
        h = h[-1].view(-1, self.d * self.hidden_size)
        # pass through the linear layer
        return self.linear(h)



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
            num_concat_dims : int = 1,
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
            num_concat_dims = num_concat_dims,
            )
        # initialize the attention layer
        self.attention = MultiHeadSelfAttention(
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
        x = self.attention(x)
        # pass through the decoder
        return self.decoder(x, h)
    


class RecurrentEncoderDecoderClassifierWithAttention(nn.Module):
    def __init__(
            self,
            num_embeddings : int,
            embedding_dim : int,
            hidden_size : int, 
            out_size : int,
            num_encoder_layers : int,
            num_decoder_layers : int,
            num_heads : int, 
            num_concat_dims : int = 1,
            dropout : float = 0.0,
            attention_dropout : float = 0.0,
            bidirectional : bool = True,
            *args, **kwargs,
            ):
        super().__init__(*args, **kwargs)
        # the number of layers depends on the number of directions
        D = 2 if bidirectional else 1

        # initialize the encoder
        self.encoder = RecurrentEncoder(
            num_embeddings = num_embeddings,
            embedding_dim = embedding_dim,
            hidden_size = hidden_size,
            num_layers = num_encoder_layers,
            dropout = dropout,
            num_concat_dims = num_concat_dims,
            )
        # initialize the attention layer
        self.attention = MultiHeadSelfAttention(
            embed_dim = D * hidden_size,
            num_heads = num_heads,
            dropout = attention_dropout,
            )
        # initialize the decoder
        self.decoder = RecurrentClassiferDecoder(
            in_size = D * hidden_size,
            hidden_size = hidden_size,
            out_size = out_size,
            num_layers = num_decoder_layers,
            dropout = dropout,
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
        x = self.attention(x)
        # pass through the decoder
        return self.decoder(x, h)