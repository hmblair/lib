# attention.py

import torch
import torch.nn as nn
from typing import Union
from .dense import DenseNetwork
from typing import Optional
from torch.nn.functional import scaled_dot_product_attention
from .embedding import IntegerEmbedding

class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self, 
            embed_dim : int, 
            num_heads : int, 
            dropout : float = 0.0,
            ) -> None:
        super().__init__()
        if not embed_dim % num_heads == 0:
            raise ValueError(
                f'Embedding dimension ({embed_dim}) must be divisible by the number of heads ({num_heads}).'
                )
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.input = DenseNetwork(embed_dim, 3 * embed_dim)
        self.output = DenseNetwork(num_heads * self.head_dim, embed_dim)

    
    def split_heads(self, x : torch.Tensor) -> torch.Tensor:
        """
        Splits the input tensor into multiple heads.

        Parameters:
        -----------
        x (torch.Tensor): 
            The input tensor of shape (batch_size, seq_len, embed_dim).
        num_heads (int):
            The number of attention heads.
        head_dim (int):
            The dimension of each attention head.

        Returns:
        --------
        torch.Tensor:
            The tensor of shape (batch_size, num_heads, seq_len, head_dim).
        """
        batch_size, seq_len, _ = x.size()
        return x.view(
            batch_size, seq_len, self.num_heads, self.head_dim
            ).permute(0, 2, 1, 3)
    

    def forward(
            self, x: torch.Tensor, bias: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
        """
        Forward pass of the multi-head self-attention layer.

        Parameters:
        -----------
        x (torch.Tensor): 
            Input tensor of shape (batch_size, seq_len, embed_dim).
        bias (torch.Tensor, optional):
            An attention bias tensor of shape (batch_size, num_heads, seq_len, seq_len). 
            If this has dtype torch.bool, the attention scores will be set to
            -inf for the corresponding positions. Defaults to None.

        Returns:
        --------
        torch.Tensor: 
            Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = x.size()

        # Linearly project the input (query, key, value).
        query, key, value = self.input(x).chunk(3, dim=-1)

        # Reshape and permute for multi-head attention.
        query, key, value = [self.split_heads(t) for t in (query, key, value)]

        # Apply scaled dot product attention.
        attn_output = scaled_dot_product_attention(
            query=query, 
            key=key, 
            value=value, 
            attn_mask=bias, 
            dropout_p=self.dropout,
            )

        # Concatenate heads and apply final linear layer.
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.embed_dim)

        return self.output(attn_output)
    

class MultiHeadSelfAttentionOld(nn.Module):
    """
    Implements a multi-head self-attention layer using the 
    nn.MultiheadAttention module. The weights are initialized using Xavier
    initialization.

    Parameters:
    -----------
    embed_dim (int): 
        The embedding dimension.
    num_heads (int):
        The number of attention heads.
    dropout (float):
        The dropout probability. Defaults to 0.0.
    """
    def __init__(
            self, 
            embed_dim : int, 
            num_heads : int, 
            dropout : float = 0.0,
            ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout)

        # initialize the input weights
        nn.init.xavier_uniform_(self.attention.in_proj_weight)
        if self.attention.in_proj_bias is not None:
            nn.init.constant_(self.attention.in_proj_bias, 0)
        # initialize the output weights
        nn.init.xavier_uniform_(self.attention.out_proj.weight)
        if self.attention.out_proj.bias is not None:
            nn.init.constant_(self.attention.out_proj.bias, 0)

    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-head self-attention layer.

        Parameters:
        -----------
        x (torch.Tensor): 
            Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
        --------
        torch.Tensor: 
            Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        return self.attention(x, x, x)[0]
    


class RelativePositionalEncoding(nn.Module):
    """
    Implements the relative positional encoding from 'Self-Attention with
    Relative Position Representations' (Shaw et al., 2018), which can be found
    at 'https://arxiv.org/abs/1803.02155'.

    Parameters:
    -----------
    max_position (int):
        The maximum position of the input tensor.
    embed_dim (int):
        The embedding dimension.
    """
    def __init__(
            self, 
            max_position : int,
            embed_dim : int, 
            ) -> None:
        super().__init__()
        self.max_position = max_position
        self.embedding = IntegerEmbedding(max_position, embed_dim)

        # construct a tensor of shape (max_position, max_position) containing
        # the relative positions
        x_vals = torch.arange(max_position)
        diff_mat = x_vals[None, :] - x_vals[:, None]
        diff_tensor = torch.clamp(diff_mat + max_position -1, 0, max_position - 1,)

        # make the tensor a parameter so that it is moved to the correct device
        self.diff_tensor = nn.Parameter(diff_tensor, requires_grad=False)


    def forward(self, batch_size : int, seq_len : int) -> torch.Tensor:
        """
        Computes the relative positional encoding of the correct shape for 
        the given input tensor, and adds it on.

        Parameters:
        -----------
        x (torch.Tensor):
            The input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
        --------
        torch.Tensor:
            Corresponding relative positional encoding of shape 
            (batch_size, embed_dim, seq_len, seq_len).
        """
        # slice the difference tensor and repeat to match the batch size
        diff_tensor = self.diff_tensor[:seq_len, :seq_len].repeat(batch_size, 1, 1)
    
        # pass through the embedding layer and reshape 
        return self.embedding(diff_tensor).view(batch_size, -1, seq_len, seq_len)
    


class InverseSquareRelativePositionalEncoding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(
            self, 
            batch_size : int, 
            seq_len : int, 
            num_heads : int,
            ) -> torch.Tensor:
        """
        Computes the inverse square relative positional encoding of the correct 
        shape for the given input tensor, and adds it on.

        Parameters:
        -----------
        x (torch.Tensor):
            The input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
        --------
        torch.Tensor:
            Corresponding inverse square relative positional encoding of shape 
            (batch_size, num_heads, seq_len, seq_len).
        """
        x_vals = torch.arange(seq_len)
        diff_mat = torch.abs(x_vals[None, :] - x_vals[:, None]) + 1
        diff_tensor = 1 / (diff_mat.float() ** 2)
        return diff_tensor.repeat(batch_size, num_heads, 1, 1)

    

class MultiHeadSelfAttentionWithRelativePositionalEncodings(MultiHeadSelfAttention):
    def __init__(
            self, 
            max_position : int, 
            embed_dim : int, 
            num_heads : int, 
            dropout : float = 0.0,
            ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.relative_positional_encoding = RelativePositionalEncoding(
            max_position, 
            num_heads,
            )
    
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-head self-attention layer.

        Parameters:
        -----------
        x (torch.Tensor): 
            Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
        --------
        torch.Tensor: 
            Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = x.size()
        bias = self.relative_positional_encoding(batch_size, seq_len)
        return super().forward(x,  bias)