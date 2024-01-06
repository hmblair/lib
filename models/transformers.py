from typing import Optional, Tuple, Union, Callable, Any
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    """
    Implements a sinusoidal positional encoding layer.

    Parameters:
    -----------
    k (int):
        The frequency for the sinusoidal positional encoding. Defaults to 10000,
        which is the default value in the paper 'Attention Is All You Need'.
    """
    def __init__(self, k : int) -> None:
        super().__init__()
        self.k = k
    

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional encoding layer.

        Parameters:
        -----------
        x (torch.Tensor): 
            Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns:
        --------
        torch.Tensor: 
            Output tensor of shape (batch_size, seq_len, embedding_dim) after 
            adding positional encoding.
        """
        # get the shape of the input
        batch_size, seq_len, embedding_dim = x.shape 

        # initialise the encoding
        encoding = torch.zeros(
            seq_len, 
            embedding_dim, 
            device=x.device, 
            requires_grad=False,
            ) 
        
        # get the positions
        pos = torch.arange(seq_len) 

        # compute the encoding 
        for i in range(embedding_dim // 2):
            encoding[:, 2*i] = torch.sin(pos / self.k ** (2 * i / embedding_dim)) 
            encoding[:, 2*i + 1] = torch.cos(pos / self.k ** (2 * i / embedding_dim)) 
        if embedding_dim % 2 == 1:
            encoding[:, -1] = torch.sin(pos / self.k) 

        # add the encoding to the input
        return x + encoding.repeat(batch_size, 1, 1) 
    
    

class BareTransformer(nn.Module):
    """
    A transformer model without positional encoding or an embedding layer.

    Parameters:
    -----------
    embed_dim (int): 
        The embedding dimension.
    output_dim (int):
        The size of the output dimension.
    num_layers (int):
        The number of layers in the transformer encoder.
    num_heads (int):
        The number of attention heads.
    dim_feedforward (int):
        The dimension of the feedforward layer in the transformer encoder.
    dropout (float):
        The dropout probability. Defaults to 0.0.
    norm_first (bool):
        Whether to apply layer normalization before the self-attention and
        feedforward layers. Defaults to False. True will yield a Pre-LN
        Transformer, as described in 'On Layer Normalization in the 
        Transformer Architecture', found at https://arxiv.org/abs/2002.04745.
    *args: 
        Variable length argument list.
    **kwargs: 
        Arbitrary keyword arguments.

    Attributes:
    ----------
    embedding (nn.Embedding): 
        The embedding layer.

    Inherits:
    ----------
    nn.Module: 
        The base PyTorch module class.
    """
    def __init__(
            self, 
            embed_dim : int,
            output_dim : int, 
            num_layers : int,
            num_heads : int,
            dim_feedforward : int,
            dropout : float = 0.0,
            norm_first : bool = False,
            *args, **kwargs
            ):
        super().__init__(*args, **kwargs)

        # transformer encoder
        self.tfe = self.build_transformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            norm_first=norm_first,
            dropout=dropout,
        )

        # final linear layer
        self.linear = nn.Linear(embed_dim, output_dim)

    
    @staticmethod
    def build_transformer(
            embed_dim : int,
            num_heads : int,
            dim_feedforward : int,
            num_layers : int,
            norm_first : bool,
            dropout : float,
            ) -> nn.Sequential:
        """
        Builds a transformer encoder from the given parameters.

        Parameters:
        -----------
        embed_dim (int):
            The embedding dimension.
        num_heads (int):
            The number of attention heads.
        dim_feedforward (int):
            The dimension of the feedforward layer in the transformer encoder.
        num_layers (int):
            The number of layers in the transformer encoder.
        norm_first (bool):
            Whether to apply layer normalization before the self-attention and
            feedforward layers. Defaults to False. True will yield a Pre-LN
            Transformer, as described in 'On Layer Normalization in the 
            Transformer Architecture', found at https://arxiv.org/abs/2002.04745.
        dropout (float):
            The dropout probability. Defaults to 0.0.

        Returns:
        --------
        nn.Sequential:
            The transformer encoder.
        """
        # initialise the list to store the layers
        tfe_layers = []
        # construct the transformer encoder layers
        for _ in range(num_layers):
            tfe_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    norm_first=norm_first,
                    dropout=dropout,
                    batch_first=True,
                    )
                )

        # join the transformer encoder layers into a sequential model
        return nn.Sequential(*tfe_layers) 
    

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # pass through the transformer encoder
        x = self.tfe(x) 

        # pass through the final layers, squeezing the output if necessary
        return self.linear(x).squeeze(-1)



class TransformerWithSinusoidalPositionalEncoding(BareTransformer):
    """
    A transformer model.

    Parameters:
    -----------
    embed_dim (int): 
        The embedding dimension.
    num_embeddings (int): 
        The maximum integer that can be embedded.
    *args: 
        Variable length argument list.
    **kwargs: 
        Arbitrary keyword arguments.

    Attributes:
    ----------
    embedding (nn.Embedding): 
        The embedding layer.

    Inherits:
    ----------
    nn.Module: 
        The base PyTorch module class.
    """
    def __init__(
            self,
            num_embeddings : int,
            embed_dim : int,
            k : int = 10000,
            *args, **kwargs
            ):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)

        # embedding layer
        self.embedding = nn.Embedding(num_embeddings, embed_dim)

        # positional encoding layer
        self.positional_encoding = SinusoidalPositionalEncoding(k)


    def embed(self, x : torch.Tensor) -> torch.Tensor:
        """
        Passes a tensor through the embedding layer.

        Parameters:
        -----------
        x (torch.Tensor): 
            Input tensor of shape (batch_size, seq_len).

        Returns:
        --------
        torch.Tensor: 
            Embedded tensor of shape (batch_size, seq_len, embed_dim).
        """
        return self.embedding(
            x.long()
        )


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input integer tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim).
        """
        # pass through the embedding layer and add on the positional encoding
        x = self.embed(x)
        x = self.positional_encoding(x)

        # pass through the transformer encoder
        x = self.tfe(x) 

        # pass through the final layers, squeezing the output if necessary
        return self.linear(x).squeeze(-1)





# class BaseTransformerWithSinusoidalPosEnc(BaseTransformer):
#     def __init__(self, k : int, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.k = k

#     def _absolute_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the positional encoding layer.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).

#         Returns:
#             torch.Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim) after adding positional encoding.
#         """
#         batch_size, seq_len, embedding_dim = x.shape # get the shape of the input

#         encoding = torch.zeros(
#             seq_len, 
#             embedding_dim, 
#             device=x.device, 
#             requires_grad=False,
#             ) # initialise the encoding
#         pos = torch.arange(seq_len) # get the positions

#         for i in range(embedding_dim // 2):
#             encoding[:, 2*i] = torch.sin(pos / self.k ** (2 * i / embedding_dim)) # compute the encoding 
#             encoding[:, 2*i + 1] = torch.cos(pos / self.k ** (2 * i / embedding_dim)) # compute the encoding

#         if embedding_dim % 2 == 1:
#             encoding[:, -1] = torch.sin(pos / self.k) # compute the encoding

#         return x + encoding.repeat(batch_size, 1, 1) # add the encoding to the input





# class Transformer(BaseTransformer):
#     """
#     Transformer model, as vanilla as it gets.

#     Parameters:
#     -----------
#     output_dim (int):
#         The output dimension.
#     feedforward_dim (int): 
#         The dimension of the feedforward layer in the transformer encoder.
#     num_heads (int): 
#         The number of attention heads in the transformer encoder.
#     num_layers (int): 
#         The number of layers in the transformer encoder.
#     norm_first (bool): 
#         Whether to apply layer normalization before the self-attention and 
#         feedforward layers. Defaults to False. True will yield a Pre-LN 
#         Transformer, as described in 'On Layer Normalization in the Transformer 
#         Architecture', found at https://arxiv.org/abs/2002.04745.
#     *args: 
#         Variable length argument list.
#     **kwargs: 
#         Arbitrary keyword arguments.

#     Attributes:
#     ----------
#     tfe (nn.TransformerEncoder): 
#         The transformer encoder.
#     linear (nn.Linear): 
#         The final linear layer.
#     """
#     def __init__(
#             self,
#             output_dim: int,
#             feedforward_dim: int,
#             num_heads: int,
#             num_layers: int,
#             norm_first: bool = False,
#             *args, **kwargs,
#             ):
#         super().__init__(*args, **kwargs)
#         tfe_layers = []
#         for _ in range(num_layers):
#             tfe_layers.append(
#                 nn.TransformerEncoderLayer(
#                     d_model=self.embed_dim,
#                     nhead=num_heads,
#                     dim_feedforward=feedforward_dim,
#                     norm_first=norm_first,
#                     )
#                     )

#         # transformer encoder
#         self.tfe = nn.Sequential(*tfe_layers) 

#         # final linear layer
#         self.linear = nn.Linear(self.embed_dim, output_dim)


    # def forward(self, x):
    #     """
    #     Forward pass of the model.

    #     Args:
    #         x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

    #     Returns:
    #         torch.Tensor: Output tensor of shape (batch_size, 1).
    #     """
    #     # pass through the embedding layer and add on the positional encoding
    #     x = self._embed(x)

    #     # pass through the transformer encoder
    #     x = self.tfe(x) 

    #     # pass through the final layers
    #     x = self.linear(x).squeeze(-1) 
    #     return torch.mean(x, dim=-1, keepdim=True)
    

class MultiHeadSelfAttention(nn.Module):
    """
    Implements a multi-head self-attention layer.

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


import torch.nn.functional as F
class MultiHeadSelfAttentionWithBias(nn.Module):
    def __init__(self, num_heads : int, embed_dim : int, dropout : float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

        # linear layers for computing the query, key, value, and output tensors
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-head self-attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = x.shape
        features_per_head = self.embed_dim // self.num_heads # compute the number of features per head

        # Compute query, key, and value tensors
        query = self.query(x).view(batch_size, seq_len, self.num_heads, features_per_head).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, features_per_head).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, features_per_head).transpose(1, 2)

        # compute the attention scores
        attention_scores = torch.einsum('bshd, bsvd -> bhsv', query, key) / (self.embed_dim // self.num_heads) ** 0.5

        # add on the bias
        attention_scores = attention_scores + self._bias(x)

        # compute the attention probabilities
        attention_probs = torch.softmax(attention_scores, dim = -1)

        # apply dropout
        attention_probs = F.dropout(attention_probs, p = self.dropout, training = self.training)

        # compute the attention output
        attention_output = torch.einsum('bhsv, bsvd -> bshd', attention_probs, value)

        # reshape the attention output
        attention_output = attention_output.reshape(batch_size, seq_len, self.embed_dim)

        return attention_output
    

    def _bias(self, x : torch.Tensor) -> Union[torch.Tensor, int]:
        """
        Computes the bias tensor for the self-attention weights. Overwrite this method in order 
        to implement different bias tensors.

        Returns:
            torch.Tensor | int: The bias, either an integer or a tensor of size (batch_size, seq_len, seq_len, num_heads).
        """
        return 0
    




class MultiHeadSelfAttentionWithRelativePosEnc(MultiHeadSelfAttentionWithBias):
    def __init__(self, max_position : int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_position = max_position
        self.embedding = nn.Embedding(max_position, 1)


    def _bias(self, x : torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x_vals = torch.arange(seq_len, device = x.device)
        diff_mat = x_vals[None, :] - x_vals[:, None]
        diff_tensor = torch.clamp(
            diff_mat.repeat(batch_size, 1, 1) + seq_len -1,
            0, self.max_position - 1
            )
    
        return self.embedding(diff_tensor).squeeze(dim = -1)



if __name__ == '__main__':
    # a simple test

    # Create an instance of the Transformer model
    model = TransformerWithSinusoidalPositionalEncoding(
        output_dim=1, 
        dim_feedforward=256, 
        num_heads=4, 
        num_layers=2, 
        embed_dim=128, 
        num_embeddings = 4,
        )

    # Generate some random input data
    batch_size = 32
    seq_len = 10
    input_data = torch.randint(0, 4, (batch_size, seq_len))

    # Perform a forward pass
    output = model.forward(input_data)

    # Print the output shape
    print(output.shape)