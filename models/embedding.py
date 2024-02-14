# embedding.py

import torch.nn as nn
import torch
from .normalisation import BatchNorm

class IntegerEmbedding(nn.Embedding):
    """
    A simple wrapper around nn.Embedding, which performs Xavier initialization
    of the weights and calls x.long() on the input before passing it to the
    embedding layer.

    Parameters:
    -----------
    num_embeddings (int):
        The maximum number of embeddings.
    embedding_dim (int):
        The dimension of the embeddings.
    num_concat_dims (bool):
        The number of dimensions to concatenate. If greater than 1, then the
        embedding dimension will be divided by this number, and the output
        will be stacked along the last dimension, so that the output will have
        final dimension equal to the original embedding dimension. Defaults to 1.
    use_batchnorm (bool):
        Whether to use batch normalisation. Defaults to False.
    *args:
        Additional arguments to nn.Embedding.
    **kwargs:
        Additional keyword arguments to nn.Embedding.
    """
    def __init__(
            self,
            num_embeddings : int,
            embedding_dim : int,
            num_concat_dims : int = 1,
            use_batchnorm : bool = False,
            *args, **kwargs,
            ) -> None:
        if embedding_dim % num_concat_dims != 0:
            raise ValueError(
                f'The embedding dimension ({embedding_dim}) must be divisible by the number of dimensions to concatenate ({num_concat_dims}).'
                )
        embedding_dim = embedding_dim // num_concat_dims
        super().__init__(num_embeddings, embedding_dim, *args, **kwargs)
        # initialize the weights of the embedding layer
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.weight, gain)
        if self.padding_idx is not None:
            nn.init.constant_(
                self.weight[self.padding_idx], 0
                )
        # initialize the batch normalisation layer
        self.batchnorm = BatchNorm(embedding_dim) if use_batchnorm else None


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        A forward pass through the embedding layer. Performs x.long() on the 
        input first.

        Parameters:
        -----------
        x (torch.Tensor):
            The input to the embedding layer, of shape (batch_size, seq_len, *d).
            It must be of a type that can be cast to a long tensor. 

        Returns:
        --------
        torch.Tensor:
            The output of the embedding layer, of dtype self.dtype. If concat_dims
            is False, then the output will have shape (batch_size, seq_len, *d, embedding_dim).
            Else, all dimensions except the first two will be collapsed into 
            the embedding dimension. 
        """
        x = super().forward(x.long())
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        return x.view(x.shape[0], x.shape[1], -1)