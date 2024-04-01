# embedding.py

import torch.nn as nn
import torch
from .normalisation import BatchNorm

class IntegerEmbedding(nn.Module):
    """
    A simple wrapper around nn.Embedding, which performs Xavier initialization
    of the weights and calls x.long() on the input before passing it to the
    embedding layer. It also supports having multiple independent embedding
    layers, which are concatenated along the final dimension.

    Parameters:
    -----------
    num_embeddings (int):
        The maximum integer index that can be used to index the embeddings.
        If a list of integers is provided, then each embedding layer will
        have a different number of embeddings. In this case, the length of
        the list must be equal to the length of embedding_dims.
    embedding_dims (list[int]):
        The dimensions of the embedding layer. Treating the final dimension of 
        the input as a feature dimension, each element of the final dimension
        is passed through a separate embedding layer, and the results are concatenated
        along the final dimension. Hence, the final dimension of the output will
        have size equal to the sum of the elements of embedding_dims.
    use_batchnorm (bool):
        Whether to use batch normalisation. Defaults to False.
    """
    def __init__(
            self,
            num_embeddings : list[int],
            embedding_dims : list[int],
            use_batchnorm : bool = False,
            *args, **kwargs,
            ) -> None:
        super().__init__(*args, **kwargs)

        # ensure that the number of embeddings and the number of embedding dimensions
        # are the same
        if not len(num_embeddings) == len(embedding_dims):
            raise ValueError(
                f'The number of embedding dimensions {len(num_embeddings)} must be the same as the number of embeddings {len(embedding_dims)}.'
                )
        
        # initialise the embedding layers
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_embedding, embedding_dim)
            for num_embedding, embedding_dim 
            in zip(num_embeddings, embedding_dims)
        ])

        # store the embedding dimensions
        self.embedding_dims = embedding_dims

        # initialize the weights of the embedding layers
        gain = nn.init.calculate_gain('relu')
        for embedding_layer in self.embedding_layers:
            nn.init.xavier_uniform_(embedding_layer.weight, gain)
            
        # initialize the batch normalisation layer
        self.batchnorm = BatchNorm(sum(embedding_dims)) if use_batchnorm else None


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        A forward pass through the embedding layer. Performs x.long() on the 
        input first.

        Parameters:
        -----------
        x (torch.Tensor):
            The input to the embedding layer, of shape (..., n), where n is the 
            length of self.embedding_dims. If n is 1, then the final dimension 
            can be omitted.

        Returns:
        --------
        torch.Tensor:
            The output of the embedding layer, of dtype self.dtype. If concat_dims
            is False, then the output will have shape (..., embedding_dim).
            Else, all dimensions except the first two will be collapsed into 
            the embedding dimension. 
        """

        # convert to a long tensor
        x = x.long()

        # if there is only one embedding layer, then we need to add a dummy dimension
        if len(self.embedding_dims) == 1:
            x = x.unsqueeze(-1)

        # pass the input through the embedding layers
        x = torch.cat([
            embedding_layer(x[..., i])
            for i, embedding_layer in enumerate(self.embedding_layers)
        ], dim = -1)

        # apply batch normalisation if necessary
        if self.batchnorm is not None:
            x = self.batchnorm(x)

        return x