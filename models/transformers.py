from typing import Optional, Tuple, Union, Callable, Any
import torch
import torch.nn as nn
from models.schedulers import LinearWarmupAndInverseSqrtDecayLR
from models.abstract_models import BaseModel


class BaseTransformer(BaseModel):
    """
    Base for building transformer models upon. Implements the embedding layer and the learning rate scheduler,
    as well as some boilerplate methods.

    Parameters:
    -----------
    embed_dim (int): 
        The embedding dimension.
    num_embeddings (int): 
        The maximum integer that can be embedded.
    lr_warmup_steps (int): 
        The number of warmup steps for the learning rate. Defaults to 4000.
    *args: 
        Variable length argument list.
    **kwargs: 
        Arbitrary keyword arguments.

    Attributes:
    ----------
    embed_dim (int): 
        The embedding dimension.
    lr_warmup_steps (int): 
        The number of warmup steps for the learning rate.
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
            num_embeddings : int = 8, 
            lr_warmup_steps : int = 8000,
            *args, **kwargs
            ):
        super().__init__(*args, **kwargs)

        # store the embedding dimension and the number of lr warmup steps
        self.embed_dim = embed_dim
        self.lr_warmup_steps = lr_warmup_steps

        # embedding layer
        self.embedding = nn.Embedding(num_embeddings, embed_dim)


    def _absolute_positional_encoding(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional encoding layer. Overwrite this method in 
        order to implement different positional encodings. The default is to not
        add any positional encoding.

        Args:
            x (torch.Tensor): Input tensor of shape 
            (batch_size, seq_len, embedding_dim).

        Returns:
            torch.Tensor: Output tensor of shape 
            (batch_size, seq_len, embedding_dim) after adding positional encoding.
        """
        return x


    def _embed(self, x : torch.Tensor) -> torch.Tensor:
        """
        Passes a tensor through the embedding layer and adds on an absoute 
        positional encoding.

        Parameters:
        -----------
        x (torch.Tensor): 
            Input tensor of shape (batch_size, seq_len).

        Returns:
        --------
        torch.Tensor: 
            Embedded tensor of shape (batch_size, seq_len, embed_dim).
        """
        x = x.long()
        x = self.embedding(x)
        return self._absolute_positional_encoding(x)


    def _get_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        """
        A learning rate warmup and decay scheduler for transformers.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: The learning rate scheduler.
        """
        return LinearWarmupAndInverseSqrtDecayLR(
            optimizer=optimizer,
            warmup_steps=self.lr_warmup_steps,
            )





class BaseTransformerWithSinusoidalPosEnc(BaseTransformer):
    def __init__(self, k : int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def _absolute_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional encoding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim) after adding positional encoding.
        """
        batch_size, seq_len, embedding_dim = x.shape # get the shape of the input

        encoding = torch.zeros(seq_len, embedding_dim, device=x.device, requires_grad=False) # initialise the encoding
        pos = torch.arange(seq_len) # get the positions

        for i in range(embedding_dim // 2):
            encoding[:, 2*i] = torch.sin(pos / self.k ** (2 * i / embedding_dim)) # compute the encoding 
            encoding[:, 2*i + 1] = torch.cos(pos / self.k ** (2 * i / embedding_dim)) # compute the encoding

        if embedding_dim % 2 == 1:
            encoding[:, -1] = torch.sin(pos / self.k) # compute the encoding

        return x + encoding.repeat(batch_size, 1, 1) # add the encoding to the input





class Transformer(BaseTransformer):
    """
    Transformer model, as vanilla as it gets.

    Args:
        embed_dim (int): The embedding dimension.
        feedforward_dim (int): The dimension of the feedforward layer in the transformer encoder.
        num_heads (int): The number of attention heads in the transformer encoder.
        num_layers (int): The number of layers in the transformer encoder.
        lr_warmup_steps (int, optional): The number of warmup steps for the learning rate. Defaults to 4000.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        lr_warmup_steps (int): The number of warmup steps for the learning rate.
        embed_dim (int): The embedding dimension.
        pos_encoder (SinusoidalPosEnc): The positional encoder.
        embedding (nn.Embedding): The embedding layer.
        tfl (nn.TransformerEncoderLayer): The transformer encoder layer.
        tfe (nn.TransformerEncoder): The transformer encoder.
        linear (nn.Linear): The final linear layer.
    """
    def __init__(
            self,
            output_dim: int,
            feedforward_dim: int,
            num_heads: int,
            num_layers: int,
            *args, **kwargs,
            ):
        super().__init__(*args, **kwargs)
        tfe_layers = []
        for _ in range(num_layers):
            tfe_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=self.embed_dim,
                    nhead=num_heads,
                    dim_feedforward=feedforward_dim
                    )
                    )

        # transformer encoder
        self.tfe = nn.Sequential(*tfe_layers) 

        # final linear layer
        self.linear = nn.Linear(self.embed_dim, output_dim)


    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # pass through the embedding layer and add on the positional encoding
        x = self._embed(x)

        # pass through the transformer encoder
        x = self.tfe(x) 

        # pass through the final layers
        x = self.linear(x).squeeze(-1) 
        return torch.mean(x, dim=-1, keepdim=True)



import torch.nn.functional as F
class PreLNTransformer(BaseTransformer):
    def __init__(
            self, 
            num_heads : int, 
            num_layers : int,
            feedforward_dim : int,
            *args, **kwargs
            ):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.num_layers = num_layers
                
        self.attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(self.embed_dim, num_heads) for _ in range(num_layers)]
        )
        
        self.feedforward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, feedforward_dim),
                nn.ReLU(),
                nn.Linear(feedforward_dim, self.embed_dim)
            ) for _ in range(num_layers)
        ])
        
        self.linear = nn.Linear(self.embed_dim, 1)
        

    def forward(self, x):
        x = self._embed(x)
        
        for i in range(self.num_layers):
            residual = x
            x = F.layer_norm(x, normalized_shape=[x.size(-1)])
            x, _ = self.attention_layers[i](x, x, x)
            x = x + residual
            
            residual = x
            x = F.layer_norm(x, normalized_shape=[x.size(-1)])
            x = self.feedforward_layers[i](x) + residual
        
        x = self.linear(x).squeeze(-1)  # pass through the final linear layer
        return torch.mean(x, dim=-1, keepdim=True)








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
    model = Transformer(output_dim=1, feedforward_dim=256, num_heads=4, num_layers=2, embed_dim=128, num_embeddings = 4)

    # Generate some random input data
    batch_size = 32
    seq_len = 10
    input_data = torch.randint(0, 4, (batch_size, seq_len))

    # Perform a forward pass
    output = model.forward(input_data)

    # Print the output shape
    print(output.shape)