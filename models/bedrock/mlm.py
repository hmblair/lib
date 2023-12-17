import torch
import torch.nn as nn
import random

from abstract_models import BaseClassifier

class BaseMLM(BaseClassifier):
    """
    Base class for Masked Language Modeling (MLM) models.

    Args:
        mask_token (int): The token used for masking.
        mask_frac (float): The fraction of tokens to be masked.
        *args: Variable length argument list passed to BaseModel.
        **kwargs: Arbitrary keyword arguments passed to BaseModel.

    Attributes:
        mask_token (int): The token used for masking.
        mask_frac (float): The fraction of tokens to be masked.
        objective (nn.CrossEntropyLoss): The loss function for MLM.

    Inherits:
        BaseModel: A base class for PyTorch Lightning models.
    """
    def __init__(
            self, 
            mask_token : int, 
            mask_frac : float,
            *args, **kwargs
            ):
        super().__init__(*args, **kwargs)
        self.mask_token = mask_token
        self.mask_frac = mask_frac
        if mask_frac < 0 or mask_frac > 1:
            raise ValueError('The masking fraction must be between 0 and 1.')
        self.cross_entropy = nn.CrossEntropyLoss()


    def _mask(self, x : torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        """
        Masks a portion of the input tensor by replacing selected tokens with a 
        mask token.

        Parameters:
        ----------
        x (torch.Tensor): 
            The input tensor to be masked.

        Returns:
        -------
        tuple[torch.Tensor, list[int]]: 
            A tuple containing the masked tensor and a list of the indices that
            were masked.
        """

        # this needs to be fixed, so that the indices are different across different elements of the batch
        seq_len = x.shape[1] # get the sequence length
        n = int(self.mask_frac * seq_len) # get the number of tokens to mask
        ix = random.sample(range(seq_len), n) # get the indices to mask
        x[:,ix] = self.mask_token # mask the input
        return x, ix # return the masked input and the indices that were masked
    

    def _compute_loss(self, batch : torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for the masked language modelling task.

        Parameters:
        ----------
        batch (torch.Tensor): 
            The input batch containing the sequences.

        Returns:
        -------
        torch.Tensor: 
            The computed cross-entropy loss at the masked indices.
        """
        x = batch['seqs'] # get the sequences
        x_masked, ix = self._mask(x) # mask the sequences
        return self._compute_cross_entropy_and_accuracy(x_masked[:,ix], x[:,ix])



def test_mask():
    # Create an instance of the BaseMLM class
    model = BaseMLM(mask_token=0, mask_frac=0.5)

    # Create a sample input tensor
    input_tensor = torch.tensor([[1, 2, 3, 4, 5]])

    # Call the _mask method
    masked_tensor, masked_indices = model._mask(input_tensor)

    # Assert that the shape of the masked tensor is the same as the input tensor
    assert masked_tensor.shape == input_tensor.shape

    # Assert that the correct number of indices were masked
    assert len(masked_indices) == int(input_tensor.shape[1] * model.mask_frac)    

    # Assert that the masked indices are within the range of the input tensor
    assert all(idx < input_tensor.shape[1] for idx in masked_indices)

    # Assert that the masked indices in the masked tensor are equal to the mask token
    assert all(masked_tensor[0, idx] == model.mask_token for idx in masked_indices)

    print("Test passed!")

if __name__ == '__main__':
    # Run the test
    test_mask()