
import torch
import torch.nn as nn

def test_backward(model : nn.Module, x : torch.Tensor, y : torch.Tensor) -> None:
    """
    Tests the backward pass of a model.

    Parameters:
    -----------
    model (nn.Module):
        The model to test.
    x (torch.Tensor):
        The input tensor.
    y (torch.Tensor):
        The target tensor.
    """
    original_parameters = [p.clone() for p in model.parameters()]

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Forward pass
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)

    # Backward pass
    loss.backward()
    optimizer.step()

    # test that the model weights are updating
    for original, (name, updated) in zip(original_parameters, model.named_parameters()):
        if original.requires_grad:
            assert not torch.equal(original, updated), f'Parameter {name} has not updated.'