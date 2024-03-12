# modules.py

from typing import Any, Callable, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from .hooks import HookList, patch_and_register_layer_hooks
from ..models.dense import DenseNetwork

class PipelineModule(pl.LightningModule):
    """
    Base class for Pytorch Lightning modules. It abstracts away much of the 
    boilerplate code of training and inference, and provides a simple interface
    for logging and checkpointing. It also provides a simple interface for
    registering hooks on layers of the model.

    Parameters:
    ----------
    model (nn.Module):
        The model to be used.
    objectives (Optional[dict[str, nn.Module]]):
        A dictionary of objectives, where the key is the name of the objective
        and the value is the objective itself. Defaults to None.
    """
    def __init__(
            self, 
            model : nn.Module,
            objectives : Optional[dict[str, nn.Module]] = None, 
            name : Optional[str] = None,
            *args, **kwargs,
            ) -> None:
        super().__init__(*args, **kwargs)
        # store the model and name
        self.model = model
        if name is not None:
            self.name = name

        # create a list to store any hooks that are registered
        self.hooks = HookList()

        # store the objectives
        if objectives is not None and 'loss' not in objectives:
            raise ValueError(
                'The objectives must contain a loss function with the key "loss", since this is the one which is used to train the model.'
                )
        self.objectives = nn.ModuleDict(objectives) if objectives is not None else None

        # save the hyperparameters, excluding the objectives
        self.save_hyperparameters(ignore=['objectives', 'model'])

    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through the model defined by the model attribute.
        """
        return self.model(*args, **kwargs)
    

    def training_step(
            self, 
            batch : Any, 
            batch_ix : list[int],
            dataloader_idx : int = 0,
            ) -> torch.Tensor:
        """
        Performs a single training step.

        Parameters:
        ----------
        batch (Any): 
            The input batch data.
        batch_ix (int): 
            The index of the current batch.
        dataloader_idx (int):
            The index of the current dataloader. Defaults to 0, which will be
            the case if there is only one dataloader.

        Returns:
        --------
        torch.Tensor: 
            The loss value for the training step.
        """
        # compute and log the losses
        loss = self._compute_and_log_losses(batch, 'train')

        # compute and log the learning rate
        lr = self._get_lr() 
        self._log('lr', lr, on_epoch=False) 
        
        return loss
    

    def validation_step(
            self, 
            batch : Any, 
            batch_ix : list[int],
            dataloader_idx : int = 0,
            ) -> None:
        """
        Perform a validation step on a batch of data.

        Parameters:
        ----------
        batch (Any): 
            The input batch data.
        batch_ix (int): 
            The index of the current batch.
        dataloader_idx (int):
            The index of the current dataloader. Defaults to 0, which will be
            the case if there is only one dataloader.
        """
        _ = self._compute_and_log_losses(batch, 'val') # compute the losses

    
    def test_step(
            self, 
            batch : Any, 
            batch_ix : list[int],
            dataloader_idx : int = 0,
            ) -> None:
        """
        Perform a test step on a batch of data.

        Parameters:
        ----------
        batch (Any): 
            The input batch data.
        batch_ix (int): 
            The index of the current batch.
        dataloader_idx (int):
            The index of the current dataloader. Defaults to 0, which will be
            the case if there is only one dataloader.
        """
        _ = self._compute_and_log_losses(batch, 'test') # compute the losses


    def predict_step(
            self, 
            batch : tuple[torch.Tensor, None], 
            batch_ix : list[int], 
            dataloader_idx : int = 0,
            ) -> torch.Tensor:
        """
        Perform a prediction step on a batch of data.

        Parameters:
        ----------
        batch (Any): 
           The input batch data.
        batch_ix (int): 
            The index of the current batch.
        dataloader_idx (int):
            The index of the current dataloader. Defaults to 0, which will be
            the case if there is only one dataloader.

        Returns:
        --------
        tuple[torch.Tensor, torch.Tensor]: 
            The inputs and predicted outputs from the model for the input batch.
        """ 
        # get the input from the batch
        x, _ = batch

        # return the input and the predicted output
        return self(x)
    

    def _compute_and_log_losses(
            self, 
            batch : tuple[torch.Tensor, torch.Tensor], 
            phase : str
            ) -> torch.Tensor:
        """
        Compute the relevant losses and log them, returning the loss that is 
        required for training, which is the output of _compute_losses() named 
        'loss'.

        Parameters:
        ----------
        batch (torch.Tensor): 
            The input batch of data.
        phase (str): 
            The current phase.

        Returns:
        --------
        torch.Tensor: 
            The primary loss value for the current step.
        """
        # get the input and target from the batch
        x, y = batch

        # get the model output
        y_hat = self(x) 

        # compute the losses
        losses = {name : obj(y_hat, y) for name, obj in self.objectives.items()}

        # loop through the losses, ensuring that they are valid and logging them
        for name, value in losses.items():
            self._log(
                phase + '_' + name, 
                value, 
                on_step = (phase == 'train'),
                )
        return losses['loss']


    def _log(
        self, 
        name: str, 
        value: torch.Tensor, 
        on_step : bool = True, 
        on_epoch : bool = True, 
        **kwargs,
        ) -> None:
        """
        Logs the given name-value pair with additional optional keyword 
        arguments.

        Parameters:
        ----------
        name (str): 
            The name of the value being logged.
        value (torch.Tensor): 
            The value to be logged.
        **kwargs: 
            Additional optional keyword arguments.
        """
        self.log(
            name=name, 
            value=value, 
            prog_bar=True, 
            sync_dist=True,
            on_epoch=on_epoch, 
            on_step=on_step, 
            **kwargs
            )

    
    def _get_lr(self) -> float:
        """
        Retrieves the current learning rate.

        Returns:
        --------
        float: 
            The current learning rate.
        """
        return self.optimizers().param_groups[0]["lr"]
    

    def patch_and_register_layer_hooks(
            self,
            layer_type : type[nn.Module],
            hook : Callable,
            transform : Optional[Callable] = None,
            patch : Optional[Callable] = None,
            ) -> None:
        """
        Register a hook on all layers of the given type in the model. Along the
        way, optionally patch the layers with the given patch function.

        Parameters:
        ----------
        layer_type (type[nn.Module]):
            The type of layer to register the hook on.
        hook (Callable):
            The hook to register.
        transform (Callable):
            A function to transform the layer before registering the hook.
            This is useful, for example, for registering hooks on the attention
            modules of a transformer layer. Defaults to None.
        patch (Optional[Callable]):
            A function to patch the layer before registering the hook.
            This is useful, for example, for guaranteeing that the attention
            modules of a transformer layer return the attention weights.
            Defaults to None.
        """
        self.hooks.extend(
            patch_and_register_layer_hooks(
                model=self,
                layer_type=layer_type,
                hook=hook,
                transform=transform,
                patch=patch,
                )
            )
    
    
    def remove_hooks(self) -> None:
        """
        Removes all hooks that were registered on the model.
        """
        self.hooks.remove_hooks()



def strip_checkpoint(path : str) -> None:
    """
    Strips the checkpoint file of all unnecessary information, leaving only the
    state dict of the model attribute.

    Parameters:
    ----------
    path (str):
        The path to the checkpoint file.
    """
    checkpoint = torch.load(path)
    state_dict = checkpoint['state_dict']
    if 'model' not in state_dict:
        raise ValueError('The state dict must contain a key "model".')
    path_stripped = path.replace('.ckpt', '_stripped.ckpt')
    torch.save(state_dict['model'], path_stripped)



class FinetuningModuleDenseHead(PipelineModule):
    """
    Attaches a dense head to the output of the model defined in the subclass.
    Freezes the model defined in the subclass, so that by default only the head 
    is trained.

    Parameters:
    ----------
    out_size (int):
        The size of the output tensor.
    embedding_dim (int, optional):
        The dimension of the embedding. If not specified, it will be inferred
        from the attribute "embedding_dim" of the model. Defaults to None.
    hidden_sizes (list[int]):
        A list of hidden layer sizes for the dense head. Defaults to an empty list.
    dropout (float):
        The dropout probability for the dense head. Defaults to 0.0.
    pooling (Optional[dict]):
        A dictionary specifying the pooling layer to use. Defaults to None.
    *args:
        Additional positional arguments to pass to the PipelineModule.
    **kwargs:
        Additional keyword arguments to pass to the PipelineModule.
    """
    def __init__(
            self, 
            out_size : int,
            embedding_dim : Optional[int] = None,
            hidden_sizes : list[int] = [],
            dropout : float = 0.0,
            pooling : Optional[dict] = None,
            *args, **kwargs
            ) -> None:
        super().__init__(*args, **kwargs)

        # get the embedding dimension
        if embedding_dim is None:
            if not hasattr(self.model, 'embedding_dim'):
                raise ValueError(
                    'The model does not have an attribute "embedding_dim", so the embedding dimension must be specified.'
                    )
            embedding_dim = self.model.embedding_dim

        # freeze the model
        self.model.eval()
        self.model.requires_grad_(False)

        # create the dense head
        self.head = DenseNetwork(
            in_size=embedding_dim,
            out_size=out_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            pooling=pooling,
            )
        
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Embeddings are passed through the model defined in the subclass, and 
        these outputs are then passed through the dense head model.
        """
        return self.head(
            super().forward(x)
            )
    


import dgl

class DenoisingDiffusionModule(pl.LightningModule):
    """
    Implements a denoising diffusion model, which is trained to predict the
    noise added to the input data during the forward diffusion process. The 
    forward method of this class samples from the distribution of the diffusion
    model, by repeatedly applying the reverse diffusion process.

    Parameters:
    ----------
    model (nn.Module):
        The model to be used as an approximate posterior for the noise. This 
        should map inputs to outputs of the same shape.
    beta (torch.Tensor):
        The betas for the diffusion process. The length of this tensor
        determines the number of steps in the forward diffusion process.
    """
    def __init__(
            self, 
            model : nn.Module,
            beta_low : float = 0.001,
            beta_high : float = 0.02,
            num_timesteps : int = 1000,
            *args, **kwargs,
            ) -> None:
        super().__init__(*args, **kwargs)

        # store the model and the objective
        self.model = model
        self.objective = nn.MSELoss()

        # store the betas and compute the alphas, registering them as buffers
        self.register_buffer('betas', torch.linspace(beta_low, beta_high, num_timesteps))
        self.register_buffer('alpha', 1 - self.betas)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, 0))

    
    def forward_diffusion(self, graph : dgl.DGLGraph, t : int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies t steps of the forward diffusion process to the coordinates
        in the input graph, and returns the noise that was added to the
        coordinates, along with the graph with the added noise.

        Parameters
        ----------
        graph : dgl.DGLGraph
            The input graph to be noised. It must have a node feature called
            'coordinates', and an edge feature called 'rel_pos'.
        t : int
            The number of forward diffusion steps to apply.
        
        Returns
        -------
        torch.Tensor
            The noise which was added to the input tensor, of the same shape as x.
        dgl.DGLGraph
            The input graph, whose coordinates have been noised.
        """
        # get the coordinates and relative positions
        x = graph.ndata['coordinates']

        # sample standard normal noise
        z = torch.randn_like(x)

        # apply the forward diffusion process
        diffuse_x = x * torch.sqrt(self.alpha_bar[t]) + z * torch.sqrt(1 - self.alpha_bar[t])

        # update the graph with the diffused coordinates
        graph.ndata['coordinates'] = diffuse_x

        return z, graph
    

    def reverse_diffusion(self, graph : dgl.DGLGraph, t : int) -> torch.Tensor:
        """
        Applies a single step of the reverse diffusion process to the coordinates
        of the input graph.

        Parameters
        ----------
        graph : dgl.DGLGraph
            The input graph whose coordinates are to be denoised. It must have a
            node feature called 'coordinates'.
        add_noise : bool, optional
            Whether to add standard normal noise to the denoised tensor. 
            Defaults to True.

        Returns
        -------
        dgl.DGLGraph
            The graph with the denoised coordinates.
        """

        # make a copy of the graph
        graph = graph.local_var()

        # get the coordinates
        x = graph.ndata['coordinates']

        # sample standard normal noise at all steps except the final step
        # (which is the first step of the forward diffusion process)
        z = 0 if t == 0 else torch.randn_like(x)

        # apply the reverse diffusion process
        x_hat = self.model(graph, t)['coordinates']
        undiffuse_x =  1 / torch.sqrt(self.alpha[t]) * (
            (x - (1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_bar[t]) * x_hat) \
                + torch.sqrt(1 - self.alpha[t]) * z
        )

        # update the graph with the undiffused coordinates
        graph.ndata['coordinates'] = undiffuse_x

        return graph
    

    def forward(self, shape : torch.Size) -> torch.Tensor:
        """
        Samples from the distribution of the diffusion model, by repeatedly
        applying the reverse diffusion process. 

        Parameters
        ----------
        shape : torch.Size
            The shape of the tensor to be sampled.

        Returns
        -------
        torch.Tensor
            A sample from the diffusion model, of the specified shape.
        """

        # initialise the sample by sampling standard normal noise
        x = torch.randn(shape, device=self.device)

        # apply the reverse diffusion process
        for t in range(len(self.betas) - 1, -1, -1):
            x = self.reverse_diffusion(x, t)

        return x
    

    def loss_step(
            self, 
            batch : torch.Tensor, 
            phase : str,
            ) -> torch.Tensor:
        """
        Sample a random timestep, and apply the forward and reverse diffusion
        processes to the input batch of data. Given that the model is trying to
        predict the noise added to the model, the output of this process is 
        copmared to the original noise that was used in the forward diffusion
        process.
        
        Parameters
        ----------
        batch : torch.Tensor
            The input batch of data, which will be noised and denoised.
        batch_ix : list[int]
            The index of the current batch.
        dataloader_idx : int
            The index of the current dataloader. Defaults to 0, which will be
            the case if there is only one dataloader.

        Returns
        -------
        torch.Tensor
            The loss value for the training step.
        """
        # sample a random timestep
        t = torch.randint(0, len(self.betas), (1,))

        # apply the forward diffusion process
        z, x = self.forward_diffusion(batch, t)

        # apply the model
        z_hat = self.model(x, t)

        # get the noise from the coordinates of z_hat
        z_hat = z_hat.ndata['coordinates']

        # compute the loss
        loss = self.objective(z_hat, z)

        # log the loss
        self.log(
            name=phase + '_loss', 
            value=loss, 
            on_step=True,
            on_epoch=True, 
            batch_size=batch.batch_size,
            )

    
    def training_step(
            self, 
            batch : torch.Tensor, 
            batch_ix : list[int],
            dataloader_idx : int = 0,
            ) -> torch.Tensor:
        """
        Performs a single training step.

        Parameters:
        ----------
        batch (Any): 
            The input batch data.
        batch_ix (int): 
            The index of the current batch.
        dataloader_idx (int):
            The index of the current dataloader. Defaults to 0, which will be
            the case if there is only one dataloader.

        Returns:
        --------
        torch.Tensor: 
            The loss value for the training step.
        """
        self.loss_step(batch, 'train')


    def validation_step(
            self, 
            batch : torch.Tensor, 
            batch_ix : list[int],
            dataloader_idx : int = 0,
            ) -> None:
        """
        Perform a validation step on a batch of data.

        Parameters:
        ----------
        batch (Any): 
            The input batch data.
        batch_ix (int): 
            The index of the current batch.
        dataloader_idx (int):
            The index of the current dataloader. Defaults to 0, which will be
            the case if there is only one dataloader.
        """
        self.loss_step(batch, 'val')


    def test_step(
            self, 
            batch : torch.Tensor, 
            batch_ix : list[int],
            dataloader_idx : int = 0,
            ) -> None:
        """
        Perform a test step on a batch of data.

        Parameters:
        ----------
        batch (Any): 
            The input batch data.
        batch_ix (int): 
            The index of the current batch.
        dataloader_idx (int):
            The index of the current dataloader. Defaults to 0, which will be
            the case if there is only one dataloader.
        """
        self.loss_step(batch, 'test')