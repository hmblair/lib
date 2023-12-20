from typing import Any, Callable, Union, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn, rank_zero_info
import psutil

from .hook import HookList, patch_and_register_layer_hooks
from .weight_init import xavier_init

# ignore the following warnings
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", "Your `IterableDataset` has `__len__` defined*")


def module_requires_grad(module: torch.nn.Module) -> bool:
    """
    Checks whether all parameters of the given module have requires_grad set to 
    True.

    Parameters:
    ----------
    module (torch.nn.Module): 
        The module to be verified.

    Returns:
    -------
    bool: 
        True if all parameters of the module have requires_grad set to True, 
        False otherwise.
    """
    return all(param.requires_grad for param in module.parameters())



def get_mem_usage() -> float:
    """
    Retrieves the current memory usage of the entire system.

    Returns:
    --------
    float: 
        The current memory usage.
    """
    process = psutil.Process()
    return float(process.memory_info().rss)


def get_gpu_mem_usage() -> tuple[float, float]:
    """
    Retrieves the current absolute and relative memory usage of the GPU.

    Returns:
    --------
    tuple[float, float]: 
        The current absolute and relative GPU memory usage.
    """
    available, total = torch.cuda.mem_get_info()
    return float(available), float(available) / float(total)


def log_mem_usage(f : Callable) -> Callable:
    """
    A decorator to log the memory usage of the system after the function is
    called.

    Parameters:
    ----------
    f (Callable): 
        The function to be decorated.

    Returns:
    --------
    Callable: 
        The decorated function.
    """
    def wrapper(self, *args, **kwargs):
        result = f(self, *args, **kwargs)
        mem_usage = get_mem_usage()
        self._log('mem_usage', mem_usage, on_epoch=False)
        if self.device.type == 'cuda':
            abs_gpu_mem_usage, rel_gpu_mem_usage = get_gpu_mem_usage()
            self._log('abs_gpu_mem_usage', abs_gpu_mem_usage, on_epoch=False)
            self._log('rel_gpu_mem_usage', rel_gpu_mem_usage, on_epoch=False)
        return result
    return wrapper



def catch_and_log_errors(f):
    """
    A decorator to catch any errors that occur in the given function and log
    them. If the model has a log method, the error is logged using that method.
    Otherwise, the error is printed.

    Parameters:
    ----------
    f (Callable): 
        The function to be decorated.

    Returns:
    --------
    Callable: 
        The decorated function.
    """
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            # Log the error
            # if hasattr(self, 'logger'):
            #     self.logger.experiment.add_text('Error Log', str(e), global_step=batch_idx)
            # else:
            print(f"Error in {f.__name__}: {e}")
            raise e
    return wrapper



from abc import ABCMeta, abstractmethod
class WeightInitialisationMetaClass(ABCMeta):
    """
    A metaclass that automatically calls the _weight_init() method of a class
    after all child classes are initialized. This is useful for initializing
    the weights of a model after it is initialized.

    Any class that inherits from this metaclass must implement the _weight_init()
    method, and have a save_attention_weights attribute.

    Inherits:
    --------
    ABCMeta: 
        A metaclass that allows abstract methods to be defined.
    """
    def __call__(cls, *args, **kwargs):
        # create an instance of the class using the __call__ method of the type 
        # class
        obj = type.__call__(cls, *args, **kwargs)
        # initialize the weights
        obj._weight_init()

        return obj



class BaseModel(pl.LightningModule, metaclass=WeightInitialisationMetaClass):
    """
    Base class for PyTorch Lightning models that abstracts away some of the 
    boilerplate code.

    Inherits:
    --------
    pl.LightningModule: 
        Base class for all PyTorch Lightning models.
    """
    def __init__(self, validate_losses : bool = True):
        super().__init__()
        self.validate_losses = validate_losses

        # create a list to store any hooks that are registered
        self.hooks = HookList()

        # save the hyperparameters
        self.save_hyperparameters()  


    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass of the model. This method should be implemented in the 
        subclass.

        Parameters:
        ----------
        *args: 
            Variable length argument list.
        **kwargs: 
            Arbitrary keyword arguments.

        Returns:
        --------
        Any: 
            The output of the model.
        """
        return super().forward(*args, **kwargs)
    

    def _weight_init(self) -> None:
        """
        Initializes the model weights. This method is called automatically
        when the model is initialized. By default, it initializes the weights
        using Xavier initialization. If you want to use a different 
        initialization method, you can override this method in your subclass.
        """
        if not self.modules():
            raise RuntimeError(
                'No modules found in the model for weight initialization.'
                )
        try:
            for m in self.modules():
                if not m.children() and module_requires_grad(m):
                    xavier_init(m)
        except Exception as e:
            raise RuntimeError(
                'An error occurred during weight initialization.'
                ) from e


    @catch_and_log_errors
    # @log_mem_usage
    def training_step(
            self, 
            batch : Any, 
            batch_ix : list[int]
            ) -> torch.Tensor:
        """
        Performs a single training step.

        Parameters:
        ----------
        batch (Any): 
            The input batch data.
        batch_ix (int): 
            The index of the current batch.

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
    

    @catch_and_log_errors
    # @log_mem_usage
    def validation_step(
            self, 
            batch : Any, 
            batch_ix : list[int],
            ) -> None:
        """
        Perform a validation step on a batch of data.

        Parameters:
        ----------
        batch (Any): 
            The input batch data.
        batch_ix (int): 
            The index of the current batch.
        """
        _ = self._compute_and_log_losses(batch, 'val') # compute the losses

    
    @catch_and_log_errors
    # @log_mem_usage
    def test_step(
            self, 
            batch : Any, 
            batch_ix : list[int],
            ) -> None:
        """
        Perform a test step on a batch of data.

        Parameters:
        ----------
        batch (Any): 
            The input batch data.
        batch_ix (int): 
            The index of the current batch.
        """
        _ = self._compute_and_log_losses(batch, 'test') # compute the losses


    @catch_and_log_errors
    def predict_step(
            self, 
            batch : Any, 
            batch_ix : list[int]
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a prediction step on a batch of data.

        Parameters:
        ----------
        batch (Any): 
           The input batch data.
        batch_ix (int): 
            The index of the current batch.

        Returns:
        --------
        tuple[torch.Tensor, torch.Tensor]: 
            The inputs and predicted outputs from the model for the input batch.
        """ 
        # get the input from the batch
        x, y = self._get_inputs_and_outputs(batch) 

        # return the input and the predicted output
        return x, self(x), y
    

    def _compute_and_log_losses(
            self, 
            batch : torch.Tensor, 
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
        # compute the losses
        losses = self._compute_losses(batch) 

        # loop through the losses, ensuring that they are valid and logging them
        for name, value in losses.items():
            if self.validate_losses:
                self._validate_losses(value, name) 
            self._log(phase + '_' + name, value) 
        return losses['loss']


    def _compute_losses(self, batch : Any) -> dict[str, torch.Tensor]:
        """
        Compute the losses for the model. The loss named 'loss' will be the one 
        which is used to train the model.

        Parameters:
        ----------
        batch (Any): 
            The input batch of data.

        Returns:
        --------
        dict[str, torch.Tensor]: 
            A dictionary containing the computed losses and their respective 
            names.
        """
        raise NotImplementedError('The _compute_losses method must be implemented.')
    

    def _log(self, name: str, value: torch.Tensor, on_epoch : bool = True, **kwargs) -> None:
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
        if on_epoch:
            self.log(
                name=name, 
                value=value, 
                prog_bar=True, 
                sync_dist=True, 
                on_epoch=True, 
                on_step=False, 
                **kwargs
                )
        else:
            self.log(
                name=name, 
                value=value, 
                prog_bar=True, 
                sync_dist=True, 
                on_epoch=False, 
                on_step=True, 
                **kwargs)


    def _validate_losses(self, loss : torch.Tensor, name : str) -> None:
        """
        Validates the loss value to ensure it is not NaN, infinite, or negative.

        Parameters:
        ----------
        loss: 
            The loss value to be validated.
        name: 
            The name of the loss value.

        Raises:
        -------
        ValueError: 
            If the loss value is NaN or infinite.
        """
        if loss.isnan():
            raise ValueError(f'The {name} is NaN.')
        if loss.isinf():
            raise ValueError(f'The {name} is infinite.')
        if loss < 0:
            rank_zero_warn(f'The {name} is negative.')


    def _get_inputs_and_outputs(self, batch : Any) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the inputs and corresponding outputs from the given batch.

        Parameters:
        ----------
        batch (Any): 
            The input batch.

        Returns:
        --------
        tuple(torch.Tensor, torch.Tensor): 
            The inputs and corresponding outputs.
        
        Examples:
            >>> x, y = self._get_inputs_and_outputs(batch)
        """
        raise NotImplementedError('The _get_inputs_and_outputs method must be implemented.')
    

    def _get_inputs(self, batch : Any) -> torch.Tensor:
        """
        Get the inputs from the given batch.

        Parameters:
        ----------
        batch (Any): 
            The input batch.

        Returns:
        --------
        torch.Tensor: 
            The inputs.
        """
        raise NotImplementedError('The _get_inputs method must be implemented.')
    

    def _get_outputs(self, batch : Any) -> torch.Tensor:
        """
        Get the outputs from the given batch.

        Parameters:
        ----------
        batch (Any): 
            The input batch.

        Returns:
        --------
        torch.Tensor: 
            The outputs.
        """
        raise NotImplementedError('The _get_outputs method must be implemented.')


    def _check_constant(self, x : torch.Tensor, eps : float = 1E-8) -> bool:
        """
        Checks whether the given tensor is constant.

        Parameters:
        ----------
        x (torch.Tensor): 
            The tensor to check.
        eps (float): 
            The tolerance for the check.

        Returns:
        --------
        bool: 
            True if the tensor is constant, else False.
        """
        return torch.allclose(x, x[0], atol=eps)

    
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
    







class BaseClassifier(BaseModel):
    """
    Base class for PyTorch Lightning models that perform classification. 

    Attributes:
    ----------
    cross_entropy (torch.nn.CrossEntropyLoss): 
        The cross-entropy loss function.

    Inherits:
    --------
    BaseModel: 
        Base class for PyTorch Lightning models that abstracts away some of the 
        boilerplate code.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cross_entropy = torch.nn.CrossEntropyLoss()


    def _compute_cross_entropy_and_accuracy(
            self, x : torch.Tensor, y : torch.Tensor
            ) -> dict[str, torch.Tensor]:
        """
        Computes the cross-entropy loss and accuracy for the given inputs and
        targets.

        Parameters:
        ----------
        x (torch.Tensor): 
            The input tensor.
        y (torch.Tensor):
            The target tensor.

        Returns:
        --------
        dict[str, torch.Tensor]: 
            A dictionary containing the computed loss and accuracy.
        """
        # get the logits via the forward step
        logits = self(x) 
        # compute the loss at the masked indices
        loss = self.cross_entropy(y, logits) 

        # compute the predicted tokens from the logits
        predictions = torch.argmax(logits, dim=-1) 
        # compute the accuracy
        accuracy = torch.sum(predictions == y) 

        return {'loss': loss, 'accuracy': accuracy}