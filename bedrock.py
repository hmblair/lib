# bedrock.py

from typing import Any, Callable, Union, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn, rank_zero_info
import psutil

from .models.utils.hook import HookList, patch_and_register_layer_hooks
from .models.utils.weight_init import xavier_init

from torch.utils.data import DataLoader
import os

from typing import Iterable, Sequence
import warnings


# ignore the following warnings
# import warnings
# warnings.filterwarnings("ignore", ".*does not have many workers.*")
# warnings.filterwarnings("ignore", "*has `__len__` defined*")


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


def catch_and_log_errors(f : Callable) -> Callable:
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






from .models.utils.weight_init import WeightInitialisationMetaClass
from abc import abstractmethod
class BedrockModel(pl.LightningModule, metaclass=WeightInitialisationMetaClass):
    """
    Base class for Pytorch Lightning modules. It abstracts away much of the 
    boilerplate code of the entire deep learning pipeline.

    Parameters:
    ----------
    batch_size (int): 
        The batch size for the dataloaders.
    num_workers (int, optional): 
        The number of workers for data loading. If set to -1, the number of
        workers is set to the number of available CPUs. Defaults to -1.
    
    Attributes:
    ----------
    data (dict[str, Sequence | Iterable]): 
        A dictionary containing the datasets for each phase. This is initialised
        empty and is populated by the setup method.
    batch_size (int): 
        The batch size for the dataloaders.
    rank (int):
        The rank of the current process, as determined by the trainer object.
        If no trainer object is found, the rank is set to 0.
    world_size (int):
        The total number of processes, as determined by the trainer object.
        If no trainer object is found, the world size is set to 1.
    num_workers (int): 
        The number of workers to be used for data loading. 

    Methods:
    -------
    _create_datasets: 
        Creates the dataset for the specified phase. Must be implemented by the 
        subclass.
    _create_dataloaders: 
        Creates the dataloader for the specified phase. Can be overwritten by 
        a subclass.
    setup: 
        Calls the _create_datasets method for the specified stage.
    train_dataloader: 
        Calls the _create_dataloaders method for the 'train' phase.
    val_dataloader: 
        Calls the _create_dataloaders method for the 'validate' phase.
    test_dataloader: 
        Calls the _create_dataloaders method for the 'test' phase.
    pred_dataloader: 
        Calls the _create_dataloaders method for the 'predict' phase.
    get_inputs:
        Extracts the inputs from the batch. Must be implemented by the subclass.
    get_targets:
        Extracts the targets from the batch. Must be implemented by the subclass.
    on_before_batch_transfer:
        Called before a batch is transferred to the device. This method extracts
        the inputs and, if the phase not 'predict', the targets from the batch.
    compute_losses:
        Computes the losses given inputs and outputs. Must be implemented by the
        subclass.
    """
    def __init__(
            self, 
            batch_size : int, 
            num_workers : int = -1,
            ) -> None:
        super().__init__()

        # initialise the data dictionary
        self.data = {}

        # save batch size
        self.batch_size = batch_size

        # determine the number of workers from the number of available CPUs if 
        # num_workers is set to -1, otherwise use the provided value
        if num_workers == -1:
            num_workers = os.cpu_count()
        self.num_workers = num_workers

        # create a list to store any hooks that are registered
        self.hooks = HookList()

        # save the hyperparameters
        self.save_hyperparameters()  
    

    def distributed_info(self) -> tuple[int, int]:
        """
        Returns the rank and world size of the current process. If this is 
        called too early, before the trainer object has been created, the rank
        is set to 0 and the world size is set to 1, which may cause issues with
        distributed training.

        Returns:
        -------
        tuple[int, int]: 
            The rank and world size of the current process.
        """
        try:
            rank = self.trainer.global_rank
            world_size = self.trainer.world_size
        except RuntimeError:
            warnings.warn(
                message = 'No trainer object found. Setting rank to 0 and world' \
                    ' size to 1. To use distributed training, please pass this' \
                    ' DataModule to a trainer object.', 
                category = UserWarning, 
                stacklevel = 2
                )
            rank = 0
            world_size = 1

        return rank, world_size


    @abstractmethod
    def _create_datasets(
        self, 
        phase : str, 
        rank : int, 
        world_size : int,
        ) -> Union[Sequence, Iterable]:
        """
        Create a dataset for the specified phase.

        Parameters:
        ----------
        phase (str): 
            The phase for which to create the datasets. Can be one of 'train', 
            'val', 'test', or 'predict'.
        rank (int):
            The rank of the current process.
        world_size (int):
            The total number of processes.

        Returns:
        -------
        (Sequence | Iterable): 
            The dataset for the specified phase.
        """
        return   
    

    def _create_dataloaders(self, phase: str) -> DataLoader:
        """
        Create a dataloader for the specified phase. Overwrite this method if 
        you want to use a custom dataloader construction, such as if batching
        is handled by the dataset itself.

        The default implementation creates a dataloader with the following 
        parameters:
        - num_workers = self.num_workers
        - batch_size = self.batch_size
        - shuffle = (phase == 'train')

        Parameters:
        phase (str): 
            The phase for which to create the dataloaders. Can be one of 
            'train', 'validate', 'test', or 'predict'.

        Returns:
        -------
        torch.utils.data.DataLoader: 
            The dataloader for the specified phase.
        """
        if phase not in ['train', 'validate', 'test', 'predict']:
            raise ValueError(
                f'Unknown phase {phase}. Please specify one of "train", "validate", "test", or "predict".'
                )
        
        if phase not in self.data:
            raise ValueError(
                f'There is no {phase} dataset. Please call the setup method with the appropriate stage first.'
                )

        return DataLoader(
            dataset = self.data[phase],
            num_workers = self.num_workers,
            batch_size = self.batch_size,
            shuffle = (phase == 'train'),
        )
    

    def setup(self, stage: str) -> None:
        """
        Creates datasets for the specified stage, and stores them in the 
        'self.data' dictionary.

        Parameters:
        ----------
        stage (str): 
            The stage of the data setup. Must be either 'fit', 'validate', 
            'test', or 'predict'.

        Raises:
        -------
        ValueError: 
            If the stage is not one of 'fit', 'validate', 'test', or 'predict'.
        """
        rank, world_size = self.distributed_info()

        if stage == 'fit':
            self.data['train'] = self._create_datasets(
                'train', rank, world_size,
                )
            self.data['validate'] = self._create_datasets(
                'validate', rank, world_size,
                )
        
        elif stage in ['test', 'validate', 'predict']:
            self.data[stage] = self._create_datasets(
                stage, rank, world_size,
                )
        else:
            raise ValueError(
                f'Invalid stage {stage}. The stage must be either "fit", "validate", "test" or "predict".'
                )


    def train_dataloader(self) -> DataLoader:
        """
        Returns the train dataloader.

        Returns:
        -------
        torch.utils.data.DataLoader: 
            The train dataloader.
        """
        return self._create_dataloaders('train')


    def val_dataloader(self) -> DataLoader:
        """
        Returns the validaiton dataloader, if a validation dataset exists. Else,
        raises a NotImplementedError.

        Returns:
        -------
        torch.utils.data.DataLoader: 
            The validation dataloader.
        """
        if self.data['validate'] is None:
            return super().val_dataloader()
        else:
            return self._create_dataloaders('validate')


    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader.

        Returns:
        -------
        torch.utils.data.DataLoader: 
            The test dataloader.
        """
        if self.data['test'] is None:
            raise NotImplementedError(
                'No test dataset found. Please ensure there is a test dataset when initialising the data module.'
                )
        return self._create_dataloaders('test')


    def predict_dataloader(self) -> DataLoader:
        """
        Returns the prediction dataloader.

        Returns:
        -------
        torch.utils.data.DataLoader: 
            The prediction dataloader.
        """
        if self.data['predict'] is None:
            raise NotImplementedError(
                'No prediction dataset found. Please ensure there is a prediction dataset when initialising the data module.'
                )
        return self._create_dataloaders('predict')
    

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
            batch : tuple[torch.Tensor, None], 
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
        x, _ = batch

        # return the input and the predicted output
        return x, self(x)
    

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
        losses = self.compute_losses(y_hat, y) 

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


    @abstractmethod
    def get_inputs(self, batch : Any) -> torch.Tensor:
        """
        Extracts the inputs from the batch. This is useful as the batch may be
        of varying types, such as a tuple or a dictionary, and so this method
        allows you to provide an interface for the model to retrieve the inputs.

        Parameters:
        ----------
        batch (Any): 
            The batch to extract the inputs from.

        Returns:
        -------
        torch.Tensor: 
            The inputs.
        """
        return
    

    @abstractmethod
    def get_targets(self, batch : Any) -> torch.Tensor:
        """
        Extracts the targets from the batch. This is useful as the batch may be
        of varying types, such as a tuple or a dictionary, and so this method
        allows you to provide an interface for the model to retrieve the targets.

        Parameters:
        ----------
        batch (Any): 
            The batch to extract the targets from.

        Returns:
        -------
        torch.Tensor: 
            The targets.
        """
        return
    

    def on_before_batch_transfer(
            self, 
            batch: Any, 
            dataloader_idx: int,
            ) -> tuple[Any, Any]:
        """
        Called before a batch is transferred to the device. This method extracts
        the inputs and, if the phase not 'predict', the targets from the batch.
        In the latter case, the targets are returned as None.

        Parameters:
        ----------
        batch (Any): 
            The batch to extract the inputs and targets from.
        dataloader_idx (int):
            The index of the dataloader.

        Returns:
        -------
        tuple[Any, Any]:
            The inputs and, if the phase is not 'predict', targets, else None.
        """
        if self.trainer.predicting:
            return self.get_inputs(batch), None
        else:
            return self.get_inputs(batch), self.get_targets(batch)
        

    def compute_losses(
            self, 
            model_outputs: torch.Tensor, 
            targets: torch.Tensor, 
            ) -> dict[str, torch.Tensor]:
        """
        Compute the losses given inputs and outputs. The loss named 'loss' will 
        be the one which is used to train the model.

        Parameters:
        ----------
        model_outputs (torch.Tensor): 
            The outputs of the model.
        targets (torch.Tensor):
            The targets.

        Returns:
        -------
        dict[str, torch.Tensor]: 
            A dictionary containing the computed losses and their respective 
            names.
        """
        raise NotImplementedError(
            'Please implement the compute_losses method in your subclass if you are going to use any phase other than "predict".'
            )
    



class LoRALayerWrapper(nn.Module):
    """
    Wraps a pre-trained nn.Module with a LoRA layer.

    Parameters:
    -----------
    base_module (nn.Module): 
        The base module to be wrapped.
    lora_rank (int): 
        The rank of the LoRA layer.
    device (str | torch.device): 
        The device to initialise the LoRA weights on.

    Attributes:
    -----------
    base_module (nn.Module): 
        The base module being wrapped.
    lora_A (nn.Parameter): 
        LoRA weight A.
    lora_B (nn.Parameter): 
        LoRA weight B.
    """
    def __init__(
            self, 
            base_module: nn.Module, 
            lora_rank: int, 
            device: Union[str, torch.device]
            ) -> None:
        super().__init__()
        self.base_module = base_module
        weight_shape = self.base_module.weight.shape

        self.lora_A = nn.Parameter(
            torch.zeros(weight_shape[0], lora_rank, device=device)
        )
                
        self.lora_B = nn.Parameter(
            torch.randn(weight_shape[1], lora_rank, device=device)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LoRALayerWrapper.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the base module and 
            adding on the LoRA perturbation.
        """
        # compute the output of the base module
        base_out = self.base_module(x)  

        # add on the LoRA perturbation
        return base_out + (x @ self.lora_A) @ self.lora_B.T 


from pytorch_lightning.utilities.model_summary import summarize
class BedrockFinetuningModel(BedrockModel):
    """
    Base class for fine-tuning pre-trained models. Adds logic for freezing and
    unfreezing the weights of the pre-trained model, as well as logic for 
    wrapping the pre-trained model with a LoRA layer.

    Parameters:
    ----------
    unfreeze_epoch (int, optional): 
        The epoch to unfreeze the pre-trained model. Defaults to None.
    lora_p (int, optional): 
        The rank of LoRA. Defaults to None.
    *args: 
        Variable length argument list.
    **kwargs: 
        Arbitrary keyword arguments.

    Attributes:
    ----------
    pt_model (nn.Module): 
        The pre-trained model.
    unfreeze_epoch (int | None): 
        The epoch to unfreeze the pre-trained model. If using LoRA, this is the
        epoch to unfreeze the LoRA parameters.
    lora_p (int | None): 
        The rank of LoRA.
    """
    def __init__(
            self, 
            unfreeze_epoch: Optional[int] = None, 
            lora_p: Optional[int] = None, 
            *args, **kwargs,
            ) -> None:
        super().__init__(*args, **kwargs)

         # load the pre-trained model
        self.pt_model = self.load_model() 
         # freeze the model to begin with
        self.freeze_pt()

        # store the epoch to unfreeze the pre-trained model and the rank of LoRA
        self.unfreeze_epoch = unfreeze_epoch  
        self.lora_p = lora_p

        # wrap the LoRA layer around the pre-trained model    
        if lora_p is not None:
            self._lora_wrapper() 


    @abstractmethod
    def load_model(self) -> nn.Module:
        """
        Loads a pre-trained model. An abstract method which must be implemented
        by the child class.
        """
        return
    

    def load_weights_from_checkpoint(
            self, 
            model_class: type[nn.Module],
            checkpoint_path: str,
            ) -> None:
        """
        Loads a pre-trained model from a checkpoint. Because of the way PyTorch
        Lightning loads models, you must pass in the model class as an argument
        to this method.

        Parameters:
        -----
        model_class (type[nn.Module]):
            The class of the model to load.
        checkpoint_path (str): 
            The path to the checkpoint.
        """
        # load the pre-trained model from the checkpoint
        return model_class.load_from_checkpoint(checkpoint_path) 


    def on_train_epoch_start(self) -> None:
        """
        Callback function called at the start of each training epoch.
        If the `unfreeze_epoch` parameter is set and the current epoch matches 
        the `unfreeze_epoch` value, unfreezes the pre-trained model by setting 
        `requires_grad` to True.
        """
        if self.unfreeze_epoch is not None and self.current_epoch == self.unfreeze_epoch:
            rank_zero_info('We have reached the unfreeze epoch.')
            self.unfreeze_pt() 

    
    def freeze_pt(self) -> None:
        """
        Freezes the pre-trained model.
        """
        # set the model to eval mode
        self.pt_model.eval() 
        # freeze the model
        self.pt_model.requires_grad_(False) 
        rank_zero_info('Freezing the pre-trained model...')
        

    def unfreeze_pt(self) -> None:
        """
        Unfreezes the pre-trained model, or the LoRA weights if using LoRA.
        """
        if self.lora_p is None:
            rank_zero_info('Unfreezing the pre-trained model...')
            for layer in self._layers_to_unfreeze():
                layer.train()
                layer.requires_grad_(True)
        else:
            for param in self.lora_params:
                param.requires_grad_(True) # unfreeze the LoRA parameters
            rank_zero_info('Unfreezing the LoRA parameters...')
            
        # print a model summary showing the updated number of trainable parameters
        rank_zero_info(summarize(self)) 
    

    def _lora_wrapper(self) -> None:
        """
        Wraps the LoRALayerWrapper around the modules in the pre-trained model.
        The LoRA weights are initialised as untrainable, and are only unfrozen
        when 'unfreeze' is called.
        """
        lora_params = []
        for module in self.pt_model.modules():
            if not module.children(): 
                # wrap the LoRA layer around the module
                module = LoRALayerWrapper(module, self.lora_p, self.device) 
                # store the LoRA parameters
                lora_params += [module.lora_A, module.lora_B] 
        # set the LoRA parameters to untrainable
        self.lora_params = nn.ParameterList(lora_params).requires_grad_(False) 


    def _layers_to_unfreeze(self) -> Iterable:
        """
        Get the layers of the pre-trained model to unfreeze. Override this 
        method if only a subset of the layers should be unfrozen. This is only 
        called if LoRA is not being used.

        Returns:
        --------
        Iterable:
            The layers to unfreeze. Defaults to the entire pre-trained model.
        """
        # yield the pre-trained model by default
        yield self.pt_model 
    







import torch
import torch.nn as nn
from typing import Callable, Optional, Union, Any
from .models.utils.hook import HookContextManager
from .visualisation.plotting import image_grid

def patch_attn_to_return_weights(
        m : nn.MultiheadAttention,
        average_attn_weights : Optional[bool] = None,
        ) -> None:
    """
    Force the attention module to return the attention weights.

    Parameters:
    ----------
    m (nn.MultiheadAttention): 
        The attention module.
    average_attn_weights (bool | None):
        Whether to average the attention weights across the heads. If None,
        then the default setting of the attention module is used. The reason 
        to use None is that some attention modules do not have this setting.
    """
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        if average_attn_weights is not None:
            kwargs["average_attn_weights"] = average_attn_weights
        return forward_orig(*args, **kwargs)

    m.forward = wrap


class AttentionWeightsHook(HookContextManager):
    """
    A hook that saves the attention weights of a transformer or attention layer.
    The weights are saved in a list, which can be retrieved using the
    get_weights method.

    Parameters:
    ----------
    model (nn.Module):
        The model to register the hook on.
    layer_type (tuple[type[nn.Module]]):
        The type of layer to register the hook on.

    Attributes:
    ----------
    outputs (list[torch.Tensor]):
        A list to store the attention weights.
    """
    def __init__(self, model : nn.Module, layer_type : tuple[type[nn.Module]]):
        super().__init__(
            module=model,
            layer_type=layer_type,
            patch=patch_attn_to_return_weights,
            )
        # a list to store the attention weights
        self.outputs = []


    def __call__(
            self, 
            module : nn.Module, 
            module_in : Any, 
            module_out : Any,
            ) -> None:
        """
        Save the attention weights of the given module.
        
        Parameters:
        ----------
        module (nn.Module):
            The module whose attention weights are to be saved.
        module_in (Any):
            The input to the module.
        module_out (Any):
            The output of the module.
        """
        self.outputs.append(
            module_out[1].detach()
            )


    def get_weights(self) -> torch.Tensor:
        """
        Get the attention weights of the model, as saved by the hook.

        Returns:
        --------
        torch.Tensor: 
            The attention weights of the model, of shape 
            (*, n_layers, n_heads, seq_len, seq_len).
        """
        return torch.stack(self.outputs, dim=1)
    


def get_attn_layer(model : type[nn.Module]) -> nn.Module:
    """
    Get the attention layer of the given model, which should be an instance of
    one of the following classes: nn.MultiheadAttention, 
    nn.TransformerEncoderLayer, nn.TransformerDecoderLayer.

    Parameters:
    ----------
    model (AttentiveModule): 
        The model to get the attention layer from.

    Returns:
    --------
    nn.Module: 
        The attention layer of the model. If the model is an instance of
        nn.MultiheadAttention, then the model itself is returned. Else, 
        the self-attention module of the model is returned.
    """
    return model if isinstance(model, nn.MultiheadAttention) else model.self_attn



class BaseAttentionModel(BedrockModel):
    def __init__(
            self, 
            attn_layer_type : tuple[type[nn.Module]] = (nn.MultiheadAttention,), 
            *args, **kwargs,
            ):
        super().__init__(*args, **kwargs)
        self.attn_layer_type = attn_layer_type

    
    def get_attn_weights(self, x : torch.Tensor) -> torch.Tensor:
        """
        Get the attention weights of each layer of the model when the given 
        input is passed through the model.

        Parameters:
        ----------
        x (torch.Tensor): 
            The input to the model, of shape (*, seq_len, d_model)

        Returns:
        --------
        torch.Tensor: 
            The attention weights of the model, of shape 
            (*, n_layers, n_heads, seq_len, seq_len)
        """
        with AttentionWeightsHook(self, self.attn_layer_type) as hook:
            self(x)
        return hook.get_weights()
    

    def plot_attention_weights(self, x : torch.Tensor) -> None:
        """
        Plot the attention weights of each layer of the model when the given 
        input is passed through the model.

        Parameters:
        ----------
        x (torch.Tensor): 
            The input to the model, of shape (*, seq_len, d_model)
        """
        weights = self.get_attn_weights(x)
        for input in weights:
            image_grid(input, grid_size=(3, 4))
