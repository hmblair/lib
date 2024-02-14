# schedulers.py

from torch.optim.lr_scheduler import _LRScheduler
from abc import abstractmethod, ABCMeta

class WarmupAndDecayLRScheduler(_LRScheduler, metaclass=ABCMeta):
    """
    A base class for learning rate schedulers that incorporate warm-up and decay steps.

    Args:
        warmup_steps (int): The number of warm-up steps.
        *args: Additional positional arguments to be passed to the parent class.
        **kwargs: Additional keyword arguments to be passed to the parent class.
    
    Attributes:
        warmup_steps (int): The number of warm-up steps.

    Inherits:
        LRScheduler: Base class for learning rate schedulers.

    Methods:
        _warmup_step(): Calculates a warm-up step for the learning rate scheduler. An abstract method that must be implemented by subclasses.
        _decay_step(): Calculates a decay step for the learning rate scheduler. An abstract method that must be implemented by subclasses.
        get_lr(): Get the learning rate for the current epoch (or step).
    """
    def __init__(self, optimizer, warmup_epochs, *args, **kwargs) -> None:
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, *args, **kwargs)


    def get_lr(self) -> list[float]:
        """
        Get the learning rate for the current epoch (or step).

        If the current epoch is less than the warm-up steps, the learning rate is calculated using the warm-up step function.
        Otherwise, the learning rate is calculated using the decay step function.

        Returns:
            list[float]: The learning rates for each parameter group.
        """
        if self.last_epoch == -1:
            pass
        elif self.last_epoch < self.warmup_epochs:
            return self._warmup_step()
        else:
            return self._decay_step()
    

    @abstractmethod
    def _warmup_step(self) -> list[float]:
        """
        Calculates a warm-up step for the learning rate scheduler.

        Returns:
        --------
        list[float]:
            The learning rates for the current warmup step.
        """
        return
    

    @abstractmethod
    def _decay_step(self) -> list[float]:
        """
        Calculates a decay step for the learning rate scheduler.
        
        Returns:
        --------
        list[float]:
            The learning rates for the current decay step.
        """
        return
    


class LinearWarmupLR(WarmupAndDecayLRScheduler):
    """
    Learning rate scheduler that implements a linear warmup. The decay remains
    an abstract method that must be implemented by subclasses.

    Parameters:
        - *args: Variable length argument list.
        - **kwargs: Arbitrary keyword arguments.

    Inherits:
        - WarmupAndDecayLRScheduler: Base class for learning rate schedulers that incorporate 
        warm-up and decay steps.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # scale down the learning rate by the number of warmup steps
        for group in self.optimizer.param_groups:
            group['lr'] = group['lr'] / (self.warmup_epochs + 1)


    def _warmup_step(self) -> list[float]:
        """
        A linear warmup step for the learning rate. The maximum learning rate is 
        equal to the learning rate provided to the optimizer.

        Returns:
        --------
        list[float]:
            The learning rates for the current warmup step.
        """
        scale = (self.last_epoch + 2) / (self.last_epoch + 1)
        return [group['lr'] * scale for group in self.optimizer.param_groups]



class LinearWarmupAndNoDecayLR(LinearWarmupLR):
    """
    Learning rate scheduler that implements a linear warmup and no decay.

    Inherits:
    ---------
    LinearWarmupLR: 
        Learning rate scheduler that implements a linear warmup. The decay remains
        an abstract method that must be implemented by subclasses.
    """
    def _decay_step(self) -> list[float]:
        """
        A linear warmup step for the learning rate. The maximum learning rate is 
        equal to the learning rate provided to the optimizer.

        Returns:
        ---------
        list[float]:
            The learning rates for the current decay step.
        """
        return [group['lr'] for group in self.optimizer.param_groups]



class LinearWarmupAndInverseSqrtDecayLR(LinearWarmupLR):
    """
    Learning rate scheduler that implements an inverse square root schedule
    with a linear warmup.

    Inherits:
    ----------
    LinearWarmupLR: 
        Learning rate scheduler that implements a linear warmup. The decay 
        remains an abstract method that must be implemented by subclasses.
    """
    def _decay_step(self) -> list[float]:
        """
        An inverse square root decay step for the learning rate. The learning rate 
        is equal to the scale factor multiplied by the current epoch to the power of -1/2.

        Returns:
        ---------
        list[float]:
            The learning rates for the current decay step.
        """
        scale = (self.last_epoch + 2) / (self.last_epoch + 1)
        return [group['lr'] * (scale ** (-1/2)) for group in self.optimizer.param_groups]