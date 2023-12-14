# distributed_helper.py

import torch
from typing import Any
import warnings

def get_worker_info() -> tuple[int, int]:
    """
    Gets the id of the current worker and the total number of workers.

    Returns:
    --------
    tuple[int, int]: 
        The id of the current worker and the total number of workers.
    """
    worker_info = torch.utils.data.get_worker_info()
    worker_id = (0 if worker_info is None else worker_info.id)
    num_workers = (1 if worker_info is None else worker_info.num_workers)
    return worker_id, num_workers


def get_device_info() -> tuple[int, int]:
    """
    Gets the id of the current device and the total number of devices.

    Returns:
    --------
    tuple[int, int]: 
        The id of the current device and the total number of devices.
    """
    if torch.distributed.is_initialized():
        device_id = torch.distributed.get_rank()
        num_devices = torch.distributed.get_world_size()
    else:
        device_id = 0
        num_devices = 1
    return device_id, num_devices