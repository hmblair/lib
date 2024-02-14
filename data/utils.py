# utils.py

import torch
import xarray as xr
from pathlib import Path

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



def xarray_to_dict(ds: xr.Dataset) -> dict[str, torch.Tensor]:
    """
    Convert an xarray dataset to a dictionary of tensors.
    """
    return {
        name : torch.Tensor(ds[name].values)
        for name in ds.data_vars
        }



def get_filename(file: str) -> str:
    """
    Get the filename from a path, without the extension, or None if the file
    is None.
    """
    return Path(file).stem if file is not None else None



def construct_slices_for_iterable_dataset(
        num_datapoints : int, 
        batch_size : int, 
        world_size : int, 
        rank : int,
        ) -> list[slice]:
    """
    Construct slices for an iterable dataset in a distributed setting. The 
    number of batches is guaranteed to be divisible by the number of devices.
    In order to accomplish this, the size of the final batch is automatically
    adjusted to ensure that there are enough batches for each device. By passing
    the rank of the current device, the slices can be distributed across the
    devices.

    Parameters:
    ----------
    num_datapoints (int):
        The number of datapoints in the dataset.
    batch_size (int):
        The batch size.
    world_size (int):
        The number of devices.
    rank (int):
        The rank of the current device.

    Returns:
    -------
    list[slice]:
        The slices for the dataset.
    """
    # get the number of datapoints per batch
    datapoints_per_batch = batch_size * world_size

    # get the number of datapoints which will not fit exactly into a batch
    overflow_datapoints = num_datapoints % datapoints_per_batch

    # construct the initial slices, ignoring the overflow datapoints
    slices = [
        slice(i, i + batch_size) 
        for i in range(0, num_datapoints - overflow_datapoints, batch_size)
        ]
    
    assert len(slices) % world_size == 0, f'The old number of batches ({len(slices)}) must be divisible by the number of devices ({world_size}).'

    # construct the slices for the overflow datapoints            
    overflow_batch_size = overflow_datapoints // world_size
    if overflow_batch_size > 0:
        slices += [
            slice(i, i + overflow_batch_size)
            for i in range(num_datapoints - overflow_datapoints, num_datapoints, overflow_batch_size)
            ]
    
    # remove any final slices that are not divisible by the number of devices
    r = len(slices) % world_size
    if not r == 0:
        slices = slices[:-r]

    # ensure that the number of batches is divisible by the number of devices
    assert len(slices) % world_size == 0, f'The new number of batches ({len(slices)}) must be divisible by the number of devices ({world_size}).'
    
    # distribute the batches across the devices
    return slices[rank::world_size]