# ploting.py

import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable, Optional, Callable
import os
import math

def find_closest_factor_pair(n : int) -> tuple[int, int]:
    """
    Finds the closest pair of factors of a given number.

    Parameters
    ----------
    n : int
        The number to find the closest factor pair for.

    Returns
    -------
    tuple[int, int]
        The closest factor pair.
    """
    closest_pair = (1, n)  
    min_difference = n - 1

    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factor1 = i
            factor2 = n // i
            difference = abs(factor1 - factor2)

            if difference < min_difference:
                min_difference = difference
                closest_pair = (factor1, factor2)

    return closest_pair


def compute_grid_size(n : int, no_whitespace : bool = False) -> tuple[int, int]:
    """
    Computes an optimal grid size for a given number of images.
    
    Parameters
    ----------
    n : int
        The number of images.
    no_whitespace : bool
        Whether to force there to be no whitespace in the grid.

    Returns
    -------
    tuple[int, int]
        The grid size.
    """
    if no_whitespace:
        x, y = find_closest_factor_pair(n)
    else:
        y = math.ceil(math.sqrt(n))
        x = math.ceil(n / y)
    return x, y


def grid(
        data: Iterable[np.ndarray],
        plotting_fn: Callable[[np.ndarray, plt.Axes], None],
        grid_size: Optional[tuple[int, int]] = None,
        title: Optional[str] = None,
        no_whitespace: bool = False,
        save_fig: bool = False,
        save_path: str = "figures",
        fig_size: Optional[tuple[int, int]] = None
        ) -> None:
    """
    Plots a grid of images.

    :param data: Iterable of np.ndarray, the images to plot.
    :param plotting_fn: Callable, a function that plots an image onto an Axes object.
    :param grid_size: tuple of two ints, the size of the grid (rows, cols). If None, computed automatically.
    :param title: str, optional title for the figure.
    :param no_whitespace: bool, whether to compute grid size with no whitespace.
    :param save_fig: bool, whether to save the figure.
    :param save_path: str, path to save the figure.
    :param fig_size: tuple of two ints, size of the figure.
    """
    if data.ndim == 0:
        raise ValueError("No data provided for plotting.")

    if grid_size is None:
        grid_size = compute_grid_size(len(data), no_whitespace)
    if math.prod(grid_size) < len(data):
        raise ValueError("grid_size is too small for the number of images.")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig, axs = plt.subplots(*grid_size, figsize=fig_size)
    axs = np.array(axs).reshape(-1)  # Ensure axs is always a flat array

    for image, ax in zip(data, axs):
        plotting_fn(image, ax)

    if title is not None:
        fig.suptitle(title, fontsize=24)
    
    plt.tight_layout()
    plt.show()

    if save_fig and title:
        fig.savefig(
            os.path.join(save_path, f"{title}.png"),
            bbox_inches='tight',
            dpi=300
        )



def image_grid(
        images : Iterable[np.ndarray], 
        grid_size : tuple[int, int] = None,
        title : Optional[str] = None,
        ) -> None:
    grid(images, lambda image, ax: ax.imshow(image), grid_size, title)


def histogram_grid(
        histograms : Iterable[np.ndarray], 
        grid_size : tuple[int, int] = None,
        title : Optional[str] = None,
        ) -> None:
    grid(histograms, lambda hist, ax: ax.hist(hist), grid_size, title)


def plot_grid(
        plots : Iterable[np.ndarray], 
        grid_size : tuple[int, int] = None,
        title : Optional[str] = None,
        no_whitespace : bool = False,
        ) -> None:
    grid(plots, lambda plot, ax: ax.plot(plot[0], plot[1]), grid_size, title, no_whitespace)


def scatter_grid(
        plots : Iterable[np.ndarray], 
        grid_size : tuple[int, int] = None,
        title : Optional[str] = None,
        no_whitespace : bool = False,
        ) -> None:
    grid(plots, lambda plot, ax: ax.scatter(plot[0], plot[1]), grid_size, title, no_whitespace)