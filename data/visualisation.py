# visualisation.py

from typing import Sequence

def print_colored_string(
        string : str, 
        floats : Sequence[int], 
        ) -> None:
    """
    Print a string with each character colored according to the intensity of
    the corresponding float in floats. The color of each character is
    interpolated between red (255,0,0) and white (255,255,255).

    Parameters:
    ----------
    string (str):
        The string to print.
    floats (Sequence[int]):
        The floats to use to color the string.
    """
    if not len(string) == len(floats):
        raise ValueError(
            'The length of the string and floats must be the same.'
            )
    min_float = min(floats)
    max_float = max(floats)
    normalized_floats = [(f - min_float) / (max_float - min_float) for f in floats]

    for char, intensity in zip(string, normalized_floats):
        # Interpolate between red (255,0,0) and white (255,255,255)
        pixel_value = int(255 * intensity)
        print(f"\x1b[38;2;255;{pixel_value};{pixel_value}m{char}", end="")

    print("\x1b[0m")  # Reset to default color at the end