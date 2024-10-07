import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def redblue(m=256):
    """
    Generate a red-blue colormap.

    This function creates a colormap that transitions from blue to white to red.
    It's useful for visualizing data that ranges from negative to positive values.

    Parameters:
    m (int): The number of color segments to generate. Default is 256.

    Returns:
    numpy.ndarray: An m x 3 array representing the RGB values of the colormap.
    """
    if m % 2 == 0:
        # Even number of segments
        # From [0 0 1] (blue) to [1 1 1] (white), then [1 1 1] to [1 0 0] (red)
        m1 = m * 0.5
        r = np.linspace(0, 1, int(m1))
        g = r.copy()
        r = np.concatenate((r, np.ones(int(m1))))
        g = np.concatenate((g, np.flip(g)))
        b = np.flip(r)
    else:
        # Odd number of segments
        # From [0 0 1] (blue) to [1 1 1] (white) to [1 0 0] (red)
        m1 = np.floor(m * 0.5)
        r = np.linspace(0, 1, int(m1))
        g = r.copy()
        r = np.concatenate((r, np.ones(int(m1) + 1)))
        g = np.concatenate((g, [1], np.flip(g)))
        b = np.flip(r)
    
    return np.column_stack((r, g, b))

# Create the colormap
redblue_cmap = LinearSegmentedColormap.from_list("redblue", redblue(256))