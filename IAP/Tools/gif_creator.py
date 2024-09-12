import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.colors import LinearSegmentedColormap

def setCustomColorMap():
    """
    create the colormap for diffraction data (the same as matlab)
    return: customized matplotlib colormap
    """
    colors = [
        (0.0, 0.0, 0.2),
        (0, 0.0875, 1),
        (0, 0.4928, 1),
        (0, 1, 0),
        (1, 0.6614, 0),
        (1, 0.4384, 0),
        (0.8361, 0, 0),
        (0.6505, 0, 0),
        (0.4882, 0, 0),
    ]

    n = 255  # Discretizes the interpolation into n bins
    cm = LinearSegmentedColormap.from_list("cmap", colors, n)
    return cm


def create_gif(data, scale='log', colormap=None, fps=1, output_filename='output.gif'):
    """
    Creates an animated GIF from a 3D numpy array with a specified colormap.

    Parameters:
        data (numpy.ndarray): A 3D numpy array where each slice along the first axis is a frame.
        colormap (str): The name of the colormap to use (e.g., 'viridis', 'plasma', 'magma').
        duration (float): Duration each frame is displayed in the GIF in seconds.
        output_filename (str): Filename for the output GIF.
    """
    data = np.log10(data+1) if scale == 'log' else data

    # Normalize the data to fit within the colormap's range
    data_normalized = (data - data.min()) / (data.max() - data.min())

    # Set up the writer for output GIF
    writer = imageio.get_writer(output_filename, mode='I', format='GIF', loop=0, fps=fps)

    if colormap is None:
        cmap = setCustomColorMap()
    else:
        # Get the colormap
        cmap = plt.get_cmap(colormap)

    # Apply the colormap to each frame and write to the GIF
    for frame in data_normalized:
        # Apply colormap
        colored_frame = cmap(frame)  # This returns RGBA values
        # Convert RGBA to 8-bit RGB suitable for imageio
        colored_image = (255 * colored_frame).astype(np.uint8)
        # Write frame
        writer.append_data(colored_image[:, :, :3],)  # Exclude alpha channel

    # Close the writer to finish the GIF
    writer.close()
    print(f"GIF created successfully: {output_filename}")