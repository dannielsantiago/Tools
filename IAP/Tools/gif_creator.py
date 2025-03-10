import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.colors import LinearSegmentedColormap
import cv2

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

def setCustomColorMap_r():
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
    cm = LinearSegmentedColormap.from_list("cmap", colors[::-1], n)
    return cm

def hsv2rgb(hsv: np.ndarray) -> np.ndarray:
    """
    Convert a 3D hsv np.ndarray to rgb (5 times faster than colorsys).
    https://stackoverflow.com/questions/27041559/rgb-to-hsv-python-change-hue-continuously
    h,s should be a numpy arrays with values between 0.0 and 1.0
    v should be a numpy array with values between 0.0 and 255.0
    :param hsv: np.ndarray of shape (x,y,3)
    :return: hsv2rgb returns an array of uints between 0 and 255.
    """
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')

def complex2rgb(u, amplitudeScalingFactor=1, scalling=1):
    """
    Preparation function for a complex plot, converting a 2D complex array into an rgb array
    :param u: a 2D complex array
    :return: an rgb array for complex plot
    """
    # hue (normalize angle)
    # if u is on the GPU, remove it as we can toss it now.
    h = np.angle(u).astype(float)
    h = (h + np.pi) / (2 * np.pi)
    # saturation  (ones)
    s = np.ones_like(h)
    # value (normalize brightness to 8-bit)
    v = np.abs(u)
    if amplitudeScalingFactor != 1:
        v[v > amplitudeScalingFactor * np.max(v)] = amplitudeScalingFactor * np.max(v)
    if scalling != 1:
        local_max = np.max(v)
        v = v / (np.max(v) + np.finfo(float).eps) * (2 ** 8 - 1)
        print(f'ratio: {local_max / scalling}, max(v): {np.max(v)}')

        v *= local_max / scalling
        print(f'max(v): {np.max(v)}')

    else:
        v = v / (np.max(v) + np.finfo(float).eps) * (2 ** 8 - 1)

    hsv = np.dstack([h, s, v])
    rgb = hsv2rgb(hsv)
    return rgb


def create_gif(data, scale='log', colormap=None, fps=1, output_filename='output.gif'):
    """
    Creates an animated GIF from a 3D numpy array with a specified colormap.

    Parameters:
        data (numpy.ndarray): A 3D numpy array where each slice along the first axis is a frame.
        colormap (str): The name of the colormap to use (e.g., 'viridis', 'plasma', 'magma').
        duration (float): Duration each frame is displayed in the GIF in seconds.
        output_filename (str): Filename for the output GIF.
    """
    # Set up the writer for output GIF
    writer = imageio.get_writer(output_filename, mode='I', format='GIF', loop=0, fps=fps)

    data = np.log10(data+1) if scale == 'log' else data

    # Normalize the data to fit within the colormap's range
    data = (data - data.min()) / (data.max() - data.min())

    if colormap is None:
        cmap = setCustomColorMap()
    else:
        # Get the colormap
        cmap = plt.get_cmap(colormap)

    # Apply the colormap to each frame and write to the GIF
    for frame in data:
        # Apply colormap
        colored_frame = cmap(frame)  # This returns RGBA values
        # Convert RGBA to 8-bit RGB suitable for imageio
        colored_image = (255 * colored_frame).astype(np.uint8)
        # Write frame
        writer.append_data(colored_image[:, :, :3],)  # Exclude alpha channel

    # Close the writer to finish the GIF
    writer.close()
    print(f"GIF created successfully: {output_filename}")



def apply_custom_colormap(image, colormap):
    """
    Apply a colormap to the image.
    - If `colormap` is a callable function (e.g., from `matplotlib`), apply it.
    - If `colormap` is a string, fall back to OpenCV's built-in colormaps.
    """
    if True:  # Custom colormap from matplotlib or user-defined
        colorized = colormap(image)
        # image = np.uint8(255 * image)  # Normalize to 0-255
        colorized = np.uint8(colorized[:, :, :3] * 255)  # Convert to uint8 (RGB)
    else:  # Use OpenCV colormap
        colormap_dict = {
            'jet': cv2.COLORMAP_JET,
            'viridis': cv2.COLORMAP_VIRIDIS,
            'hot': cv2.COLORMAP_HOT,
            'gray': cv2.COLORMAP_BONE,
            'default': cv2.COLORMAP_JET
        }
        color_map = colormap_dict.get(colormap, cv2.COLORMAP_JET)
        image = np.uint8(255 * image)  # Normalize to 0-255
        colorized = cv2.applyColorMap(image, color_map)

    return colorized


def create_video(data, scale='log', colormap=None, fps=1, output_filename='video.mp4'):
    """
    Create a video from a sequence of frames.

    Parameters:
    - data: List or NumPy array of 2D images.
    - scale: 'log' (default) or 'linear' for normalization.
    - colormap: A string for OpenCV colormaps or a callable function for custom colormaps.
    - fps: Frames per second.
    - output_filename: Output file name.
    """
    output_path = output_filename
    height, width = data[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if colormap is None:
        cmap = setCustomColorMap()
    else:
        # Get the colormap
        cmap = plt.get_cmap(colormap)

    frames = []
    if np.iscomplexobj(data):
        pass
    else:
        # data = normalize_data(data, scale)
        data = np.log10(data + 1) if scale == 'log' else data
        # Normalize the data to fit within the colormap's range
        data = (data - data.min()) / (data.max() - data.min())

    for i in range(len(data)):
        frame = data[i]

        if np.iscomplexobj(frame):
            frame = complex2rgb(frame)  # Your function for handling complex data
        else:
            # Apply colormap
            colored_frame = cmap(frame)  # This returns RGBA values
            # colored_frame = cv2.applyColorMap(frame, colormap)
            colored_image = (255 * colored_frame).astype(np.uint8)
            rgb_frame = colored_image[:, :, :3]
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        frames.append(frame)

    # Write frames
    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    print(f"VIDEO created successfully: {output_filename}")
