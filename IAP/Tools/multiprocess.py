import numpy as np
from scipy.interpolate import griddata
import os
from propagators import propagate

NUM_CPU_CORES = os.cpu_count()
print("Number of CPU cores:", NUM_CPU_CORES)

def RS_diffraction_integral(args):
    """
    Compute the Rayleigh-Sommerfeld diffraction integral using a vectorized implementation.

    Parameters:
        U_source (array): Complex amplitude of the wave at the source plane.
        Xs (array): X-coordinates on the source plane.
        Ys (array): Y-coordinates on the source plane.
        x (float): X-coordinate on the observation plane.
        y (float): Y-coordinate on the observation plane.
        z (float): Propagation distance.
        wavelength (float): Wavelength of the wave.

    Returns:
        complex: Complex amplitude of the wave at the observation plane.
    """
    U_source, Xs, Ys, x, y, z, wavelength, dx = args
    k = 2 * np.pi / wavelength
    r = np.sqrt((x - Xs)**2 + (y - Ys)**2 + z**2)
    phase_term = np.exp(1j * k * r) * z / r**2
    second_term = 1/(2 * np.pi) - 1j/wavelength
    U_observation = np.sum(U_source * phase_term * second_term * dx**2)
    return U_observation


def parallel_RS_diffraction(U_source, Xs, Ys, Xq, Yq, Zq, wavelength, dx, num_processes=int(NUM_CPU_CORES / 5)):
    observation_coordinates = np.stack((Xq.flatten(), Yq.flatten(), Zq.flatten()), axis=-1)
    # observation_coordinates_flat = np.stack([coord.flatten() for coord in observation_coordinates], axis=-1)

    args_list = [(U_source, Xs, Ys, x, y, z, wavelength, dx) for x, y, z in observation_coordinates]

    with Pool(processes=num_processes) as pool:
        U_observation_chunks = pool.map(RS_diffraction_integral, args_list)

    U_observation = np.array(U_observation_chunks).reshape(Xq.shape[-2], Xq.shape[-1])

    return U_observation

def interpolate_tilted_plane(points, frame, XX, YY, XXu, YYu):
    """
    :param points: initial grid points points = np.array([Xd.flatten(), Yd.flatten()]).T
    :param frame: 2d - intensity frame
    :param XX: destination grid usually Xq_t = T_inv(Xd, Yd, zo, theta) if interp from tilted plane, T
    :param YY: destination grid, usually same as Yd
    :return: interpolated frame
    """
    # intensity_interp = griddata(points, frame.flatten(), (XX, YY), method='linear', fill_value=0)
    # I_t = griddata(input_points, I_t.flatten(), (Q_x, Q_y), method='cubic', fill_value=0)
    # intensity_interp = np.nan_to_num(intensity_interp)
    # intensity_interp = np.clip(intensity_interp, 0, None)

    # input_points = np.array([XX.flatten(), YY.flatten()]).T
    intensity_interp = griddata(points, frame.flatten(), (XXu, YYu), method='linear', fill_value=0, rescale=False)
    # intensity_interp = np.nan_to_num(intensity_interp)
    # intensity_interp[intensity_interp < 0] = 0
    return intensity_interp


def binning_parallel(arr, binFactor):
    shape = (arr.shape[0] // binFactor, binFactor,
             arr.shape[1] // binFactor, binFactor)
    return arr.reshape(shape).sum(-1).sum(1)

def propagate_(field, dx, dz, wavelength, method = 'aspw'):
    u1 = propagate(field, dx=dx, dz=dz, method=method, wavelength=wavelength)
    return np.abs(u1).astype(np.float32)