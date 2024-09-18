import numpy as np
try:
    import cupy as cp
except:
    print('no CUPY available')

from scipy.interpolate import griddata
import os
from .propagators import propagate
from multiprocessing import Pool

NUM_CPU_CORES = os.cpu_count()
print("Number of CPU cores:", NUM_CPU_CORES)

def RS_diffraction_integral_cpu(args):
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


def parallel_RS_diffraction_cpu(U_source, Xs, Ys, Xq, Yq, Zq, wavelength, dx, num_processes=int(NUM_CPU_CORES / 5)):
    observation_coordinates = np.stack((Xq.flatten(), Yq.flatten(), Zq.flatten()), axis=-1)
    # observation_coordinates_flat = np.stack([coord.flatten() for coord in observation_coordinates], axis=-1)

    args_list = [(U_source, Xs, Ys, x, y, z, wavelength, dx) for x, y, z in observation_coordinates]

    with Pool(processes=num_processes) as pool:
        U_observation_chunks = pool.map(RS_diffraction_integral_cpu, args_list)

    U_observation = np.array(U_observation_chunks).reshape(Xq.shape[-2], Xq.shape[-1])

    return U_observation


# Cupy accelerated RS integral
def RS_diffraction_integral_gpu(U_source, Xs, Ys, Xq, Yq, Zq, wavelength, dx):
    """
    Compute the Rayleigh-Sommerfeld diffraction integral using a vectorized implementation on the GPU.

    Parameters:
        U_source (array): Complex amplitude of the wave at the source plane.
        Xs (array): X-coordinates on the source plane.
        Ys (array): Y-coordinates on the source plane.
        Xq (array): X-coordinates on the observation plane (2D array).
        Yq (array): Y-coordinates on the observation plane (2D array).
        Zq (array): Propagation distances (scalar or 2D array).
        wavelength (float): Wavelength of the wave.
        dx (float): Sampling interval on the source plane.

    Returns:
        array: Complex amplitude of the wave at the observation plane.
    """
    k = 2 * cp.pi / wavelength

    # Broadcast Xq, Yq, Zq to match the shape of Xs and Ys
    r = cp.sqrt((Xq[..., cp.newaxis, cp.newaxis] - Xs) ** 2 +
                (Yq[..., cp.newaxis, cp.newaxis] - Ys) ** 2 +
                Zq[..., cp.newaxis, cp.newaxis] ** 2)

    # Compute the phase term
    phase_term = cp.exp(1j * k * r) * Zq[..., cp.newaxis, cp.newaxis] / r ** 2

    # Additional constant term
    second_term = 1 / (2 * cp.pi) - 1j / wavelength

    # Compute the Rayleigh-Sommerfeld integral
    U_observation = cp.sum(U_source * phase_term * second_term * dx ** 2, axis=(-1, -2))

    return U_observation

def parallel_RS_diffraction_gpu(U_source, Xs, Ys, Xq, Yq, Zq, wavelength, dx):
    """
    Parallelized Rayleigh-Sommerfeld diffraction integral computation using GPU.

    Parameters:
        U_source (array): Complex amplitude of the wave at the source plane.
        Xs (array): X-coordinates on the source plane.
        Ys (array): Y-coordinates on the source plane.
        Xq (array): X-coordinates on the observation plane (2D array).
        Yq (array): Y-coordinates on the observation plane (2D array).
        Zq (array): Propagation distances (2D array).
        wavelength (float): Wavelength of the wave.
        dx (float): Sampling interval on the source plane.

    Returns:
        array: Complex amplitude of the wave at the observation plane.
    """
    # Move data to GPU
    U_source_gpu = cp.asarray(U_source)
    Xs_gpu = cp.asarray(Xs)
    Ys_gpu = cp.asarray(Ys)
    Xq_gpu = cp.asarray(Xq)
    Yq_gpu = cp.asarray(Yq)
    Zq_gpu = cp.asarray(Zq)

    # Compute the Rayleigh-Sommerfeld diffraction integral on the GPU
    U_observation_gpu = RS_diffraction_integral_gpu(U_source_gpu, Xs_gpu, Ys_gpu, Xq_gpu, Yq_gpu, Zq_gpu, wavelength,
                                                    dx)

    # Move result back to CPU
    U_observation = cp.asnumpy(U_observation_gpu)

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