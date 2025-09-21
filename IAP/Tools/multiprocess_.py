import numpy as np
try:
    import cupy as cp
except:
    print('no CUPY available')

from scipy.interpolate import griddata
import os
from multiprocessing import Pool

NUM_CPU_CORES = os.cpu_count()
print("Number of CPU cores:", NUM_CPU_CORES)

# Globals inside workers
_G = {}

def _init_worker(U_src, Xs, Ys, wavelength, dx2, k, second_term):
    _G['U'] = U_src
    _G['Xs'] = Xs
    _G['Ys'] = Ys
    _G['dx2'] = dx2
    _G['k'] = k
    _G['second_term'] = second_term

def _worker_observe(pt):
    x, y, z = pt
    U = _G['U']; Xs = _G['Xs']; Ys = _G['Ys']
    k = _G['k']; second_term = _G['second_term']; dx2 = _G['dx2']
    r = np.sqrt((x - Xs)**2 + (y - Ys)**2 + z**2)
    phase_term = np.exp(1j * k * r) * (z / r**2)
    return np.sum(U * phase_term * second_term * dx2)

def parallel_RS_diffraction_cpu(U_source, Xs, Ys, Xq, Yq, Zq, wavelength, dx,
                                     threshold_dB=None,
                                     num_processes=int(NUM_CPU_CORES / 4), chunksize=1024):
    # 1) mask once
    mag = np.abs(U_source); m = mag > 0
    if threshold_dB is not None and mag.max() > 0:
        m = (20*np.log10(m / mag.max()) > threshold_dB)
    U = U_source[m].ravel(); Xs_m = Xs[m].ravel(); Ys_m = Ys[m].ravel()
    if U.size == 0:
        return np.zeros(Xq.shape[-2:], dtype=complex)

    # 2) precompute constants & build observation list (batched by chunksize via map)
    dx2 = dx*dx
    k = 2*np.pi / wavelength
    second_term = (1.0/(2*np.pi)) - (1j/wavelength)
    obs = list(zip(Xq.ravel(), Yq.ravel(), Zq.ravel()))

    # 3) start pool with initializer (arrays sent ONCE per worker)
    with Pool(processes=num_processes,
              initializer=_init_worker,
              initargs=(U, Xs_m, Ys_m, wavelength, dx2, k, second_term)) as pool:
        # imap with chunksize batches many points per task
        out = pool.imap(_worker_observe, obs, chunksize=chunksize)
        U_flat = np.fromiter(out, dtype=np.complex128, count=len(obs))

    return U_flat.reshape(Xq.shape[-2], Xq.shape[-1])


# Helper to build a mask (amplitude-referenced dB)
def _mask_by_threshold_db(U_source, threshold_dB):
    """
    Returns a boolean mask keeping points with 20*log10(|U|/max|U|) > threshold_dB.
    If threshold_dB is None, returns a mask of all True.
    """
    if threshold_dB is None:
        return np.ones(U_source.shape, dtype=bool)
    mag = np.abs(U_source)
    max_mag = np.max(mag)
    # Guard against completely zero source
    if max_mag == 0:
        return np.zeros_like(mag, dtype=bool)
    rel_db = 20.0 * np.log10(mag / max_mag)
    return rel_db > threshold_dB

def RS_diffraction_integral_cpu(args):
    """
    Compute the Rayleigh-Sommerfeld diffraction integral for one (x,y,z) point.
    Expected pre-masked U_source, Xs, Ys (1D or flattened arrays).
    """
    U_source, Xs, Ys, x, y, z, wavelength, dx = args
    k = 2 * np.pi / wavelength
    r = np.sqrt((x - Xs)**2 + (y - Ys)**2 + z**2)
    # Avoid division warnings for r==0 (shouldn't happen for distinct planes, but safe):
    # add a tiny epsilon if desired; here assume z>0 so r>0.
    phase_term = np.exp(1j * k * r) * (z / r**2)
    second_term = (1.0 / (2 * np.pi)) - (1j / wavelength)
    U_observation = np.sum(U_source * phase_term * second_term * (dx**2))
    return U_observation

def parallel_RS_diffraction_cpu_1(
    U_source, Xs, Ys, Xq, Yq, Zq, wavelength, dx,
    threshold_dB=None,
    num_processes=int(NUM_CPU_CORES / 4)
):
    """
    threshold_dB: keep only source points with 20*log10(|U|/max|U|) > threshold_dB.
                  Use None to disable masking. Typical values: -40, -60, etc.
    """

    # 1) Build and apply the mask once, before spawning workers
    mask = _mask_by_threshold_db(U_source, threshold_dB)
    # Flatten to minimize pickle/copy size
    U_src_m = U_source[mask].ravel()
    Xs_m    = Xs[mask].ravel()
    Ys_m    = Ys[mask].ravel()

    # Optional: early exit if nothing passes the threshold
    if U_src_m.size == 0:
        return np.zeros(Xq.shape[-2:], dtype=complex)

    # 2) Prepare observation coordinates
    observation_coordinates = np.stack(
        (Xq.flatten(), Yq.flatten(), Zq.flatten()), axis=-1
    )

    # 3) Build args for each observation point
    base = (U_src_m, Xs_m, Ys_m, wavelength, dx)
    args_list = [(base[0], base[1], base[2], x, y, z, base[3], base[4])
                 for x, y, z in observation_coordinates]

    # 4) Parallel map
    with Pool(processes=num_processes) as pool:
        U_observation_flat = pool.map(RS_diffraction_integral_cpu, args_list)

    # 5) Reshape to the field grid shape
    U_observation = np.array(U_observation_flat, dtype=complex).reshape(
        Xq.shape[-2], Xq.shape[-1]
    )
    return U_observation


def RS_diffraction_integral_cpu_bk(args):
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

def parallel_RS_diffraction_cpu_bk(
        U_source, Xs, Ys, Xq, Yq, Zq, wavelength, dx,
        num_processes=int(NUM_CPU_CORES / 4)):
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
    intensity_interp = griddata(points, frame.flatten(), (XXu, YYu), method='linear', fill_value=0, rescale=False)
    return intensity_interp


def binning_parallel(arr, binFactor):
    shape = (arr.shape[0] // binFactor, binFactor,
             arr.shape[1] // binFactor, binFactor)
    return arr.reshape(shape).sum(-1).sum(1)

def propagate_(field, dx, dz, wavelength, method = 'aspw'):
    dy = dx
    u = field

    k = 2 * np.pi / wavelength
    N, M = u.shape[-2], u.shape[-1]  # Shape of spatial grid

    # linspacex = np.linspace(-N / 2, N / 2, N, endpoint=False).reshape(1, N)
    # Fx = linspacex / (N*dx)
    # Fy = linspacex.reshape(N, 1) / (N*dy)
    fx = np.fft.fftshift(np.fft.fftfreq(M, dx))
    fy = np.fft.fftshift(np.fft.fftfreq(N, dy))
    Fx, Fy = np.meshgrid(fx, fy, indexing='xy')
    # Fx = np.fft.fftshift(np.fft.fftfreq(M, dx)).reshape(1, M)
    # Fy = np.fft.fftshift(np.fft.fftfreq(N, dy)).reshape(N, 1)

    # Correct elliptical bandlimit radius
    f_max_x = 1 / (2 * dx)
    f_max_y = 1 / (2 * dy)
    W = ((Fx / f_max_x) ** 2 + (Fy / f_max_y) ** 2) <= 1  # Elliptical support mask
    # f_max = 1 / (wavelength * np.sqrt(1 + (2 * dz / max(N*dx, M*dx)) ** 2))
    # W = circ(Fx, Fy, 2 * f_max)
    # w accounts for circular symmetry of transfer function and imposes bandlimit to avoid sampling issues
    w = 1 / wavelength ** 2 - Fx ** 2 - Fy ** 2
    w[w >= 0] = np.sqrt(w[w >= 0])
    w[w < 0] = 0
    # w = np.sqrt(w, dtype=complex)
    H = np.exp(1.j * 2 * np.pi * dz * w) * W

    U = fft2c(u)

    u1 = ifft2c(U * H)

    return np.abs(u1).astype(np.float32)

if __name__ == "__main__":
    pass
