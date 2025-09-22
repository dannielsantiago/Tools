"""
Created on Thu Apr 23 22:20:38 2020
@author: r2d2
"""
import numpy as np
import cupy as cp
from scipy.sparse.linalg import svds
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates
from scipy import linalg
from math import pi
from dataclasses import dataclass
from .multiprocess_ import parallel_RS_diffraction_gpu, parallel_RS_diffraction_cpu
import gc

@dataclass
class PropagationParams:
    wavelength: float = None  # in meters
    dz: float = None  # propagation distance in meters
    dx: float = None  # pixel size (in meters) in source coordinate grid
    dy: float = None  # pixel size (in meters) in source coordinate grid
    dq: float = None # pixel size in destination grid
    bandlimit: bool = True # used by default in fourier-based methods such as ASPW
    thetax: float = 0  # x-tilt-angle in degrees for shift-sas propagation
    thetay: float = 0 # y-tilt-angle in degrees for shift-sas propagation
    X0: float = 0  # shift coordinate in meters for destination grind in shiftASPW
    Y0: float = 0  # shift coordinate in meters for destination grind in shiftASPW
    pad_factor_x: int = 2 #default padding factor for sas propagator
    pad_factor_y: int = 2 #default padding factor for sas propagator
    s_angle: float = 0  # incidence angle for reflection propagation using RS_integral, ReflectionASPW, eulerdecomposition
    d_angle: float = 0  # incidence angle for reflection propagation using RS_integral, ReflectionASPW, eulerdecomposition
    backward_step_ok: bool = False  #whether or not to use the negative mid-plane in fresnel two step propagator
    Nq: int = None  # number of pixels in destination grid for RS_integral
    threshold_dB: float = None # threshold in dB to filter out source points for RS_integral cpu implementation
    theta_rot_deg: float = 0 # rotation angle in degrees for in_plane_rotation method
    recenter_carrier: bool = True # re-centering of carrier frequency after rotation

def zero_pad(arr, pad_factor_x=2, pad_factor_y=2):
    '''
    Pad arr with zeros to enlarge the size by pad_factor_x and pad_factor_y times.
    First dim is assumed to be batch dim which won't be changed.

    Args:
    arr (numpy.ndarray): Input array to be padded.
    pad_factor_x (int): The factor by which to enlarge the array in the x-direction.
    pad_factor_y (int): The factor by which to enlarge the array in the y-direction.

    Returns:
    numpy.ndarray: The padded array.
    '''
    # Calculate the new size of the array
    new_shape = (arr.shape[-2] * pad_factor_y, arr.shape[-1] * pad_factor_x)

    # Create the output array with zeros
    out_arr = np.zeros(new_shape, dtype=arr.dtype)

    # Calculate the starting indices for the original array in the new array
    as1 = (new_shape[-2] - arr.shape[-2]) // 2
    as2 = (new_shape[-1] - arr.shape[-1]) // 2

    # Place the original array in the center of the output array
    out_arr[as1:as1 + arr.shape[-2], as2:as2 + arr.shape[-1]] = arr
    return out_arr

def zero_unpad(arr, original_shape):
    '''
    Strip off padding of arr with zeros to halve the size. First dim is assumed to be batch dim which
    won't be changed.
    '''
    # as1 = (original_shape[-2] + 1) // 2
    # as2 = (original_shape[-1] + 1) // 2
    as1 = (arr.shape[-1] - original_shape[-1]) // 2
    as2 = (arr.shape[-2] - original_shape[-2]) // 2
    return arr[as1:as1 + original_shape[-2], as2:as2 + original_shape[-1]]



def circ(x, y, D):
    """
    generate a circle on a 2D grid
    :param x: 2D x coordinate, normally calculated from meshgrid: x,y = np.meshgird((,))
    :param y: 2D y coordinate, normally calculated from meshgrid: x,y = np.meshgird((,))
    :param D: diameter
    :return: a 2D array
    """
    circle = (x ** 2 + y ** 2) < (D / 2) ** 2
    return circle

def rect(arr, threshold = 0.5):
    """
    generate a binary array containing a rectangle on a 2D grid
    :param x: 2D x coordinate, normally calculated from meshgrid: x,y = np.meshgird((,))
    :param threshold: threshold value to binarilize the input array, default value 0.5
    :return: a binary array
    """
    arr = abs(arr)
    return arr<threshold

def ifft2c(array):
    """
    performs 2 - dimensional inverse Fourier transformation, where energy is reserved abs(G)**2==abs(fft2c(g))**2
    if G is two - dimensional, fft2c(G) yields the 2D iDFT of G
    if G is multi - dimensional, fft2c(G) yields the 2D iDFT of G along the last two axes
    :param array:
    :return:
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(array), norm='ortho'))


def fft2c(array):
    """
    performs 2 - dimensional unitary Fourier transformation, where energy is reserved abs(g)**2==abs(fft2c(g))**2
    if g is two - dimensional, fft2c(g) yields the 2D DFT of g
    if g is multi - dimensional, fft2c(g) yields the 2D DFT of g along the last two axes
    :param array:
    :return:
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(array), norm='ortho'))


def orthogonalizeModes(probe, numModes=None):
    '''
    [probe, normalizedEigenvalues] = orthogonalizeModes(probe, numModes)
    imposes orthogonality through singular value decomposition

    Data compression strategies for ptychographic diffraction imaging
    Advanced Optical Technologies - Nov 2017
    '''
    p = np.reshape(probe, (probe.shape[-3], np.product((probe.shape[-2], probe.shape[-1]))))

    if numModes is None:
        numModes = min(probe.shape)
        u, s, vt = linalg.svd(p.T, full_matrices=False)  # computes all modes, s is sorted
    else:
        u, s, vt = svds(p.T, numModes)  # computes the specified modes but the s may be not sorted

    probe = np.reshape((u * s).T, (len(s), probe.shape[-2], probe.shape[-1]))

    normalizedEigenvalues = np.divide(s ** 2, np.sum(s ** 2))

    return probe, normalizedEigenvalues


def aspwMultiMode(u, z, lambda_illu, L):
    '''
    ASPW wave propagation
    u: field distribution at z = 0 (u is assumed to be square, i.e. N x N)
    z: propagation distance
    lambda: wavelength
    FOVp: total size [m] of input field

    returns propagated field
    following: Matsushima et al., "Band-Limited Angular Spectrum Method for
    Numerical Simulation of Free-Space
    Propagation in Far and Near Fields", Optics Express, 2009
    '''

    k = 2 * np.pi / lambda_illu
    N = u.shape[-1]
    fx = np.arange(-N // 2, N // 2) / L
    [Fy, Fx] = np.meshgrid(fx, fx)
    f_max = L / (lambda_illu * np.sqrt(L ** 2 + 4 * z ** 2))
    W = np.logical_and((abs(Fx / f_max) < 1), (abs(Fy / f_max) < 1))
    H = np.exp(1j * k * z * np.sqrt(1 - (Fx * lambda_illu) ** 2 - (Fy * lambda_illu) ** 2))
    U = nip.ft2d(u) * H * W
    uNew = nip.ift2d(U)
    return uNew


def fresnelPropagator(N, dx, w, dz):
    k = 2 * np.pi / w
    # source coordinates, this assumes that the field is NxN pixels
    L = N * dx
    x = np.arange(-N / 2, N / 2) * dx
    [Y, X] = np.meshgrid(x, x)

    # target coordinates
    dq = w * dz / L
    q = np.arange(-N / 2, N / 2) * dq
    [Qy, Qx] = np.meshgrid(q, q)
    Qin = np.exp(1j * k / (2 * dz) * (np.square(X) + np.square(Y)))
    # u_new = fft2c(Qin * u)
    return Qin


def angularPropagator(N, dx, w, dz):
    k = 2 * np.pi / w
    # source coordinates, this assumes that the field is NxN pixels
    L = N * dx
    x = np.arange(-N / 2, N / 2) * dx
    [Y, X] = np.meshgrid(x, x)

    X = np.arange(-N / 2, N / 2) / L
    Fx, Fy = np.meshgrid(X, X)
    f_max = L / (w * np.sqrt(L ** 2 + 4 * dz ** 2))
    # note: see the paper above if you are not sure what this bandlimit has to do here
    # W = rect(Fx/(2*f_max)) .* rect(Fy/(2*f_max));
    W = circ(Fx, Fy, 2 * f_max)
    # note: accounts for circular symmetry of transfer function and imposes bandlimit to avoid sampling issues
    H = np.exp(1.j * k * dz * np.sqrt(1 - (Fx * w) ** 2 - (Fy * w) ** 2))
    # u_new = ifft2c(fft2c(u) * H * W)
    return H * W


def fastPropagate(u, method='angular', Qin=None, HW=None, GPU=False):
    if not GPU:
        if method == 'angular':
            u_new = ifft2c(fft2c(u) * HW)
        elif method == 'fresnel':
            u_new = fft2c(Qin * u)
    else:
        if method == 'angular':
            u_new = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u), norm='ortho')) * HW
            u_new = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(u_new), norm='ortho'))
        elif method == 'fresnel':
            u_new = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Qin * u), norm='ortho'))
    return u_new

def binning(arr, binFactor, method='sum'):
    shape = (arr.shape[0] // binFactor, binFactor,
             arr.shape[1] // binFactor, binFactor)
    if method == 'sum':
        return arr.reshape(shape).sum(-1).sum(1)
    elif method == 'mean':
        return arr.reshape(shape).mean(-1).mean(1)

def _RS_diffraction_integral_gpu(U_source, Xs, Ys, Xq, Yq, Zq, wavelength, dx):
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

def parallel_RS_diffraction_gpux(U_source, Xs, Ys, Xq, Yq, Zq, wavelength, dx):
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
    U_observation_gpu = _RS_diffraction_integral_gpu(U_source_gpu, Xs_gpu, Ys_gpu, Xq_gpu, Yq_gpu, Zq_gpu, wavelength,
                                                    dx)

    # Move result back to CPU
    U_observation = cp.asnumpy(U_observation_gpu)

    return U_observation





def propagate(u, method='fourier', **kwargs):
    '''
    propagates a guiven field using diferent methods

    u:          input field to propagate
    method:     'fourier','fresnel','angular'(angular spectrum)
    **kargs: see dataclass propagationParams
    returns propagated wavefront u'
    '''
    params = PropagationParams(**kwargs)

    if method == 'fourier':
        u_new = fft2c(u) if params.dz > 0 else ifft2c(u)

    elif method == 'aspw':
        wavelength = params.wavelength
        dx = params.dx
        dy = getattr(params, 'dy', None) or dx  # Allow for anisotropic sampling
        dz = params.dz

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
        u_new = ifft2c(U * H)

    elif method == 'shift_aspw_1':
        wavelength = params.wavelength
        dx = params.dx
        dz = params.dz
        x0 = params.X0
        y0 = params.Y0

        k = 2 * np.pi / wavelength
        N = u.shape[-1]  # assumes square array
        L = dx * N
        df = 1 / L
        f = np.arange(-N / 2, N / 2) * df
        fx, fy = np.meshgrid(f, f)
        # Transfer function

        # Calculate w, fu, fv
        w = np.sqrt((1 / wavelength ** 2 - fx ** 2 - fy ** 2) + 0j)  # keep complex
        fu = (x0 - dz * fx / (w + 0j))  # no eps in the math; handle zeros with where()
        fv = (y0 - dz * fy / (w + 0j))
        c=2
        # mask safely (example)
        denx = c * np.abs(fu)
        deny = c * np.abs(fv)
        Wx = np.where(denx > 0, df <= 1 / denx, False)
        Wy = np.where(deny > 0, df <= 1 / deny, False)
        W = Wx & Wy

        root = np.sqrt((1 - (fx * wavelength) ** 2 - (fy * wavelength) ** 2) + 0j)
        exponent = 1j * k * dz * root
        a = np.real(exponent)
        b = np.imag(exponent)
        a_clipped = np.clip(a, -700.0, 700.0)
        exp_stable = np.exp(a_clipped) * (np.cos(b) + 1j * np.sin(b))

        H = exp_stable * W
        H *= np.exp(1j * 2 * np.pi * (x0 * fx + y0 * fy))
        u_new = ifft2c(fft2c(u) * H)

        # w = np.sqrt((1 / wavelength ** 2 - fx ** 2 - fy ** 2) + 0j)  # keep complex
        # #
        # # # w = np.real(np.sqrt((1 / wavelength ** 2 - fx ** 2 - fy ** 2).astype(complex)))
        # # # fu = dz * (x0 - fx / (w + np.finfo(float).eps)) + np.finfo(float).eps  # Adding epsilon to avoid division by zero
        # # # fv = dz * (y0 - fy / (w + np.finfo(float).eps)) + np.finfo(float).eps
        # fu = (x0 - dz * fx / (w + np.finfo(float).eps)) + np.finfo(float).eps  # Adding epsilon to avoid division by zero
        # fv = (y0 - dz * fy / (w + np.finfo(float).eps)) + np.finfo(float).eps
        # #
        # c = 2
        # W = (df <= 1 / (c * np.abs(fu))) & (df <= 1 / (c * np.abs(fv)))
        # # W = (np.abs(fx) <= 1 / (c * np.abs(fu))) & (np.abs(fy) <= 1 / (c * np.abs(fv)))
        #
        # #
        # # # Compute H
        # H = np.ones_like(fx, dtype=complex)
        # root = np.sqrt((1 - (fx * wavelength) ** 2 - (fy * wavelength) ** 2).astype(complex))
        # exponent = 1j * k * dz * root  # complex: a + i b
        # a = np.real(exponent)
        # b = np.imag(exponent)
        # # clip the real part to avoid overflow (np.exp blows up around ~709 for float64)
        # a_clipped = np.clip(a, -700.0, 700.0)
        # # # stable complex exp = exp(a)*(cos b + i sin b)
        # exp_stable = np.exp(a_clipped) * (np.cos(b) + 1j * np.sin(b))
        # H *= exp_stable * W
        # #
        # H *= np.exp(1j * 2 * np.pi * (x0 * fx + y0 * fy))
        # #
        # u_new = H#ifft2c(fft2c(u) * H )

    elif method == 'shift_aspw_2':
        #Working better than shift_aspw1
        wavelength = params.wavelength
        dx = params.dx
        dz = params.dz
        thetax = params.thetax
        thetay = params.thetay

        k = 2 * np.pi / wavelength
        N = u.shape[-1]  # assumes square array
        L = dx * N

        sx = np.sin(np.deg2rad(thetax))
        sy = np.sin(np.deg2rad(thetay))
        tx = np.tan(np.deg2rad(thetax))
        ty = np.tan(np.deg2rad(thetay))

        # Fourier coordinates
        df = 1 / L
        f = np.arange(-N / 2, N / 2) * df
        Fx, Fy = np.meshgrid(f, f)

        # Transfer function
        f_max = L / (2 * wavelength * dz)
        sqrt_chi = np.sqrt((1 / wavelength ** 2 - (Fx + sx / wavelength) ** 2 - (Fy + sy / wavelength) ** 2).astype(
            complex)) + np.finfo(float).eps
        Omegax = dz * (tx - (Fx + sx / wavelength) / sqrt_chi) + np.finfo(float).eps
        Omegay = dz * (ty - (Fy + sy / wavelength) / sqrt_chi) + np.finfo(float).eps

        c = 2
        W = (df <= 1 / np.abs(c * Omegax)) & (df <= 1 / (c * np.abs(Omegay)))

        H = np.ones_like(Fx, dtype=complex)
        root = np.sqrt((1 - (Fx * wavelength + sx) ** 2 - (Fy * wavelength + sy) ** 2).astype(complex))  # complex
        exponent = 1j * k * dz * root  # complex: a + i b
        a = np.real(exponent)
        b = np.imag(exponent)
        # clip the real part to avoid overflow (np.exp blows up around ~709 for float64)
        a_clipped = np.clip(a, -700.0, 700.0)
        # stable complex exp = exp(a)*(cos b + i sin b)
        exp_stable = np.exp(a_clipped) * (np.cos(b) + 1j * np.sin(b))
        # H *= np.exp(1j * k * dz * root) * W
        H *= exp_stable * W
        H *= np.exp(1j * 2 * np.pi * dz * (tx * Fx + ty * Fy))
        H = H.astype(complex)

        # Propagate
        u_new = ifft2c(H * fft2c(u))

    elif method == 'fresnel':
        wavelength = params.wavelength
        dx = params.dx
        dz = params.dz

        k = 2 * np.pi / wavelength
        # source coordinates, this assumes that the field is NxN pixels
        N = u.shape[-1]
        L = N * dx

        linspacex = np.linspace(-N / 2, N / 2, N, endpoint=False).reshape(1, N)
        X = linspacex * dx
        Y = X.reshape(N, 1)

        # target coordinates
        dq = wavelength * dz / L
        Qx = linspacex * dq
        Qy = Qx.reshape(N, 1)

        Q1 = np.exp(1j * k / (2 * dz) * (X ** 2 + Y ** 2))
        Q2 = np.exp(1j * k / (2 * dz) * (Qx ** 2 + Qy ** 2))

        # pre-factor
        A = 1 / (1j * wavelength * dz)

        # Fresnel-Kirchhoff integral
        u_new = A * Q2 * fft2c(u * Q1)

    elif method == 'shift_fresnel':
        wavelength = params.wavelength
        dx = params.dx
        dz = params.dz
        X0 = params.X0
        Y0 = params.Y0
        thetax = params.thetax

        k = 2 * np.pi / wavelength
        # source coordinates, this assumes that the field is NxN pixels
        N = u.shape[-1]
        L = N * dx

        linspacex = np.linspace(-N / 2, N / 2, N, endpoint=False).reshape(1, N)
        Xs = linspacex * dx
        Ys = Xs.reshape(N, 1)

        # target coordinates
        dq = wavelength * dz / L
        Xq = linspacex * dq
        Yq = Xq.reshape(N, 1)

        dz_ = dz * np.cos(np.deg2rad(thetax))
        Q1 = np.exp(1j * k / (2 * dz) * (Xs ** 2 + Ys ** 2))
        Q2 = np.exp(1j * k / (2 * dz) * ((Xq+X0) ** 2 + (Yq+Y0) ** 2))
        Q3 = np.exp(-1j * k / dz * (Xs*X0 + Ys*Y0))

        # pre-factor
        A = 1 / (1j * wavelength * dz)

        # Fresnel-Kirchhoff integral
        u_new = A * Q2 * fft2c(u * Q1 * Q3)

    elif method == 'sas':
        """
        Scalable angular spectrum propagation
        :param u: a 2D square input field
        :param z: propagation distance
        :param wavelength: propagation wavelength
        :param dx: grid spacing in original plane (u)
        :return: propagated field and two quadratic phases
        
        for details see: 
        Heintzmann, R., Loetgering, L., & Wechsler, F. (2023). 
        Scalable angular spectrum propagation. Optica, 10(11), 1407. 
        https://doi.org/10.1364/optica.497809
        MIT License. Copyright (c) 2023 Felix Wechsler (info@felixwechsler.science), Rainer Heintzmann, Lars Lötgering
        """
        wavelength = params.wavelength
        dx = params.dx
        dz = params.dz

        N = u.shape[-1]
        L = N * dx
        k = 2 * np.pi / wavelength

        z_limit = (- 4 * L * np.sqrt(8 * L ** 2 / N ** 2 + wavelength ** 2) * np.sqrt(
            L ** 2 * 1 / (8 * L ** 2 + N ** 2 * wavelength ** 2)) \
                   / (wavelength * (-1 + 2 * np.sqrt(2) * np.sqrt(L ** 2 * 1 / (8 * L ** 2 + N ** 2 * wavelength ** 2)))))
        print(f'z_limit:{z_limit}')
        # print(f'NA: {L/(2*z)}')
        # assert dz <= z_limit

        # don't change this pad_factor, only 2 is supported
        pad_factor = 8
        L_new = pad_factor * L
        N_new = pad_factor * N
        M = wavelength * dz * N / L ** 2 / 2
        u_p = zero_pad(u, pad_factor)
        # helper varaibles
        # df = 1 / L_new
        # Lf = N_new * df

        # freq space coordinates for padded array
        # f_y = np.fft.fftshift(np.fft.fftfreq(N_new, dx).reshape(1, N_new).astype(np.float32))
        f_y = np.linspace(-N_new / 2, N_new / 2, N_new, endpoint=False).reshape(1, N_new).astype(np.float32) / L_new
        f_x = f_y.reshape(N_new, 1)
        # real space coordinates for padded array
        x = np.linspace(-L_new / 2, L_new / 2, N_new, endpoint=False).reshape(1, N_new).astype(np.float32)
        y = x.reshape(N_new, 1)

        # bandlimit helper
        cx = wavelength * f_x
        cy = wavelength * f_y
        tx = L_new / 2 / dz + np.abs(wavelength * f_x)
        ty = L_new / 2 / dz + np.abs(wavelength * f_y)

        # bandlimit filter for precompensation, not smoothened!
        W = (cx ** 2 * (1 + tx ** 2) / tx ** 2 + cy ** 2 <= 1) * (cy ** 2 * (1 + ty ** 2) / ty ** 2 + cx ** 2 <= 1)

        # calculate kernels
        w_as = 1 - np.abs(f_x * wavelength) ** 2 - np.abs(f_y * wavelength) ** 2
        # w_as[w_as < 0] = 0

        H_AS = np.sqrt(w_as, dtype=complex)
        H_Fr = 1 - np.abs(f_x * wavelength) ** 2 / 2 - np.abs(f_y * wavelength) ** 2 / 2
        delta_H = W * np.exp(1j * k * dz * (H_AS - H_Fr))

        # apply precompensation
        u_precomp = ifft2c(fft2c(u_p) * delta_H)
        # u_precomp = zero_unpad(u_precomp, u.shape)
        # print(u_precomp.shape)
        dq = wavelength * dz / (u_precomp.shape[-1]*dx)
        # dq = wavelength * dz / L
        Q = dq * N * pad_factor

        # output coordinates
        q_x = np.linspace(-Q / 2, Q / 2, N_new, endpoint=False).reshape(1, N_new)
        q_y = q_x.reshape(N_new, 1)

        H_1 = np.exp(1j * k / (2 * dz) * (x ** 2 + y ** 2))

        if True: #skip final phase term, in only intensity is to be considered of u_new
            u_p_final = fft2c(H_1 * u_precomp)
        else:
            H_2 = np.exp(1j * k * dz) * np.exp(1j * k / (2 * z) * (q_x ** 2 + q_y ** 2))
            u_p_final = H_2 * fft2c(H_1 * u_precomp)

        # u_new = zero_unpad(u_p_final, u.shape)
        u_new = binning(u_p_final, pad_factor)

    elif method == 'shift_sas':
        """
            Scalable angular spectrum propagation
            :param u: a 2D square input field
            :param z: propagation distance
            :param wavelength: propagation wavelength
            :param dx: grid spacing in original plane (u)
            :return: propagated field and two quadratic phases

            for details see:
            Heintzmann, R., Loetgering, L., & Wechsler, F. (2023).
            Scalable angular spectrum propagation. Optica, 10(11), 1407.
            https://doi.org/10.1364/optica.497809
            MIT License. Copyright (c) 2023 Felix Wechsler (info@felixwechsler.science), Rainer Heintzmann, Lars Lötgering
            """
        wavelength = params.wavelength
        dx = params.dx
        dz = params.dz
        pad_factor_x = params.pad_factor_x
        pad_factor_y = params.pad_factor_y
        thetax = params.thetax
        thetay = params.thetay

        N = u.shape[-1]
        L = N * dx
        k = 2 * np.pi / wavelength

        z_limit = (- 4 * L * np.sqrt(8 * L ** 2 / N ** 2 + wavelength ** 2) * np.sqrt(
            L ** 2 * 1 / (8 * L ** 2 + N ** 2 * wavelength ** 2)) \
                   / (wavelength * (
                            -1 + 2 * np.sqrt(2) * np.sqrt(L ** 2 * 1 / (8 * L ** 2 + N ** 2 * wavelength ** 2)))))

        # assert z <= z_limit

        # don't change this pad_factor, only 2 is supported
        # pad_factor
        Lx_new = pad_factor_x * L
        Ly_new = pad_factor_y * L
        Nx_new = pad_factor_x * N
        Ny_new = pad_factor_y * N
        u_p = zero_pad(u, pad_factor_x, pad_factor_y)
        dfx = 1 / Lx_new
        dfy = 1 / Ly_new

        sx = np.sin(np.deg2rad(thetax))
        sy = np.sin(np.deg2rad(thetay))
        tx = np.tan(np.deg2rad(thetax))
        ty = np.tan(np.deg2rad(thetay))

        # z-distance needs to be adjusted for the precompensation and for the propagation
        z_ = dz * np.cos(np.deg2rad(thetax))

        # freq space coordinates for padded array
        Fx = np.fft.fftshift(np.fft.fftfreq(Nx_new, dx).reshape(1, Nx_new).astype(np.float32))
        Fy = np.fft.fftshift(np.fft.fftfreq(Ny_new, dx).reshape(Ny_new, 1).astype(np.float32))

        # real space coordinates for padded array
        x = np.linspace(-Lx_new / 2, Lx_new / 2, Nx_new, endpoint=False).reshape(1, Nx_new)
        y = np.linspace(-Ly_new / 2, Ly_new / 2, Ny_new, endpoint=False).reshape(Ny_new, 1)

        # Transfer function with bandlimits
        sqrt_chi = np.sqrt((1 / wavelength ** 2 - (Fx + sx / wavelength) ** 2 - (Fy + sy / wavelength) ** 2).astype(
            complex)) + np.finfo(float).eps
        HFresnel = np.exp(-1j * np.pi * dz / wavelength * ((wavelength * Fx) ** 2 + (wavelength * Fy) ** 2))
        Omegax = dz * (tx - (Fx + sx / wavelength) / sqrt_chi + wavelength * Fx) + np.finfo(float).eps
        Omegay = dz * (ty - (Fy + sy / wavelength) / sqrt_chi + wavelength * Fy) + np.finfo(float).eps

        c = 2
        W = (dfx <= 1 / np.abs(c * Omegax)) & (dfy <= 1 / (c * np.abs(Omegay)))

        H = np.exp(
            1j * k * z_ * np.sqrt((1 - (Fx * wavelength + sx) ** 2 - (Fy * wavelength + sy) ** 2).astype(complex))) * W
        H *= np.conj(HFresnel) * np.exp(1j * 2 * np.pi * z_ * (tx * Fx + ty * Fy))

        # apply precompensation
        u_precomp = ifft2c(fft2c(u_p) * H)

        # quadratic phase term for Fresnel Propagation
        Q1 = np.exp(1j * k / (2 * z_) * (x ** 2 + y ** 2))

        if False:  # skip final phase term, in only intensity is to be considered of u_new
            u_p_final = fft2c(Q1 * u_precomp)
        else:
            dqx = wavelength * dz / Lx_new
            dqy = wavelength * dz / Ly_new

            Qx = dqx * Nx_new  # * pad_factor
            Qy = dqy * Ny_new  # * pad_factor

            # output coordinates
            q_x = np.linspace(-Qx / 2, Qx / 2, Nx_new, endpoint=False).reshape(1, Nx_new)
            q_y = np.linspace(-Qy / 2, Qy / 2, Ny_new, endpoint=False).reshape(Ny_new, 1)

            Q2 = np.exp(1j * k * z_) * np.exp(1j * k / (2 * z_) * (q_x ** 2 + q_y ** 2))
            u_p_final = Q2 * fft2c(Q1 * u_precomp)

        # 1) Crop FOV, not desired
        # u_new = zero_unpad(u_p_final, u.shape)

        # 2) Artifacts
        # u_new = binning(u_p_final, pad_factor, 'mean')

        # 3) Artifacts
        # real_binned = binning(u_p_final.real, pad_factor, 'mean')
        # imag_binned = binning(u_p_final.imag, pad_factor, 'mean')
        # u_new1 = real_binned + 1j * imag_binned

        # 4) Smooth but slower than 5)
        # real_binned = binning(np.abs(u_p_final), pad_factor, 'mean')
        # imag_binned = binning(np.angle(u_p_final), pad_factor, 'mean')
        # u_new2 = real_binned * np.exp(1j*imag_binned)

        # 5) Fastest
        u_new_r = cv2.resize(u_p_final.real, u.shape, interpolation=cv2.INTER_LINEAR)
        u_new_i = cv2.resize(u_p_final.imag, u.shape, interpolation=cv2.INTER_LINEAR)
        u_new = u_new_r + 1j * u_new_i
        # u_new = u_p_final

    elif method == 'scaledASP':
        """
        Angular spectrum propagation with customized grid spacing dq
        :param u: a 2D square input field
        :param z: propagation distance
        :param wavelength: propagation wavelength
        :param dx: grid spacing in original plane (u)
        :param dq: grid spacing in destination plane (Uout)
        :return: propagated field and two quadratic phases

        note: to be analytically correct, add Q3 (see below)
        if only intensities matter, leave it out
        """
        wavelength = params.wavelength
        dx = params.dx
        dz = params.dz
        dq = params.dq

        # optical wavenumber
        k = 2 * np.pi / wavelength
        # assume square grid
        N = u.shape[-1]
        L = N * dx

        # source plane coordinates
        linspacex = np.linspace(-N / 2, N / 2, N, endpoint=False).reshape(1, N)
        X = linspacex * dx
        Y = X.reshape(N, 1)

        r1sq = X ** 2 + Y ** 2
        # spatial frequencies(of source plane)
        Fx = linspacex / L
        Fy = Fx.reshape(N, 1)

        fsq = Fx ** 2 + Fy ** 2
        # scaling parameter
        m = dq / dx

        # quadratic phase factors
        Q1 = np.exp(1.j * (k / 2) * ((1 - m) / dz) * r1sq)
        Q2 = np.exp(1.j * (np.pi ** 2) * (2 * (-dz) / (m * k)) * fsq)

        if params.bandlimit:
            if m != 1:
                r1sq_max = wavelength * dz / (2 * dx * (1 - m))
                Wr = np.array(circ(X, Y, 2 * r1sq_max))
                Q1 = Q1 * Wr

            fsq_max = m / (2 * dz * wavelength * (1 / (N * dx)))
            Wf = np.array(circ(Fx, Fy, 2 * fsq_max))
            Q2 = Q2 * Wf

        # note: to be analytically correct, add Q3 (see below)
        # if only intensities matter, leave it out
        X2 = linspacex * dq
        Y2 = X2.reshape(N, 1)
        r2sq = X2 ** 2 + Y2 ** 2
        Q3 = np.exp(1.j * k / 2 * (m - 1) / (m * dz) * r2sq)

        # compute the propagated field
        if dz > 0:
            # u_new = ifft2c(Q2 * fft2c(Q1 * u))
            u_new = Q3 * ifft2c(Q2 * fft2c(Q1 * u))
        else:
            # u_new = np.conj(Q1) * ifft2c(np.conj(Q2) * fft2c(u))
            u_new = np.conj(Q1) * ifft2c(np.conj(Q2) * fft2c(u * np.conj(Q3)))

    elif method == 'twoStepFresnel':
        # unpack parameters
        wavelength = params.wavelength
        dx = params.dx
        dq_out = params.dq  # user’s desired final pixel size
        dz = params.dz
        m = dq_out / dx

        # choose split: "+" ⇒ both dz_a,dz2>0; "-" ⇒ one leg back‐prop
        if getattr(params, 'backward_step_ok', False):
            dz_a = dz / (1 - m)
        else:
            dz_a = dz / (1 + m)
        dz2 = dz - dz_a

        # wavenumber
        k = 2 * np.pi / wavelength

        # grid setup
        N = u.shape[-1]
        L = N * dx
        lin = np.linspace(-N / 2, N / 2, N, endpoint=False).reshape(1, N)
        X = lin * dx
        Y = X.T

        # intermediate‐plane spacing & coords
        dx_a = wavelength * dz_a / L
        X_a = lin * dx_a
        Y_a = X_a.T

        # final‐plane coords (for Q4)
        dq1 = wavelength * dz2 / (N * dx_a)  # actual output spacing
        Qx = lin * dq1
        Qy = Qx.T

        # quadratic phase factors
        Q1 = np.exp(1j * k / (2 * dz_a) * (X ** 2 + Y ** 2))
        Q23 = np.exp(1j * k * (1 + m) / (2 * dz2) * (X_a ** 2 + Y_a ** 2))
        Q4 = np.exp(1j * k / (2 * dz2) * (Qx ** 2 + Qy ** 2))

        # band‐limit each chirp if requested
        if getattr(params, 'bandlimit', False):
            # 1) source‐plane mask on Q1
            r1 = wavelength * abs(dz_a) / (2 * dx)
            Q1 *= circ(X, Y, 2 * r1)

            # 2) intermediate‐plane mask on Q23
            # denom = |1 ∓ m| depending on split
            denom = abs(1 - m) if getattr(params, 'backward_step_ok', False) else (1 + m)
            r2 = wavelength * abs(dz2) / (2 * dx_a * denom)
            Q23 *= circ(X_a, Y_a, 2 * r2)

            # 3) final‐plane mask on Q4
            r3 = wavelength * abs(dz2) / (2 * dq1)
            Q4 *= circ(Qx, Qy, 2 * r3)

        # prefactors
        A1 = 1 / (1j * wavelength * dz_a)
        A2 = 1 / (1j * wavelength * dz2)

        # propagation
        if dz2 > 0:
            # two forward Fresnel hops
            u1 = fft2c(u * Q1)
            u2 = fft2c(A1 * Q23 * u1)
            u_new = A2 * Q4 * u2
        else:
            # second hop is a back-propagation
            u1 = fft2c(u * Q1)
            u2 = ifft2c(np.conj(Q23) * u1)
            u_new = np.conj(Q4) * ifft2c(A1 * u2)

    elif method == 'reflectionASPW':
        """
        1. First do a coordinate transformtation with sourceAngle
        2. Then a scaledASP to propagate field to the detector
        3. Then another coordinate rotation if detector is tilted detectAngle
        """
        wavelength = params.wavelength
        dx = params.dx
        dz = params.dz
        dq = params.dq
        s_angle = params.s_angle
        d_angle = params.d_angle
        bandlimit = params.bandlimit
        # 1st step
        rad = np.deg2rad(s_angle)# *-1?
        k = 2 * np.pi / wavelength
        N = u.shape[-1]
        L = dx * N
        z = 0
        
        linspace = np.linspace(-N / 2, N / 2, N, endpoint=False).reshape(1, N)
        Fx = linspace / L
        Fy = Fx.reshape(N,1)

        f_max = 1 / wavelength
        W = np.logical_and((abs(Fx / f_max) < 1), (abs(Fy / f_max) < 1))

        # propagate and create rotated coordinates for reference plane
        Ud = fft2c(u)
        # FxRot = np.cos(rad) * Fx + np.sin(rad) * omega
        # FxRot = (1 / wavelength) * (np.cos(rad) * wavelength * Fx + np.sin(rad) * omega)
        # FxRot -= np.sin(rad) / wavelength  # frequency shift for u_0? Check paper again
        jacobian = (np.cos(rad) - Fx / (1 - Fx ** 2 - Fy ** 2) * np.sin(rad))

        # interpolate Fourier spectrum on tilted plane
        # method 1 (fastest)
        # inverting the scaling seem to do the correct stretching on the beam
        # normal scaling for propagation towards detector, elliptic beam becomes circle
        fxRot = (1 / wavelength) * ((Fx * np.cos(rad)) * wavelength + np.sin(rad)) # Fx instead?
        fxRot -= np.sin(rad) / wavelength  # frequency shift for u_0? Check paper again
        URot = np.empty_like(Ud)

        fx = np.squeeze(Fx)
        if np.ndim(URot)>2:
            for i in range(URot.shape[0]):
                interp_spline_r = RectBivariateSpline(fx, fx, np.real(Ud[i, ...]))
                interp_spline_i = RectBivariateSpline(fx, fx, np.imag(Ud[i, ...]))
                URot[i, ...] = interp_spline_r(fx, fxRot) + 1j * interp_spline_i(fx, fxRot)
        else:
            interp_spline_r = RectBivariateSpline(fx, fx, np.real(Ud))
            interp_spline_i = RectBivariateSpline(fx, fx, np.imag(Ud))
            URot = interp_spline_r(fx, fxRot) + 1j * interp_spline_i(fx, fxRot)

        u = ifft2c(URot * W * jacobian)

        # 2. Step scaled ASP
        f_max = L / (wavelength * np.sqrt(L ** 2 + 4 * dz ** 2))
        # source plane coordinates
        X1 = linspace * dx  # fxRot*L*dx
        Y1 = X1.reshape(N, 1)
        r1sq = X1 ** 2 + Y1 ** 2
        # spatial frequencies(of source plane)
        FX = Fx  # Rot
        FY = Fy

        W = circ(FX, FY, 2 * f_max)
        fsq = FX ** 2 + FY ** 2
        # scaling parameter
        # dq = wavelength * dz / L
        m = dq / dx

        # quadratic phase factors
        Q1 = np.exp(1.j * (k / 2) * ((1 - m) / dz) * r1sq)
        Q2 = np.exp(1.j * (np.pi ** 2) * (2 * (-dz) / (m * k)) * fsq)  # * W
        
        if bandlimit:
            if m != 1:
                r1sq_max = wavelength * dz / (2 * dx * (1 - m))
                Wr = np.array(circ(X1, Y1, 2 * r1sq_max))
                Q1 = Q1 * Wr

            fsq_max = m / (2 * dz * wavelength * (1 / (N * dx)))
            Wf = np.array(circ(FX, FY, 2 * fsq_max))
            Q2 = Q2 * Wf

        # note: to be analytically correct, add Q3 (see below)
        # if only intensities matter, leave it out
        x2 = np.arange(-N / 2, N / 2) * dq
        X2, Y2 = np.meshgrid(x2, x2)
        r2sq = X2 ** 2 + Y2 ** 2
        Q3 = np.exp(1.j * k / 2 * (m - 1) / (m * dz) * r2sq)

        # compute the propagated field
        if dz > 0:
            u_new = ifft2c(Q2 * fft2c(Q1 * u))
            # u_new = Q3 * ifft2c(Q2 * fft2c(Q1 * u))
        else:
            # u_new = np.conj(Q1) * ifft2c(np.conj(Q2) * fft2c(u))
            u_new = np.conj(Q1) * ifft2c(np.conj(Q2) * fft2c(u * np.conj(Q3)))

        dz = 0

        # 3. Step rotation
        rad = d_angle / 180 * np.pi
        L = N * dq
        fx = np.arange(-N // 2, N // 2) / L
        Fx, Fy = np.meshgrid(fx, fx)
        omega = np.sqrt(1 - (Fx * wavelength) ** 2 - (Fy * wavelength) ** 2)
        f_max = L / (wavelength * np.sqrt(L ** 2 + 4 * dz ** 2))
        W = circ(Fx, Fy, 2 * f_max)
        H = np.exp(1j * k * dz * np.sqrt(1 - (Fx * wavelength) ** 2 - (Fy * wavelength) ** 2))

        # propagate and create rotated coordinates for reference plane
        Ud = fft2c(u_new) * H
        jacobian = (np.cos(rad) - Fx / (1 - Fx ** 2 - Fy ** 2) * np.sin(rad))
        # inverting the scaling seem to do the correct stretching on the beam
        fxRot = (1 / wavelength) * ((fx / np.cos(rad)) * wavelength + np.sin(rad))
        fxRot -= np.sin(rad) / wavelength  # frequency shift for u_0
        URot = np.empty_like(Ud)

        if np.ndim(URot)>2:
            for i in range(URot.shape[0]):
                interp_spline_r = RectBivariateSpline(fx, fx, np.real(Ud[i, ...]))
                interp_spline_i = RectBivariateSpline(fx, fx, np.imag(Ud[i, ...]))
                URot[i, ...] = interp_spline_r(fx, fxRot) + 1j * interp_spline_i(fx, fxRot)
        else:
            interp_spline_r = RectBivariateSpline(fx, fx, np.real(Ud))
            interp_spline_i = RectBivariateSpline(fx, fx, np.imag(Ud))
            URot = interp_spline_r(fx, fxRot) + 1j * interp_spline_i(fx, fxRot)

        # interpolate Fourier spectrum on tilted plane
        u_new = ifft2c(URot * W * jacobian)

    elif method == 'eulerDecomposition':
        """
        1. First do a coordinate transformtation with sourceAngle
        2. Then a scaledASP to propagate field to the detector
        3. Then another coordinate rotation if detector is tilted detectAngle
        """
        wavelength = params.wavelength
        dx = params.dx
        dz = params.dz
        dq = params.dq
        s_angle = params.s_angle
        d_angle = params.d_angle
        bandlimit = params.bandlimit

        # 1st step
        rad = -s_angle / 180 * np.pi
        k = 2 * np.pi / wavelength
        N = u.shape[-1]
        L = dx * N
        z = 0
        fx = np.arange(-N // 2, N // 2) / L
        Fx, Fy = np.meshgrid(fx, fx)
        f_max = L / (wavelength * np.sqrt(L ** 2 + 4 * z ** 2))
        W = np.logical_and((abs(Fx / f_max) < 1), (abs(Fy / f_max) < 1))

        # rotate around the x axis in the source plane
        Ud = fft2c(u)
        URot = np.empty_like(Ud)
        if np.ndim(URot)>2:
            for i in range(URot.shape[0]):
                URot[i, ...] = rotate_around_x(Fx, Fy, Ud[i, ...], wavelength, rad, Fx, Fy)[0]
        else:
            URot = rotate_around_x(Fx, Fy, Ud, wavelength, rad, Fx, Fy)[0]

        u = ifft2c(URot * W)

        # 2. Step scaled ASP
        # source plane coordinates
        x1 = np.arange(-N // 2, N // 2) * dx  # fxRot*L*dx
        y1 = np.arange(-N // 2, N // 2) * dx
        X1, Y1 = np.meshgrid(x1, y1)
        r1sq = X1 ** 2 + Y1 ** 2
        # spatial frequencies(of source plane)
        FX = Fx  # Rot
        FY = Fy

        W = circ(FX, FY, 2 * f_max)
        fsq = FX ** 2 + FY ** 2
        # scaling parameter
        # dq = wavelength * dz / L
        m = dq / dx

        # quadratic phase factors
        Q1 = np.exp(1.j * (k / 2) * ((1 - m) / dz) * r1sq)
        Q2 = np.exp(1.j * (np.pi ** 2) * (2 * (-dz) / (m * k)) * fsq)  # * W
        # Q1 = np.exp(1.j * k / 2 * (1 - m) / dz * r1sq)
        # Q2 = np.exp(-1.j * np.pi ** 2 * 2 * dz / m / k * fsq)

        if bandlimit:
            if m != 1:
                r1sq_max = wavelength * dz / (2 * dx * (1 - m))
                Wr = np.array(circ(X1, Y1, 2 * r1sq_max))
                Q1 = Q1 * Wr

            fsq_max = m / (2 * dz * wavelength * (1 / (N * dx)))
            Wf = np.array(circ(FX, FY, 2 * fsq_max))
            Q2 = Q2 * Wf

        # note: to be analytically correct, add Q3 (see below)
        # if only intensities matter, leave it out
        x2 = np.arange(-N / 2, N / 2) * dq
        X2, Y2 = np.meshgrid(x2, x2)
        r2sq = X2 ** 2 + Y2 ** 2
        Q3 = np.exp(1.j * k / 2 * (m - 1) / (m * dz) * r2sq)
        # compute the propagated field

        # # compute the propagated field
        # Uout = np.conj(Q1) * ifft2c(np.conj(Q2) * fft2c(u*np.conj(Q3)))

        if dz > 0:
            u_new = ifft2c(Q2 * fft2c(Q1 * u))
            # u_new = Q3 * ifft2c(Q2 * fft2c(Q1 * u))
        else:
            # u_new = np.conj(Q1) * ifft2c(np.conj(Q2) * fft2c(u))
            u_new = np.conj(Q1) * ifft2c(np.conj(Q2) * fft2c(u * np.conj(Q3)))

        u = u_new

        # 3. Step rotation
        dz = 0
        rad = d_angle / 180 * np.pi
        L = N * dq
        fx = np.arange(-N // 2, N // 2) / L
        Fx, Fy = np.meshgrid(fx, fx)
        f_max = L / (wavelength * np.sqrt(L ** 2 + 4 * dz ** 2))
        W = circ(Fx, Fy, 2 * f_max)

        # rotate around the x axis in the detector plane
        Ud = fft2c(u)
        URot = np.empty_like(Ud)
        if np.ndim(URot) > 2:
            for i in range(URot.shape[0]):
                URot[i, ...] = rotate_around_x(Fx, Fy, Ud[i, ...], wavelength, rad, Fx, Fy)[0]
        else:
            URot = rotate_around_x(Fx, Fy, Ud, wavelength, rad, Fx, Fy)[0]

        u_new = ifft2c(URot * W)

    elif method == 'RS_integral':
        wavelength = params.wavelength
        threshold_dB = params.threshold_dB
        # define source coordinates
        N = u.shape[-1]
        dx = params.dx
        x_prime = np.arange(-N / 2, N / 2) * dx
        X, Y = np.meshgrid(x_prime, x_prime)

        # detector coordinates
        dq = params.dq
        dz = params.dz
        Nq = getattr(params, 'Nq', None) or N
        x = np.arange(-Nq / 2, Nq / 2) * dq
        Xq, Yq = np.meshgrid(x, x)
        Zq = np.zeros_like(Xq)

        Zq = Zq + dz

        s_angle = params.s_angle #source angle. (illu angle)
        d_angle = params.d_angle #detection angle
        theta_r = np.deg2rad(s_angle - d_angle)

        # Gimbal Rotation matrices
        R_y = np.array([[np.cos(theta_r), 0, np.sin(theta_r)],
                        [0, 1, 0],
                        [-np.sin(theta_r), 0, np.cos(theta_r)]])

        def unpack(coordinates):
            Za, Ya, Xa = np.split(coordinates, 3, axis=-1)
            Za = np.squeeze(Za)
            Ya = np.squeeze(Ya)
            Xa = np.squeeze(Xa)
            return Za, Ya, Xa

        ZYXq = np.stack((Zq * 0, Yq, Xq), axis=-1)
        # rotate detector coordinates
        ZYXq_tilted = ZYXq @ R_y

        Zq_t, Yq_t, Xq_t = unpack(ZYXq_tilted)
        # adjust offsets
        X0 = np.sin(np.deg2rad(s_angle)) * dz
        Z0 = np.cos(np.deg2rad(s_angle)) * dz

        # center tilted detector
        Zq_t += Z0
        Xq_t += X0
        # adds phase ramp
        sx = np.sin(np.deg2rad(s_angle))
        tilted_probe = np.exp(1j * 2 * np.pi / wavelength * (sx * X))
        u = u * tilted_probe
        # propagate with RS integral to tilted plane
        memory_error = False
        if np.ndim(u) > 2:
            npsm = u.shape[-3]
            u_new = np.zeros(shape=(npsm, Nq, Nq), dtype=complex)
            for p in range(npsm):
                if not memory_error:
                    try:
                        u_new[..., p, :, :] = parallel_RS_diffraction_gpu(U_source=u[..., p, :, :], Xs=X, Ys=Y,
                                                                          Xq=Xq_t, Yq=Yq_t, Zq=Zq_t,
                                                                          wavelength=wavelength, dx=dx, )
                    except cp.cuda.memory.OutOfMemoryError as e:
                        memory_error = True
                        print('out of memory error occurred:', e)
                        # Free up GPU memory
                        cp.get_default_memory_pool().free_all_blocks()
                        cp.get_default_pinned_memory_pool().free_all_blocks()

                        # Optionally, you can run garbage collection
                        gc.collect()

                        print('using parallel_RS_diffraction_cpu instead')
                        u_new[..., p, :, :] = parallel_RS_diffraction_cpu(U_source=u[..., p, :, :], Xs=X, Ys=Y,
                                                                          Xq=Xq_t, Yq=Yq_t, Zq=Zq_t,
                                                                          wavelength=wavelength, dx=dx,
                                                                          threshold_dB=threshold_dB)
                else:
                    u_new[..., p, :, :] = parallel_RS_diffraction_cpu(U_source=u[..., p, :, :], Xs=X, Ys=Y,
                                                                      Xq=Xq_t, Yq=Yq_t, Zq=Zq_t,
                                                                      wavelength=wavelength, dx=dx,
                                                                      threshold_dB=threshold_dB)

                # ensures same photon_counts after propagation
                temp_photons = np.sum(np.square(np.abs(u[..., p, :, :])))
                u_new[..., p, :, :] *= np.sqrt(temp_photons) / np.sqrt(np.sum(np.square(np.abs(u_new[..., p, :, :]))))
        else:
            u_new = np.zeros(shape=u.shape, dtype=complex)
            u_new[..., :, :] = parallel_RS_diffraction_cpu(U_source=u[..., :, :], Xs=X, Ys=Y,
                                                              Xq=Xq_t, Yq=Yq_t, Zq=Zq_t,
                                                              wavelength=wavelength, dx=dx,
                                                              threshold_dB=threshold_dB)

        # ensures same photon_counts after propagation
        temp_photons = np.sum(np.square(np.abs(u[..., :, :])))
        u_new[..., :, :] *= np.sqrt(temp_photons) / np.sqrt(np.sum(np.square(np.abs(u_new[..., :, :]))))
    
    elif method == 'in_plane_rotation':
        """
            Propagate complex field u(x,y) from z=0 to a plane rotated by +theta about y,
            pivoting through the origin. Optionally shift the tilted plane by s_normal
            along its own normal (positive toward +n').

            Parameters
            ----------
            u : array, shape (C, Ny, Nx) or (Ny, Nx)
                Complex field(s) at z=0.
            theta_deg : float
                Rotation angle (degrees). +θ tilts the plane normal toward +x.
            dx, dy : float
                Sample spacings [m].
            wavelength : float
                Wavelength [m].
            s_normal : float, optional
                Displacement of the tilted plane along its own normal [m].
                s_normal=0 gives the plane that cuts the origin.
            recenter_carrier : bool, optional
                If True, remove the carrier so DC stays centered by subtracting sinθ/λ in Fx'.

            Returns
            -------
            u_out : array, same shape as u
                Field on the tilted (and optionally shifted) plane sampled on the same (x,y) grid.
            """
        dx = params.dx
        dy = getattr(params, 'dy', None) or dx  # Allow for anisotropic sampling
        wavelength = params.wavelength
        theta_deg = params.theta_rot_deg
        recenter_carrier = params.recenter_carrier
        # shape handling
        U_in = u
        if u.ndim == 2:
            U_in = u[None, ...]
        C, Ny, Nx = U_in.shape

        # frequency grids (cycles/m)
        fx = np.fft.fftshift(np.fft.fftfreq(Nx, dx))
        fy = np.fft.fftshift(np.fft.fftfreq(Ny, dy))
        FX, FY = np.meshgrid(fx, fy, indexing='xy')

        fmax = 1.0 / wavelength
        # Forward spectrum
        U = fft2c(U_in)
        # U = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U_in, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))

        # Source light-cone (propagating)
        cone_src = (FX ** 2 + FY ** 2) <= fmax ** 2
        FZ = np.zeros_like(FX)
        FZ[cone_src] = np.sqrt(fmax ** 2 - (FX[cone_src] ** 2 + FY[cone_src] ** 2))

        # Rotation parameters
        th = np.deg2rad(theta_deg)
        c, s = np.cos(th), np.sin(th)

        # ---- Build target (rotated-plane) frequency grid ----
        FXp = FX.copy()
        FYp = FY.copy()
        if recenter_carrier:
            # shift to keep DC centered on the tilted plane
            FXp = FXp + 0.0  # start from same grid
            FXp += 0.0  # clarity only
            # Note: we incorporate the '−sinθ/λ' by INVERTING the mapping below (preferred)
        # Target cone and its normal component
        cone_tgt = (FXp ** 2 + FYp ** 2) <= fmax ** 2
        FZp = np.zeros_like(FXp)
        FZp[cone_tgt] = np.sqrt(fmax ** 2 - (FXp[cone_tgt] ** 2 + FYp[cone_tgt] ** 2))

        # ---- Inverse mapping: (Fx',Fy') -> (Fx,Fy) on source plane ----
        # Rotation of wavevector:
        # Fx' = Fx*c + Fz*s   ;   Fz' = -Fx*s + Fz*c   ;   Fy' = Fy
        # => Fx = Fx'*c - Fz'*s   ;   Fy = Fy'   ;   Fz = Fx'*s + Fz'*c
        FX_src = FXp * c - FZp * s
        FY_src = FYp

        # Optional “recenter carrier”: subtract sinθ/λ inside the *target* Fx'
        # which is equivalent to heterodyning exp(+i 2π (sinθ/λ) x) in real space.
        if recenter_carrier:
            FX_src += (s / wavelength)  # because we used inverse mapping

        # Compute Fz on the *source* points for the Jacobian
        cone_src_eval = (FX_src ** 2 + FY_src ** 2) <= fmax ** 2
        FZ_src = np.zeros_like(FX_src)
        FZ_src[cone_src_eval] = np.sqrt(fmax ** 2 - (FX_src[cone_src_eval] ** 2 + FY_src[cone_src_eval] ** 2))

        # Jacobian (dΩ conservation) evaluated at source (Fx,Fy)
        with np.errstate(divide='ignore', invalid='ignore'):
            J = c - (FX_src / FZ_src) * s
            J = np.nan_to_num(J)

        # Build index arrays for map_coordinates (y,x order)
        # Map physical freq -> index in [0..N-1]
        def to_index(arr, fmin, fmax, N):
            return (arr - fmin) / (fmax - fmin) * (N - 1)

        fx_min, fx_max = fx.min(), fx.max()
        fy_min, fy_max = fy.min(), fy.max()
        x_idx = to_index(FX_src, fx_min, fx_max, Nx)
        y_idx = to_index(FY_src, fy_min, fy_max, Ny)

        # Interpolate spectrum on rotated grid
        UR = np.zeros_like(U, dtype=np.complex128)
        for cidx in range(C):
            real_vals = map_coordinates(np.real(U[cidx]), [y_idx, x_idx], order=3, mode='constant', cval=0.0)
            imag_vals = map_coordinates(np.imag(U[cidx]), [y_idx, x_idx], order=3, mode='constant', cval=0.0)
            UR[cidx] = (real_vals + 1j * imag_vals) * J

        # Apply target cone (evanescent rejection) on the rotated plane
        UR *= cone_tgt

        # Inverse FFT
        # u_new = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(UR, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
        u_new = ifft2c(UR)
        if u.ndim == 2:
            u_new = u_new[0]
            
    return u_new


def rotate_around_y_bk(Xin, Yin, Ein, waveLen, phi, Xout, Yout, linx_in=0, liny_in=0):
    # Copy field
    Ex = Ein

    # original sampling points and field
    old_ny, old_nx = Ex.shape

    # zero padding
    Ex = np.pad(Ex, ((int(old_ny / 2), int(old_nx / 2)), (int(old_ny / 2), int(old_nx / 2))), 'constant',
                constant_values=(0, 0))

    # new sampling points
    ny, nx = Ex.shape

    # extended spatial coordinates and spatial frequencies
    dx1 = Xin[0, 1] - Xin[0, 0]
    dy1 = Yin[1, 0] - Yin[0, 0]
    x1 = np.fft.fftshift(np.fft.fftfreq(nx, 1)) * nx * dx1
    y1 = np.fft.fftshift(np.fft.fftfreq(ny, 1)) * ny * dy1
    X, Y = np.meshgrid(x1, y1)
    Z = 0 * X

    # extended spatial frequencies
    dx1 = Xin[0, 1] - Xin[0, 0]
    dy1 = Yin[1, 0] - Yin[0, 0]
    sx1 = np.fft.fftshift(np.fft.fftfreq(nx, dx1))
    sy1 = np.fft.fftshift(np.fft.fftfreq(ny, dy1))
    Sx, Sy = np.meshgrid(sx1, sy1)

    # calculation of the spectrum
    Gxm = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ex)))
    Kx = Sx * 2 * pi
    Ky = Sy * 2 * pi

    # coupled coordinates z' = z(y)
    z2y = Yout[:, 1] * np.tan(phi) * np.cos(phi)

    # scaled coordinates in Y'
    Ys = Yout * np.cos(phi)

    # output sampling points
    my, mx = np.shape(Xout)

    # analytical phase
    Sx = Sx + linx_in
    Sy = Sy + liny_in

    # parameters for the chirp-z
    dx2 = np.abs(Xout[0, 1] - Xout[0, 0])
    rxb = (dx1 / dx2) * (nx / mx)
    if rxb < 1:
        raise Exception('Zoom out is not possible. Please change the output sampling grid.')

    sx = np.fft.fftshift(np.fft.fftfreq(nx, dx1))
    dkxb = (sx[1] - sx[0]) * 2 * pi
    dxb = 2 * pi / (dkxb * nx)
    kxminb = -np.ceil((mx - 1) / 2) * dxb / rxb
    kxmaxb = np.floor((mx - 1) / 2) * dxb / rxb
    kxMaxb = mx * dxb
    Ax = np.exp(-1j * pi * (2 * kxminb / kxMaxb + 0 * dxb / kxMaxb))
    Wx = np.exp(1j * 2 * pi * (kxmaxb + dxb / rxb - kxminb) / ((kxMaxb) * (mx)))

    # calculation for Ex
    Gx2 = np.zeros((my, nx), dtype=complex)
    Mx = Gxm * np.exp(2 * np.pi * 1j * (Ys[0, 0] * Sy + z2y[0] * np.sqrt(1 / (waveLen) ** 2 - Sx ** 2 - Sy ** 2)))
    dMx = np.exp(2 * np.pi * 1j * (
                (Ys[1, 0] - Ys[0, 0]) * Sy + (z2y[1] - z2y[0]) * np.sqrt(1 / (waveLen) ** 2 - Sx ** 2 - Sy ** 2)))
    Gx2[0, :] = np.sum(Mx, axis=0) / ny
    for jy in range(my - 1):
        Mx = Mx * dMx
        Gx2[jy + 1, :] = np.sum(Mx, axis=0) / ny

    dim2 = 2
    Ex = chirpz2Daxis(Gx2, Ax, Wx, mx, dim2) / nx

    # analytical linear phase,
    liny_out = liny_in - np.cos(phi) * np.tan(phi)
    linx_out = linx_in

    # substract the linear phase
    Eout = Ex * np.exp(-1j * 2 * pi * Ys * np.tan(phi) / waveLen)

    return Eout, Xout, Yout, linx_out, liny_out

#
# def rotate_around_y_new(Xin, Yin, Ein, waveLen, phi, Xout, Yout, linx_in=0, liny_in=0):
#     # Copy field
#     Ex = Ein.copy()
#
#     # original sampling points and field
#     old_ny, old_nx = Ex.shape
#
#     # zero padding
#     Ex = np.pad(Ex, ((old_ny//2, old_ny//2), (old_nx//2, old_nx//2)), 'constant', constant_values=(0, 0))
#
#     # new shape
#     ny, nx = Ex.shape
#
#     # coordinate spacing
#     dx1 = Xin[0, 1] - Xin[0, 0]
#     dy1 = Yin[1, 0] - Yin[0, 0]
#
#     # spatial coordinates
#     x1 = np.fft.fftshift(np.fft.fftfreq(nx, 1)) * nx * dx1
#     y1 = np.fft.fftshift(np.fft.fftfreq(ny, 1)) * ny * dy1
#     X, Y = np.meshgrid(x1, y1)
#
#     # spatial frequencies
#     sx1 = np.fft.fftshift(np.fft.fftfreq(nx, dx1))
#     sy1 = np.fft.fftshift(np.fft.fftfreq(ny, dy1))
#     Sx, Sy = np.meshgrid(sx1, sy1)
#
#     # spectrum
#     Gxm = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ex)))
#     Kx = Sx * 2 * pi
#     Ky = Sy * 2 * pi
#
#     # Tilt: rotation around y => variation in x
#     z2x = Xout[1, :] * np.tan(phi) * np.cos(phi)
#     Xs = Xout * np.cos(phi)
#
#     # output sampling
#     my, mx = Xout.shape
#
#     # adjust input frequencies
#     Sx += linx_in
#     Sy += liny_in
#
#     # chirp-z parameters
#     dy2 = np.abs(Yout[1, 0] - Yout[0, 0])
#     ryb = (dy1 / dy2) * (ny / my)
#     if ryb < 1:
#         raise Exception('Zoom out not possible; change output sampling.')
#
#     sy = np.fft.fftshift(np.fft.fftfreq(ny, dy1))
#     dkyb = (sy[1] - sy[0]) * 2 * pi
#     dyb = 2 * pi / (dkyb * ny)
#     kyminb = -np.ceil((my-1)/2) * dyb / ryb
#     kymaxb = np.floor((my-1)/2) * dyb / ryb
#     kyMaxb = my * dyb
#
#     Ay = np.exp(-1j * pi * (2 * kyminb / kyMaxb))
#     Wy = np.exp(1j * 2 * pi * (kymaxb + dyb/ryb - kyminb) / (kyMaxb * my))
#
#     # build propagated spectrum
#     Gy2 = np.zeros((ny, mx), dtype=complex)
#     Mx = Gxm * np.exp(2 * pi * 1j * (Xs[0, 0] * Sx + z2x[0] * np.sqrt(1/(waveLen**2) - Sx**2 - Sy**2)))
#     dMx = np.exp(2 * pi * 1j * ((Xs[0, 1] - Xs[0, 0]) * Sx + (z2x[1] - z2x[0]) * np.sqrt(1/(waveLen**2) - Sx**2 - Sy**2)))
#
#     Gy2[:, 0] = np.sum(Mx, axis=1) / nx
#     for jx in range(1, mx):
#         Mx *= dMx
#         Gy2[:, jx] = np.sum(Mx, axis=1) / nx
#
#     # apply chirp-z in Y direction (dim=1)
#     dim2 = 1
#     Ex = chirpz2Daxis(Gy2, Ay, Wy, my, dim2) / ny
#
#     # analytical linear phase correction
#     linx_out = linx_in - np.cos(phi) * np.tan(phi)
#     liny_out = liny_in
#
#     Eout = Ex * np.exp(-1j * 2 * pi * Xs * np.tan(phi) / waveLen)
#
#     return Eout, Xout, Yout, linx_out, liny_out
#
# def rotate_around_x_new(Xin, Yin, Ein, waveLen, phi, Xout, Yout, linx_in=0, liny_in=0):
#     # Copy field
#     Ex = Ein
#
#     # original sampling points and field
#     old_ny, old_nx = Ex.shape
#
#     # zero padding
#     Ex = np.pad(Ex, ((int(old_ny/2), int(old_nx/2)), (int(old_ny/2), int(old_nx/2))), 'constant', constant_values=(0, 0))
#
#     # new sampling points
#     ny, nx = Ex.shape
#
#     # extended spatial coordinates and spatial frequencies
#     dx1 = Xin[0, 1] - Xin[0, 0]
#     dy1 = Yin[1, 0] - Yin[0, 0]
#     x1 = np.fft.fftshift(np.fft.fftfreq(nx, 1)) * nx * dx1
#     y1 = np.fft.fftshift(np.fft.fftfreq(ny, 1)) * ny * dy1
#     X, Y = np.meshgrid(x1, y1)
#     Z = 0 * X
#
#     # extended spatial frequencies
#     dx1 = Xin[0, 1] - Xin[0, 0]
#     dy1 = Yin[1, 0] - Yin[0, 0]
#     sx1 = np.fft.fftshift(np.fft.fftfreq(nx, dx1))
#     sy1 = np.fft.fftshift(np.fft.fftfreq(ny, dy1))
#     Sx, Sy = np.meshgrid(sx1, sy1)
#
#     # calculation of the spectrum
#     Gxm = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ex)))
#     Kx = Sx * 2 * pi
#     Ky = Sy * 2 * pi
#
#     # coupled coordinates z' = z(y)
#     z2y = Yout[:, 1] * np.tan(phi) * np.cos(phi)
#
#     # scaled coordinates in Y'
#     Ys = Yout * np.cos(phi)
#
#     # output sampling points
#     my, mx = np.shape(Xout)
#
#     # analytical phase
#     Sx = Sx + linx_in
#     Sy = Sy + liny_in
#
#     # parameters for the chirp-z
#     dx2 = np.abs(Xout[0, 1] - Xout[0, 0])
#     rxb = (dx1 / dx2) * (nx / mx)
#     if rxb < 1:
#         raise Exception('Zoom out is not possible. Please change the output sampling grid.')
#
#     sx = np.fft.fftshift(np.fft.fftfreq(nx,dx1))
#     dkxb = (sx[1] - sx[0])*2*pi
#     dxb = 2*pi/(dkxb*nx)
#     kxminb = -np.ceil((mx-1)/2)*dxb/rxb
#     kxmaxb = np.floor((mx-1)/2)*dxb/rxb
#     kxMaxb = mx*dxb
#     Ax = np.exp(-1j*pi*(2*kxminb/kxMaxb + 0*dxb/kxMaxb))
#     Wx = np.exp(1j*2*pi*(kxmaxb + dxb/rxb - kxminb)/((kxMaxb)*(mx)))
#
#     # calculation for Ex
#     Gx2 = np.zeros((my, nx), dtype=complex)
#     Mx = Gxm*np.exp(2*np.pi*1j*(Ys[0,0]*Sy + z2y[0]*np.sqrt(1/(waveLen)**2 - Sx**2 - Sy**2)))
#     dMx = np.exp(2*np.pi*1j*((Ys[1,0] - Ys[0,0])*Sy + (z2y[1] - z2y[0])*np.sqrt(1/(waveLen)**2 - Sx**2 - Sy**2)))
#     Gx2[0,:] = np.sum(Mx, axis=0)/ny
#     for jy in range(my-1):
#         Mx = Mx*dMx
#         Gx2[jy+1,:] = np.sum(Mx, axis=0)/ny
#
#     dim2 =2
#     Ex =  chirpz2Daxis(Gx2, Ax, Wx, mx, dim2)/nx
#
#     # analytical linear phase,
#     liny_out = liny_in - np.cos(phi) * np.tan(phi)
#     linx_out = linx_in
#
#     # substract the linear phase
#     Eout = Ex * np.exp(-1j * 2 * pi * Ys * np.tan(phi) / waveLen)
#
#     return Eout, Xout, Yout, linx_out, liny_out
#
# def chirpz2Daxis(U1, A, W, M, dim=1):
#     """
#     Returns the chirp-z transform of an array along the given axis for specified parameters.
#     """
#
#     if dim == 1: #y-axis
#         N = U1.shape[0]
#         My = M
#         Mx = U1.shape[1]
#         Lx = 2 ** int(np.ceil(np.log2(N + M - 1)))
#         Ly = 2 ** int(np.ceil(np.log2(N + M - 1)))
#         L = Ly
#
#     elif dim == 2: #x-axis
#         N = U1.shape[1]
#         Mx = M
#         My = U1.shape[0]
#         Lx = 2 ** int(np.ceil(np.log2(N + M - 1)))
#         Ly = N
#         L = Lx
#
#     nn = np.linspace(0, N - 1, N)
#     mn = np.linspace(0, M - 1, M)
#     ln = np.linspace(L - N + 1, L - 1, N - 1)
#     be = np.linspace(M, L - N, L - N - M + 1)
#
#     W1 = W ** (((nn) ** 2) / 2)
#     W2 = W ** (-((mn) ** 2) / 2)
#     W3 = W ** (-((L - ln) ** 2) / 2)
#     W4 = 0 * be
#
#     A1 = A ** (-nn)
#     V = np.fft.fft(np.concatenate((W2, W4, W3), axis=0), axis=0)
#     Am = A1 * W1
#     Vm = V
#
#     # select correct axis
#     if dim == 1:
#         Am = np.transpose(Am)
#         Vm = np.transpose(Vm)
#     else:
#         Am = Am
#         Vm = Vm
#
#     # linear convolution
#     Y = np.fft.fft(np.pad(U1 * Am, ((0, Ly - U1.shape[0]), (0, Lx - U1.shape[1]))), axis=dim - 1)
#     U2g = np.fft.ifft(Y * Vm, axis=dim - 1)
#
#     # phase factor for centering
#     fak = (W2 ** -1) * ((A ** -1) * (W ** (np.linspace(0, M - 1, M))) ** (-1 * (int(np.floor(N / 2)))))
#
#     # output field
#     U2g_extr = U2g[:My, :Mx]
#     U2 = U2g_extr * fak
#
#     return U2

def rotate_around_y_new(Xin, Yin, Ein, waveLen, phi, Xout, Yout, linx_in=0, liny_in=0):
    Ex = Ein.copy()
    old_ny, old_nx = Ex.shape
    Ex = np.pad(Ex, ((old_ny//2, old_ny//2), (old_nx//2, old_nx//2)), 'constant')
    ny, nx = Ex.shape

    dx1 = Xin[0, 1] - Xin[0, 0]
    dy1 = Yin[1, 0] - Yin[0, 0]

    sx1 = np.fft.fftshift(np.fft.fftfreq(nx, dx1))
    sy1 = np.fft.fftshift(np.fft.fftfreq(ny, dy1))
    Sx, Sy = np.meshgrid(sx1, sy1)

    Gxm = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ex)))
    Gxm *= np.exp(2 * pi * 1j * (linx_in * Sx + liny_in * Sy))

    my, mx = Xout.shape
    z2x = Xout[1, :] * np.tan(phi) * np.cos(phi)
    Xs = Xout * np.cos(phi)

    dy2 = np.abs(Yout[1, 0] - Yout[0, 0])
    ryb = (dy1 / dy2) * (ny / my)
    if ryb < 1:
        raise Exception('Zoom out not possible; change output sampling.')

    sy = np.fft.fftshift(np.fft.fftfreq(ny, dy1))
    dkyb = (sy[1] - sy[0]) * 2 * pi
    dyb = 2 * pi / (dkyb * ny)
    kyminb = -np.ceil((my-1)/2) * dyb / ryb
    kymaxb = np.floor((my-1)/2) * dyb / ryb
    kyMaxb = my * dyb

    Ay = np.exp(-1j * pi * (2 * kyminb / kyMaxb))
    Wy = np.exp(1j * 2 * pi * (kymaxb + dyb/ryb - kyminb) / (kyMaxb * my))

    Gy2 = np.zeros((ny, mx), dtype=complex)
    Mx = Gxm * np.exp(2 * pi * 1j * (Xs[0, 0] * Sx + z2x[0] * np.sqrt(1/(waveLen**2) - Sx**2 - Sy**2)))
    dMx = np.exp(2 * pi * 1j * ((Xs[0, 1] - Xs[0, 0]) * Sx + (z2x[1] - z2x[0]) * np.sqrt(1/(waveLen**2) - Sx**2 - Sy**2)))

    Gy2[:, 0] = np.sum(Mx, axis=1) / nx
    for jx in range(1, mx):
        Mx *= dMx
        Gy2[:, jx] = np.sum(Mx, axis=1) / nx

    dim2 = 0
    Ex = chirpz2Daxis(Gy2, Ay, Wy, my, dim2) / ny

    linx_out = linx_in - np.cos(phi) * np.tan(phi)
    liny_out = liny_in
    Eout = Ex * np.exp(-1j * 2 * pi * Xs * np.tan(phi) / waveLen)
    return Eout, Xout, Yout, linx_out, liny_out

def rotate_around_x_new(Xin, Yin, Ein, waveLen, phi, Xout, Yout, linx_in=0, liny_in=0):
    Ex = Ein.copy()
    old_ny, old_nx = Ex.shape
    Ex = np.pad(Ex, ((old_ny//2, old_ny//2), (old_nx//2, old_nx//2)), 'constant')
    ny, nx = Ex.shape

    dx1 = Xin[0, 1] - Xin[0, 0]
    dy1 = Yin[1, 0] - Yin[0, 0]

    sx1 = np.fft.fftshift(np.fft.fftfreq(nx, dx1))
    sy1 = np.fft.fftshift(np.fft.fftfreq(ny, dy1))
    Sx, Sy = np.meshgrid(sx1, sy1)

    Gxm = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ex)))
    Gxm *= np.exp(2 * pi * 1j * (linx_in * Sx + liny_in * Sy))

    my, mx = Xout.shape
    z2y = Yout[:, 1] * np.tan(phi) * np.cos(phi)
    Ys = Yout * np.cos(phi)

    dx2 = np.abs(Xout[0, 1] - Xout[0, 0])
    rxb = (dx1 / dx2) * (nx / mx)
    if rxb < 1:
        raise Exception('Zoom out is not possible. Please change the output sampling grid.')

    sx = np.fft.fftshift(np.fft.fftfreq(nx, dx1))
    dkxb = (sx[1] - sx[0])*2*pi
    dxb = 2*pi/(dkxb*nx)
    kxminb = -np.ceil((mx-1)/2)*dxb/rxb
    kxmaxb = np.floor((mx-1)/2)*dxb/rxb
    kxMaxb = mx*dxb
    Ax = np.exp(-1j*pi*(2*kxminb/kxMaxb))
    Wx = np.exp(1j*2*pi*(kxmaxb + dxb/rxb - kxminb)/(kxMaxb * mx))

    Gx2 = np.zeros((my, nx), dtype=complex)
    Mx = Gxm * np.exp(2*pi*1j*(Ys[0, 0]*Sy + z2y[0]*np.sqrt(1/(waveLen**2) - Sx**2 - Sy**2)))
    dMx = np.exp(2*pi*1j*((Ys[1, 0] - Ys[0, 0])*Sy + (z2y[1] - z2y[0])*np.sqrt(1/(waveLen**2) - Sx**2 - Sy**2)))
    Gx2[0, :] = np.sum(Mx, axis=0) / ny
    for jy in range(1, my):
        Mx *= dMx
        Gx2[jy, :] = np.sum(Mx, axis=0) / ny

    dim2 = 1
    Ex = chirpz2Daxis(Gx2, Ax, Wx, mx, dim2) / nx

    liny_out = liny_in - np.cos(phi) * np.tan(phi)
    linx_out = linx_in
    Eout = Ex * np.exp(-1j * 2 * pi * Ys * np.tan(phi) / waveLen)
    return Eout, Xout, Yout, linx_out, liny_out

def chirpz2Daxis(U1, A, W, M, dim=0):
    if dim == 0:
        N = U1.shape[0]
        L = 2 ** int(np.ceil(np.log2(N + M - 1)))
        W1 = W ** ((np.arange(N) ** 2) / 2)
        W2 = W ** (-(np.arange(M) ** 2) / 2)
        W3 = W ** (-(np.arange(L - N + 1, L) ** 2) / 2)
        V = np.fft.fft(np.concatenate((W2, np.zeros(L - N - M + 1), W3)))
        A1 = A ** (-np.arange(N))
        Am = A1 * W1
        U1_padded = np.pad(U1 * Am[:, np.newaxis], ((0, L - N), (0, 0)))
        Y = np.fft.fft(U1_padded, axis=0)
        U2g = np.fft.ifft(Y * V[:, np.newaxis], axis=0)
        fak = W2 ** -1 * (A ** -1 * W ** np.arange(M)) ** (-int(np.floor(N / 2)))
        return U2g[:M, :] * fak[:, np.newaxis]
    elif dim == 1:
        N = U1.shape[1]
        L = 2 ** int(np.ceil(np.log2(N + M - 1)))
        W1 = W ** ((np.arange(N) ** 2) / 2)
        W2 = W ** (-(np.arange(M) ** 2) / 2)
        W3 = W ** (-(np.arange(L - N + 1, L) ** 2) / 2)
        V = np.fft.fft(np.concatenate((W2, np.zeros(L - N - M + 1), W3)))
        A1 = A ** (-np.arange(N))
        Am = A1 * W1
        U1_padded = np.pad(U1 * Am[np.newaxis, :], ((0, 0), (0, L - N)))
        Y = np.fft.fft(U1_padded, axis=1)
        U2g = np.fft.ifft(Y * V[np.newaxis, :], axis=1)
        fak = W2 ** -1 * (A ** -1 * W ** np.arange(M)) ** (-int(np.floor(N / 2)))
        return U2g[:, :M] * fak[np.newaxis, :]

def generate_ProbeModes(illu, wavelength, pinhole, Np, Xp, Yp, zs, nModes=None, verbose=True, Xs=None, Ys=None):
    """
    simulate partially coherent illumination:
    simulate (mutually uncorrelated) spherical wavelets
    from each source point resulting in source plane

    :param illu: 2D array of fourier space-map of coherence of illumination
    :param wavelength: in meters [m] ie. 450e-9
    :param pinhole: 2D array of describing the pinhole
    :param Np: pixels along one direction in pinhole
    :param Xp: position coordinates in meters of pinhole pixels
    :param Yp: position coordinates in meters of pinhole pixels
    :param zs: distance of source to pinhole
    :return: probeModes, sphericalWavelets in pupil
    """
    if Xs is None and Ys is None:
        Xs, Ys = Xp, Yp

    # source coordinates
    sources = np.array(np.where(illu > 0))
    # number of source points
    nsp = len(sources[0, :])
    # (mutually uncorrelated) spherical wavelets
    sphericalWavelets = np.zeros((nsp, Np, Np), dtype=complex)

    # get orthogonal modes by orthogonalizing spherical waves
    for i in range(nsp):
        # evaluate Greens function (Rayleigh-Sommerfeld) for each point source
        # R = np.sqrt(np.square(Xp - Xs[sources[:, i][-2], sources[:, i][-1]]) +
        #             np.square(Yp - Ys[sources[:, i][-2], sources[:, i][-1]]) + zs ** 2)
        # phaase = np.exp((1j * 2 * np.pi / wavelength) * R)
        # phase = (phaase * zs) / (np.dot(R, R))
        # Extract the source indices
        x_index = sources[:, i][-1]
        y_index = sources[:, i][-2]

        # Calculate the squared distances
        dx_squared = np.square(Xp - Xs[y_index, x_index])
        dy_squared = np.square(Yp - Ys[y_index, x_index])
        dz_squared = zs ** 2

        # Sum the squared distances
        R_squared = dx_squared + dy_squared + dz_squared

        # Compute R (distance)
        R = np.sqrt(R_squared)

        # evaluate Greens function (Rayleigh-Sommerfeld) for each point source
        greens = np.exp((1j * 2 * np.pi / wavelength) * R) * zs / R_squared

        sphericalWavelets[i, :, :] = illu[y_index, x_index] * greens
        # multiply each spherical wave with pinhole in entrance pupil
        sphericalWavelets[i, :, :] *= pinhole

    # krank = min(self.nsp, maxNumModes)       # determines how many modes are calculated in SVD

    probe, normalizedEigenvalues = orthogonalizeModes(sphericalWavelets)
    purity = np.sqrt(np.sum(normalizedEigenvalues ** 2)) / sum(normalizedEigenvalues)  # coherence measure

    # retain only effective number of modes
    effModes = int(min(np.ceil(2 / purity ** 2), len(normalizedEigenvalues)))
    energy = np.sum(100 * normalizedEigenvalues[0:effModes])

    if nModes is None:
        # nModes = 9
        nModes = effModes
    probeModes = probe[0:nModes, :, :]

    if verbose:
        print('purity of source: 1(perfect coherent) --> 0(incoherent)')
        print(purity)
        print('eff. number of modes: ' + str(effModes))
        print('energy contained therein: %.2f' % energy + '%')
        print('--Probe modes completed--')

    # normalizedEigenvalues = np.resize(normalizedEigenvalues,(nModes,))
    normalizedEigenvalues.resize(nModes, )
    return sphericalWavelets, probeModes, normalizedEigenvalues, purity


def createProbe(Np, dxp, wavelength, diameter, f=None):
    Lp = Np * dxp
    xp = np.arange(-Np / 2, Np / 2) * dxp
    Yp, Xp = np.meshgrid(xp, xp)
    # pinhole = nip.rr((Np, Np)) < (diameter / (2 * dxp))
    # blur the edges by convolving with a small kernel i.e 5
    if f is not None:
        probe = pinhole * np.exp(1j * 2 * np.pi / wavelength * (Xp**2 + Yp**2) / 2 / f)
        return probe
    else:
        return probe

def u_limit(L,dz,w, x0=0, sign='+'):
    if sign == '+':
        u_limit = 1 / (w * np.sqrt(1 + (dz ** 2) / (x0 + L) ** 2))
    if sign == '-':
        u_limit = 1 / (w * np.sqrt(1 + (dz ** 2) / (x0 - L) ** 2))
    # print(f'u_limit{x0}{sign}:{u_limit}')
    return u_limit

def table2avoidAliasing(x0,L,dz,w):
    """
    See Table 1 in:
    K. Matsushima, “Shifted angular spectrum method for
    off-axis numerical propagation,” Opt. Express,
    vol. 18, no. 17, p. 18453, 2010, doi: 10.1364/oe.18.018453.
    """
    u_plus = u_limit(L, dz, w, x0, '+')
    u_minus = u_limit(L, dz, w, x0, '-')
    Sx = 2*L
    if Sx < x0:
        u0 = (u_plus + u_minus) / 2
        u_width = u_plus - u_minus
    if -Sx <= x0 < Sx:
        u0 = (u_plus - u_minus) / 2
        u_width = u_plus + u_minus
    if x0 <= -Sx:
        u0 = -(u_plus + u_minus) / 2
        u_width = u_minus - u_plus

    return u0, u_width

def shif_ASPW(u, N, dx, w, dz, x0=0, y0=0, bandlimit=True):
    k = 2 * np.pi / w
    L =  N * dx
    X = np.arange(-N / 2, N / 2) / L
    Fx, Fy = np.meshgrid(X, X)

    W = np.ones_like(u)
    if bandlimit and x0 == 0 and y0 == 0:
        f_max = u_limit(L,dz,w)
        W = circ(Fx, Fy, 2 * f_max)

    # observation plane shifted by x0 and y0
    Wx = np.ones_like(u, dtype=np.float32)
    Wy = np.ones_like(u, dtype=np.float32)
    if x0 != 0:
        u0, u_width = table2avoidAliasing(x0, L, dz, w)
        Wx = rect(Fx - u0, u_width)
    if y0 != 0:
        v0, v_width = table2avoidAliasing(y0, L, dz, w)
        Wy = rect(Fy - v0, v_width)

    # note: accounts for circular symmetry of transfer function and imposes bandlimit to avoid sampling issues
    # H = np.exp(1.j * k * (x0*Fx*w + y0*Fy*w + dz * np.sqrt(1 - (Fx * w) ** 2 - (Fy * w) ** 2)))
    arg = np.power(w, -2) - Fx ** 2 - Fy ** 2
    arg[arg<0]=0
    H = np.exp(1.j * 2*np.pi * (x0 * Fx + y0 * Fy + dz * np.sqrt(arg))) * W * Wx * Wy
    u_new = ifft2c(fft2c(u) * H)

    # plt.figure()
    # plt.title('WxWy')
    # plt.imshow(Wx*Wy)
    # plt.colorbar()
    # plt.show()

    return u_new, H

def complex_colorize(z, rmin=False, rmax=False, hue_start=180):
    """ Returns a real RGB image base on some complex input. """
    from matplotlib.colors import hsv_to_rgb

    # get amplitude of z and limit to [rmin, rmax]
    amp = np.abs(z)
    if not rmin and not rmax:
        rmin = 0
        rmax = np.max(amp)
    amp = np.where(amp < rmin, rmin, amp)
    amp = np.where(amp > rmax, rmax, amp)
    ph = np.angle(z, deg=1) + hue_start

    # HSV are values in range [0,1]
    h = (ph % 360) / 360
    s = 0.80 * np.ones_like(h)
    v = (amp - rmin) / (rmax - rmin)
    return hsv_to_rgb(np.dstack((h, s, v)))

def viz(image, title='none', x_dim=False, y_dim=False):
    ''' Visualizes a 2D image, array etc. (can be complex) '''
    import matplotlib.pyplot as plt

    is_complex = False

    if image.dtype == 'complex128' or image.dtype == 'complex64':
        image = complex_colorize(image)
        is_complex = True

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    if x_dim and y_dim:
        plt.imshow(image, extent=[-y_dim/2, y_dim/2,
                                  -x_dim/2, x_dim/2],cmap='gray')
        plt.xlabel('x ($\mu m$)')
        plt.ylabel('y ($\mu m$)')
    else:
        plt.imshow(image,cmap='inferno')
    if title != 'none':
        plt.title(title)
    if not is_complex:
        plt.colorbar()
    plt.show()

    return

# off-axis propagation using shifted planes angular spectrum. pre-calculate the propagator
def precalc_prop_np(u, dl, lmb, z, sx=0, sy=0, x0=0, y0=0,NA_max=1):
    '''

    Parameters
    ----------
    shape : (int, int)
        shape of the field, usually square.
    lmb : double
        wavelength of light.
    z : double
        distance field propagated in micro metre.
    dl : double
        pixel size in micro metre.
    sx : double, optional
        sensor area size x direction in micro metre. The default is 0.
    sy : double, optional
        sensor area size y direction in micro metre. The default is 0.
    x0 : double, optional
        centre of field shifted in x direction in micro metre. The default is 0.
    y0 : double, optional
        centre of field shifted in y direction in micro metre. The default is 0.
    NA_max : double, optional
        maximum of numerical aperture. The default is 1.

    Returns
    -------
    complex 128
        Returns an array which has the propagator values. Multiplication with
        a Fourier space field will propagate the field by z distance.
        eg. propagate the field from the sample plane to the sensor plane.

        This method is based on work by K. Matsushima:
        https://doi-org.proxy.library.uu.nl/10.1364/OE.18.018453

    '''

    def U0(x0, lmb, z, Sx, up, um):
        if Sx < x0:
            return (up + um) / 2
        elif x0 <= -Sx:
            return -(up + um) / 2
        else:
            return (up - um) / 2

    def Uwidth(x0, lmb, z, Sx, up, um):
        if Sx < x0:
            return up - um
        elif x0 <= -Sx:
            return um + up
        else:
            return up + um

    def rect(x):
        if np.abs(x) > 0.5:
            return 0
        elif np.abs(x) == 0.5:
            return 0.5
        else:
            return 1

    [m, n] = np.shape(u)

    if sx == 0:
        sx = dl * m
    if sy == 0:
        sy = dl * n

    [x, y] = np.meshgrid(np.arange(-n / 2, n / 2),
                         np.arange(-m / 2, m / 2))

    fx = (x / (dl * m))  # frequency space width [1/m]
    fy = (y / (dl * n))  # frequency space height [1/m]

    freq = np.fft.fftfreq(n, dl)
    stepsize = freq[1]

    if True:
        up = np.power((np.power((x0 + 1 / (2 * (2.0 * sx) ** -1)), -2) * z ** 2 + 1), -0.5) / lmb
        um = np.power((np.power((x0 - 1 / (2 * (2.0 * sx) ** -1)), -2) * z ** 2 + 1), -0.5) / lmb
        vp = np.power((np.power((y0 + 1 / (2 * (2.0 * sy) ** -1)), -2) * z ** 2 + 1), -0.5) / lmb
        vm = np.power((np.power((y0 + 1 / (2 * (2.0 * sy) ** -1)), -2) * z ** 2 + 1), -0.5) / lmb

        u0 = U0(x0, lmb, z, sx, up, um)
        uwidth = Uwidth(x0, lmb, z, sx, up, um)

        v0 = U0(y0, lmb, z, sy, vp, vm)
        vwidth = Uwidth(y0, lmb, z, sy, vp, vm)

        arrX = np.zeros((m, n), dtype='complex128')
        arrY = np.zeros((m, n), dtype='complex128')

        for i in range(m):
            for j in range(n):
                arrX[i, j] = rect((fx[i, j] - u0) / (uwidth))
                arrY[i, j] = rect((fy[i, j] - v0) / (vwidth))

        fourier_crop = arrX * arrY

    k0 = 2 * np.pi / lmb
    alpha = 2 * np.pi * fx
    beta = 2 * np.pi * fy
    if False:
        # NA filtering

        # Implementing a slight smoothing filter of the propagator
        filter_init = np.zeros(shape, dtype=np.complex_)
        smoothmax = 1.1
        smoothmin = 0.9
        dist2 = alpha ** 2 + beta ** 2
        X, Y = np.where(smoothmin ** 2 * NA_max ** 2 * k0 ** 2 > alpha ** 2 + beta ** 2)
        filter_init[X, Y] = 1
        xgauss, ygauss = np.where(np.logical_and(smoothmin ** 2 * NA_max ** 2 * k0 ** 2 < dist2,
                                                 smoothmax ** 2 * NA_max ** 2 * k0 ** 2 > dist2))
        filter_init[xgauss, ygauss] = np.exp(-(dist2[xgauss, ygauss] - NA_max ** 2 * k0 ** 2 * smoothmin ** 2) / (
                    k0 ** 2 * NA_max ** 2 * (smoothmax ** 2 - smoothmin ** 2) * 0.5))
        filter_init = np.roll(filter_init, int(u0 / stepsize), 1)

    gamma = np.sqrt(k0 ** 2 - alpha ** 2 - beta ** 2)
    # to prevent an datatype error cast gamma to Ein datatype
    gamma = gamma.astype(dtype='complex128')
    H = np.exp(1j * gamma * z) * np.exp(1j * 2 * np.pi * (x0 * fx + y0 * fy)) * fourier_crop #* filter_init
    u_new = ifft2c(fft2c(u) * H)

    return u_new, propagator

if __name__ == "__main__":
    pass
