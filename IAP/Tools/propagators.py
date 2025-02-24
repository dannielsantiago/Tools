"""
Created on Thu Apr 23 22:20:38 2020
@author: r2d2
"""
import numpy as np
import cupy as cp
from scipy.sparse.linalg import svds
from scipy import linalg
from math import pi

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

def circ2(x,y, D):
    """
    generate a circle on a 2D grid
    :param x: 2D x coordinate, normally calculated from meshgrid: x,y = np.meshgird((,))
    :param y: 2D y coordinate, normally calculated from meshgrid: x,y = np.meshgird((,))
    :param D: diameter
    :return: a 2D array
    """
    circle = np.sqrt(x ** 2 + y ** 2) < (D / 2) ** 2
    return circle

def ellipse(x, y, width, height):
    """
    Generate an ellipse on a 2D grid
    :param x: 2D x coordinate, normally calculated from meshgrid: x,y = np.meshgrid((,))
    :param y: 2D y coordinate, normally calculated from meshgrid: x,y = np.meshgrid((,))
    :param width: width of the ellipse (major axis)
    :param height: height of the ellipse (minor axis)
    :return: a 2D array
    """
    ellipse = ( x**2 / width**2 +  y**2 / height**2) < 1
    return ellipse

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

def fft2d_single_column(matrix, column_index):
    # Step 1: Compute the FFT along the rows
    row_fft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(matrix), axis=1, norm='ortho'))

    # Step 2: Extract the column of interest from the row-wise FFT result
    column_of_interest = row_fft[:, column_index]
    # Step 3: Compute the FFT along the column of interest
    column_fft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(column_of_interest), norm='ortho'))
    return column_fft

def ifft2d_single_column(matrix, column_index):
    # Step 1: Compute the FFT along the rows
    row_fft = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(matrix), axis=1, norm='ortho'))

    # Step 2: Extract the column of interest from the row-wise FFT result
    column_of_interest = row_fft[:, column_index]
    # Step 3: Compute the FFT along the column of interest
    column_fft = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(column_of_interest), norm='ortho'))
    return column_fft


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
    U_observation_gpu = _RS_diffraction_integral_gpu(U_source_gpu, Xs_gpu, Ys_gpu, Xq_gpu, Yq_gpu, Zq_gpu, wavelength,
                                                    dx)

    # Move result back to CPU
    U_observation = cp.asnumpy(U_observation_gpu)

    return U_observation


def propagate(u, method='fourier', dx=None, wavelength=None, dz=None, dq=None, bandlimit=True, thetax=0, thetay=0, X0=0, Y0=0, pad_factor_x=2, pad_factor_y=2):
    '''
    propagates a guiven field using diferent methods

    u:          input field to propagate
    method:     'fourier','fresnel','angular'(angular spectrum)
    dx:         pixel spacing of the input field
    wavelength: illumination wavelength
    dz:         distance to propagate

    returns propagated wavefront u'
    '''
    if method == 'fourier':
        if dz > 0:
            # u_new = np.fft.fftshift(np.fft.fft2(u, norm='ortho'))
            # u_new = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u), norm='ortho'))
            # u_new = nip.ft2d(u, norm='ortho')
            u_new = fft2c(u)

        else:
            # u_new = np.fft.ifft2(np.fft.ifftshift(u), norm="ortho")
            # u_new = u_new = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u), norm='ortho'))
            # u_new = nip.ift2d(u, norm='ortho')
            u_new = ifft2c(u)

    elif method == 'aspw':
        k = 2 * np.pi / wavelength
        # source coordinates, this assumes that the field is NxN pixels
        N = u.shape[-1]
        L = N * dx
        linspacex = np.linspace(-N / 2, N / 2, N, endpoint=False).reshape(1, N)
        Fx = linspacex / L
        Fy = Fx.reshape(N, 1)

        f_max = 1 / (wavelength * np.sqrt(1 + (2 * dz / L) ** 2))
        W = circ(Fx, Fy, 2 * f_max)
        # w accounts for circular symmetry of transfer function and imposes bandlimit to avoid sampling issues
        w = 1 / wavelength ** 2 - Fx ** 2 - Fy ** 2
        w[w >= 0] = np.sqrt(w[w >= 0])
        w[w < 0] = 0
        H = np.exp(1.j * 2 * np.pi * dz * w) * W

        U = fft2c(u)
        u_new = ifft2c(U * H)

    elif method == 'fresnel':
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
        # print(u_new.shape)
        # u_new = u_p_final

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

        if bandlimit:
            if m is not 1:
                r1sq_max = wavelength * dz / (2 * dx * (1 - m))
                Wr = np.array(circ(X, Y, 2 * r1sq_max))
                Q1 = Q1 * Wr

            fsq_max = m / (2 * dz * wavelength * (1 / (N * dx)))
            Wf = np.array(circ2(Fx, Fy, 2 * fsq_max))
            # Wf = (np.abs(Fx) < np.abs(2*fsq_max)) * (np.abs(Fy) < np.abs(2*fsq_max))
            # Wf = np.array(circ(Fx, Fy, 2 * fsq_max * np.sqrt(2)))
            # Wf = np.array(circ(Fx*0, Fy, 2 * fsq_max)) * np.array(circ(Fx, Fy*0, 2 * fsq_max))
            # Wf = np.array(circ(fsq, fsq, 2 * fsq_max))
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
    
    elif method == 'reflectionASPW':
        """
        1. First do a coordinate transformtation with sourceAngle
        2. Then a scaledASP to propagate field to the detector
        3. Then another coordinate rotation if detector is tilted detectAngle
        """

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
        for i in range(URot.shape[0]):
            interp_spline_r = RectBivariateSpline(fx, fx, np.real(Ud[i, ...]))
            interp_spline_i = RectBivariateSpline(fx, fx, np.imag(Ud[i, ...]))
            URot[i, ...] = interp_spline_r(fx, fxRot) + 1j * interp_spline_i(fx, fxRot)

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

        for i in range(URot.shape[0]):
            interp_spline_r = RectBivariateSpline(fx, fx, np.real(Ud[i, ...]))
            interp_spline_i = RectBivariateSpline(fx, fx, np.imag(Ud[i, ...]))
            URot[i, ...] = interp_spline_r(fx, fxRot) + 1j * interp_spline_i(fx, fxRot)

        # interpolate Fourier spectrum on tilted plane
        u_new = ifft2c(URot * W * jacobian)

    elif method == 'eulerDecomposition':
        """
        1. First do a coordinate transformtation with sourceAngle
        2. Then a scaledASP to propagate field to the detector
        3. Then another coordinate rotation if detector is tilted detectAngle
        """

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
        for i in range(URot.shape[0]):
            URot[i, ...] = rotate_around_x(Fx, Fy, Ud[i, ...], wavelength, rad, Fx, Fy)[0]
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
        for i in range(URot.shape[0]):
            URot[i, ...] = rotate_around_x(Fx, Fy, Ud[i, ...], wavelength, rad, Fx, Fy)[0]
        u_new = ifft2c(URot * W)

    return u_new

def rotate_around_y(Xin, Yin, Ein, waveLen, phi, Xout, Yout, linx_in=0, liny_in=0):
    # Copy field
    Ex = Ein

    # original sampling points and field
    old_ny, old_nx = Ex.shape

    # zero padding
    Ex = np.pad(Ex, ((int(old_ny/2), int(old_nx/2)), (int(old_ny/2), int(old_nx/2))), 'constant', constant_values=(0, 0))

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

    sx = np.fft.fftshift(np.fft.fftfreq(nx,dx1))
    dkxb = (sx[1] - sx[0])*2*pi
    dxb = 2*pi/(dkxb*nx)
    kxminb = -np.ceil((mx-1)/2)*dxb/rxb
    kxmaxb = np.floor((mx-1)/2)*dxb/rxb
    kxMaxb = mx*dxb
    Ax = np.exp(-1j*pi*(2*kxminb/kxMaxb + 0*dxb/kxMaxb))
    Wx = np.exp(1j*2*pi*(kxmaxb + dxb/rxb - kxminb)/((kxMaxb)*(mx)))

    # calculation for Ex
    Gx2 = np.zeros((my, nx), dtype=complex)
    Mx = Gxm*np.exp(2*np.pi*1j*(Ys[0,0]*Sy + z2y[0]*np.sqrt(1/(waveLen)**2 - Sx**2 - Sy**2)))
    dMx = np.exp(2*np.pi*1j*((Ys[1,0] - Ys[0,0])*Sy + (z2y[1] - z2y[0])*np.sqrt(1/(waveLen)**2 - Sx**2 - Sy**2)))
    Gx2[0,:] = np.sum(Mx, axis=0)/ny
    for jy in range(my-1):
        Mx = Mx*dMx
        Gx2[jy+1,:] = np.sum(Mx, axis=0)/ny 

    dim2 = 2
    Ex =  chirpz2Daxis(Gx2, Ax, Wx, mx, dim2)/nx

    # analytical linear phase,
    liny_out = liny_in - np.cos(phi) * np.tan(phi)
    linx_out = linx_in

    # substract the linear phase
    Eout = Ex * np.exp(-1j * 2 * pi * Ys * np.tan(phi) / waveLen)

    return Eout, Xout, Yout, linx_out, liny_out

def rotate_around_x(Xin, Yin, Ein, waveLen, phi, Xout, Yout, linx_in=0, liny_in=0):
    # Copy field
    Ex = Ein

    # original sampling points and field
    old_ny, old_nx = Ex.shape

    # zero padding
    Ex = np.pad(Ex, ((int(old_ny/2), int(old_nx/2)), (int(old_ny/2), int(old_nx/2))), 'constant', constant_values=(0, 0))

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

    sx = np.fft.fftshift(np.fft.fftfreq(nx,dx1))
    dkxb = (sx[1] - sx[0])*2*pi
    dxb = 2*pi/(dkxb*nx)
    kxminb = -np.ceil((mx-1)/2)*dxb/rxb
    kxmaxb = np.floor((mx-1)/2)*dxb/rxb
    kxMaxb = mx*dxb
    Ax = np.exp(-1j*pi*(2*kxminb/kxMaxb + 0*dxb/kxMaxb))
    Wx = np.exp(1j*2*pi*(kxmaxb + dxb/rxb - kxminb)/((kxMaxb)*(mx)))

    # calculation for Ex
    Gx2 = np.zeros((my, nx), dtype=complex)
    Mx = Gxm*np.exp(2*np.pi*1j*(Ys[0,0]*Sy + z2y[0]*np.sqrt(1/(waveLen)**2 - Sx**2 - Sy**2)))
    dMx = np.exp(2*np.pi*1j*((Ys[1,0] - Ys[0,0])*Sy + (z2y[1] - z2y[0])*np.sqrt(1/(waveLen)**2 - Sx**2 - Sy**2)))
    Gx2[0,:] = np.sum(Mx, axis=0)/ny
    for jy in range(my-1):
        Mx = Mx*dMx
        Gx2[jy+1,:] = np.sum(Mx, axis=0)/ny 

    dim2 =2
    Ex =  chirpz2Daxis(Gx2, Ax, Wx, mx, dim2)/nx

    # analytical linear phase,
    liny_out = liny_in - np.cos(phi) * np.tan(phi)
    linx_out = linx_in

    # substract the linear phase
    Eout = Ex * np.exp(-1j * 2 * pi * Ys * np.tan(phi) / waveLen)

    return Eout, Xout, Yout, linx_out, liny_out

def chirpz2Daxis(U1, A, W, M, dim=1):
    """
    Returns the chirp-z transform of an array along the given axis for specified parameters.
    """

    if dim == 1:
        N = U1.shape[0]
        My = M
        Mx = U1.shape[1]
        Lx = 2 ** int(np.ceil(np.log2(N + M - 1)))
        Ly = 2 ** int(np.ceil(np.log2(N + M - 1)))
        L = Ly

    elif dim == 2:
        N = U1.shape[1]
        Mx = M
        My = U1.shape[0]
        Lx = 2 ** int(np.ceil(np.log2(N + M - 1)))
        Ly = N
        L = Lx

    nn = np.linspace(0, N - 1, N)
    mn = np.linspace(0, M - 1, M)
    ln = np.linspace(L - N + 1, L - 1, N - 1)
    be = np.linspace(M, L - N, L - N - M + 1)

    W1 = W ** (((nn) ** 2) / 2)
    W2 = W ** (-((mn) ** 2) / 2)
    W3 = W ** (-((L - ln) ** 2) / 2)
    W4 = 0 * be

    A1 = A ** (-nn)
    V = np.fft.fft(np.concatenate((W2, W4, W3), axis=0), axis=0)
    Am = A1 * W1
    Vm = V

    # select correct axis
    if dim == 1:
        Am = np.transpose(Am)
        Vm = np.transpose(Vm)
    else:
        Am = Am
        Vm = Vm

    # linear convolution
    Y = np.fft.fft(np.pad(U1 * Am, ((0, Ly - U1.shape[0]), (0, Lx - U1.shape[1]))), axis=dim - 1)
    U2g = np.fft.ifft(Y * Vm, axis=dim - 1)

    # phase factor for centering
    fak = (W2 ** -1) * ((A ** -1) * (W ** (np.linspace(0, M - 1, M))) ** (-1 * (int(np.floor(N / 2)))))

    # output field
    U2g_extr = U2g[:My, :Mx]
    U2 = U2g_extr * fak

    return U2


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


if __name__ == "__main__":
    pass