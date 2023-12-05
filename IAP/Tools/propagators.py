#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 22:20:38 2020

@author: r2d2
"""
import numpy as np
from scipy.sparse.linalg import svds
from scipy import linalg

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


def propagate(u, method='fourier', dx=None, wavelength=None, dz=None, dq=None, bandlimit=True):
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
        X = np.arange(-N / 2, N / 2) / L
        Fx, Fy = np.meshgrid(X, X)
        f_max = L / (wavelength * np.sqrt(L ** 2 + 4 * dz ** 2))
        W = circ(Fx, Fy, 2 * f_max)
        # note: see the paper above if you are not sure what this bandlimit has to do here
        # note: accounts for circular symmetry of transfer function and imposes bandlimit to avoid sampling issues
        w_ = 1 - (Fx * wavelength) ** 2 - (Fy * wavelength) ** 2
        w_[w_ >= 0] = np.sqrt(w_[w_ >= 0])
        w_[w_ < 0] = 0
        H = np.exp(1.j * k * dz * w_ * W)
        U = fft2c(u)
        u_new = ifft2c(U * H)

    elif method == 'fresnel':
        k = 2 * np.pi / wavelength
        # source coordinates, this assumes that the field is NxN pixels
        N = u.shape[-1]
        L = N * dx
        x = np.arange(-N // 2, N // 2) * dx
        [Y, X] = np.meshgrid(x, x)

        # target coordinates
        dq = wavelength * dz / L
        q = np.arange(-N // 2, N // 2) * dq
        [Qy, Qx] = np.meshgrid(q, q)

        Q1 = np.exp(1j * k / (2 * dz) * (X ** 2 + Y ** 2))
        Q2 = np.exp(1j * k / (2 * dz) * (Qx ** 2 + Qy ** 2))

        # pre-factor
        A = 1 / (1j * wavelength * dz)

        # Fresnel-Kirchhoff integral
        u_new = A * Q2 * fft2c(u * Q1)

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
        f_max = L / (wavelength * np.sqrt(L ** 2 + 4 * dz ** 2))

        # source plane coordinates
        x1 = np.arange(-N // 2, N // 2) * dx
        X1, Y1 = np.meshgrid(x1, x1)
        r1sq = X1 ** 2 + Y1 ** 2
        # spatial frequencies(of source plane)
        f = np.arange(-N // 2, N // 2) / (N * dx)
        FX, FY = np.meshgrid(f, f)

        W = circ(FX, FY, 2 * f_max)
        fsq = FX ** 2 + FY ** 2
        # scaling parameter
        # dq = wavelength * dz / L
        m = dq / dx

        # quadratic phase factors
        Q1 = np.exp(1.j * (k / 2) * ((1 - m) / dz) * r1sq)
        Q2 = np.exp(1.j * (np.pi ** 2) * (2 * (-dz) / (m * k) ) * fsq) #* W
        # Q1 = np.exp(1.j * k / 2 * (1 - m) / dz * r1sq)
        # Q2 = np.exp(-1.j * np.pi ** 2 * 2 * dz / m / k * fsq)

        if bandlimit:
            if m is not 1:
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

    return u_new


def generate_ProbeModes(illu, wavelength, pinhole, Np, Xp, Yp, zs, nModes=None, verbose=True):
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
    # source coordinates
    sources = np.array(np.where(illu > 0))
    # number of source points
    nsp = len(sources[0, :])
    # (mutually uncorrelated) spherical wavelets
    sphericalWavelets = np.zeros((nsp, Np, Np), dtype=np.complex)

    # get orthogonal modes by orthogonalizing spherical waves
    for i in range(nsp):
        # evaluate Greens function (Rayleigh-Sommerfeld) for each point source
        R = np.sqrt(np.square(Xp - Xp[sources[:, i][-2], sources[:, i][-1]]) +
                    np.square(Yp - Yp[sources[:, i][-2], sources[:, i][-1]]) + zs ** 2)
        phaase = np.exp((1j * 2 * np.pi / wavelength) * R)
        phase = (phaase * zs) / (np.dot(R, R))
        sphericalWavelets[i, :, :] = illu[sources[:, i][-2], sources[:, i][-1]] * phase
        # multiply each spherical wave with pinhole in entrance pupil
        sphericalWavelets[i, :, :] *= pinhole

    # krank = min(self.nsp, maxNumModes)       # determines how many modes are calculated in SVD

    probe, normalizedEigenvalues = orthogonalizeModes(sphericalWavelets)
    purity = np.sqrt(np.sum(normalizedEigenvalues ** 2)) / sum(normalizedEigenvalues)  # coherence measure

    # retain only effective number of modes
    effModes = int(min(np.ceil(2 / purity ** 2), len(normalizedEigenvalues)))
    energy = np.sum(100 * normalizedEigenvalues[0:effModes])

    if nModes is None:
        nModes = 9
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


def createProbe(Np, dxp, wavelength, diameter, phaseFactor=20):
    Lp = Np * dxp
    xp = np.arange(-Np / 2, Np / 2) * dxp
    Yp, Xp = np.meshgrid(xp, xp)
    pinhole = nip.rr((Np, Np)) < (diameter / (2 * dxp))
    # blur the edges by convolving with a small kernel i.e 5
    pinhole = nip.convolve(pinhole, nip.rr(pinhole.shape) < 5)
    phase1 = np.exp(1j * phaseFactor * np.pi / wavelength * (np.square(Xp) + np.square(Yp)))
    # self.probe = nip.ift2d(self.pinhole*phase1)
    probe = pinhole * phase1
    return probe, Yp, Xp, Lp

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
    H = np.exp(1.j * 2*np.pi * (x0 * Fx + y0 * Fy + dz * np.sqrt(np.power(w, -2) - Fx ** 2 - Fy ** 2)))
    u_new = ifft2c(fft2c(u) * H * W * Wx * Wy)

    # plt.figure()
    # plt.title('WxWy')
    # plt.imshow(Wx*Wy)
    # plt.colorbar()
    # plt.show()

    return u_new

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
def precalc_prop_np(u, shape, lmb, z, dl, sx=0, sy=0, x0=0, y0=0,
                    NA_max=1, visual=False):
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
    visual : boolean, optional
        toggle visualisation of propagator. The default is False.

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

    [m, n] = shape

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

    # NA filtering
    alpha = 2 * np.pi * fx
    beta = 2 * np.pi * fy

    k0 = 2 * np.pi / lmb

    if False:
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

    gamma = np.power((k0 ** 2 - alpha ** 2 - beta ** 2), 0.5)
    # to prevent an datatype error cast gamma to Ein datatype
    gamma = gamma.astype(dtype='complex128')
    if True:
        propagator = np.exp(1j * gamma * z) \
                     * np.exp(1j * 2 * np.pi * (x0 * fx + y0 * fy)) * fourier_crop  # *filter_init
        if visual:
            if x0 == 0:
                viz(propagator, "on-axis propagator")
            else:
                viz(propagator, "off-axis propagator")
    else:
        propagator = np.exp(1j * gamma * z) * filter_init
        viz(propagator, "on-axis propagator")
    return ifft2c(fft2c(u) * propagator)
    # return np.fft.ifftshift(propagator)

if __name__ == "__main__":
    pass