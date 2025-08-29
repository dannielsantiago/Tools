"""
Created on Thu Apr 23 22:19:47 2020
@author: r2d2
"""
import numpy as np
from math import log, log10, pi
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation as register_translation
from scipy.ndimage import shift, gaussian_filter, center_of_mass
from numpy.fft import fft2, fftshift
import scipy.ndimage as ndi
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import center_of_mass
import multiprocessing
from scipy.signal.windows import tukey
import matplotlib
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


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


def setCustomColorMap2(wl):
    """
    create the colormap for diffraction data (the same as matlab)
    return: customized matplotlib colormap
    """
    rgb = wavelength_to_rgb(wl)
    R = rgb[0]/255
    G = rgb[1]/255
    B = rgb[2]/255
    colors = [
        (0.0, 0.0, 0.0),
        (R, G, B),
    ]
    n = 255  # Discretizes the interpolation into n bins
    cm = LinearSegmentedColormap.from_list("cmap", colors, n)
    return cm

CMAP_DIFFRACTION = setCustomColorMap()

# plt.rcParams['text.usetex'] = True
# # plt.rcParams['font.size'] = 15
# # plt.rcParams['legend.fontsize'] = 18
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# plt.rcParams['font.family'] = 'serif'


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


def binning_1d(arr, binFactor):
    # Calculate the length of the new array after binning
    new_length = arr.shape[0] // binFactor

    # Reshape the array to a 2D array where each row has binFactor elements
    reshaped_array = arr[:new_length * binFactor].reshape((new_length, binFactor))

    # Sum along the rows to bin the data
    binned_array = reshaped_array.sum(axis=1)

    return binned_array

def binning(arr, binFactor, method='sum'):
    shape = (arr.shape[0] // binFactor, binFactor,
             arr.shape[1] // binFactor, binFactor)
    if method == 'sum':
        return arr.reshape(shape).sum(-1).sum(1)
    elif method == 'mean':
        return arr.reshape(shape).mean(-1).mean(1)

def bin2(X):
    """
    perform 2-by-2 binning.
    :Params X: input 2D image for binning
    return: Y: output 2D image after 2-by-2 binning
    """
    # simple 2-fold binning
    m, n = X.shape
    Y = np.sum(X.reshape(2, m // 2, 2, n // 2), axis=(0, 2))
    return Y


def generateFermatGrid(n, radius, minStep):
    """
    see https://en.wikipedia.org/wiki/Fermat%27s_spiral
    :param n: number of points generated
    :param radius: radius of spiral in meters
    :return: scanPositions
    """
    # golden ratio
    base = np.append(np.arange(n), 0)
    base = np.arange(n)

    r = np.sqrt(base)
    theta0 = (137.508 / 180) * np.pi
    theta = base * theta0

    Xpos = (r * np.cos(theta))
    Ypos = (r * np.sin(theta))

    scanPos = np.array((Ypos, Xpos)).T
    scanPos *= radius / np.amax(scanPos)

    scanPos = scanPos // minStep
    scanPos *= minStep
    # scanPos = np.around(scanPos*1e6, decimals=2)

    return scanPos


def zero_pad(arr):
    """
    Pad arr with zeros to double the size. Only the last 2 dimensions are affected.
    """
    # Determine the new shape with doubled size in the last two dimensions
    new_shape = arr.shape[:-2] + (arr.shape[-2] * 2, arr.shape[-1] * 2)
    out_arr = np.zeros(new_shape, dtype=arr.dtype)

    # Compute the starting indices for the original array within the padded array
    as1 = (arr.shape[-2] + 1) // 2
    as2 = (arr.shape[-1] + 1) // 2

    # Place the original array in the center of the new zero-padded array
    out_arr[..., as1:as1 + arr.shape[-2], as2:as2 + arr.shape[-1]] = arr
    return out_arr


def zero_unpad(arr, original_shape):
    """
    Strip off padding of arr with zeros to halve the size. Only the last 2 dimensions are affected.
    """
    # Compute the starting indices for the subarray to extract
    as1 = (original_shape[-2] + 1) // 2
    as2 = (original_shape[-1] + 1) // 2

    # Extract the subarray that corresponds to the original array's shape
    return arr[..., as1:as1 + original_shape[-2], as2:as2 + original_shape[-1]]


def generateRectangularGrid(step, minStep, Lx, Ly, noiseP=0.15):
    """
    step: distance in [m] of the scanning step. ie = 20e-6
    minStep: min step distance supported by the XYstage. ie = 5e-6
    Lx: length [m] of the grid in the x-direction. ie = 300e-6
    Ly: lenght [m] of the grid in the y-direction ie= 200e-6
    noiseP: percentage of noise to add to the grid, if None = 0.15 [15%]
    """
    ratio = abs(Lx / Ly)
    Nx = int(Lx / step) + 1
    Ny = Nx + int((Nx - 1) * ((1 / ratio) - 1))
    n = int(Nx * Ny)
    print(f'n:{n}, Nx:{Nx}, Ny:{Ny}, ratio:{ratio}')
    x = np.linspace(-1, 1, Nx) * Lx / 2
    y = np.linspace(-1, 1, Ny) * Ly / 2
    Y, X = np.meshgrid(y, x)
    Y[1::2, :] = np.flip(Y[1::2, :])

    x = np.reshape(X, (1, n))
    y = np.reshape(Y, (1, n))

    scanPos = np.concatenate((y, x)).T
    center = np.expand_dims(scanPos[0, :] * 0, axis=0)
    scanPos = np.concatenate((center, scanPos))

    variation = 1 + np.random.rand(*scanPos.shape) * noiseP
    variation *= step
    scanPos += variation

    scanPos = scanPos // minStep
    scanPos *= minStep
    scanPos -= scanPos[0, :]
    # scanPos = np.around(scanPos * 1e6, decimals=2)

    return scanPos


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

def circ_px(N, D):
    """
    generate a circle on a 2D grid
    :param N: lateral size of array in px
    :param D: diameter in px
    :return: a 2D array
    """
    x = np.linspace(-N//2, N//2, N, endpoint=False).reshape(1,N)
    y = x.reshape(N,1)
    circle = (x ** 2 + y ** 2) < (D / 2) ** 2
    return circle

def rect_px(N, D):
    """
    generate a rectngle on a 2D grid
    :param N: lateral size of array in px
    :param D: lateral size of square in px
    :return: a 2D array
    """
    x = np.linspace(-N // 2, N // 2, N, endpoint=False).reshape(1, N)
    y = x.reshape(N, 1)
    square = (x**2 <= (D / 2)**2) * (y**2 <= (D / 2)**2)
    return square


def rect(arr, threshold = 0.5):
    """
    generate a binary array containing a rectangle on a 2D grid
    :param x: 2D x coordinate, normally calculated from meshgrid: x,y = np.meshgird((,))
    :param threshold: threshold value to binarilize the input array, default value 0.5
    :return: a binary array
    """
    arr = abs(arr)
    return arr<threshold

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


def complex2rgb(u, amplitudeScalingFactor=1, scaling=1):
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
    if scaling != 1:
        local_max = np.max(v)
        v = v / (np.max(v) + np.finfo(float).eps) * (2 ** 8 - 1)
        print(f'ratio: {local_max / scaling}, max(v): {np.max(v)}')

        v *= local_max * scaling
        print(f'max(v): {np.max(v)}')

    else:
        v = v / (np.max(v) + np.finfo(float).eps) * (2 ** 8 - 1)

    hsv = np.dstack([h, s, v])
    rgb = hsv2rgb(hsv)
    return rgb


def complex2rgb2(u, amplitudeScalingFactor=1, scalling=1):
    """
    Preparation function for a complex plot, converting a 2D complex array into an rgb array
    :param u: a 2D complex array
    :return: an rgb array for complex plot
    """
    # hue (normalize angle)
    # if u is on the GPU, remove it as we can toss it now.
    h = np.angle(u)
    # plot cos(x)
    h = np.real(np.exp(1j * h))
    h = (h + 1) / (4)
    # plot arcos(x)
    # h = np.arccos(np.cos(h))
    # h /= 2*np.pi
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


path_avg_overlap = lambda c, d: np.mean(1 - np.array([np.linalg.norm(c[p] - c[p + 1]) for p in range(len(c) - 1)]) / d)


def makeGrating2(gratingFreq, shape=(500, 500), dxp=1e-6, binary=True):
    period = 1 / (dxp * gratingFreq * 1000)  # *1000 mmm/m
    L = shape[-1]
    nLines = int(L / period)
    hole = int(2)
    wall = int(period - hole)

    grating = np.zeros(shape)
    # spacing=shape[-1]//(gratingFreq*2)
    for i in range(nLines + 1):
        a = int(i * period + shape[-1] // 2 - hole // 2)
        b = a + hole
        c = b + wall  # int((i + 1) * period + shape[-1] // 2 + hole)
        grating[:, a:b] = 1
        grating[:, b:c] = 0

    grating[:, shape[-1] // 2:0:-1] = grating[:, shape[-1] // 2::]

    return grating


def get_m1_m2(x, y):
    dx = np.zeros_like(y)

    for i in range(len(x) - 1):
        dx[i] = x[i + 1] - x[i]

    '''calculate mean'''
    m1 = np.sum(x * y * dx) / np.sum(y * dx)
    '''calculate variance'''
    m2 = np.sum(((x - m1) ** 2) * y * dx) / np.sum(y * dx)

    return m1, m2


def get_m1(x, y):
    dx = np.zeros_like(y)

    for i in range(len(x) - 1):
        dx[i] = x[i + 1] - x[i]

    '''calculate mean'''
    m1 = np.sum(x * y * dx) / np.sum(y * dx)

    return m1


def lcoh(w, dw, type='gaussian', a=1):
    lcoh = w ** 2 / dw
    # lcoh = np.outer(w**2,1/dw)

    if type == 'gaussian':  # ~0.4
        a = (2 * log(2) / pi)
    if type == 'lorentzian':  # ~0.62
        a = 2 / pi
    if type == 'other':
        a = a

    return a * lcoh


def find_nearest(array, arrax, value):
    array = np.asarray(array)
    try:
        idxs = np.argwhere(np.logical_and(np.abs(array) > value * 0.95, np.abs(array) < value * 1.05))
        idx = np.amax(idxs)
    except:
        idx = np.amax(np.argwhere(array == np.amin(array)))
        # print(f'lcoh:{arrax[idx]*1e6:.1f} um')
    return idx


def compute_Autocorrelation_and_get_lcoh(l, spectrum):
    # print(l.shape)
    # print(spectrum.shape)
    # l = np.expand_dims(l, axis=0)
    # spectrum = np.expand_dims (l, axis=0)
    K = 1000000
    lcoh_guess = 50e-6
    c = 225563910  # speed of light (m/s)
    x = np.zeros((1, K))
    x[0, :] = np.linspace(0, 4 * lcoh_guess, K)
    t = x / c
    gamma_t2 = np.zeros((1, K))

    delta_f = np.zeros_like(l)
    for i in range(l.shape[-1] - 1):
        delta_f[i] = c / (l[i + 1] - l[i])

    # power spectral density
    G_f = (l ** 2 / c) * spectrum
    # normlaized PSD
    G_f /= np.sum(G_f)

    gamma_t2[0, :] = 2 * np.real(
        np.sum(np.exp(1j * 2 * pi * np.outer((c / l) * G_f * delta_f, t[0, :])), axis=-2))
    gamma_t2[0, :] /= np.amax(gamma_t2[0, :])

    plt.figure()
    plt.plot(x[0, :] * 1e6, gamma_t2[0, :])
    plt.show()
    idx = find_nearest(gamma_t2[0, :], x[0, :], 1 / np.exp(1))
    coherence_lenght = x[0, idx]
    return coherence_lenght


def unwrap1D(phase):
    unwrapped = np.zeros_like(phase)
    unwrapped[0] = phase[0]

    phase *= -1
    carrier = 0
    center = len(phase) // 2
    for i in range(len(phase) - 1):
        cond = phase[i + 1] - phase[i]

        if np.abs(cond) >= np.pi:
            carrier += -2 * np.sign(cond) * np.pi
        unwrapped[i + 1] = phase[i + 1] + carrier

    # for i in range(len(phase)//2-1):
    #     cond = phase[center + i + 1] - phase[center + i]
    #
    #     if np.abs(cond) >= np.pi:
    #         carrier += -2 * np.sign(cond) * np.pi
    #     unwrapped[center + i + 1] = phase[center + i + 1] + carrier

    return unwrapped

def rotate_around_z(Xin, Yin, Ein, phi, linx_in=0, liny_in=0):
    # Copy field
    Ex = Ein

    # original sampling points and field
    old_ny, old_nx = np.shape(Ex)

    # zero padding
    Ex = np.pad(Ex, ((int(old_ny/2), int(old_nx/2)), (int(old_ny/2), int(old_nx/2))), 'constant', constant_values=(0, 0))

    # new sampling points
    ny, nx = np.shape(Ex)

    # extended spatial coordinates and spatial frequencies
    dx1 = Xin[0, 1] - Xin[0, 0]
    dy1 = Yin[1, 0] - Yin[0, 0]
    x1 = np.fft.fftshift(np.fft.fftfreq(nx, 1)) * nx * dx1
    y1 = np.fft.fftshift(np.fft.fftfreq(ny, 1)) * ny * dy1
    X, Y = np.meshgrid(x1, y1)

    # extended spatial frequencies
    dx1 = X[0, 1] - X[0, 0]
    dy1 = Y[1, 0] - Y[0, 0]
    sx1 = np.fft.fftshift(np.fft.fftfreq(nx, dx1))
    sy1 = np.fft.fftshift(np.fft.fftfreq(ny, dy1))
    Sx, Sy = np.meshgrid(sx1, sy1)

    # calculation of the shearing parameters
    Shx1 = Y * np.tan(phi / 2)
    Shy1 = X * np.sin(phi)

    # rotation by three shearing transforms
    Gx = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Ex)))
    Exm = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Gx * np.exp(2 * np.pi * 1j * Shx1 * Sx))))
    
    Gxm = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Exm)))
    Exm = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Gxm * np.exp(-2 * np.pi * 1j * Shy1 * Sy))))
    
    Gxm = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Exm)))
    Ex = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Gxm * np.exp(2 * np.pi * 1j * Shx1 * Sx))))

    # add analytical linear phase change
    linx_out = linx_in
    liny_out = liny_in

    liny_out = liny_out - linx_out*np.tan(phi/2)
    linx_out = linx_out + liny_out*np.sin(phi)
    liny_out = liny_out - linx_out*np.tan(phi/2)

    # undo zeropadding
    Ex = Ex[int(old_ny/2):int(old_ny/2)+old_ny, int(old_nx/2):int(old_nx/2)+old_nx]
    Xout = Xin
    Yout = Yin
    Eout = Ex
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

    dim2 = 2
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
    Y = np.fft.fft(np.pad(U1 * Am, ((0, Ly - U1.shape[0]), (0, Lx - U1.shape[1]))), axis=dim-1)
    U2g = np.fft.ifft(Y * Vm, axis=dim-1)
    
    #phase factor for centering
    fak = (W2 ** -1) * ((A ** -1) * (W ** (np.linspace(0, M - 1, M))) ** (-1 * (int(np.floor(N / 2)))))
    
    # output field
    U2g_extr = U2g[:My, :Mx]
    U2 = U2g_extr * fak

    return U2

def wavelength_to_rgb(wavelength, gamma=0.8, opacity=200):

    '''This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 800:
        attenuation = 0.3 + 0.7 * (800 - wavelength) / (800 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 1.0
        G = 1.0
        B = 1.0
    R *= 255
    G *= 255
    B *= 255
    return (int(R), int(G), int(B), opacity)

def OAM_phase(X, Y, l=1):
    """
    Calculate the phase profile of an Optical Angular Momentum (OAM) beam with a given charge.

    :param X: 2D numpy array, X coordinates of the meshgrid
    :param Y: 2D numpy array, Y coordinates of the meshgrid
    :param l: int, charge of the OAM beam (default is 1)
    :return: 2D numpy array, complex phase profile of the OAM beam
    """
    # Calculate the azimuthal angle theta
    theta = np.arctan2(Y, X)
    # Calculate the phase profile
    phase_profile = np.exp(1j * l * theta)
    return phase_profile


def spiral_phase(X, Y, f, wavelength, n_blades=1):
    """
    Calculate the phase profile of a spiral phase plate with given parameters.

    :param X: 2D numpy array, X coordinates of the meshgrid
    :param Y: 2D numpy array, Y coordinates of the meshgrid
    :param f: float, focal length
    :param wavelength: float, wavelength of the light
    :param n_blades: int, number of blades in the spiral phase plate (default is 1)
    :return: 2D numpy array, complex phase profile of the spiral phase plate
    """
    # Calculate the azimuthal angle theta
    theta = np.arctan2(Y, X)
    # Calculate the radial distance r
    r = np.sqrt(X ** 2 + Y ** 2)
    # Calculate the phase profile
    data = np.exp(-1j * np.pi * r ** 2 / (f * wavelength)) * np.exp(1j * n_blades * theta)
    return data

def spiral_blade_mask(wavelength=13.5e-9, f=0.6e-3, N=256, dx=10e-9, n_blades=3, blades_diameter=8e-6, angle=None, factor=0):
    """
    :param wavelength: target wavelength
    :param f: focus distance, the smaller --> more twisting of the blades around the center
    :param N: #pixels along 1-direction
    :param dx: pixel space
    :param n_blades: # of blades to generate
    :param blades_diameter: extension of the blades
    :param angle: used to strech along x-direction the pattern
    :param factor: [0-1] increases the fill factor of the spiral, default=0 is 50%, factor=0.6 is 70% fill factor
    :return: binary array NxN where 0 represents the blade structures
    """
    if angle is not None:
        stretching_factor = 1 / np.cos(np.deg2rad(angle))
    else:
        stretching_factor = 1

    x = np.arange(-N / 2, N / 2) * dx
    y = np.copy(x)
    x_grid, y_grid = np.meshgrid(x, y)
    x_grid /= stretching_factor
    # y_grid *= stretching_factor
    r = abs(x_grid ** 2 + y_grid ** 2) ** (1 / 2)
    # r = (x_grid ** 2 + (y_grid*stretching_factor) ** 2) ** (1 / 2)
    # r = ((x_grid/stretching_factor) ** 2 + (y_grid) ** 2) ** (1 / 2)


    phi = np.arctan2(y_grid, x_grid)
    # phi = np.arctan2(y_grid*stretching_factor, x_grid)
    # phi = np.arctan2(y_grid, x_grid/stretching_factor)

    # n_blades = n_blades % 5 + 2  #  minimum amount of blades = 2, increases to 6 anc cycles back
    data = np.exp(-1j * np.pi * r ** 2 / f / wavelength) * np.exp(1j * n_blades * phi)

    binary = np.real(data) < factor

    circ = x_grid ** 2 + y_grid ** 2 < (blades_diameter / 2) ** 2
    # circ = (x_grid/np.sqrt(2)) ** 2 + y_grid ** 2 < (blades_diameter / 2) ** 2

    binary = circ * binary
    binary = (binary).astype(int)

    return binary


def remove_phase_ramp(myObject):
    # find center of mass
    ftobj = fft2c(myObject) * np.conj(fft2c(myObject))
    ftobj = np.real(ftobj)
    cy, cx = ndi.center_of_mass(ftobj)
    # re_center using fft or
    object_1_centered1 = ifft2c(re_center_ptychogram(fft2c(myObject), center_coord=np.array([cy, cx])))

    # using phase ramp multiplication
    No = myObject.shape[-1]
    xp = np.linspace(-No // 2, No // 2, No)
    Xp, Yp = np.meshgrid(xp, xp)
    xcoord = (Xp - np.amin(Xp))
    ycoord = (Yp - np.amin(Yp))
    thetax = -(cx - No // 2)
    thetay = -(cy - No // 2)
    phase_ramp = np.exp(1.j * (2 * np.pi / No) * xcoord * thetax) * np.exp(1.j * (2 * np.pi / No) * ycoord * thetay)
    object_1_centered2 = myObject * phase_ramp
    return object_1_centered1, object_1_centered2


def crop(data, center_coordinate):
    if center_coordinate[0] < data.shape[0] / 2:
        data = data[:int(round(2 * center_coordinate[0])), :]
    else:
        data = data[int(round(2 * center_coordinate[0] - data.shape[0])):, :]
    if center_coordinate[1] < data.shape[1] / 2:
        data = data[:, :int(round(2 * center_coordinate[1]))]
    else:
        data = data[:, int(round(2 * center_coordinate[1] - data.shape[1])):]
    return data

def re_center_ptychogram(data, center_coord):
    """
    re-centers ptychogram to a given center_coord
    """
    center_coord = np.around(center_coord, decimals=0)#.astype(np.int32)
    shape = data.shape
    centered = np.zeros_like(data)

    if center_coord[0] <= shape[0] / 2:
        ylen = 2 * center_coord[0]
        if center_coord[1] <= shape[1] / 2:
            xlen = 2 * center_coord[1]
        else:
            xlen = (shape[1] - center_coord[1]) * 2
    else:
        ylen = (shape[0] - center_coord[0]) * 2
        if center_coord[1] <= shape[1] / 2:
            xlen = 2 * center_coord[1]
        else:
            xlen = (shape[1] - center_coord[1]) * 2

    cropped_data = crop(data, center_coord)
    ymin = int((shape[0] - ylen) / 2)
    ymax = int((shape[0] + ylen) / 2)
    xmin = int((shape[1] - xlen) / 2)
    xmax = int((shape[1] + xlen) / 2)
    centered[ymin: ymax, xmin: xmax] = cropped_data
    return centered




def cropCenter(ptychogram, size, shift_x=None, shift_y=None, fill_value=0, center_of_mass_flag=False):
    """
    Crop (and, if needed, pad) an array in its last two dimensions to a square of side 'size'.
    Padding is applied when the requested (possibly shifted) window would fall outside the array.
    """
    # Spatial dims are the last two
    dim_y, dim_x = ptychogram.shape[-2], ptychogram.shape[-1]

    # --- 1) choose center (float)
    if center_of_mass_flag:
        if ptychogram.ndim > 2:
            proj = np.mean(ptychogram, axis=0)
        else:
            proj = ptychogram

        com = center_of_mass(proj)
        if np.any(np.isnan(com)):
            cy, cx = dim_y / 2.0, dim_x / 2.0
        else:
            cy, cx = float(com[0]), float(com[1])
    else:
        cy, cx = dim_y / 2.0, dim_x / 2.0

    # --- 2) apply shifts
    if shift_y is not None:
        cy += float(shift_y)
    if shift_x is not None:
        cx += float(shift_x)

    # --- 3) desired window [start, end) BEFORE padding
    half = size / 2.0
    start_y = int(np.floor(cy - half))
    start_x = int(np.floor(cx - half))
    end_y   = start_y + size
    end_x   = start_x + size

    # --- 4) compute required padding on each side to accommodate the window
    pad_top    = max(0, -start_y)
    pad_left   = max(0, -start_x)
    pad_bottom = max(0, end_y - dim_y)
    pad_right  = max(0, end_x - dim_x)

    if any(v > 0 for v in (pad_top, pad_bottom, pad_left, pad_right)):
        # build pad spec: pad only last two dims
        pad_spec = [(0, 0)] * (ptychogram.ndim - 2) + [(pad_top, pad_bottom), (pad_left, pad_right)]
        ptychogram = np.pad(ptychogram, pad_spec, mode='constant', constant_values=fill_value)
        # update dims after padding
        dim_y += pad_top + pad_bottom
        dim_x += pad_left + pad_right
        # shift window into padded coordinates
        start_y += pad_top
        start_x += pad_left
        end_y = start_y + size
        end_x = start_x + size

    # --- 5) safe-guard (should be inside now, but clamp just in case)
    start_y = max(0, min(start_y, dim_y - size))
    start_x = max(0, min(start_x, dim_x - size))
    end_y   = start_y + size
    end_x   = start_x + size

    # --- 6) slice
    cropped = ptychogram[..., start_y:end_y, start_x:end_x]
    return cropped
def phase_ramp(slope_x, slope_y, offset, shape):
    """
    Adds a phase rampt to the object.
    :param slope_x: Slope along x direction
    :param slope_y: Slope along y direction
    :param offset: constant phase offset.
    """
    y = np.linspace(-1, 1, shape[0]) * shape[0] * np.pi
    x = np.linspace(-1, 1, shape[1]) * shape[1] * np.pi

    x_grid, y_grid = np.meshgrid(x, y)
    ramp = x_grid * slope_x/shape[1] + y_grid *slope_y/shape[0]

    return ramp + offset

def ringfunc(r, y_shape, x_shape):
    """
    Returns a ring with the radius r.

    :param r: Radius of the ring
    :param y_shape: Shape in y direction of the windw.
    :param x_shape: Shape inx direction of the window
    :return:
    """
    x = np.arange(0, x_shape, 1) - x_shape / 2 + 0.5
    y = np.arange(0, y_shape, 1) - y_shape / 2 + 0.5
    y_grid, x_grid = np.meshgrid(y, x)

    circ_outer = y_grid**2 + x_grid**2 > r**2
    circ_inner = y_grid**2 + x_grid**2 <= (r - 1)**2
    ring = circ_outer + circ_inner
    ring = (ring - 1.) * -1.
    return ring

def one_bit_criterion(N):
    return (0.5 + 2.41/np.sqrt(N)) / (1.5 + 1.41/np.sqrt(N))

def half_bit_criterion(N):
    return (0.2071 + 1.91 / np.sqrt(N)) / (1.2071 + 0.9102 / np.sqrt(N))


def error(reconstruction, simulation):
    return np.sum(np.abs(reconstruction - simulation) ** 2) / np.sum(np.abs(simulation) ** 2)

def FRC(image_1, image_2, filter=True, global_phase_pos=None, filter_radius=100, mask_phase=False, show_difference=False,
        pramp_1=None, pramp_2=None, norm_pos=None):
    """
    Image_1: reconstructed image,
    image_2: Reference
    return frc, one_bit_crit, half_bit_crit, error(image_1, image_2)
    """
    shift_order = 5
    # normalize images to the same avg value
    image_2 /= np.abs(cropCenter(image_2, 2 * filter_radius)).mean()
    image_1 /= np.abs(cropCenter(image_1, 2 * filter_radius)).mean()

    #check error metric before shift
    error_bs = error(image_1, image_2)

    # First check if both images have the same center:
    shift_distance = register_translation(np.abs(image_1), np.abs(image_2), upsample_factor=100)[0]
    print("Shift distance: " + str(shift_distance))
    # shift_distance += np.array([0, .3])
    temp = shift(np.real(image_2), shift_distance, order=shift_order) + 1j * shift(np.imag(image_2), shift_distance, order=shift_order)

    #check error after shift
    error_as = error(image_1, temp)

    if error_as < error_bs:
        image_2 = temp
    image_2 = temp
    if filter:
        image_1 = cropCenter(image_1, 2 * filter_radius)
        image_2 = cropCenter(image_2, 2 * filter_radius)

    print(global_phase_pos)

    if global_phase_pos is not None:
        print("Adjusting global phase")
        # Substracts a global phase offset from a given position for both images
        image_1 *= np.exp(-1j * np.angle(image_1[global_phase_pos]))
        image_2 *= np.exp(-1j * np.angle(image_2[global_phase_pos]))

    if norm_pos is not None:
        image_1 /= np.abs(image_2[norm_pos])
        image_2 /= np.abs(image_2[norm_pos])

    # plt.figure("Original image")
    # plt.imshow(np.abs(image_1))
    #
    # plt.figure("Original image phase")
    # plt.imshow(np.angle(image_1))
    #
    # plt.figure("Image compare")
    # plt.imshow(np.abs(image_2))
    #
    # plt.figure("Image compare phase")
    # plt.imshow(np.angle(image_2))

    if pramp_1 is not None:
        p_ramp = phase_ramp(pramp_1[0], pramp_1[1], 0, image_1.shape)
        image_1 *= np.exp(1j * p_ramp)

    if pramp_2 is not None:
        p_ramp = phase_ramp(pramp_2[0], pramp_2[1], 0, image_2.shape)
        image_2 *= np.exp(1j * p_ramp)

    print("--------------")
    print("FRC")

    y_shape = image_1.shape[0]
    x_shape = image_1.shape[1]
    R = image_1.shape[0]/2
    window_func = np.hanning(y_shape).reshape(1, -1) * np.hanning(x_shape).reshape(-1, 1)

    fft_image1 = fftshift(fft2(fftshift(image_1 * window_func)))
    fft_image2 = fftshift(fft2(fftshift(image_2 * window_func)))
    fft_image1 /= np.max(np.abs(fft_image1))
    fft_image2 /= np.max(np.abs(fft_image2))
    conj_image1 = np.conj(fft_image1)

    frc = np.zeros(int(np.min([x_shape, y_shape])/2), dtype=complex)
    one_bit_crit = np.zeros_like(frc)
    half_bit_crit = np.zeros_like(frc)

    # plt.figure()
    # plt.subplot(131)
    # plt.title("Fourier recon")
    # plt.imshow(np.log10(np.abs(fft_image1)))
    #
    # plt.subplot(132)
    # plt.title("Fourier reference")
    # plt.imshow(np.log10(np.abs(fft_image2)))
    #
    # plt.subplot(133)
    # plt.title("Differences")
    # plt.imshow(np.abs(fft_image2 - fft_image1))

    if show_difference:
        plt.figure()
        plt.subplot(121)
        plt.title("Phase difference abs. value")
        plt.imshow(np.abs(np.angle(image_1) - np.angle(image_2)), interpolation="none", vmin=0, vmax=1)
        plt.subplot(122)
        plt.title("Ampl. difference")
        plt.imshow(np.abs(image_1) - np.abs(image_2), interpolation="none")

    max = np.max(np.abs(image_1)) / 3.

    # if mask_phase:
    #     plt.figure()
    #     plt.imshow(np.angle(np.where(np.abs(image_1) > max, image_1, 0)))
    # else:
    #     plt.figure()
    #     plt.imshow(np.angle(image_1))

    for r in range(int(R)):
        r += 1
        # print('x shape')
        # print(x_shape)
        ring = ringfunc(r, x_shape, y_shape)
        one_bit_crit[r-1] = one_bit_criterion(np.sum(ring))
        half_bit_crit[r-1] = half_bit_criterion(np.sum(ring))
        # check complex value/real value
        frc[r-1] = np.sum(ring * conj_image1 * fft_image2) / np.sqrt(np.sum(ring * np.abs(fft_image1)**2) * np.sum(ring * np.abs(fft_image2)**2))

    return frc, one_bit_crit, half_bit_crit, error(image_1, image_2)


def remove_ramp_three_point(phase_map, points, area_size=3,):
    """
    Remove a linear phase ramp from a 2D phase map using three reference points.

    Parameters
    ----------
    phase_map : 2D ndarray
        Input phase map (in radians).
    points : list of tuples
        List of (y, x) coordinates for three reference points.
    area_size : int, optional
        Size of the square region (n×n) around each point to average phase.
    unwrap : bool, optional
        If True, unwrap the phase before ramp removal.

    Returns
    -------
    phase_corrected : 2D ndarray
        Phase map with ramp removed.
    ramp : 2D ndarray
        The fitted ramp that was subtracted.
    coeffs : tuple
        (a, b, c) coefficients of the fitted plane: a*x + b*y + c
    """
    phase = np.array(phase_map, dtype=float)

    # Average phase in area around each point
    coords = []
    values = []
    r = area_size // 2

    for (y, x) in points:
        y_min, y_max = max(0, y - r), min(phase.shape[0], y + r + 1)
        x_min, x_max = max(0, x - r), min(phase.shape[1], x + r + 1)
        area_phase = phase[y_min:y_max, x_min:x_max]
        coords.append((x, y))
        values.append(np.mean(area_phase))

    # Fit plane to these 3 averaged points
    coords = np.array(coords)
    values = np.array(values)
    A = np.c_[coords[:, 0], coords[:, 1], np.ones(3)]
    coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)  # a, b, c

    # Compute ramp over full grid
    ny, nx = phase.shape
    Y, X = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    ramp = (coeffs[0] * X + coeffs[1] * Y + coeffs[2])

    # Subtract ramp
    phase_corrected = phase - ramp

    return phase_corrected

def remove_phase_ramp_three_point_complex(field, points, area_size=3):
    """
    Remove a linear phase ramp from a 2D complex field using three reference points.

    Parameters
    ----------
    field : 2D ndarray (complex)
        Input complex field.
    points : list of tuples
        List of (y, x) coordinates for three reference points.
    area_size : int, optional
        Size of the square region (n×n) around each point to average phase.

    Returns
    -------
    corrected_field : 2D ndarray (complex)
        Complex field with linear phase ramp removed.
    ramp_phase : 2D ndarray (float)
        The phase ramp (in radians) that was subtracted.
    coeffs : tuple of floats
        (a, b, c) coefficients of the fitted phase plane: a*x + b*y + c
    """
    if np.iscomplexobj(field) is False:
        raise ValueError("Input must be a complex-valued array.")

    phase = np.angle(field)
    ny, nx = field.shape
    r = area_size // 2

    # Extract averaged phases from 3 areas
    coords = []
    values = []

    for (y, x) in points:
        y_min, y_max = max(0, y - r), min(ny, y + r + 1)
        x_min, x_max = max(0, x - r), min(nx, x + r + 1)
        region = phase[y_min:y_max, x_min:x_max]
        coords.append((x, y))
        values.append(np.mean(region))

    coords = np.array(coords)
    values = np.unwrap(values)  # unwrap 1D to avoid ambiguity

    A = np.c_[coords[:, 0], coords[:, 1], np.ones(3)]
    coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)  # a, b, c

    # Build ramp
    Y, X = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    ramp_phase = coeffs[0] * X + coeffs[1] * Y + coeffs[2]

    # Subtract ramp
    corrected_field = field * np.exp(-1j * ramp_phase)

    return corrected_field, ramp_phase

def equalize_phase_ramp_roi(
    obj1: np.ndarray,
    obj2: np.ndarray,
    roi: tuple,                 # (y0, y1, x0, x1)
    weight: str = "amp",        # "none", "amp" (|o1|*|o2|), or "amp2" ((|o1|*|o2|)^2)
    amp_thresh: float = 0.0,    # relative threshold on |o1| and |o2| inside ROI (e.g. 0.1)
    apply: str = "obj2",        # "obj2" (default), "both" (split half to each)
    return_ramp: bool = False,  # return ramp map and coeffs
):
    """
    Fit a phase plane to the phase difference in a rectangular ROI and subtract it
    (to minimize L2 error of phase difference in that ROI).

    Assumes obj1 and obj2 are already spatially aligned and same shape.
    """
    if obj1.shape != obj2.shape:
        raise ValueError("obj1 and obj2 must have the same shape")

    y0, y1, x0, x1 = roi
    if not (0 <= y0 < y1 <= obj1.shape[0] and 0 <= x0 < x1 <= obj1.shape[1]):
        raise ValueError("ROI is out of bounds")

    # Phase difference (wrapped), then unwrap crudely along both axes
    ph_diff_full = np.angle(obj1 * np.conj(obj2))
    ph_roi = ph_diff_full[y0:y1, x0:x1]
    ph_roi_unw = np.unwrap(np.unwrap(ph_roi, axis=0), axis=1)

    # Weights
    if weight == "amp":
        w_roi = (np.abs(obj1[y0:y1, x0:x1]) * np.abs(obj2[y0:y1, x0:x1]))
    elif weight == "amp2":
        w_roi = (np.abs(obj1[y0:y1, x0:x1]) * np.abs(obj2[y0:y1, x0:x1]))**2
    else:
        w_roi = np.ones_like(ph_roi_unw)

    if amp_thresh > 0:
        a1 = np.abs(obj1[y0:y1, x0:x1])
        a2 = np.abs(obj2[y0:y1, x0:x1])
        m = (a1 > amp_thresh * a1.max()) & (a2 > amp_thresh * a2.max())
        # avoid all-zero mask
        if np.any(m):
            w_roi = w_roi * m.astype(float)

    # Build coordinates over ROI (global indices)
    yy, xx = np.mgrid[y0:y1, x0:x1]

    # Weighted least squares fit to ph ~ a*x + b*y + c
    X = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.size)]
    y = ph_roi_unw.ravel()
    W = np.sqrt(w_roi.ravel() + 1e-12)  # sqrt weights for WLS
    Xw = X * W[:, None]
    yw = y * W
    coeffs, *_ = np.linalg.lstsq(Xw, yw, rcond=None)  # [a, b, c]

    a, b, c = coeffs

    # Build full-image ramp using global coordinates
    Yfull, Xfull = np.indices(obj1.shape)
    ramp_full = a * Xfull + b * Yfull + c

    if apply == "obj2":
        obj1_corr = obj1
        obj2_corr = obj2 * np.exp(-1j * ramp_full)
    elif apply == "both":
        half = 0.5 * ramp_full
        obj1_corr = obj1 * np.exp(+1j * half)
        obj2_corr = obj2 * np.exp(-1j * half)
    else:
        raise ValueError("apply must be 'obj2' or 'both'")

    if return_ramp:
        return obj1_corr, obj2_corr, ramp_full, (a, b, c)
    return obj1_corr, obj2_corr

def equalize_phase_ramp_via_fft_shift(u1: np.ndarray, u2: np.ndarray, window=True, upsample=100):
    """
    Estimate linear phase ramp between u1 and u2 by registering the shift
    between their FFT magnitudes, then remove that ramp from u2.

    Returns
    -------
    u1_corr, u2_corr, kappa : (ndarray, ndarray, tuple)
        Corrected fields and (Δky, Δkx) in cycles/pixel.
    """
    if u1.shape != u2.shape:
        raise ValueError("u1 and u2 must have the same shape")

    H, W = u1.shape

    # 1) Apodization to reduce leakage (optional but recommended)
    if window:
        wy = np.hanning(H)
        wx = np.hanning(W)
        win = np.outer(wy, wx)
    else:
        win = 1.0

    u1w = u1 * win
    u2w = u2 * win

    # 2) FFT magnitudes
    U1 = fftshift(fft2(u1w))
    U2 = fftshift(fft2(u2w))
    A1 = np.abs(U1)
    A2 = np.abs(U2)

    # 3) Subpixel shift between spectra magnitudes (in pixels in k-space grid)
    # phase_cross_correlation returns (shift_y, shift_x) to apply to A2 to align to A1
    dky_px, dkx_px = register_translation(A1, A2, upsample_factor=upsample)[0]

    # 4) Convert "pixel shift in FFT image" to spectral shift (cycles/pixel)
    # fftshifted spectrum is on an integer pixel grid; 1 pixel in FFT corresponds to 1/W (x) or 1/H (y) cycles/pixel
    dkx = dkx_px / W
    dky = dky_px / H

    # 5) Build real-space ramp: exp(-i 2π (dkx * X + dky * Y)) and apply to u2
    Y, X = np.indices((H, W))
    ramp = np.exp(-2j * np.pi * (dkx * X + dky * Y))
    u2_corr = u2 * ramp

    return u1, u2_corr

def align_objects(obj1_ref, obj2, preference='auto', shift_tol=1e-3, error_tol=1e-4, method='spline'):
    """
    Aligns  obj2 to  obj1_ref based on amplitude or phase, using residual shift and image error criteria.

    Parameters
    ----------
    preference : str
        Alignment mode:
            'auto'      : use best shift based on residual + error metric
            'phase'     : accept phase shift if both criteria improve
            'amplitude' : accept amplitude shift if both criteria improve
    shift_tol : float
        Minimum reduction in shift norm required to accept alignment
    error_tol : float
        Minimum reduction in image error required to accept alignment

    Returns
    -------
    bool
        True if no shift was applied (alignment rejected), False otherwise.
    """

    verbose = True
    def fourier_shift2d_complex(u: np.ndarray, shift_vec):
        """
        Coherent subpixel shift via Fourier shift theorem.
        shift_vec = (dy, dx) in pixel units (same sign convention as register_translation).
        """
        ny, nx = u.shape
        dy, dx = float(shift_vec[0]), float(shift_vec[1])

        # frequency grids in cycles/pixel
        fy = np.fft.fftfreq(ny)  # 0, 1/ny, ..., (ny-1)/ny with negative freqs wrapped
        fx = np.fft.fftfreq(nx)
        FY, FX = np.meshgrid(fy, fx, indexing='ij')

        # phase ramp exp(-2πi (fx*dx + fy*dy))
        ramp = np.exp(-2j * np.pi * (FX * dx + FY * dy))

        U = fft2c(u)
        U_shifted = U * ramp
        u_shifted = ifft2c(U_shifted)

        # if input is real-space non-periodic, consider windowing/padding before calling
        return u_shifted

    def error_metric(image1, image2):
        return np.sum(np.abs(image1 - image2) ** 2) / np.sum(np.abs(image1) ** 2)

    def apply_shift(obj, shift_vec, method):
        if method =='fourier':
            return fourier_shift2d_complex(obj, shift_vec)
        elif method=='spline':
            return shift(np.real(obj), shift_vec, order=5) + 1j * shift(np.imag(obj), shift_vec, order=5)

    def compute_residual_shift(a, b):
        return register_translation(np.abs(a), np.abs(b), upsample_factor=100)[0]

    # Compute initial shifts
    shift_amp = register_translation(np.abs(  obj1_ref), np.abs(  obj2), upsample_factor=100)[0]
    shift_phs = register_translation(np.angle(  obj1_ref), np.angle(  obj2), upsample_factor=100)[0]

    # Apply shifts
    object2_amp_shifted = apply_shift(  obj2, shift_amp, method)
    object2_phs_shifted = apply_shift(  obj2, shift_phs, method)

    # Residual shift
    resid_shift_amp = compute_residual_shift(  obj1_ref, object2_amp_shifted)
    resid_shift_phs = compute_residual_shift(  obj1_ref, object2_phs_shifted)

    # Shift norms
    norm_init_amp = np.linalg.norm(shift_amp)
    norm_init_phs = np.linalg.norm(shift_phs)
    norm_resid_amp = np.linalg.norm(resid_shift_amp)
    norm_resid_phs = np.linalg.norm(resid_shift_phs)

    # Errors
    err_amp = error_metric(  obj1_ref, object2_amp_shifted)
    err_phs = error_metric(  obj1_ref, object2_phs_shifted)
    err_init = error_metric(  obj1_ref,   obj2)

    if verbose:
        print(f"Initial shift (amplitude): {shift_amp}, norm={norm_init_amp:.3e}")
        print(f"Initial shift (phase):     {shift_phs}, norm={norm_init_phs:.3e}")
        print(f"Residual shift (amp):      {resid_shift_amp}, norm={norm_resid_amp:.3e}")
        print(f"Residual shift (phs):      {resid_shift_phs}, norm={norm_resid_phs:.3e}")
        print(f"Initial error: {err_init:.4e}")
        print(f"Amp error:     {err_amp:.4e}")
        print(f"Phs error:     {err_phs:.4e}")

    def criteria_ok(resid_norm, init_norm, err, err_init):
        # return (resid_norm < init_norm - shift_tol) and (err < err_init - error_tol)
        return err < (err_init - error_tol)
        # return resid_norm < init_norm - shift_tol

    # Decision logic
    if preference == 'phase':
        if criteria_ok(norm_resid_phs, norm_init_phs, err_phs, err_init):
            if verbose:
                print("Using phase-based shift (forced).")
            obj2 = object2_phs_shifted
            return False, obj2

    elif preference == 'amplitude':
        if criteria_ok(norm_resid_amp, norm_init_amp, err_amp, err_init):
            if verbose:
                print("Using amplitude-based shift (forced).")
            obj2 = object2_amp_shifted
            return False, obj2

    elif preference == 'auto':
        phase_ok = criteria_ok(norm_resid_phs, norm_init_phs, err_phs, err_init)
        amp_ok = criteria_ok(norm_resid_amp, norm_init_amp, err_amp, err_init)

        if phase_ok and (not amp_ok or err_phs < err_amp):
            if verbose:
                print("Using phase-based shift (auto).")
            obj2 = object2_phs_shifted
            return False, obj2

        elif amp_ok:
            if verbose:
                print("Using amplitude-based shift (auto).")
            obj2 = object2_amp_shifted
            return False, obj2

    else:
        raise ValueError(f"Invalid preference '{preference}'. Use 'auto', 'phase', or 'amplitude'.")

    if verbose:
        print("No alignment accepted — neither strategy improved both shift and error.")
    return True,  obj2



class MyFRC:
    """
    Example to use:
    myFRC = MyFRC(object_1, object_2, dx)
    myFRC.show_raw_data()
    myFRC.normalize_amplitude()
    myFRC.remove_phase_ramp()
    myFRC.show_comparison_after_phase_ramp_removal(id=0)
    myFRC.show_comparison_after_phase_ramp_removal(id=1)

    #based on plotted results, choose the best result for each object
    myFRC.choose_phase_ramp_result(id=0, result=0)
    myFRC.choose_phase_ramp_result(id=1, result=0)

    region = slice(0,50), slice(0,50)
    myFRC.remove_global_phase_from_avg_region(region)

    myFRC.align_objects()
    myFRC.show_centered_objects()
    myFRC.clip_filter_objects(filter_radius=180)
    myFRC.show_clipped_objects()
    myFRC.calculateFRC()
    myFRC.plotFRC()
    myFRC.get_spatial_resolution()
    """
    def __init__(self, object1, object2, dx, FRC_obj_size):
        self.object1_raw = object1
        self.object2_raw = object2
        self.dx = dx
        self.FRC_obj_size = FRC_obj_size
        self.FRC_sx = 0
        self.FRC_sy = 0
        self._match_shape()

    def _match_shape(self):
        s1 = self.object1_raw.shape
        s2 = self.object2_raw.shape
        min_shape = min(s1,s2)
        if s1 != min_shape:
            # crop obj1 to obj2's shape
            self.object1_raw = cropCenter(self.object1_raw, size=min_shape[0])
        if s2 != min_shape:
            #crop obj2 to obj1's shape
            self.object2_raw = cropCenter(self.object2_raw, size=min_shape[0])
            
        self.object1_processed = np.copy(self.object1_raw)
        self.object2_processed = np.copy(self.object2_raw)

    def show_raw_data(self):
        # plot raw data
        fig, axes = plt.subplots(1, 2)
        axes = axes.flatten()
        fig.suptitle('raw files')
        axes[0].imshow(complex2rgb(self.object1_raw))
        axes[0].set_axis_off()
        axes[1].imshow(complex2rgb(self.object2_raw))
        axes[1].set_axis_off()
        fig.canvas.draw()#(block=False)
    
    def show_processed_data(self, ROI=False):
        if ROI:
            obj1 = cropCenter(self.object1_processed, self.FRC_obj_size, shift_x=self.FRC_sx, shift_y=self.FRC_sy)
            obj2 = cropCenter(self.object2_processed, self.FRC_obj_size, shift_x=self.FRC_sx, shift_y=self.FRC_sy)
        else:
            obj1 = self.object1_processed
            obj2 = self.object2_processed

        h, w = obj1.shape
        out = np.empty_like(obj1)
        cut = w // 2
        out[:, :cut] = obj1[:, :cut]
        out[:, cut:] = obj2[:, cut:]

        abs_difference = np.abs(abs(obj1) - abs(obj2)) ** 2 / np.amax(
            np.abs(obj1) ** 2)
        ang_difference = np.angle(obj1 *np.conj(obj2))
        # plot raw data
        fig, axes = plt.subplots(1, 3)
        axes = axes.flatten()
        fig.suptitle('processed files')
        axes[0].imshow(complex2rgb(out))
        axes[0].set_axis_off()
        eps = 1e-12
        im2 = axes[1].imshow(abs_difference, cmap='viridis')
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_axis_off()
        im3 = axes[2].imshow(ang_difference, cmap='twilight')
        fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        axes[2].set_axis_off()
        axes[1].set_title(f'log amp diff')
        axes[2].set_title(f'phase diff')
        fig.show()

    def normalize_amplitude(self):
        N = self.object1_processed.shape[-1] // 2
        W = int(N * 0.4)
        self.object1_processed /= np.abs(self.object1_processed[N - W // 2:N + W // 2, N - W // 2:N + W // 2]).mean()
        self.object2_processed /= np.abs(self.object2_processed[N - W // 2:N + W // 2, N - W // 2:N + W // 2]).mean()

    def remove_phase_ramp(self, points):

        # remove phase ramp
        self.obj1_v1, ramp = remove_phase_ramp_three_point_complex(self.object1_raw, points)
        self.obj2_v1  = self.object2_raw * np.exp(-1j*ramp)#remove_phase_ramp_three_point_complex(self.object2_raw, points)

        _, self.obj1_v2 = remove_phase_ramp(self.object1_raw)
        _, self.obj2_v2 = remove_phase_ramp(self.object2_raw)

        self.results_phase_ramp = [[self.object1_raw, self.obj1_v1, self.obj1_v2],
                                   [self.object2_raw, self.obj2_v1, self.obj2_v2]]

    
    def remove_phase_ramp_three_points(self, points):
        self.object1_processed, _ = remove_phase_ramp_three_point_complex(self.object1_processed, points)
        self.object2_processed, _ = remove_phase_ramp_three_point_complex(self.object2_processed, points)
    def equalize_phase_ramps(self, **kwargs):
        # self.object1_processed, self.object2_processed = equalize_phase_ramp_roi(self.object1_processed,
        #                                                                          self.object2_processed,
        #                                                                          **kwargs)
        self.object1_processed, self.object2_processed = equalize_phase_ramp_via_fft_shift(self.object1_processed,
                                                                                 self.object2_processed)

    def normalize_amplitude_region(self, point: tuple, size: int) -> np.ndarray:
        """
        Normalize the amplitude of a complex field so that the mean amplitude
        in a given square region is 1.

        Parameters
        ----------
        point : tuple
            (y, x) coordinates of the region center (in pixels).
        size : int
            Size of the square region (in pixels).

        Returns
        -------
        u_norm : np.ndarray
            Amplitude-normalized complex array.
        """
        y0, x0 = point
        half = size // 2
        h, w = self.object1_processed.shape
        # Ensure bounds
        y_min, y_max = max(0, y0 - half), min(h, y0 + half + (size % 2))
        x_min, x_max = max(0, x0 - half), min(w, x0 + half + (size % 2))

        # Compute mean amplitude in region
        self.object1_processed /= np.abs(self.object1_processed[y_min:y_max, x_min:x_max]).mean()
        self.object2_processed /= np.abs(self.object2_processed[y_min:y_max, x_min:x_max]).mean()
        phase1 = np.angle(self.object1_processed[y_min:y_max, x_min:x_max]).mean()
        phase2 = np.angle(self.object2_processed[y_min:y_max, x_min:x_max]).mean()
        self.object1_processed *= np.exp(-1j * phase1)
        self.object2_processed *= np.exp(-1j * phase2)

    def normalize_amplitudes_and_remove_global_phase(self, obj1, obj2):
        N = obj1.shape[-1] // 2
        W = int(N * 0.4)
        obj1 /= np.abs(obj1[N-W//2:N+W//2,N-W//2:N+W//2]).mean()
        obj2 /= np.abs(obj2[N-W//2:N+W//2,N-W//2:N+W//2]).mean()
        # Align global phase between complex fields
        phase_offset = np.angle(np.vdot(obj1, obj2))  # ⟨obj1, obj2⟩ inner product
        print(f'global phase offset: {phase_offset}')
        obj2 = obj2 * np.exp(-1j * phase_offset)
        return obj1, obj2


    def remove_global_phase_from_avg_region(self, region):
        # same phase
        ref_phase = np.angle(self.object1p[region])
        ref_phase = np.mean(ref_phase)
        self.object1p *= np.exp(-1j * ref_phase)
        ref_phase = np.angle(self.object2p[region])
        ref_phase = np.mean(ref_phase)
        self.object2p *= np.exp(-1j * ref_phase)


    def fourier_shift2d_complex(self, u: np.ndarray, shift_vec):
        """
        Coherent subpixel shift via Fourier shift theorem.
        shift_vec = (dy, dx) in pixel units (same sign convention as register_translation).
        """
        ny, nx = u.shape
        dy, dx = float(shift_vec[0]), float(shift_vec[1])

        # frequency grids in cycles/pixel
        fy = np.fft.fftfreq(ny)  # 0, 1/ny, ..., (ny-1)/ny with negative freqs wrapped
        fx = np.fft.fftfreq(nx)
        FY, FX = np.meshgrid(fy, fx, indexing='ij')

        # phase ramp exp(-2πi (fx*dx + fy*dy))
        ramp = np.exp(-2j * np.pi * (FX * dx + FY * dy))

        U = fft2c(u)
        U_shifted = U * ramp
        u_shifted = ifft2c(U_shifted)

        # if input is real-space non-periodic, consider windowing/padding before calling
        return u_shifted

    def align_objects(self, preference='auto', shift_tol=1e-3, error_tol=1e-4, method='spline'):
        """
        Aligns object2_processed to object1_processed based on amplitude or phase, using residual shift and image error criteria.

        Parameters
        ----------
        preference : str
            Alignment mode:
                'auto'      : use best shift based on residual + error metric
                'phase'     : accept phase shift if both criteria improve
                'amplitude' : accept amplitude shift if both criteria improve
        shift_tol : float
            Minimum reduction in shift norm required to accept alignment
        error_tol : float
            Minimum reduction in image error required to accept alignment

        Returns
        -------
        bool
            True if no shift was applied (alignment rejected), False otherwise.
        """

        verbose = True

        def error_metric(image1, image2):
            return np.sum(np.abs(image1 - image2) ** 2) / np.sum(np.abs(image1) ** 2)

        def apply_shift(obj, shift_vec, method):
            if method =='fourier':
                return self.fourier_shift2d_complex(obj, shift_vec)
            elif method=='spline':
                return shift(np.real(obj), shift_vec, order=5) + 1j * shift(np.imag(obj), shift_vec, order=5)

        def compute_residual_shift(a, b):
            return register_translation(np.abs(a), np.abs(b), upsample_factor=100)[0]

        # Compute initial shifts
        shift_amp = register_translation(np.abs(self.object1_processed), np.abs(self.object2_processed), upsample_factor=100)[0]
        shift_phs = register_translation(np.angle(self.object1_processed), np.angle(self.object2_processed), upsample_factor=100)[0]

        # Apply shifts
        object2_amp_shifted = apply_shift(self.object2_processed, shift_amp, method)
        object2_phs_shifted = apply_shift(self.object2_processed, shift_phs, method)

        # Residual shift
        resid_shift_amp = compute_residual_shift(self.object1_processed, object2_amp_shifted)
        resid_shift_phs = compute_residual_shift(self.object1_processed, object2_phs_shifted)

        # Shift norms
        norm_init_amp = np.linalg.norm(shift_amp)
        norm_init_phs = np.linalg.norm(shift_phs)
        norm_resid_amp = np.linalg.norm(resid_shift_amp)
        norm_resid_phs = np.linalg.norm(resid_shift_phs)

        # Errors
        err_amp = error_metric(self.object1_processed, object2_amp_shifted)
        err_phs = error_metric(self.object1_processed, object2_phs_shifted)
        err_init = error_metric(self.object1_processed, self.object2_processed)

        if verbose:
            print(f"Initial shift (amplitude): {shift_amp}, norm={norm_init_amp:.3e}")
            print(f"Initial shift (phase):     {shift_phs}, norm={norm_init_phs:.3e}")
            print(f"Residual shift (amp):      {resid_shift_amp}, norm={norm_resid_amp:.3e}")
            print(f"Residual shift (phs):      {resid_shift_phs}, norm={norm_resid_phs:.3e}")
            print(f"Initial error: {err_init:.4e}")
            print(f"Amp error:     {err_amp:.4e}")
            print(f"Phs error:     {err_phs:.4e}")

        def criteria_ok(resid_norm, init_norm, err, err_init):
            # return (resid_norm < init_norm - shift_tol) and (err < err_init - error_tol)
            return err < (err_init - error_tol)
            # return resid_norm < init_norm - shift_tol

        # Decision logic
        if preference == 'phase':
            if criteria_ok(norm_resid_phs, norm_init_phs, err_phs, err_init):
                if verbose:
                    print("Using phase-based shift (forced).")
                self.object2_processed = object2_phs_shifted
                return False

        elif preference == 'amplitude':
            if criteria_ok(norm_resid_amp, norm_init_amp, err_amp, err_init):
                if verbose:
                    print("Using amplitude-based shift (forced).")
                self.object2_processed = object2_amp_shifted
                return False

        elif preference == 'auto':
            phase_ok = criteria_ok(norm_resid_phs, norm_init_phs, err_phs, err_init)
            amp_ok = criteria_ok(norm_resid_amp, norm_init_amp, err_amp, err_init)

            if phase_ok and (not amp_ok or err_phs < err_amp):
                if verbose:
                    print("Using phase-based shift (auto).")
                self.object2_processed = object2_phs_shifted
                return False

            elif amp_ok:
                if verbose:
                    print("Using amplitude-based shift (auto).")
                self.object2_processed = object2_amp_shifted
                return False

        else:
            raise ValueError(f"Invalid preference '{preference}'. Use 'auto', 'phase', or 'amplitude'.")

        if verbose:
            print("No alignment accepted — neither strategy improved both shift and error.")
        return True


    def clip_filter_objects(self):
        self.object_1c = cropCenter(self.object1_processed, self.FRC_obj_size, shift_x=self.FRC_sx, shift_y=self.FRC_sy)
        self.object_2c = cropCenter(self.object2_processed, self.FRC_obj_size, shift_x=self.FRC_sx, shift_y=self.FRC_sy)

    def show_clipped_objects(self):
        fig, axes = plt.subplots(1, 2)
        axes = axes.flatten()
        fig.suptitle('clipped objects')
        axes[0].imshow(complex2rgb(self.object_1c))
        axes[0].set_axis_off()
        axes[1].imshow(complex2rgb(self.object_2c))
        axes[1].set_axis_off()
        fig.tight_layout()
        fig.show()

    def compute_custom_frc(self, mask, fft_image1, fft_image2):
        if np.sum(mask) == 0:
            return  0
        else:
            num = np.sum(np.conj(fft_image1[mask]) * fft_image2[mask])
            den = np.sqrt(
                np.sum(np.abs(fft_image1[mask]) ** 2) *
                np.sum(np.abs(fft_image2[mask]) ** 2)
            )
            return num / den if den != 0 else 0

    def calculateFRC_bk(self):
        y_shape = self.object_1c.shape[0]
        x_shape = self.object_1c.shape[1]
        R = self.object_1c.shape[0] / 2

        # Create coordinate grids
        center_y, center_x = y_shape // 2, x_shape // 2
        Y, X = np.indices((y_shape, x_shape))
        dx = X - center_x
        dy = Y - center_y
        r_grid = np.sqrt(dx ** 2 + dy ** 2)
        angle_grid = np.arctan2(dy, dx)  # angle in radians, range (-pi, pi]
        # Convert tolerance to radians
        tol = np.deg2rad(15)

        window_func = np.hanning(y_shape).reshape(1, -1) * np.hanning(x_shape).reshape(-1, 1)
        
        fft_image1 = fftshift(fft2(fftshift(self.object_1c * window_func)))
        fft_image2 = fftshift(fft2(fftshift(self.object_2c * window_func)))
        # fft_image1 /= np.max(np.abs(fft_image1))
        # fft_image2 /= np.max(np.abs(fft_image2))

        # Apply same normalization BEFORE using fft_image1 and fft_image2
        norm_factor = np.max(np.abs(fft_image2))  # or pick a consistent scale
        fft_image1 /= norm_factor
        fft_image2 /= norm_factor

        conj_image1 = np.conj(fft_image1)


        self.prtf = np.zeros(int(np.min([x_shape, y_shape]) / 2), dtype=float)  # new array for PRTF
        self.prtf_x1 = np.zeros_like(self.prtf)
        self.prtf_x2 = np.zeros_like(self.prtf)
        self.prtf_y = np.zeros_like(self.prtf)

        for r in range(int(R)):
            r += 1
            ring = ringfunc(r, x_shape, y_shape)
            ring_mask = ring.astype(bool)
            N = np.sum(ring_mask)

            # Compute radial average of |FFT_recon| and |FFT_measured|
            amp_recon = np.abs(fft_image1[ring_mask])
            amp_meas = np.abs(fft_image2[ring_mask])

            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            ratio = amp_recon / (amp_meas + epsilon)

            self.prtf[r - 1] = np.mean(ratio)

        self.prtf = 1/self.prtf


        self.frc = np.zeros(int(np.min([x_shape, y_shape]) / 2), dtype=complex)
        self.frc_y = np.zeros_like(self.frc)
        self.frc_x1 = np.zeros_like(self.frc)
        self.frc_x2 = np.zeros_like(self.frc)
        self.radial_ft1 = np.zeros_like(self.frc)
        self.radial_ft2 = np.zeros_like(self.frc)

        self.one_bit_crit = np.zeros_like(self.frc)
        self.half_bit_crit = np.zeros_like(self.frc)
    
        for r in range(int(R)):
            r += 1
            ring = ringfunc(r, x_shape, y_shape)
            ring_mask = ring.astype(bool)
            self.one_bit_crit[r-1] = one_bit_criterion(np.sum(ring))
            self.half_bit_crit[r-1] = half_bit_criterion(np.sum(ring))
            # check complex value/real value
            self.frc[r-1] = np.sum(ring * conj_image1 * fft_image2) / np.sqrt(np.sum(ring * np.abs(fft_image1)**2) * np.sum(ring * np.abs(fft_image2)**2))

            # X-axis slit: select pixels with angles near 0 or π.
            # (i.e. Fourier components along the horizontal axis)
            slit_mask_x1 = ring_mask  & (np.abs(angle_grid) < tol)
            slit_mask_x2 = ring_mask  & (np.abs(np.pi - np.abs(angle_grid)) < tol)
            # Y-axis slit: select pixels with angles near π/2 or -π/2.
            # (i.e. Fourier components along the vertical axis)
            slit_mask_y = ring_mask  & (
                    (np.abs(angle_grid - np.pi / 2) < tol) | (np.abs(angle_grid + np.pi / 2) < tol)
            )
            self.frc_x1[r - 1] = self.compute_custom_frc(slit_mask_x1, fft_image1, fft_image2)
            self.frc_x2[r - 1] = self.compute_custom_frc(slit_mask_x2, fft_image1, fft_image2)
            self.frc_y[r - 1] = self.compute_custom_frc(slit_mask_y, fft_image1, fft_image2)

            # 🔍 Add: Radially averaged FT magnitude
            self.radial_ft1[r - 1] = np.mean(np.abs(fft_image1[ring_mask]))
            self.radial_ft2[r - 1] = np.mean(np.abs(fft_image2[ring_mask]))

    def compute_custom_prtf(self, mask, fft_recon, fft_meas):
        if np.sum(mask) == 0:
            return 0
        amp_recon = np.abs(fft_recon[mask])
        amp_meas = np.abs(fft_meas[mask])
        epsilon = 1e-10
        ratio = amp_recon / (amp_meas + epsilon)
        return 1 / np.mean(ratio)

    def calculateFRC(self, mode='complex'):
        """
        Compute FRC and PRTF between object_1c and object_2c.

        Parameters
        ----------
        mode : str, optional
            Which component of the complex field to use.
            Options: 'complex' (default), 'amplitude', 'phase'
        """
        self.clip_filter_objects()
        y_shape, x_shape = self.object_1c.shape
        R = y_shape / 2

        center_y, center_x = y_shape // 2, x_shape // 2
        Y, X = np.indices((y_shape, x_shape))
        dx = X - center_x
        dy = Y - center_y
        r_grid = np.sqrt(dx ** 2 + dy ** 2)
        angle_grid = np.arctan2(dy, dx)
        tol = np.deg2rad(15)

        # Window function
        window_func = np.hanning(y_shape).reshape(1, -1) * np.hanning(x_shape).reshape(-1, 1)

        # Choose image input based on mode
        if mode == 'amplitude':
            im1 = np.abs(self.object_1c)
            im2 = np.abs(self.object_2c)
        elif mode == 'phase':
            im1 = self.object_1c/np.abs(self.object_1c)
            im2 = self.object_2c/np.abs(self.object_2c)

        elif mode == 'complex':
            im1 = self.object_1c
            im2 = self.object_2c

        else:
            raise ValueError("Invalid mode. Choose from 'complex', 'amplitude', or 'phase'.")

        # FFTs
        fft_image1 = fftshift(fft2(fftshift(im1 * window_func)))
        fft_image2 = fftshift(fft2(fftshift(im2 * window_func)))

        # amp1 = np.abs(im1)
        # amp2 = np.abs(im2)
        # threshold = 0.1 * max(np.amax(amp1), np.amax(amp2))
        # mask = (amp1 > threshold) & (amp2 > threshold)
        #
        # # Multiply both with mask before FFT
        # fft_image1 = fftshift(fft2(fftshift(im1 * mask * window_func)))
        # fft_image2 = fftshift(fft2(fftshift(im2 * mask * window_func)))

        # fft_image1 = np.fft.fft2(im1*window_func, norm='ortho')
        # fft_image2 = np.fft.fft2(im2*window_func, norm='ortho')

        norm_factor1 = np.max(np.abs(fft_image1))
        norm_factor2 = np.max(np.abs(fft_image2))

        fft_image1 /= norm_factor1
        fft_image2 /= norm_factor1
        conj_image1 = np.conj(fft_image1)

        n_rings = int(np.min([x_shape, y_shape]) / 2)

        # Initialize outputs
        self.frc = np.zeros(n_rings, dtype=complex)
        self.frc_x1 = np.zeros_like(self.frc)
        self.frc_x2 = np.zeros_like(self.frc)
        self.frc_y = np.zeros_like(self.frc)

        self.prtf = np.zeros_like(self.frc)
        self.prtf_x1 = np.zeros_like(self.frc)
        self.prtf_x2 = np.zeros_like(self.frc)
        self.prtf_y = np.zeros_like(self.frc)

        self.radial_ft1 = np.zeros_like(self.frc)
        self.radial_ft2 = np.zeros_like(self.frc)

        self.one_bit_crit = np.zeros_like(self.frc)
        self.half_bit_crit = np.zeros_like(self.frc)

        for r in range(n_rings):
            r_index = r + 1
            ring = ringfunc(r_index, x_shape, y_shape)
            ring_mask = ring.astype(bool)

            N = np.sum(ring_mask)
            self.one_bit_crit[r] = one_bit_criterion(N)
            self.half_bit_crit[r] = half_bit_criterion(N)

            # FRC computation
            self.frc[r] = np.sum(ring * conj_image1 * fft_image2) / np.sqrt(
                np.sum(ring * np.abs(fft_image1) ** 2) * np.sum(ring * np.abs(fft_image2) ** 2)
            )

            # Radial magnitudes
            self.radial_ft1[r] = np.mean(np.abs(fft_image1[ring_mask]))
            self.radial_ft2[r] = np.mean(np.abs(fft_image2[ring_mask]))

            # Global PRTF (1/PRTF convention)
            amp_recon = np.abs(fft_image1[ring_mask])
            amp_meas = np.abs(fft_image2[ring_mask])
            epsilon = 1e-10
            ratio = amp_recon / (amp_meas + epsilon)
            self.prtf[r] = 1 / np.mean(ratio)

            # Directional masks (X and Y)
            slit_mask_x1 = ring_mask & (np.abs(angle_grid) < tol)
            slit_mask_x2 = ring_mask & (np.abs(np.pi - np.abs(angle_grid)) < tol)
            slit_mask_y = ring_mask & (
                    (np.abs((np.pi / 2) - angle_grid) < tol) | (np.abs((-np.pi / 2) - angle_grid) < tol)
            )

            # Directional FRC and PRTF
            self.frc_x1[r] = self.compute_custom_frc(slit_mask_x1, fft_image1, fft_image2)
            self.frc_x2[r] = self.compute_custom_frc(slit_mask_x2, fft_image1, fft_image2)
            self.frc_y[r] = self.compute_custom_frc(slit_mask_y, fft_image1, fft_image2)

            self.prtf_x1[r] = self.compute_custom_prtf(slit_mask_x1, fft_image1, fft_image2)
            self.prtf_x2[r] = self.compute_custom_prtf(slit_mask_x2, fft_image1, fft_image2)
            self.prtf_y[r] = self.compute_custom_prtf(slit_mask_y, fft_image1, fft_image2)

    def plotFRC(self):
        n_ticks = 5
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=150)
        fig.suptitle(f'FRC')
        ax.plot(self.frc, label=f'FRC')
        # ax.plot(self.frc_x1, label=f'FRC_x1')
        # ax.plot(self.frc_x2, label=f'FRC_x2')
        # ax.plot(self.frc_y, label=f'FRC_y')
        # ax.plot(np.abs(np.angle(self.frc)), label=f'FRC_angle')
        # ax.plot(np.abs(self.frc), label=f'FRC_abs')
        ax.plot(self.one_bit_crit, '--', label='1 bit')
        ax.plot(self.half_bit_crit, '--', label='1/2 bit')
        # ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='PRTF = 0.5')

        self.qmax = 1 / (2 * self.dx * 1e6)
        self.Fx = np.round(np.linspace(0, self.qmax, n_ticks), decimals=3)
        plt.xticks(np.linspace(0, len(self.frc), n_ticks), labels=self.Fx)
        ax.set_ylabel('FRC')
        ax.set_xlabel(r'Spatial freq. ($\mu m^{-1}$)')
        ax.grid(alpha=0.5)
        ax.minorticks_on()
        ax.legend()
        fig.tight_layout()
        fig.show()

    def plotPRTF(self):
        n_ticks = 5
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=150)
        fig.suptitle('PRTF')

        # Plot global and directional PRTFs
        ax.plot(self.prtf, label='PRTF')
        # ax.plot(self.prtf_x1, label='PRTF_x1')
        # ax.plot(self.prtf_x2, label='PRTF_x2')
        # ax.plot(self.prtf_y, label='PRTF_y')

        # Add threshold line
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='PRTF = 0.5')

        # Set spatial frequency ticks
        self.qmax = 1 / (2 * self.dx * 1e6)
        self.Fx = np.round(np.linspace(0, self.qmax, n_ticks), decimals=3)
        plt.xticks(np.linspace(0, len(self.prtf), n_ticks), labels=self.Fx)

        ax.set_ylabel('PRTF')
        ax.set_xlabel(r'Spatial freq. ($\mu m^{-1}$)')
        ax.grid(alpha=0.5)
        ax.minorticks_on()
        ax.legend()
        fig.tight_layout()
        fig.show()

    def get_FRC_plot_data(self):
        return self.frc, self.half_bit_crit, self.dx

    def get_spatial_resolution_prtf(self):
        # Build your spatial frequency axis
        qm = np.linspace(0, self.qmax, self.prtf.size)

        # Find the first radial-frequency index where PRTF < 0.5
        below = np.where(self.prtf < 0.5)[0]
        if below.size > 0:
            idx = below[0]
        else:
            # never falls below 0.5 → use the highest frequency
            idx = qm.size - 1

        # resolution = 1 / (2 * q)
        res = 1.0 / (2.0 * qm[idx])
        print(f'PRTF-based resolution: {res:.3f} µm')
        return res

    def get_spatial_resolution(self):
        # Build your spatial frequency axis
        qm = np.linspace(0, self.qmax, self.frc.size)

        # Find the first index where FRC < half-bit criterion
        below = np.where(self.frc < self.half_bit_crit)[0]

        if below.size > 0:
            idx = below[0]
        else:
            # never falls below criterion → use highest freq
            idx = qm.size - 1

        res = 1.0 / (2.0 * qm[idx])
        return res

    def get_res(self, index, qm):
        if len(index) > 0:
            index = index[0]
            res = 1 / (2 * qm[index])  # (um)
        else:
            res = 1 / (2 * qm[-1])  # (um)
        print(f'resolution: {res} um')
        return res

    def get_spatial_resolution_individual(self):
        # half bit criteria
        qm = np.linspace(0, self.qmax, self.frc.shape[-1])
        index = np.argwhere(self.frc < self.half_bit_crit)
        index_x1 = np.argwhere(self.frc_x1 < self.half_bit_crit)
        index_x2 = np.argwhere(self.frc_x2 < self.half_bit_crit)
        index_y = np.argwhere(self.frc_y < self.half_bit_crit)

        res = self.get_res(index, qm)
        res_x1 = self.get_res(index_x1, qm)
        res_x2 = self.get_res(index_x2, qm)
        res_y = self.get_res(index_y, qm)

        return res, res_x2, res_x1, res_y


def tappering_window(array, method='fraction', taper_value=0.3):
    """
    Apply Hanning tapering to the edges of a 2D array.

    Parameters:
        array (2D np.ndarray): The input array to be tapered.
        method (str): The tapering method. Options are 'fraction' 'pixels' or 'fixed'.
                      'fraction' applies tapering based on a fraction of the array size.
                      'pixels' applies tapering based on a fixed number of pixels.
                      'fixed' applies a binary threshold on the fixed number of pixels.
        taper_value (float or int): The amount of tapering. If 'fraction', this is the
                                    fraction of the array to taper (e.g., 0.3 for 30%).
                                    If 'pixels', this is the number of pixels to taper at the edges.

    Returns:
        np.ndarray: The tapered array.
    """
    rows, cols = array.shape

    if method == 'fraction':
        # Fraction-based tapering
        taper_row = np.hanning(int(rows * taper_value))
        taper_col = np.hanning(int(cols * taper_value))

        # Pad with ones for the inner region
        pad_row = np.ones(rows - 2 * len(taper_row) // 2)
        pad_col = np.ones(cols - 2 * len(taper_col) // 2)

        hanning_row = np.concatenate([taper_row[:len(taper_row) // 2], pad_row, taper_row[len(taper_row) // 2:]])
        hanning_col = np.concatenate([taper_col[:len(taper_col) // 2], pad_col, taper_col[len(taper_col) // 2:]])

    elif method == 'pixels':
        # Pixel-based tapering
        taper_row = np.hanning(taper_value * 2)  # Create Hanning window for the number of pixels
        taper_col = np.hanning(taper_value * 2)

        # Pad with ones in the center
        pad_row = np.ones(rows - taper_value * 2)
        pad_col = np.ones(cols - taper_value * 2)

        hanning_row = np.concatenate([taper_row[:taper_value], pad_row, taper_row[taper_value:]])
        hanning_col = np.concatenate([taper_col[:taper_value], pad_col, taper_col[taper_value:]])

    elif method == 'fixed':
        # Ensure param is an integer
        border_width = int(taper_value)
        # Set the borders to zero
        array[:border_width, :] = 0  # Top border
        array[-border_width:, :] = 0  # Bottom border
        array[:, :border_width] = 0  # Left border
        array[:, -border_width:] = 0  # Right border

        return array

    else:
        raise ValueError("Invalid method. Use 'fraction' 'pixels' or 'fixed'.")

    # Create a 2D Hanning window using outer product
    hanning_2d = np.outer(hanning_row, hanning_col)

    # Apply the tapering by element-wise multiplication
    return array * hanning_2d



def generate_rectangular_spiral_indices(rows, cols):
    indices = []

    top = 0
    bottom = rows - 1
    left = 0
    right = cols - 1

    while top <= bottom and left <= right:
        # Move right along the top row
        for j in range(left, right + 1):
            indices.append((top, j))

        # Move down along the right column (excluding the top element)
        for i in range(top + 1, bottom + 1):
            indices.append((i, right))

        # Move left along the bottom row (excluding the rightmost element)
        if top < bottom:
            for j in range(right - 1, left - 1, -1):
                indices.append((bottom, j))

        # Move up along the left column (excluding the bottom element)
        if left < right:
            for i in range(bottom - 1, top, -1):
                indices.append((i, left))

        top += 1
        bottom -= 1
        left += 1
        right -= 1

    return np.array(indices)[::-1]

def find_closest_to_center(points, center=None):
    if center is None:
        center = np.mean(points, axis=0)

    centered_points = points - center
    distances = np.linalg.norm(centered_points, axis=1)
    closest_index = np.argmin(distances)
    return closest_index

def com_from_2sets(cluster1, cluster2):
    # Compute the center of mass of each cluster
    com1 = np.mean(cluster1, axis=0)
    com2 = np.mean(cluster2, axis=0)
    # return (com1+com2)/2

    # Compute the weighted center of mass of the two clusters
    total_mass = cluster1.shape[0] + cluster2.shape[0]
    weighted_com = (com1 * cluster1.shape[0] + com2 * cluster2.shape[0]) / total_mass
    return weighted_com

def divide_scan_and_optimize_tsp(xy, J, K):
    # Find the minimum and maximum values of x and y
    min_x = np.min(xy[:, -1])
    max_x = np.max(xy[:, -1])
    min_y = np.min(xy[:, -2])
    max_y = np.max(xy[:, -2])

    # Calculate the width and height of each subset in the grid
    subset_width = (max_x - min_x) / J
    subset_height = (max_y - min_y) / K

    # Create an empty dictionary to hold the points in each subset
    subsets = []

    # sets the indices into an s pattern
    indices = generate_rectangular_spiral_indices(J, K)
    # Loop over each subset in the grid
    for indice in indices:
        i = indice[0]
        j = indice[1]
        # Calculate the boundaries of the subset
        subset_x_min = min_x + i * subset_width
        subset_x_max = min_x + (i + 1) * subset_width
        subset_y_min = min_y + j * subset_height
        subset_y_max = min_y + (j + 1) * subset_height
        print('----')
        print(f'{i}{j}')
        print(subset_x_min)
        print(subset_x_max)
        print(subset_y_min)
        print(subset_y_max)
        print('----')
        # Find the indices of the points that fall within the subset

        subset_indices = np.where((xy[:, -1] >= subset_x_min) &
                                  (xy[:, -1] <= subset_x_max) &
                                  (xy[:, -2] >= subset_y_min) &
                                  (xy[:, -2] <= subset_y_max))[0]

        # Get the subset points using the indices
        subset_points = xy[subset_indices]
        # Add the subset points to the dictionary
        subsets.append(subset_points)

    # finds the center of the first subset and set it to be first position:
    c_index = find_closest_to_center(subsets[0])
    center = (subsets[0][c_index]) + 0
    subsets[0][c_index] = subsets[0][0]
    subsets[0][0] = center

    # finds the closest point for each neighbor cluster and set it to the end/initial point
    for i in range(len(subsets) - 1):
        weighted_com = com_from_2sets(subsets[i], subsets[i + 1])
        c_index1 = find_closest_to_center(subsets[i], weighted_com)
        c_index2 = find_closest_to_center(subsets[i + 1], weighted_com)
        # replace c_index1 at the end of its cluster
        temp_closest = (subsets[i][c_index1]) + 0
        subsets[i][c_index1] = subsets[i][-1] + 0
        subsets[i][-1] = temp_closest + 0
        # #replace c_index2 at the beginning of its cluster
        temp_closest = (subsets[i + 1][c_index2]) + 0
        subsets[i + 1][c_index2] = subsets[i + 1][0]
        subsets[i + 1][0] = temp_closest + 0

    # creates a multiprocessing pool
    pool = multiprocessing.Pool()

    # Apply the tbs function to each element in the list using map_async
    results = pool.map_async(apply_tsp, subsets)

    # get the results as a list
    subsets = results.get()

    # close the pool
    pool.close()

    # Stack the subsets together
    xy = np.concatenate(subsets, axis=0)
    # use the center to re-align the scanning grid
    xy -= center

    return xy

def apply_tsp(xy):
    # new_xy = tsp(xy)
    # new_xy = new_xy[:-1,:]
    # new_route = two_opt(new_xy)
    # new_xy = new_xy[new_route]

    # R = xy[:, 0]
    # C = xy[:, 1]
    # pre_route = precondition_route(R, C)
    # Rnew = R[pre_route]
    # Cnew = C[pre_route]
    # xy = np.vstack((Rnew, Cnew)).T
    new_route = two_opt(xy)
    new_xy = xy[new_route]
    return new_xy


def two_opt(cities): # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
    route = np.arange(cities.shape[0]) # Make an array of row numbers corresponding to cities.
    improvement_factor = 1 # Initialize the improvement factor.
    best_distance = path_distance(route,cities) # Calculate the distance of the initial path.
    i = 0
    progress = True
    while progress: # If the route is still improving, keep going!
        progress = False
        i += 1
        print(f'{i} {best_distance}', '.' * i, end='\r')
        for swap_first in range(1,len(route)-1): # From each city except the first and last,
            for swap_last in range(swap_first+1,len(route)-1): # to each of the cities following,
                new_route = two_opt_swap(route,swap_first,swap_last) # try reversing the order of these cities
                new_distance = path_distance(new_route,cities) # and check the total distance with this modification.
                if new_distance < best_distance: # If the path distance is an improvement,
                    route = new_route # make this the accepted best route
                    best_distance = new_distance # and update the distance corresponding to this route.
                    progress=True
    return route # When the route is no longer improving, stop searching and return the route.

path_distance = lambda r,c: np.sum([np.linalg.norm(c[r[p]]-c[r[p-1]]) for p in range(len(r))])

two_opt_swap = lambda r,i,k: np.concatenate((r[0:i],r[k:-len(r)+i-1:-1],r[k+1:len(r)]))


def farthest_point_sampling(points, subset_size):
    """
    Selects a subset of points using farthest point sampling.

    Parameters:
        points (np.array): Array of shape (n_points, n_dimensions).
        subset_size (int): Number of points to sample.

    Returns:
        np.array: Subset of points.
    """
    n_points = len(points)
    if subset_size >= n_points:
        return points

    # Initialize: choose a random starting index
    indices = [np.random.randint(0, n_points)]
    indices = [0]
    # Initialize distance array with infinity
    distances = np.full(n_points, np.inf)

    for _ in range(1, subset_size):
        last_selected = points[indices[-1]]
        # Update distances: compute the distance from the last selected point to all points
        d = np.linalg.norm(points - last_selected, axis=1)
        # Keep the minimum distance to any selected point so far
        distances = np.minimum(distances, d)
        # Select the next point as the one with the maximum distance to the selected set
        next_index = np.argmax(distances)
        indices.append(next_index)

    return points[indices], indices

def select_subset_indices(points, size, TSP=False):
    set, indices = farthest_point_sampling(points, size)
    if TSP:
        route = two_opt(set)
        return np.array(indices)[route.astype(int)]
    else:
        return np.array(indices)
def calculate_distances(points):
    # Function to calculate Euclidean distances between consecutive points
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    return distances

def calcuate_avg_step_size(points):
    """
    :param points: encoder points [npos, 2]
    :return: avg step-size, std
    """
    distances = calculate_distances(points)
    return np.mean(distances), np.std(distances)


class myPtychoSetup:
    def __init__(self, wavelength, z0, dq, N, theta=0):
        self.wavelength= wavelength
        self.z0=z0
        self.dq=dq
        self.N=N
        self.theta=theta
        self.dx = wavelength*z0/(N*dq)
        self.Lq = N*dq
        self.k0 = 2*np.pi/wavelength
        self.NAd = (N*dq/2)/z0

    def get_DoF(self):
        """expected Depth of field"""
        # DoF = self.wavelength / self.NAd ** 2
        DoF = 5.2 * self.dx**2 / self.wavelength

        return DoF

    def get_z_tolerance(self, r):
        """
        min distance that shifts the r-position a single pixel shift
        :param r: farthest eucledian distance from origin
        :return:
        """
        dz = (self.wavelength * self.z0 ** 2) / (r * self.Lq - self.wavelength * self.z0)
        return dz

    def get_theta_tolerance(self, theta):
        """
        computes the change in theta that will cause the outhermost detected frequency
        to shift one single frequency spacing delta_Fq in the target regular frequency spacing grid
        :return: \delta_theta
        """
        theta = np.deg2rad(theta)
        x = np.arange(-self.N/2, self.N/2) * self.dq
        z = np.zeros_like(x) + self.z0
        r = np.sqrt(x ** 2 + z ** 2)
        Fq_x = self.k0 * x / r
        Fq_uniform_x = np.linspace(np.amin(Fq_x), np.amax(Fq_x), self.N)
        delta_Fq = Fq_uniform_x[-1] - Fq_uniform_x[-2]

        # derivative of X_t = (L/2) cos(theta) + z sin(theta)  #this is the rotation transformation
        # d/dtheta [X_t/r] = dX_t_dtheta / r, since r is constant
        dX_t_dtheta = - (self.Lq / 2) * np.sin(theta) + self.z0 * np.cos(theta)
        # d_term = dX_t_dtheta / r
        d_term = dX_t_dtheta / self.z0
        dQ_dtheta = self.k0 * (d_term - np.cos(theta))
        # Predicted delta_theta (radians) such that Q_xt shifts by Delta_qu:
        delta_theta_pred = delta_Fq / np.abs(dQ_dtheta)
        return np.rad2deg(delta_theta_pred)


    def get_beta_tolerance(self, r):
        """
        Computes the min beta angle that the sample's plane can be missmatched
        to avoid a pixe-shif coordinates
        :param r:
        :return:
        """
        beta = np.arctan(np.sqrt(2*self.dx/r))
        return np.rad2deg(beta)

    def evaluate_beta_alpha_tol(self, x, beta, alpha=0, y=0, theta=0):
        """
        evaluates if the given alpha, beta result in a pixel shif modification of
        the input coordinate
        :param x: max x_pos in meters
        :param beta: tilt angle in degrees around y-axis
        :param alpha: tilt angle in degrees around x-axis
        :param y: max y_pos in meters
        :return: True, if condition is met, meaning remapping of positions is needed
        """
        r = np.sqrt(x**2 + y**2)
        condition= 2*r*self.dx
        if theta==0: #perpendicular probe incidence
            lhs = x**2*np.tan(np.deg2rad(beta))**2 + y**2*np.tan(np.deg2rad(alpha))**2
        else:
            dz = self.get_dz_eff(x,y,beta,alpha,theta)
            lhs = ((x+ dz*np.tan(np.deg2rad(theta)))/np.cos(np.deg2rad(beta)))**2 - x**2 + y**2*np.tan(np.deg2rad(alpha))**2
        return lhs >= condition, lhs/condition

    def get_dz_eff(self, x,beta,y=0,alpha=0,theta=0):
        """
        evaluates the dz-distance offset needed to compensate the propagation of the probe
        due to tilted surface geometry
        :param x: pos in meters
        :param beta: tilt angle around y (meters)
        :param y: pos in meters
        :param alpha: tilt angle around x (meters)
        :param theta: incidence angle probe (degrees)
        :return:
        """
        num = -x*np.tan(np.deg2rad(beta))-y*np.tan(np.deg2rad(alpha))
        den = 1 + np.tan(np.deg2rad(theta))*np.tan(np.deg2rad(beta))

        return num/den

def beatufy_axes(ax):
    # Hide top and left spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Move bottom and right spines outward a bit
    # ax.spines['bottom'].set_position(('outward', 5))
    ax.spines['right'].set_position(('outward', 5))

    # X-axis triangle arrowhead (→), at right edge
    ax.arrow(
        1, 0,  # x, y start (axes fraction)
        0.02, 0,  # dx, dy (small x-shift)
        head_width=0.02,
        head_length=0.03,
        fc='black', ec='black',
        transform=ax.transAxes,
        length_includes_head=True,
        clip_on=False
    )

    # Y-axis triangle arrowhead (↑), at top edge
    ax.arrow(
        0, 1,  # x, y start (axes fraction)
        0, 0.02,  # dx, dy (small y-shift)
        head_width=0.02,
        head_length=0.03,
        fc='black', ec='black',
        transform=ax.transAxes,
        length_includes_head=True,
        clip_on=False
    )
def plot_hist(myarray, title='', x_label='', cmap='RdBu_r', bins=None, step=0.335e-9, savename=None):
    """
    Plot a histogram where each bar is colored by its bin center using a colormap.

    Parameters
    ----------
    myarray : array-like
        Data to histogram.
    title : str
        Plot title.
    x_label : str
        X-axis label.
    cmap : str or matplotlib.colors.Colormap
        Colormap name (e.g. 'viridis', 'RdBu_r') or a Colormap instance.
    bins : int or sequence of scalars, optional
        If given, passed directly to np.histogram. If None, bins are computed
        so each bin has width == `step`.
    step : float
        Target bin width. Must be in the same units as `myarray`. Default 0.335e-9.
    savename : str or Path-like, optional
        If provided, save the figure to this path.
    """
    pc1_flat = np.ravel(myarray)
    pc1_flat = pc1_flat[~np.isnan(pc1_flat)]  # ignore NaNs

    if pc1_flat.size == 0:
        raise ValueError("`myarray` contains no finite values.")

    # Build step-aligned bin edges if bins not specified
    if bins is None:
        if step <= 0:
            raise ValueError("`step` must be positive.")
        data_min = np.min(pc1_flat)
        data_max = np.max(pc1_flat)

        if data_min == data_max:
            # Degenerate case: single-valued data -> make one bin centered on the value
            half = step / 2.0
            bins = np.array([data_min - half, data_min + half])
        else:
            # Align edges to multiples of `step` so every bin width == step
            start = np.floor(data_min / step) * step
            stop  = np.ceil(data_max  / step) * step
            # Ensure the rightmost edge includes the max
            bins = np.arange(start, stop + step * 0.5, step)

    # Histogram data
    hist_vals, bin_edges = np.histogram(pc1_flat, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Resolve the colormap
    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)
    elif not isinstance(cmap, matplotlib.colors.Colormap):
        raise TypeError("`cmap` must be a string colormap name or a Colormap instance.")

    # Normalize for colormap scaling
    norm = matplotlib.colors.Normalize(vmin=bin_centers.min(), vmax=bin_centers.max())
    colors = cmap(norm(bin_centers))

    fig, ax = plt.subplots(figsize=(6.6, 4))

    bar_width = (bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else step
    for i in range(len(hist_vals)):
        ax.bar(bin_centers[i], hist_vals[i],
               width=bar_width,
               color=colors[i], edgecolor='gray', linewidth=0.5)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of pixels")
    ax.grid(True, alpha=0.5, linestyle=':')
    ax.set_axisbelow(True)

    beatufy_axes(ax)  # keeping your helper as-is

    fig.tight_layout()

    if savename is not None:
        fig.savefig(savename, dpi=300, bbox_inches='tight')

    plt.show()
    return fig, ax, hist_vals, bin_centers

def plot_hist2(myarray, title='', x_label='', cmap='RdBu_r', bins=100, savename=None):
    """
    Plot a histogram where each bar is colored by its bin center using a colormap.

    Parameters
    ----------
    myarray : array-like
        Data to histogram.
    title : str
        Plot title.
    x_label : str
        X-axis label.
    cmap : str or matplotlib.colors.Colormap
        Colormap **name** (e.g. 'viridis', 'RdBu_r') or a Colormap instance.
    bins : int
        Number of histogram bins.
    savename : str or Path-like, optional
        If provided, save the figure to this path.
    """
    pc1_flat = np.ravel(myarray)

    # Histogram data
    hist_vals, bin_edges = np.histogram(pc1_flat, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Resolve the colormap
    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)
    elif not isinstance(cmap, matplotlib.colors.Colormap):
        raise TypeError("`cmap` must be a string colormap name or a Colormap instance.")

    # Normalize for colormap scaling
    norm = matplotlib.colors.Normalize(vmin=bin_centers.min(), vmax=bin_centers.max())
    colors = cmap(norm(bin_centers))

    fig, ax = plt.subplots(figsize=(6.6, 4))

    bar_width = (bin_edges[1] - bin_edges[0])
    for i in range(len(hist_vals)):
        ax.bar(bin_centers[i], hist_vals[i],
               width=bar_width,
               color=colors[i], edgecolor='gray', linewidth=0.5)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of pixels")
    ax.grid(True, alpha=0.5, linestyle=':')
    ax.set_axisbelow(True)

    beatufy_axes(ax)

    fig.tight_layout()

    if savename is not None:
        fig.savefig(savename, dpi=300, bbox_inches='tight')

    plt.show()

def plot_with_scalebar(data, dx=1, scalebar_size=10, cmap='RdBu', title='',
                       vmin=None, vmax=None, scale_bar_color='black',label_cbar='',
                       show_color_bar=True, dpi=100, save_path='', **kwargs):

    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)
    elif not isinstance(cmap, matplotlib.colors.Colormap):
        raise TypeError("`cmap` must be a string colormap name or a Colormap instance.")

    fig, ax = plt.subplots(dpi=dpi)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    ax.set_title(title)
    # Add scale bar (e.g., 5 µm corresponds to N pixels)
    bar_length_pixels = int(scalebar_size / dx)  # Convert to pixels
    scalebar = AnchoredSizeBar(ax.transData,
                               size=bar_length_pixels,
                               label=f'{scalebar_size*1e6:.1f} µm',
                               loc='lower right',
                               pad=0.5,
                               color=scale_bar_color,
                               frameon=False,
                               size_vertical=int(data.shape[0]/50),
                               fontproperties=fm.FontProperties(size=12), )
    ax.add_artist(scalebar)
    ax.set_xticks([])  # Remove x tick labels
    ax.set_yticks([])  # Remove y tick labels
    # Add colorbar
    if show_color_bar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(label_cbar)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)  # Save without frame
    plt.show()

def get_binary_cmap(cmap):
    cmap = matplotlib.pyplot.get_cmap(cmap)
    N = 256  # number of colors
    colors = cmap(np.linspace(0, 1, N))

    # 3. Convert colors to grayscale (lightness)
    # Simple method: weighted average of RGB channels
    # Using standard luminance formula
    grayscale = np.dot(colors[:, :3], [0.299, 0.587, 0.114])

    # 4. Build new grayscale colormap
    # Expand back to RGB
    gray_colors = np.vstack([grayscale, grayscale, grayscale]).T
    gray_colors = np.hstack([gray_colors, colors[:, 3][:, np.newaxis]])  # preserve alpha channel

    binary_flag = LinearSegmentedColormap.from_list('binary_flag', gray_colors)
    return binary_flag


if __name__ == "__main__":
    pass
