import os
import numpy as np
from scipy.interpolate import interp1d
import xraydb

def _rs(n1, n2, cos_i, cos_t):
    return (n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t)
def _rp(n1, n2, cos_i, cos_t):
    return (n2*cos_i - n1*cos_t) / (n2*cos_i + n1*cos_t)
def _ts(n1, n2, cos_i, cos_t):
    return (2 * n1 * cos_i) / (n1 * cos_i + n2 * cos_t)
def _tp(n1, n2, cos_i, cos_t):
    return (2*n1*cos_i) / (n2*cos_i + n1*cos_t)

def _get_cos_t(n1,n2,sin_i):
    # Snell: n1*sin(theta_i) = n2*sin(theta_t) => cos(theta_t) = sqrt(1 - (n1/n2*sin)^2)
    eta = ((n1 / n2) * sin_i)#.astype(np.complex128)

    cos_t = np.sqrt(1.0 - eta ** 2)
    # Physical branch: ensure Im(cos_tt) >= 0
    if np.any(np.iscomplex(cos_t)):
        cos_t = np.where(np.imag(cos_t) < 0, -cos_t, cos_t)
    return cos_t

def _physical_sqrt(z):
    # sqrt with branch such that Im(sqrt) ≥ 0  (decaying into absorbing media)
    w = np.sqrt(np.asarray(z, dtype=np.complex128))
    return np.where(np.imag(w) < 0, -w, w)

def kz_physical(n, theta_from_normal_rad, k0):
    # kz = k0 * n * cos(theta_n), with cos theta_n chosen s.t. Im(kz) >= 0 (decay into lossy media)
    s0 = np.sin(theta_from_normal_rad)
    root = np.sqrt(1.0 - (s0 / n)**2)       # cos(theta_n)
    # choose physical branch:
    root = np.where(np.imag(root) < 0, -root, root)
    kz = k0 * n * root
    return kz

def _kz_physical(n, theta_from_normal_rad, k0):
    """kz = k0 * n * cos(theta_n), choose physical branch (Im kz >= 0)."""
    s0 = np.sin(theta_from_normal_rad)
    cosn = np.sqrt(1.0 - (s0/n)**2)
    cosn = np.where(np.imag(cosn) < 0, -cosn, cosn)
    return k0 * n * cosn

def fresnel_rs_rp(n1, n2, theta_rad):
    """
    Fresnel reflection coefficients r_s, r_p for a single flat interface n1 -> n2 (possibly complex).
    Vectorized in theta and n2 (broadcasting rules apply).
    We pick the physical branch for cos(theta_t) so that Im(cos_t) >= 0.
    theta in rad
    """
    theta = np.asarray(theta_rad)
    sin_i = np.sin(theta)
    cos_i = np.cos(theta)

    cos_t = _get_cos_t(n1,n2,sin_i)

    rs = _rs(n1,n2, cos_i, cos_t)
    rp = _rp(n1,n2, cos_i, cos_t)

    return rs, -rp

class element_crxo:
    def __init__(self, element):
        self.element, self.rho = xraydb.get_material(element)

    def get_rho(self):
        return self.rho
    def get_n(self, energy):
        delta, beta, _ = xraydb.xray_delta_beta(self.element, self.rho, energy)
        n = 1-delta + 1j*beta
        return n
    def get_delta_beta(self, energy):
        delta, beta, _ = xraydb.xray_delta_beta(self.element, self.rho, energy)
        return delta, beta

    def get_rs_rp(self, angle_deg, energy):
        n = self.get_n(energy)
        rs, rp = fresnel_rs_rp(n , np.deg2rad(angle_deg))
        return rs, rp

    def get_rs(self, angle_deg, energy):
        n = self.get_n(energy)
        rs, _ = fresnel_rs_rp(1, n, np.deg2rad(angle_deg))
        return rs

    def get_rp(self, angle_deg, energy):
        n = self.get_n(energy)
        _, rp = fresnel_rs_rp(1, n, np.deg2rad(angle_deg))
        return rp

def reflectivity_parratt_bk(layers, wavelength, angle_deg, pol='s', angle_mode='grazing'):
    """
    layers: [(n, d), ...] including ambient at 0 and substrate at -1, with d=None for semi-infinite.
    wavelength in meters, angle_deg is the *grazing* angle α (CXRO’s convention).
    Returns complex reflection amplitude r.
    """
    k0 = 2*np.pi / wavelength
    if angle_mode == 'normal':
        alpha = np.deg2rad(90-angle_deg)
    else:
        alpha = np.deg2rad(angle_deg)             # grazing
    cos_a = np.cos(alpha)
    kiz = k0*np.sin(alpha)                     # vacuum

    n = np.array([n_ for n_, _ in layers], dtype=np.complex128)
    d = [d_ for _, d_ in layers]

    # kz in each layer the CXRO way
    kz = k0 * np.sqrt(n**2 - cos_a**2)

    # start at substrate interface (last film -> substrate)
    # build only through the finite-thickness films; the semi-infinite ones have d=None
    # layers: 0 (ambient, None), 1..N-2 films (finite d), N-1 substrate (None)
    N = len(layers)

    # Fresnel at substrate interface
    if pol == 's':
        r = (kz[N-2] - kz[N-1])/(kz[N-2] + kz[N-1])
    else:  # 'p'
        # substrate interface (N-2 → N-1)
        r = (kz[N - 2] / (n[N - 2] ** 2) - kz[N - 1] / (n[N - 1] ** 2)) / \
            (kz[N - 2] / (n[N - 2] ** 2) + kz[N - 1] / (n[N - 1] ** 2))

    # recurse upward through films i = N-3,...,1
    for i in range(N-3, 0, -1):
        if pol == 's':
            r_i = (kz[i] - kz[i+1])/(kz[i] + kz[i+1])
        else:
            r_i = (kz[i] / (n[i] ** 2) - kz[i + 1] / (n[i + 1] ** 2)) / \
                  (kz[i] / (n[i] ** 2) + kz[i + 1] / (n[i + 1] ** 2))

        p2 = np.exp(2j * kz[i+1] * d[i+1])     # phase in the layer below (i+1)
        r  = (r_i + r*p2) / (1 + r_i*r*p2)

    # surface (ambient -> first film) and top film propagation
    if pol == 's':
        r0 = (kiz - kz[1])/(kiz + kz[1])
    else:
        # r0 = (kiz - kz[1]/n[1])/(kiz + kz[1]/n[1])
        r0 = (kiz - kz[1] / (n[1] ** 2)) / (kiz + kz[1] / (n[1] ** 2))

    p2_top = np.exp(2j * kz[1] * d[1]) if d[1] is not None else 1.0
    r = (r0 + r*p2_top) / (1 + r0*r*p2_top)

    if pol=='p':
        r=r
    return r

def reflectivity_parratt(layers, wavelength, angle_deg, pol='s', angle_mode='grazing'):
    k0 = 2*np.pi / wavelength
    if angle_mode == 'normal':
        theta = np.deg2rad(angle_deg)
        alpha = np.deg2rad(90.0) - theta
    else:
        alpha = np.deg2rad(angle_deg)

    n = np.array([n_ for n_, _ in layers], dtype=np.complex128)
    d = [d_ for _, d_ in layers]
    N = len(layers)

    # CXRO-style kz with physical branch
    kz = k0 * _physical_sqrt(n**2 - np.cos(alpha)**2)

    kiz = k0*np.sin(alpha)

    # --- handle the single-interface case early ---
    if N == 2:
        if pol == 's':
            return (kiz - kz[1])/(kiz + kz[1])
        else:
            return (kiz - kz[1]/(n[1]**2)) / (kiz + kz[1]/(n[1]**2))

    # start at substrate interface (last finite -> substrate)
    if pol == 's':
        r = (kz[N-2] - kz[N-1])/(kz[N-2] + kz[N-1])
    else:
        b_up = kz[N-2]/(n[N-2]**2)
        b_dn = kz[N-1]/(n[N-1]**2)
        r = (b_up - b_dn)/(b_up + b_dn)

    # recurse upward through films i = N-3,...,1
    for i in range(N-3, 0, -1):
        if pol == 's':
            r_i = (kz[i] - kz[i+1])/(kz[i] + kz[i+1])
        else:
            b_i   = kz[i]/(n[i]**2)
            b_ip1 = kz[i+1]/(n[i+1]**2)
            r_i = (b_i - b_ip1)/(b_i + b_ip1)
        p2 = 1.0 if (d[i+1] is None or d[i+1] == 0) else np.exp(2j * kz[i+1] * d[i+1])
        r  = (r_i + r*p2) / (1 + r_i*r*p2)

    # surface (ambient -> first film)
    if pol == 's':
        r0 = (kiz - kz[1])/(kiz + kz[1])
    else:
        r0 = (kiz - kz[1]/(n[1]**2)) / (kiz + kz[1]/(n[1]**2))

    p2_top = 1.0 if d[1] is None else np.exp(2j * kz[1] * d[1])
    return (r0 + r*p2_top) / (1 + r0*r*p2_top)

class MultilayerStack:
    """
    Characteristic-matrix (Yeh/Heavens) multilayer.
    layers: [(n0, None), (n1, d1), ..., (n_{N-2}, d_{N-2}), (n_{N-1}, None)]
            Ambient (0) and substrate (N-1) are semi-infinite with d=None.
            Thicknesses in meters. Angle is from the normal (degrees).
    """
    def __init__(self, layers, wavelength_m, angle_deg_from_normal):
        self.layers = layers
        self.k0 = 2*np.pi / wavelength_m
        self.theta0 = np.deg2rad(angle_deg_from_normal)

    def _kz_q_delta(self, pol='s'):
        n = np.array([n for n,_ in self.layers], dtype=np.complex128)
        d = [d for _,d in self.layers]
        kz = _kz_physical(n, self.theta0, self.k0)
        if pol == 's':
            q = kz / self.k0
        else:  # 'p'
            q = (self.k0 * n**2) / kz
        delta = np.array([0 if (di is None or di==0) else kz[i]*di
                          for i,di in enumerate(d)], dtype=np.complex128)
        return kz, q, delta

    @staticmethod
    def _film_M_q(delta_i, q_i):
        c, s = np.cos(delta_i), np.sin(delta_i)
        return np.array([[c, 1j*s/q_i],
                         [1j*q_i*s, c]], dtype=np.complex128)

    def get_r(self, pol='s'):
        kz, q, delta = self._kz_q_delta(pol=pol)
        # Build characteristic matrix over the finite films (indices 1..N-2)
        M = np.eye(2, dtype=np.complex128)
        for i in range(1, len(self.layers)-1):
            if delta[i] == 0:
                continue
            M = M @ self._film_M_q(delta[i], q[i])

        # Input admittance for mapping [E;H]_bottom = M [E;H]_top
        A,B,C,D = M[0,0], M[0,1], M[1,0], M[1,1]
        q0, qs = q[0], q[-1]
        Yin = (qs*A - C) / (D - qs*B)
        r = (q0 - Yin) / (q0 + Yin)
        if pol=='p':
            r = -r
        return r

    def get_R(self, pol='s'):
        return abs(self.get_r(pol=pol))**2

class MultilayerStack2_bk:
    """
    Parratt recursion in CXRO style.
    layers: [(n0, None), (n1, d1), ..., (n_{N-2}, d_{N-2}), (n_{N-1}, None)]
    wavelength_m in meters.
    angle_deg is grazing α if angle_mode='grazing', else angle from normal.
    """
    def __init__(self, layers, wavelength_m, angle_deg, angle_mode='grazing'):
        self.layers = layers
        self.k0 = 2*np.pi / wavelength_m
        self.theta0 = (np.deg2rad(90.0 - angle_deg)
                       if angle_mode == 'grazing' else np.deg2rad(angle_deg))

    def _kz(self):
        n = np.array([n for n,_ in self.layers], dtype=np.complex128)
        kz = _kz_physical(n, self.theta0, self.k0)
        return kz, n

    def get_r(self, pol='s'):
        kz, n = self._kz()
        d = [d for _,d in self.layers]
        N = len(self.layers)

        # start at substrate interface (last film -> substrate)
        if pol == 's':
            r = (kz[N-2] - kz[N-1])/(kz[N-2] + kz[N-1])
        else:
            b_up = kz[N-2]/(n[N-2]**2)
            b_dn = kz[N-1]/(n[N-1]**2)
            r = (b_up - b_dn)/(b_up + b_dn)

        # recurse upward through films i = N-3,...,1
        for i in range(N-3, 0, -1):
            if pol == 's':
                r_i = (kz[i] - kz[i+1])/(kz[i] + kz[i+1])
            else:
                b_i = kz[i]/(n[i]**2)
                b_ip1 = kz[i+1]/(n[i+1]**2)
                r_i = (b_i - b_ip1)/(b_i + b_ip1)
            p2 = np.exp(2j * kz[i+1] * d[i+1])  # phase in layer below (i+1)
            r  = (r_i + r*p2) / (1 + r_i*r*p2)

        # surface (ambient -> first film) and top film propagation
        if pol == 's':
            r0 = (self.k0*np.sin(np.deg2rad(90.0)-self.theta0) - kz[1]) / \
                 (self.k0*np.sin(np.deg2rad(90.0)-self.theta0) + kz[1])
        else:
            kiz = self.k0*np.sin(np.deg2rad(90.0)-self.theta0)
            r0 = (kiz - kz[1]/(n[1]**2)) / (kiz + kz[1]/(n[1]**2))

        p2_top = np.exp(2j * kz[1] * d[1]) if d[1] is not None else 1.0

        r = (r0 + r*p2_top) / (1 + r0*r*p2_top)
        if pol=='p':
            r = r
        return r

    def get_R(self, pol='s'):
        return abs(self.get_r(pol=pol))**2

class MultilayerStack2:
    """
    Parratt recursion in CXRO style.
    layers: [(n0, None), (n1, d1), ..., (n_{N-2}, d_{N-2}), (n_{N-1}, None)]
    wavelength_m in meters.
    angle_deg is grazing α if angle_mode='grazing', else angle from normal.
    """
    def __init__(self, layers, wavelength_m, angle_deg, angle_mode='grazing'):
        self.layers = layers
        self.k0 = 2*np.pi / wavelength_m
        if angle_mode == 'grazing':
            self.alpha  = np.deg2rad(angle_deg)            # grazing (radians)
            self.theta0 = np.deg2rad(90.0) - self.alpha    # from normal
        else:
            self.theta0 = np.deg2rad(angle_deg)            # from normal
            self.alpha  = np.deg2rad(90.0) - self.theta0   # grazing

    def _kz(self):
        n = np.array([n for n,_ in self.layers], dtype=np.complex128)
        kz = _kz_physical(n, self.theta0, self.k0)
        return kz, n

    def get_r(self, pol='s'):
        kz, n = self._kz()
        d = [d for _, d in self.layers]
        N = len(self.layers)

        # ---------- single interface: ambient (0) <-> substrate (1)
        if N == 2:
            kiz = self.k0 * np.sin(self.alpha)  # alpha stored in radians
            if pol == 's':
                return (kiz - kz[1])/(kiz + kz[1])
            else:  # p
                return (kiz - kz[1]/(n[1]**2)) / (kiz + kz[1]/(n[1]**2))

        # ---------- bottom interface (last finite film -> substrate)
        if pol == 's':
            r = (kz[N-2] - kz[N-1])/(kz[N-2] + kz[N-1])
        else:
            b_up = kz[N-2]/(n[N-2]**2)
            b_dn = kz[N-1]/(n[N-1]**2)
            r = (b_up - b_dn)/(b_up + b_dn)

        # ---------- climb upward through films i = N-3,...,1
        for i in range(N-3, 0, -1):
            if pol == 's':
                r_i = (kz[i] - kz[i+1])/(kz[i] + kz[i+1])
            else:
                b_i   = kz[i]/(n[i]**2)
                b_ip1 = kz[i+1]/(n[i+1]**2)
                r_i = (b_i - b_ip1)/(b_i + b_ip1)

            p2 = 1.0 if (d[i+1] is None or d[i+1] == 0) else np.exp(2j * kz[i+1] * d[i+1])
            r  = (r_i + r*p2) / (1 + r_i*r*p2)

        # ---------- surface (ambient -> first film) and top propagation
        kiz = self.k0 * np.sin(self.alpha)
        if pol == 's':
            r0 = (kiz - kz[1])/(kiz + kz[1])
        else:
            r0 = (kiz - kz[1]/(n[1]**2)) / (kiz + kz[1]/(n[1]**2))

        p2_top = 1.0 if d[1] is None else np.exp(2j * kz[1] * d[1])
        return (r0 + r*p2_top) / (1 + r0*r*p2_top)


def kz_from_normal(n, th_deg):
    th = np.deg2rad(th_deg)
    # Enforce physical branch:
    val = n*np.sqrt(1.0 - (np.sin(th)/n)**2)
    # choose branch with Im(kz)>=0
    if np.imag(val) < 0: val = -val
    return val

def R_film(layers, wavelength, theta_deg, pol='s'):
    k0 = 2*np.pi/wavelength
    kz = lambda n: kz_from_normal(n, theta_deg)
    n0 = layers[0][0]
    n_film = layers[1][0]
    n_sub = layers[2][0]
    d_m = layers[1][1]

    if pol=='s':
        r01 = (kz(n0)-kz(n_film))/(kz(n0)+kz(n_film))
        r12 = (kz(n_film)-kz(n_sub))/(kz(n_film)+kz(n_sub))
    else:
        r01 = (kz(n0)/n0**2 - kz(n_film)/n_film**2)/(kz(n0)/n0**2 + kz(n_film)/n_film**2)
        r12 = (kz(n_film)/n_film**2 - kz(n_sub)/n_sub**2)/(kz(n_film)/n_film**2 + kz(n_sub)/n_sub**2)
    beta = (2*np.pi/wavelength)*kz(n_film)*d_m
    r = (r01 + r12*np.exp(2j*beta)) / (1 + r01*r12*np.exp(2j*beta))
    return r

if __name__ == "__main__":
    pass
