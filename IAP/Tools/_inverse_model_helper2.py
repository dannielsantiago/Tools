
"""
K=2 (pure s,p) EUV reflection ptychography helper.

Assumptions:
- amp_ref_stack, phase_ref_stack have shape (2,H,W) or (2,W)
  k=0 -> s-pol, k=1 -> p-pol
- phase_ref_stack is already unwrapped (radians) and referenced consistently.

Implements:
1) Build complex reflectivities R_s, R_p.
2) Form per-pixel ratios qp = R_p / R_s (topography cancels).
3) Soft or hard label averaging of qp to estimate eta per label using a reference label.
4) Invert eta -> (delta,beta) per label via Fresnel single-interface grid search.
5) Soft-mix nc per pixel to create delta/beta maps.
6) Unwrapped topography from s-pol reference phase minus soft-mixed material phase.
7) Optional Monte-Carlo uncertainty clouds per label.

Author: ChatGPT
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional, Any

# --------------------------- Shape helpers --------------------------- #

def _flatten_inputs(
    amp_stack: np.ndarray,
    phase_stack: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Flatten stacks into (K,P) while remembering original spatial shape.

    Returns:
      amp_flat:   (K,P)
      phase_flat: (K,P)
      info: dict with keys {"img_shape","is_1d"}
    """
    if amp_stack.shape != phase_stack.shape:
        raise ValueError("amp_stack and phase_stack must have same shape")
    if amp_stack.ndim == 3:  # (K,H,W)
        K, H, W = amp_stack.shape
        if K != 2:
            raise ValueError("This helper expects K=2 (s,p). Got %d" % K)
        amp_flat = amp_stack.reshape(K, H*W)
        phase_flat = phase_stack.reshape(K, H*W)
        info = {"img_shape": (H,W), "is_1d": False}
    elif amp_stack.ndim == 2:  # (K,W)
        K, W = amp_stack.shape
        if K != 2:
            raise ValueError("This helper expects K=2 (s,p). Got %d" % K)
        amp_flat = amp_stack.copy()
        phase_flat = phase_stack.copy()
        info = {"img_shape": (W,), "is_1d": True}
    else:
        raise ValueError("amp_stack must be (2,H,W) or (2,W)")
    return amp_flat, phase_flat, info

def _unflatten_image(arr_flat: np.ndarray, info: dict) -> np.ndarray:
    return arr_flat.reshape(info["img_shape"])

# --------------------------- Basic utilities --------------------------- #

def to_complex_stack(amp_flat: np.ndarray, phase_flat: np.ndarray) -> np.ndarray:
    return amp_flat * np.exp(1j * phase_flat)

def as_n(delta_map, beta_map):
    return 1.0 - delta_map + 1j*beta_map

# ---------------------- Fresnel helpers (single interface) ---------------------- #

def fresnel_rs_rp(nc: complex, theta_deg: float) -> Tuple[complex, complex]:
    """Field Fresnel coefficients (vacuum -> medium with index nc) at theta from normal."""
    theta = np.deg2rad(theta_deg)
    cos_t = np.cos(theta)
    sin_t2 = np.sin(theta) ** 2
    gamma = np.sqrt(nc**2 - sin_t2)
    if np.imag(gamma) < 0:
        gamma = -gamma
    rs = (cos_t - gamma) / (cos_t + gamma)
    rp = (nc**2 * cos_t - gamma) / (nc**2 * cos_t + gamma)
    return rs, rp

def fresnel_eta_from_nc(nc: complex, theta_deg: float) -> complex:
    rs, rp = fresnel_rs_rp(nc, theta_deg)
    return rp / rs

def invert_eta_to_delta_beta(
    eta_meas: complex,
    theta_deg: float,
    delta_range: Tuple[float, float]=(1e-6, 2e-1),
    beta_range: Tuple[float, float]=(1e-6, 2e-1),
    n_coarse: int=400,
    n_refine: int=200,
) -> Tuple[float, float, complex, float]:
    """Brute-force coarse+refine inversion of eta -> (delta,beta)."""
    d_grid = np.linspace(delta_range[0], delta_range[1], n_coarse)
    b_grid = np.linspace(beta_range[0],  beta_range[1],  n_coarse)
    D, B = np.meshgrid(d_grid, b_grid, indexing="ij")
    nc_grid = (1.0 - D) + 1j*B
    # vectorized eta prediction
    eta_pred = np.vectorize(lambda z: fresnel_eta_from_nc(z, theta_deg))(nc_grid)
    err = np.abs(eta_pred - eta_meas)

    if False:
        import matplotlib.pyplot as plt
        dmin, dmax = delta_range  # δ range
        bmin, bmax = beta_range  # β range
        plt.figure(figsize=(3.5,3), dpi=100)
        plt.imshow(
            err.T,
            extent=[dmin, dmax, bmin, bmax],  # [xmin, xmax, ymin, ymax]
            origin='lower',  # so that β increases upward
            aspect='auto',  # adjust as you like
            cmap='magma'  # or your preferred colormap
        )

        plt.xlabel('δ')
        plt.ylabel('β')
        plt.title('Error map across\n δ–β parameter space')
        plt.colorbar(label='Error value')
        plt.tight_layout()
        plt.show()  # err = np.abs(eta_pred - np.conj(eta_meas)) #when using -1j*B convention


    i0, j0 = np.unravel_index(np.argmin(err), err.shape)
    d0, b0 = float(D[i0, j0]), float(B[i0, j0])

    def clipwin(v0, rng, frac=0.2):
        span = (rng[1]-rng[0]) * frac
        return max(rng[0], v0-span), min(rng[1], v0+span)

    d_min, d_max = clipwin(d0, delta_range)
    b_min, b_max = clipwin(b0, beta_range)
    d_ref = np.linspace(d_min, d_max, n_refine)
    b_ref = np.linspace(b_min, b_max, n_refine)
    D2, B2 = np.meshgrid(d_ref, b_ref, indexing="ij")
    nc2 = (1.0 - D2) + 1j*B2
    eta2 = np.vectorize(lambda z: fresnel_eta_from_nc(z, theta_deg))(nc2)
    err2 = np.abs(eta2 - eta_meas)

    i1, j1 = np.unravel_index(np.argmin(err2), err2.shape)
    delta_hat, beta_hat = float(D2[i1, j1]), float(B2[i1, j1])
    nc_hat = (1.0 - delta_hat) + 1j*beta_hat
    return delta_hat, beta_hat, nc_hat, float(err2[i1, j1])

# --------------------------- Soft label handling --------------------------- #

def _flatten_labels_map(
    proba: Optional[np.ndarray],
    info: dict,
    labels: Optional[np.ndarray]=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      proba_flat: (M,P) soft probabilities aligned to labels_indices
      labels_indices: (M,) label IDs for rows of proba_flat

    Accepts proba as (M,H,W), (H,W,M), (M,W), (W,M), or (M,P).
    If proba is None, builds one-hot from hard labels (labels required).
    """
    img_shape = info["img_shape"]
    is_1d = info["is_1d"]

    if proba is None:
        if labels is None:
            raise ValueError("labels must be provided when proba is None.")
        labs = np.array(sorted([int(v) for v in np.unique(labels) if v >= 0]), dtype=int)
        if labs.size == 0:
            raise ValueError("No labels found.")
        labels_indices = labs 
        # flatten labels -> one-hot
        if not is_1d:
            H, W = img_shape
            lab_flat = labels.reshape(H*W)
            P = H*W
        else:
            W = img_shape[0]
            lab_flat = labels.reshape(W)
            P = W
        M = labels_indices.size
        proba_flat = np.zeros((M, P), dtype=float)
        for i, L in enumerate(labels_indices):
            proba_flat[i, lab_flat == L] = 1.0
            
        return proba_flat, labels_indices

    arr = np.asarray(proba)
    # Determine P
    if not is_1d:
        H, W = img_shape
        P = H*W
    else:
        W = img_shape[0]
        P = W

    # Move/reshape to (M,P)
    if arr.ndim == 3:
        # (M,H,W) or (H,W,M)
        if (not is_1d) and arr.shape[0] == H and arr.shape[1] == W:
            proba_MHW = np.moveaxis(arr, -1, 0)
        else:
            proba_MHW = arr if arr.shape[1:] == (H,W) else np.moveaxis(arr, -1, 0)
        M = proba_MHW.shape[0]
        proba_flat = proba_MHW.reshape(M, P)
    elif arr.ndim == 2:
        M2, N2 = arr.shape
        if N2 == P:
            proba_flat = arr
        elif M2 == P:
            proba_flat = arr.T
        else:
            # (M,W) or (W,M)
            if N2 == W:
                proba_flat = arr
            elif M2 == W:
                proba_flat = arr.T
            else:
                raise ValueError(f"proba shape {arr.shape} incompatible with image shape {img_shape}.")
    else:
        raise ValueError("proba must be 2D or 3D.")

    # labels_indices
    labels_indices = np.arange(0, proba_flat.shape[0], dtype=int)

    # normalize columns
    den = proba_flat.sum(axis=0, keepdims=True)
    den[den == 0] = 1.0
    proba_flat = proba_flat / den
    return proba_flat.astype(float), labels_indices


def _soft_label_mean_qp(
    qp: np.ndarray,          # (P,) complex
    proba_flat: np.ndarray,  # (M,P) weights
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Soft-weighted mean and std of qp per class.

    Returns:
      mu:     (M,) complex mean
      sigma:  (M,) float std magnitude in complex plane (sqrt(E|z-mu|^2))
      neff:   (M,) effective sample size = sum(weights)
    """
    M, P = proba_flat.shape
    mu = np.zeros(M, dtype=np.complex128)
    sigma = np.zeros(M, dtype=np.float64)
    neff = np.zeros(M, dtype=np.float64)

    for m in range(M):
        w = proba_flat[m]
        wsum = float(np.sum(w) + 1e-30)
        neff[m] = wsum
        mu_m = np.sum(w * qp) / wsum
        mu[m] = mu_m
        var_m = np.sum(w * np.abs(qp - mu_m)**2) / wsum
        sigma[m] = float(np.sqrt(var_m))
    return mu, sigma, neff

# --------------------------- Eta estimation --------------------------- #

def estimate_eta_from_q_ratios(
    qp: np.ndarray,               # (P,) complex ratios Rp/Rs
    proba_flat: np.ndarray,       # (M,P)
    labels_pos: np.ndarray,       # (M,)
    label_ref: int,
    eta_ref: complex,
    use_double_ratios: bool=True,
) -> Tuple[Dict[int, complex], Dict[int, Tuple[complex,float,float]]]:
    """
    Estimate eta per label from qp means.

    Returns:
      eta_by_label: dict[label] -> eta
      stats_by_label: dict[label] -> (qp_mean, qp_sigma, neff)
    """
    mu, sigma, neff = _soft_label_mean_qp(qp, proba_flat)
    stats_by_label: Dict[int, Tuple[complex,float,float]] = {}
    for i, L in enumerate(labels_pos):
        stats_by_label[int(L)] = (mu[i], sigma[i], neff[i])
        print(f"Label={L}\n mu={mu[i]}, sigma={sigma[i]}, neff={neff[i]}")

    eta_by_label: Dict[int, complex] = {}
    if use_double_ratios:
        # locate reference
        try:
            iref = int(np.where(labels_pos == label_ref)[0][0])
        except IndexError:
            raise ValueError("label_ref not found in labels_pos.")
        mu_ref = mu[iref]
        for i, L in enumerate(labels_pos):
            eta_by_label[int(L)] = (mu[i] / (mu_ref + 1e-30)) * eta_ref
    else:
        for i, L in enumerate(labels_pos):
            eta_by_label[int(L)] = mu[i]
    # lock reference exactly
    eta_by_label[int(label_ref)] = eta_ref
    return eta_by_label, stats_by_label

# --------------------------- Soft mixing per pixel --------------------------- #

def _per_pixel_nc_from_soft(
    nc_by_label: Dict[int, complex],
    proba_flat: np.ndarray,
    labels_pos: np.ndarray,
) -> np.ndarray:
    nc_vec = np.array([nc_by_label[int(L)] for L in labels_pos], dtype=np.complex128)
    return nc_vec @ proba_flat  # (P,)

def _per_pixel_phi_mat_ref_from_soft(
    nc_by_label: Dict[int, complex],
    proba_flat: np.ndarray,
    labels_pos: np.ndarray,
    theta_deg: float,
) -> np.ndarray:
    """Soft-mixed material phase at s-pol reference (k=0)."""
    M = proba_flat.shape[0]
    rs_per_label = np.zeros(M, dtype=np.complex128)
    for i, L in enumerate(labels_pos):
        rs, _ = fresnel_rs_rp(nc_by_label[int(L)], theta_deg)
        rs_per_label[i] = rs
    rs_mix = rs_per_label @ proba_flat  # (P,)
    return np.angle(rs_mix)

def phase_to_height(phi_topo: np.ndarray, lam_nm: float, theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    return -phi_topo * lam_nm / (4.0*np.pi*np.cos(theta))

# --------------------------- Monte Carlo clouds --------------------------- #

def mc_deltabeta_clouds(
    qp: np.ndarray,                 # (P,)
    proba_flat: np.ndarray,         # (M,P)
    labels_pos: np.ndarray,         # (M,)
    label_ref: int,
    eta_ref: complex,
    theta_deg: float,
    delta_range: Tuple[float,float],
    beta_range: Tuple[float,float],
    n_mc: int=200,
    mc_seed: Optional[int]=None,
    mc_mode: str="bootstrap_pixels",  # "bootstrap_pixels" or "gaussian_qp"
    stats_by_label: Optional[Dict[int, Tuple[complex,float,float]]]=None,
    use_double_ratios: bool=True,
) -> Dict[int, np.ndarray]:
    """
    Return dict[label] -> (n_mc,2) array of (delta,beta) samples.
    """
    if n_mc <= 0:
        return {int(L): None for L in labels_pos}

    rng = np.random.default_rng(mc_seed)
    M, P = proba_flat.shape

    # pre-find reference index
    try:
        iref = int(np.where(labels_pos == label_ref)[0][0])
    except IndexError:
        raise ValueError("label_ref not found in labels_pos for MC.")
    clouds: Dict[int, np.ndarray] = {int(L): np.zeros((n_mc,2), float) for L in labels_pos}

    if mc_mode == "bootstrap_pixels":
        # Precompute per-class sampling probs
        probs = proba_flat / (proba_flat.sum(axis=1, keepdims=True) + 1e-30)
        for t in range(n_mc):
            # bootstrap mean qp for each label
            mu_bs = np.zeros(M, dtype=np.complex128)
            for m in range(M):
                p_m = probs[m]
                idx = rng.choice(P, size=max(2, int(np.sum(proba_flat[m]))), replace=True, p=p_m)
                mu_bs[m] = np.mean(qp[idx])
            mu_ref = mu_bs[iref]
            for m, L in enumerate(labels_pos):
                if int(L) == int(label_ref):
                    eta = eta_ref
                else:
                    eta = (mu_bs[m] / (mu_ref + 1e-30)) * eta_ref if use_double_ratios else mu_bs[m]
                d,b,_,_ = invert_eta_to_delta_beta(
                    eta, theta_deg, delta_range=delta_range, beta_range=beta_range
                )
                clouds[int(L)][t] = (d,b)

    elif mc_mode == "gaussian_qp":
        if stats_by_label is None:
            raise ValueError("stats_by_label required for gaussian_qp MC.")
        mu = np.array([stats_by_label[int(L)][0] for L in labels_pos], dtype=np.complex128)
        sigma = np.array([stats_by_label[int(L)][1] for L in labels_pos], dtype=float)
        neff = np.array([stats_by_label[int(L)][2] for L in labels_pos], dtype=float)  # (M,)

        # standard error of the mean per label
        sigma_mu = sigma / np.sqrt(np.maximum(neff, 1.0))  # (M,)
        sigma_mu = 50 * sigma / np.sqrt(neff)
        sigma_mu = sigma **2

        iref = int(np.where(labels_pos == label_ref)[0][0])

        for t in range(n_mc):
            # draw complex mean per label
            mu_draw = mu + (sigma_mu / np.sqrt(2.0)) * (
                    rng.standard_normal(M) + 1j * rng.standard_normal(M)
            )

            mu_ref_draw = mu_draw[iref]
            for m, L in enumerate(labels_pos):
                if use_double_ratios:
                    eta = (mu_draw[m] / (mu_ref_draw + 1e-30)) * eta_ref
                else:
                    eta = mu_draw[m]

                if int(L) == int(label_ref):
                    eta = eta_ref

                d, b, _, _ = invert_eta_to_delta_beta(
                    eta, theta_deg, delta_range=delta_range, beta_range=beta_range
                )
                clouds[int(L)][t] = (d, b)
    else:
        raise ValueError("mc_mode must be 'bootstrap_pixels' or 'gaussian_qp'")

    return clouds

# --------------------------- Top-level K=2 pipeline --------------------------- #

def run_pipeline(
    amp_ref_stack: np.ndarray,      # (2,H,W) or (2,W)
    phase_ref_stack: np.ndarray,    # same shape, unwrapped
    labels: np.ndarray,             # hard labels (H,W) or (W,)
    lam_nm: float,
    theta_deg: float,
    label_ref: int,
    nc_ref: complex,
    proba: Optional[np.ndarray]=None,
    proba_labels: Optional[np.ndarray]=None,
    use_double_ratios: bool=True,
    amp_thresh: float=0.0,
    delta_range: Tuple[float,float]=(1e-3, 0.9),
    beta_range: Tuple[float,float]=(1e-3, 0.9),
    n_mc: int=0,
    mc_seed: Optional[int]=None,
    mc_mode: str="bootstrap_pixels",
) -> Dict[str, Any]:
    """
    K=2 simplified pipeline with optional soft labels and MC clouds.

    Returns dict containing:
      eta_by_label, stats_by_label (qp mean/std), nc_by_label, deltabeta_by_label,
      delta_map, beta_map, phi_topo, height_nm, deltabeta_cloud_by_label (if n_mc>0).
    """
    # Flatten stacks (no labels here)
    amp_flat, phase_flat, info = _flatten_inputs(amp_ref_stack, phase_ref_stack)
    K, P = amp_flat.shape

    # Flatten labels for internal hard uses
    if not info["is_1d"]:
        H, W = info["img_shape"]
        labels_flat = labels.reshape(H*W)
    else:
        W = info["img_shape"][0]
        labels_flat = labels.reshape(W)

    # Complex reflectivities and ratios
    R = to_complex_stack(amp_flat, phase_flat)  # (2,P)
    Rs, Rp = R[0], R[1]
    valid_ref = np.abs(Rs) > amp_thresh
    qp = np.full(P, np.nan + 1j*np.nan, dtype=np.complex128)
    qp[valid_ref] = Rp[valid_ref] / Rs[valid_ref]

    # Soft probabilities (or one-hot from labels)
    proba_flat, labels_pos = _flatten_labels_map(
        proba, info, labels=labels,
    )

    # Reference eta from known nc_ref
    rs0, rp0 = fresnel_rs_rp(nc_ref, theta_deg)
    eta_ref = rp0 / rs0

    # Estimate eta per label + store qp stats
    eta_by_label, stats_by_label = estimate_eta_from_q_ratios(
        qp, proba_flat, labels_pos, label_ref, eta_ref, use_double_ratios=use_double_ratios
    )

    # Invert eta -> (delta,beta)
    nc_by_label: Dict[int, complex] = {}
    deltabeta_by_label: Dict[int, Tuple[float,float]] = {}
    for L in labels_pos:
        if int(L) == int(label_ref):
            nc_by_label[int(L)] = complex(nc_ref)
            deltabeta_by_label[int(L)] = (float(1-np.real(nc_ref)), float(np.imag(nc_ref)))
        else:
            d,b,nc_hat,_ = invert_eta_to_delta_beta(
                eta_by_label[int(L)], theta_deg,
                delta_range=delta_range, beta_range=beta_range
            )
            nc_by_label[int(L)] = nc_hat
            deltabeta_by_label[int(L)] = (d,b)

    # Per-pixel soft-mixed nc -> delta/beta maps
    nc_pix_flat = _per_pixel_nc_from_soft(nc_by_label, proba_flat, labels_pos)
    delta_map_flat = 1.0 - np.real(nc_pix_flat)
    beta_map_flat  = np.imag(nc_pix_flat)

    # Unwrapped topography using s-pol material phase
    phi_ref_meas = phase_flat[0]  # s-pol is reference
    phi_mat_ref_flat = _per_pixel_phi_mat_ref_from_soft(
        nc_by_label, proba_flat, labels_pos, theta_deg=theta_deg
    )
    phi_topo_flat = phi_ref_meas - phi_mat_ref_flat
    height_nm_flat = phase_to_height(phi_topo_flat, lam_nm=lam_nm, theta_deg=theta_deg)

    # Unflatten
    delta_map = _unflatten_image(delta_map_flat, info)
    beta_map  = _unflatten_image(beta_map_flat,  info)
    phi_topo  = _unflatten_image(phi_topo_flat,  info)
    height_nm = _unflatten_image(height_nm_flat, info)

    # Optional MC clouds
    deltabeta_cloud_by_label = mc_deltabeta_clouds(
        qp=qp, proba_flat=proba_flat, labels_pos=labels_pos,
        label_ref=label_ref, eta_ref=eta_ref, theta_deg=theta_deg,
        delta_range=delta_range, beta_range=beta_range,
        n_mc=n_mc, mc_seed=mc_seed, mc_mode=mc_mode,
        stats_by_label=stats_by_label, use_double_ratios=use_double_ratios
    ) if n_mc > 0 else {int(L): None for L in labels_pos}

    return {
        "eta_by_label": eta_by_label,
        "stats_by_label": stats_by_label,  # qp_mean, qp_sigma, neff
        "nc_by_label": nc_by_label,
        "deltabeta_by_label": deltabeta_by_label,
        "delta_map": delta_map,
        "beta_map": beta_map,
        "phi_topo": phi_topo,
        "height_nm": height_nm,
        "labels_pos": labels_pos,
        "qp": qp,
        "deltabeta_cloud_by_label": deltabeta_cloud_by_label,
    }
