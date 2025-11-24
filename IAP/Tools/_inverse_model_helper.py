"""
Minimal utilities for EUV reflection ptychography (soft labels), keeping ONLY what
run_pipeline3_soft requires.

Provided entry point:
    run_pipeline3_soft(...)

Outputs:
    - per-label:   eta_by_label, nc_by_label, deltabeta_by_label
    - per-pixel:   delta_map, beta_map, phi_topo, height_nm
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional, Any

# --------------------------- Shape helpers --------------------------- #

def _flatten_inputs(
    amp_stack: np.ndarray,
    phase_stack: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Flatten stacks/images into (K, P) and labels into (P,) while remembering the original shape.

    Returns:
      amp_flat:   (K, P)
      phase_flat: (K, P)
      info: dict with keys {"img_shape", "is_1d", "labels_flat"}
    """
    if amp_stack.shape != phase_stack.shape:
        raise ValueError("amp_stack and phase_stack must have the same shape")

    if amp_stack.ndim == 3:  # (K,H,W)
        K, H, W = amp_stack.shape
        img_shape = (H, W)
        amp_flat = amp_stack.reshape(K, H * W)
        phase_flat = phase_stack.reshape(K, H * W)
        lab_flat = labels.reshape(H * W)
        is_1d = False
    elif amp_stack.ndim == 2:  # (K,W)
        K, W = amp_stack.shape
        img_shape = (W,)
        amp_flat = amp_stack.copy()
        phase_flat = phase_stack.copy()
        lab_flat = labels.reshape(W)
        is_1d = True
    else:
        raise ValueError("amp_stack must be (K,H,W) or (K,W)")

    return amp_flat, phase_flat, {"img_shape": img_shape, "is_1d": is_1d, "labels_flat": lab_flat}


def _unflatten_image(arr_flat: np.ndarray, info: dict) -> np.ndarray:
    """Reshape a flattened array back to (H,W) or (W,) according to info."""
    return arr_flat.reshape(info["img_shape"])

# --------------------------- Utility functions --------------------------- #

def to_complex_stack(amp_flat: np.ndarray, phase_flat: np.ndarray) -> np.ndarray:
    """Return complex stack R with shape (K, P) from amplitude/phase stacks (K, P)."""
    return amp_flat * np.exp(1j * phase_flat)


def polarization_weights(pol_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Weights for linear polarization: w_s = cos^2(ψ), w_p = sin^2(ψ)."""
    w_s = np.cos(pol_angles) ** 2
    w_p = np.sin(pol_angles) ** 2
    return w_s, w_p

# ---------------------- Fresnel helpers (single interface) ---------------------- #
from .reflection_model import fresnel_rs_rp


def fresnel_eta_from_nc(nc: complex, theta_deg: float) -> complex:
    rs, rp = fresnel_rs_rp(1, nc, np.deg2rad(theta_deg))
    return rp / rs


def invert_eta_to_delta_beta(
    eta_meas: complex,
    theta_deg: float,
    delta_range: Tuple[float, float] = (0.01, 0.4),
    beta_range: Tuple[float, float] = (0.01, 0.4),
    n_coarse: int = 400,
    n_refine: int = 200,
) -> Tuple[float, float, complex, float]:
    """Invert η -> (delta, beta) for a semi-infinite medium (single interface).
    Returns (delta, beta, nc, residual). Simple coarse+refine grid, SciPy-free.
    """
    d_grid = np.linspace(delta_range[0], delta_range[1], n_coarse)
    b_grid = np.linspace(beta_range[0],  beta_range[1],  n_coarse)
    D, B = np.meshgrid(d_grid, b_grid, indexing='ij')
    nc_grid = (1.0 - D) + 1j * B
    # vectorized evaluation
    eta_pred = np.empty_like(nc_grid, dtype=np.complex128)
    it = np.nditer(nc_grid, flags=['multi_index'])
    while not it.finished:
        eta_pred[it.multi_index] = fresnel_eta_from_nc(it[0], theta_deg)
        it.iternext()
    err = np.abs(eta_pred - eta_meas)
    # err = abs(np.log(eta_pred) - np.log(eta_meas))

    if True:
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

    # refine around (d0,b0)
    def clipwin(v0, rng, frac=0.2):
        span = (rng[1] - rng[0]) * frac
        return max(rng[0], v0 - span), min(rng[1], v0 + span)

    d_min, d_max = clipwin(d0, delta_range)
    b_min, b_max = clipwin(b0, beta_range)
    d_ref = np.linspace(d_min, d_max, n_refine)
    b_ref = np.linspace(b_min, b_max, n_refine)
    D2, B2 = np.meshgrid(d_ref, b_ref, indexing='ij')
    nc2 = (1.0 - D2) + 1j * B2
    eta2 = np.empty_like(nc2, dtype=np.complex128)
    it2 = np.nditer(nc2, flags=['multi_index'])
    while not it2.finished:
        eta2[it2.multi_index] = fresnel_eta_from_nc(it2[0], theta_deg)
        it2.iternext()
    err2 = np.abs(eta2 - eta_meas)

    i1, j1 = np.unravel_index(np.argmin(err2), err2.shape)
    delta_hat, beta_hat = float(D2[i1, j1]), float(B2[i1, j1])
    nc_hat = (1.0 - delta_hat) + 1j * beta_hat
    return delta_hat, beta_hat, nc_hat, float(err2[i1, j1])

# ----------------------- Topography (height) ----------------------- #

def phase_to_height(phi_topo: np.ndarray, lam_nm: float, theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    return -phi_topo * lam_nm / (4.0 * np.pi * np.cos(theta))

# --- Soft-label helpers -------------------------------------------- #

def _flatten_soft_proba(
    proba: Optional[np.ndarray],
    info: dict,
    labels_flat: np.ndarray,
    proba_labels: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns:
      proba_flat: (M, P)  soft probabilities aligned to class order in labels_pos
      labels_pos: (M,)    label IDs for the rows of proba_flat

    Accepts proba as (M,H,W), (H,W,M), (M,W), or (W,M).
    If proba is None, builds one-hot from hard labels.
    """
    # one-hot fallback from hard labels
    if proba is None:
        labs = np.array(sorted([int(v) for v in np.unique(labels_flat) if v >= 0]), dtype=int)
        if labs.size == 0:
            return None, None
        P = labels_flat.size
        M = labs.size
        proba_flat = np.zeros((M, P), dtype=float)
        for i, L in enumerate(labs):
            proba_flat[i, labels_flat == L] = 1.0
        return proba_flat, labs

    arr = np.asarray(proba)
    img_shape = info["img_shape"]
    is_1d = info["is_1d"]

    if arr.ndim == 3:
        # make class the first axis -> (M,H,W)
        if not is_1d and arr.shape[0] == img_shape[0] and arr.shape[1] == img_shape[1]:
            proba_MHW = np.moveaxis(arr, -1, 0)
        elif not is_1d and arr.shape[1] == img_shape[0] and arr.shape[2] == img_shape[1]:
            proba_MHW = arr
        elif is_1d and arr.shape[0] == img_shape[0]:
            proba_MHW = np.moveaxis(arr, -1, 0)
        elif is_1d and arr.shape[1] == img_shape[0]:
            proba_MHW = arr
        else:
            raise ValueError("proba 3D shape incompatible with inputs.")
        M, *sp = proba_MHW.shape
        proba_flat = proba_MHW.reshape(M, -1)

    elif arr.ndim == 2:
        # Accept (M,P) or (M,W); if last dim mismatches, try transpose
        proba_flat = arr
        if not is_1d:
            P = img_shape[0]*img_shape[1]
        else:
            P = img_shape[0]
        if proba_flat.shape[1] != P and proba_flat.T.shape[1] == P:
            proba_flat = proba_flat.T
        if proba_flat.shape[1] != P:
            raise ValueError("proba 2D shape incompatible with image size.")
    else:
        raise ValueError("proba must be 2D or 3D.")

    # labels_pos (row→label ID)
    if proba_labels is not None:
        labels_pos = np.asarray(proba_labels, dtype=int)
        if labels_pos.size != proba_flat.shape[0]:
            raise ValueError("len(proba_labels) must equal number of classes (M).")
    else:
        labels_pos = np.arange(proba_flat.shape[0], dtype=int)

    # normalize columns to sum to 1
    denom = np.sum(proba_flat, axis=0, keepdims=True)
    denom[denom == 0] = 1.0
    proba_flat = proba_flat / denom

    return proba_flat.astype(float), labels_pos


def _soft_label_means_q(
    q: np.ndarray,              # (K,P) complex ratios
    proba_flat: np.ndarray,     # (M,P) soft weights for M labels
) -> np.ndarray:
    """Soft-weighted label means: y[m, k] = sum_p π_{m,p} q[k,p] / sum_p π_{m,p}."""
    K, P = q.shape
    M, P2 = proba_flat.shape
    if P2 != P:
        raise ValueError("proba_flat and q have mismatched P.")
    wsum = np.sum(proba_flat, axis=1).reshape(M, 1) + 1e-30
    y = (proba_flat @ q.T) / wsum  # (M,K)
    return y.astype(np.complex128)


def f_k_of_eta(eta: complex, w_s: np.ndarray, w_p: np.ndarray, ref_k: int) -> np.ndarray:
    """Model f_k(η) = (w_s + w_p η) / (w_s[ref] + w_p[ref] η)."""
    den = w_s[ref_k] + w_p[ref_k]*eta
    return (w_s + w_p*eta) / (den + 1e-30)


def solve_eta_linear_weighted(
    y: np.ndarray,
    w_s: np.ndarray,
    w_p: np.ndarray,
    ref_k: int,
    w_fit: Optional[np.ndarray] = None,
    good_k: Optional[np.ndarray] = None
) -> complex:
    """Solve for η from y_k ≈ f_k(η) using weighted LS in the complex plane."""
    if good_k is not None:
        y = y[good_k]
        w_s = w_s[good_k]
        w_p = w_p[good_k]
        # adjust ref_k into the reduced index space
        idxs = np.flatnonzero(good_k)
        try:
            ref_k = int(np.where(idxs == ref_k)[0][0])
        except IndexError:
            # if ref angle was masked out, put it back with unit weight
            y = np.r_[y, y[0:1]]
            w_s = np.r_[w_s, w_s[0:1]]
            w_p = np.r_[w_p, w_p[0:1]]
            ref_k = y.size - 1
            if w_fit is not None:
                w_fit = np.r_[w_fit[good_k], [1.0]]

        if w_fit is not None:
            w_fit = w_fit[good_k]

    w_s0, w_p0 = w_s[ref_k], w_p[ref_k]
    a = y * w_p0 - w_p
    b = w_s - y * w_s0
    if w_fit is None:
        num = np.vdot(a, b)                     # conj(a)·b
        den = np.vdot(a, a) + 1e-30
    else:
        num = np.sum(w_fit * np.conj(a) * b)
        den = np.sum(w_fit * np.abs(a)**2) + 1e-30
    return num / den


def estimate_eta_from_labels(
    q: np.ndarray,                 # (K,P)
    w_s: np.ndarray,               # (K,)
    w_p: np.ndarray,               # (K,)
    proba_flat: np.ndarray,        # (M,P) soft weights aligned to positive labels
    labels_pos: np.ndarray,        # (M,) the label ids corresponding to the rows of proba_flat
    ref_k: int,
    label_ref: int,
    eta_ref: complex,
    use_double_ratios: bool = True,
) -> Dict[int, complex]:
    """Soft-label analogue: build soft-weighted means y[m,k] and fit η per label."""
    y = _soft_label_means_q(q, proba_flat)  # (M,K)

    eta_by_label: Dict[int, complex] = {}
    if use_double_ratios:
        try:
            iref = int(np.where(labels_pos == label_ref)[0][0])
        except IndexError:
            raise ValueError("label_ref not present in labels_pos / proba")

        f_ref = f_k_of_eta(eta_ref, w_s, w_p, ref_k)  # (K,)
        y_ref = y[iref, :]
        for mi, L in enumerate(labels_pos):
            y_eff = (y[mi, :] / y_ref) * f_ref
            eta_by_label[int(L)] = solve_eta_linear_weighted(y_eff, w_s, w_p, ref_k)
    else:
        for mi, L in enumerate(labels_pos):
            eta_by_label[int(L)] = solve_eta_linear_weighted(y[mi, :], w_s, w_p, ref_k)

    return eta_by_label

# --- Per-pixel soft mixtures ---------------------------------------- #

def _per_pixel_nc_from_soft(
    nc_by_label: Dict[int, complex],
    proba_flat: np.ndarray,
    labels_pos: np.ndarray,
) -> np.ndarray:
    """Linear soft mixture of complex index per pixel: n_c(p) = Σ_m π_{m,p} * n_c(label_m)."""
    nc_vec = np.array([nc_by_label[int(L)] for L in labels_pos], dtype=np.complex128)  # (M,)
    return nc_vec @ proba_flat  # (P,)

def _per_pixel_phi_mat_ref_from_soft(
    w_s_ref: float,
    w_p_ref: float,
    nc_by_label: Dict[int, complex],
    proba_flat: np.ndarray,
    labels_pos: np.ndarray,
    theta_deg: float,
) -> np.ndarray:
    """Soft-mixed material phase at the reference angle via complex averaging."""
    M = proba_flat.shape[0]
    M_ref_per_label = np.zeros(M, dtype=np.complex128)
    for i, L in enumerate(labels_pos):
        rs, rp = fresnel_rs_rp(1, nc_by_label[int(L)], np.deg2rad(theta_deg))
        M_ref_per_label[i] = w_s_ref * rs + w_p_ref * rp
    Mmix = M_ref_per_label @ proba_flat  # (P,)
    return np.angle(Mmix)  # (P,)

# ------------------------------ Top-level: run_pipeline3_soft ------------------------------ #

def run_pipeline(
        amp_ref_stack: np.ndarray,      # amplitudes (K,H,W) or (K,W)
        phase_ref_stack: np.ndarray,    # unwrapped phases (K,H,W) or (K,W)
        pol_angles_rad: np.ndarray,     # (K,)
        labels: np.ndarray,             # hard labels (H,W) or (W,)
        lam_nm: float,
        theta_deg: float,
        label_ref: int,
        nc_ref: complex,
        ref_k: int = 0,
        use_double_ratios: bool = True,
        amp_thresh: float = 0.0,
        delta_range: tuple = (1e-6, 2e-1),
        beta_range: tuple = (1e-6, 2e-1),
        w_s: np.ndarray = None,
        w_p: np.ndarray = None,
        proba: Optional[np.ndarray] = None,   # soft map (M,H,W) or (M,W); if None -> one-hot from labels
        return_mode: str = "all",             # "labels", "maps", or "all"
) -> Dict[str, Any]:
    """
    Soft-label variant of pipeline3. If `proba` is None, we auto-build one-hot from `labels`.
    Returns:
      - per-label: eta_by_label, nc_by_label, deltabeta_by_label
      - per-pixel (soft-mixed): delta_map, beta_map
      - unwrapped topography: phi_topo, height_nm (using soft-mixed material phase)
    """
    # Flatten inputs
    amp_flat, phase_flat, info = _flatten_inputs(amp_ref_stack, phase_ref_stack, labels)
    labels_flat = info["labels_flat"]
    K, P = amp_flat.shape

    # polarization weights & complex stack
    if w_s is None and w_p is None:
        w_s, w_p = polarization_weights(pol_angles_rad)

    R = to_complex_stack(amp_flat, phase_flat)

    # ratios against reference
    ref = R[ref_k]
    valid_ref = np.abs(ref) > amp_thresh
    q = np.full_like(R, np.nan + 1j*np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        q[:, valid_ref] = R[:, valid_ref] / ref[valid_ref]

    # soft probabilities (or one-hot from hard labels)
    proba_flat, labels_pos = _flatten_soft_proba(proba, info, labels_flat)
    if proba_flat is None:
        raise ValueError("No positive labels found to estimate materials.")

    # Step 2: η per label via soft means
    rs0, rp0 = fresnel_rs_rp(1, nc_ref, np.deg2rad(theta_deg))
    eta_ref = rp0 / rs0

    eta_by_label = estimate_eta_from_labels(
        q, w_s, w_p, proba_flat, labels_pos,
        ref_k=ref_k, label_ref=label_ref, eta_ref=eta_ref,
        use_double_ratios=use_double_ratios,
    )

    # Step 3: invert η → (δ,β) per label
    nc_by_label: Dict[int, complex] = {}
    deltabeta_by_label: Dict[int, Tuple[float, float]] = {}
    for L in labels_pos:
        if int(L) == int(label_ref):
            nc_by_label[int(L)] = complex(nc_ref)
            deltabeta_by_label[int(L)] = (float(1 - np.real(nc_ref)), float(np.imag(nc_ref)))
        else:
            d, b, nc_hat, _ = invert_eta_to_delta_beta(
                eta_by_label[int(L)], theta_deg,
                delta_range=delta_range, beta_range=beta_range
            )
            nc_by_label[int(L)] = nc_hat
            deltabeta_by_label[int(L)] = (d, b)

    # Step 4: per-pixel soft-mixed nc → delta/beta maps
    nc_pix_flat = _per_pixel_nc_from_soft(nc_by_label, proba_flat, labels_pos)  # (P,)
    delta_map_flat = 1.0 - np.real(nc_pix_flat)
    beta_map_flat  = np.imag(nc_pix_flat)

    # Step 5: unwrapped topography using soft-mixed material phase at the ref angle
    phi_ref_meas = phase_flat[ref_k]  # unwrapped gauge
    phi_mat_ref_flat = _per_pixel_phi_mat_ref_from_soft(
        w_s_ref=w_s[ref_k], w_p_ref=w_p[ref_k],
        nc_by_label=nc_by_label, proba_flat=proba_flat,
        labels_pos=labels_pos, theta_deg=theta_deg
    )
    phi_topo_flat   = phi_ref_meas - phi_mat_ref_flat
    height_nm_flat  = phase_to_height(phi_topo_flat, lam_nm=lam_nm, theta_deg=theta_deg)

    # Unflatten maps
    delta_map = _unflatten_image(delta_map_flat, info)
    beta_map  = _unflatten_image(beta_map_flat,  info)
    phi_topo  = _unflatten_image(phi_topo_flat,  info)
    height_nm = _unflatten_image(height_nm_flat, info)

    out = {
        "eta_by_label": eta_by_label,
        "nc_by_label": nc_by_label,
        "deltabeta_by_label": deltabeta_by_label,
        "delta_map": delta_map,
        "beta_map": beta_map,
        "phi_topo": phi_topo,
        "height_nm": height_nm,
        "labels_pos": labels_pos,      # label ids order used in proba
    }
    if return_mode == "labels":
        out = {k: out[k] for k in ("eta_by_label","nc_by_label","deltabeta_by_label")}
    elif return_mode == "maps":
        out = {k: out[k] for k in ("delta_map","beta_map","phi_topo","height_nm")}
    return out
