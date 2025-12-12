import numpy as np
from scipy.ndimage import shift as nd_shift
from scipy.ndimage import fourier_shift
from scipy.optimize import minimize
from skimage.registration import phase_cross_correlation

def align_objects(
    O1, O2, *,
    unwrap_phase=None,          # callable(img, method='scipy') -> unwrapped phase (needed if use_unwrap=True)
    roi=None, mask=None,        # ROI=(y0,y1,x0,x1) or boolean mask
    fit_amp_scale=True,         # fit amplitude scale alpha
    fit_ramp=True,              # fit linear phase plane ax, ay
    quadratic_ramp=False,       # also fit quadratic terms qx, qy, qxy
    precorrect_ramp=True,       # pre-apply the ramp seed to O2 before optimizing
    use_unwrap=False,            # <-- turn unwrapping-based ramp seed on/off
    shift_method="fourier",     # 'fourier' (coherent) or 'spline' (real-space)
    shift_order=5,              # spline order if shift_method='spline'
    shift_mode="reflect",       # spline boundary mode
    seed_with="auto",           # 'auto' (amp vs phase), 'amplitude', 'phase', or None
    upsample_seed=100,
    amp_weight=True,            # amplitude-product weights
    amp_thresh=0.0              # ignore pixels below this relative amplitude in ROI
):
    """
    Jointly estimate subpixel shift (dx,dy), global phase (phi0),
    optional linear (and quadratic) phase ramp, and amplitude scale to align complex fields.

    Returns:
      dict with keys: 'params', 'O2_aligned', 'success', 'fun', 'nit', 'ramp_seed_model'
    """

    # ----------------- small helpers (local scope) -----------------
    def _apply_shift(u, dx, dy, method="fourier", order=5, mode="reflect"):
        if method == "fourier":
            return np.fft.ifft2(fourier_shift(np.fft.fft2(u), shift=(dy, dx)))
        elif method == "spline":
            ur = nd_shift(np.real(u), shift=(dy, dx), order=order, mode=mode, prefilter=True)
            ui = nd_shift(np.imag(u), shift=(dy, dx), order=order, mode=mode, prefilter=True)
            return ur + 1j * ui
        else:
            raise ValueError("shift_method must be 'fourier' or 'spline'")

    def _build_mask_from_roi(shape, roi_):
        if roi_ is None:
            return np.ones(shape, dtype=bool)
        y0, y1, x0, x1 = roi_
        m = np.zeros(shape, dtype=bool)
        m[y0:y1, x0:x1] = True
        return m

    def _seed_shift(O1_, O2_, roi_=None, up=100):
        msk = _build_mask_from_roi(O1_.shape, roi_)
        A1, A2 = np.abs(O1_)*msk, np.abs(O2_)*msk
        (dyA, dxA), _, _ = phase_cross_correlation(A1, A2, upsample_factor=up)

        def quick_cost(dx, dy):
            O2s = _apply_shift(O2_, dx, dy, method="fourier")
            return np.sum((np.abs(O1_[msk]) - np.abs(O2s[msk]))**2)

        cA = quick_cost(dxA, dyA)

        try:
            P1, P2 = np.angle(O1_)*msk, np.angle(O2_)*msk
            (dyP, dxP), _, _ = phase_cross_correlation(P1, P2, upsample_factor=up)
            cP = quick_cost(dxP, dyP)
            if cP < cA:
                return float(dxP), float(dyP), "phase"
        except Exception:
            pass
        return float(dxA), float(dyA), "amplitude"

    def _fit_phase_ramp_wrapped(O1_, O2_, roi_, weight="amp", amp_thresh_=0.0, quadratic=False):
        """Fit ramp on WRAPPED phase diff using least squares around principal branch."""
        H, W = O1_.shape
        msk = _build_mask_from_roi((H, W), roi_)
        ph = np.angle(O1_ * np.conj(O2_))  # [-pi, pi]
        # weights
        if weight == "amp":
            Wgt = (np.abs(O1_) * np.abs(O2_)).astype(float)
        else:
            Wgt = np.ones_like(ph, dtype=float)
        if amp_thresh_ > 0:
            thr1 = amp_thresh_ * np.max(np.abs(O1_[msk]))
            thr2 = amp_thresh_ * np.max(np.abs(O2_[msk]))
            keep = (np.abs(O1_) >= thr1) & (np.abs(O2_) >= thr2)
            Wgt *= keep
        Wgt *= msk
        if not np.any(Wgt):
            return {"coeffs": (0,0,0), "model": "none", "ramp": np.zeros_like(ph)}

        yy, xx = np.indices((H, W))
        sel = Wgt > 0
        x = xx[sel].ravel(); y = yy[sel].ravel()
        z = ph[sel].ravel()
        w = np.sqrt(Wgt[sel].ravel() + 1e-12)

        # For wrapped phases, fitting linear model directly is an approximation.
        # Works best for small residuals (yours are ~±0.4 rad).
        if quadratic:
            X = np.c_[x, y, np.ones_like(x), x**2, y**2, x*y]
            coeffs, *_ = np.linalg.lstsq(X * w[:, None], z * w, rcond=None)
            a,b,c,qx,qy,qxy = coeffs
            ramp = a*xx + b*yy + c + qx*(xx**2) + qy*(yy**2) + qxy*(xx*yy)
            return {"coeffs": (a,b,c,qx,qy,qxy), "model": "quad", "ramp": ramp}
        else:
            X = np.c_[x, y, np.ones_like(x)]
            coeffs, *_ = np.linalg.lstsq(X * w[:, None], z * w, rcond=None)
            a,b,c = coeffs
            ramp = a*xx + b*yy + c
            return {"coeffs": (a,b,c), "model": "plane", "ramp": ramp}

    def _fit_phase_ramp_unwrapped(O1_, O2_, unwrap_phase_, roi_, weight="amp", amp_thresh_=0.0, quadratic=False):
        """Fit ramp on UNWRAPPED phase diff in ROI."""
        H, W = O1_.shape
        msk = _build_mask_from_roi((H, W), roi_)
        phi1 = unwrap_phase_(np.angle(O1_), method='scipy')
        phi2 = unwrap_phase_(np.angle(O2_), method='scipy')
        dphi = phi1 - phi2

        if weight == "amp":
            Wgt = (np.abs(O1_) * np.abs(O2_)).astype(float)
        else:
            Wgt = np.ones_like(dphi, dtype=float)
        if amp_thresh_ > 0:
            thr1 = amp_thresh_ * np.max(np.abs(O1_[msk]))
            thr2 = amp_thresh_ * np.max(np.abs(O2_[msk]))
            keep = (np.abs(O1_) >= thr1) & (np.abs(O2_) >= thr2)
            Wgt *= keep
        Wgt *= msk
        if not np.any(Wgt):
            return {"coeffs": (0,0,0), "model": "none", "ramp": np.zeros_like(dphi)}

        yy, xx = np.indices((H, W))
        sel = Wgt > 0
        x = xx[sel].ravel(); y = yy[sel].ravel()
        z = dphi[sel].ravel()
        w = np.sqrt(Wgt[sel].ravel() + 1e-12)

        if quadratic:
            X = np.c_[x, y, np.ones_like(x), x**2, y**2, x*y]
            coeffs, *_ = np.linalg.lstsq(X * w[:, None], z * w, rcond=None)
            a,b,c,qx,qy,qxy = coeffs
            ramp = a*xx + b*yy + c + qx*(xx**2) + qy*(yy**2) + qxy*(xx*yy)
            return {"coeffs": (a,b,c,qx,qy,qxy), "model": "quad", "ramp": ramp}
        else:
            X = np.c_[x, y, np.ones_like(x)]
            coeffs, *_ = np.linalg.lstsq(X * w[:, None], z * w, rcond=None)
            a,b,c = coeffs
            ramp = a*xx + b*yy + c
            return {"coeffs": (a,b,c), "model": "plane", "ramp": ramp}

    # ----------------- validations -----------------
    if O1.shape != O2.shape:
        raise ValueError("O1 and O2 must have the same shape")

    H, W = O1.shape
    Y, X = np.indices((H, W))

    # ----------------- build mask/weights -----------------
    if mask is None:
        mask = _build_mask_from_roi((H, W), roi)
    mask = mask.astype(bool)
    if not np.any(mask):
        raise ValueError("ROI/mask excludes all pixels; nothing to fit.")

    if amp_weight:
        Wgt = (np.abs(O1) * np.abs(O2)).astype(float)
    else:
        Wgt = np.ones_like(O1, dtype=float)

    if amp_thresh > 0:
        thr1 = amp_thresh * np.max(np.abs(O1[mask]))
        thr2 = amp_thresh * np.max(np.abs(O2[mask]))
        keep = (np.abs(O1) >= thr1) & (np.abs(O2) >= thr2)
        Wgt *= keep

    Wgt *= mask
    if not np.any(Wgt):
        raise ValueError("All weights are zero after amp_thresh/mask; relax thresholds.")

    # ----------------- seed shift -----------------
    if seed_with in ("auto", "amplitude", "phase"):
        if seed_with == "amplitude":
            (dy0, dx0), _, _ = phase_cross_correlation(np.abs(O1)*mask, np.abs(O2)*mask, upsample_factor=upsample_seed)
            dx0, dy0 = float(dx0), float(dy0)
            seed_mode = "amplitude"
        elif seed_with == "phase":
            (dy0, dx0), _, _ = phase_cross_correlation(np.angle(O1)*mask, np.angle(O2)*mask, upsample_factor=upsample_seed)
            dx0, dy0 = float(dx0), float(dy0)
            seed_mode = "phase"
        else:
            dx0, dy0, seed_mode = _seed_shift(O1, O2, roi_=roi, up=upsample_seed)
    else:
        dx0 = dy0 = 0.0
        seed_mode = "none"

    # ----------------- ramp seed (unwrap or wrapped) -----------------
    if use_unwrap:
        if unwrap_phase is None:
            raise ValueError("use_unwrap=True but unwrap_phase callable is None")
        ramp_info = _fit_phase_ramp_unwrapped(
            O1, O2, unwrap_phase, roi_=roi,
            weight="amp" if amp_weight else "none",
            amp_thresh_=amp_thresh, quadratic=quadratic_ramp
        )
    else:
        ramp_info = _fit_phase_ramp_wrapped(
            O1, O2, roi_=roi,
            weight="amp" if amp_weight else "none",
            amp_thresh_=amp_thresh, quadratic=quadratic_ramp
        )

    ramp_seed = ramp_info["ramp"]
    if ramp_info["model"] == "quad":
        a0, b0, c0, qx0, qy0, qxy0 = ramp_info["coeffs"]
    elif ramp_info["model"] == "plane":
        a0, b0, c0 = ramp_info["coeffs"]; qx0 = qy0 = qxy0 = 0.0
    else:
        a0 = b0 = c0 = qx0 = qy0 = qxy0 = 0.0

    # pre-correct O2 by subtracting the ramp seed (helps convergence)
    O2_seed = O2 * (np.exp(-1j * ramp_seed) if precorrect_ramp else 1.0)

    # ----------------- define cost and optimize -----------------
    def cost(theta):
        i = 0
        dx, dy, phi0 = theta[i], theta[i+1], theta[i+2]; i += 3
        if fit_ramp:
            ax, ay = theta[i], theta[i+1]; i += 2
        else:
            ax = ay = 0.0
        if quadratic_ramp:
            qx_, qy_, qxy_ = theta[i], theta[i+1], theta[i+2]; i += 3
        else:
            qx_ = qy_ = qxy_ = 0.0
        alpha = theta[i] if fit_amp_scale else 1.0

        O2s = _apply_shift(O2_seed, dx, dy, method=shift_method, order=shift_order, mode=shift_mode)
        phase_model = phi0 + ax*X + ay*Y + qx_*(X**2) + qy_*(Y**2) + qxy_*(X*Y)
        diff = O1 - alpha * O2s * np.exp(1j * phase_model)
        return np.sum(Wgt * (np.abs(diff)**2))

    theta0 = [dx0, dy0, 0.0]
    if fit_ramp:
        theta0 += [0.0 if precorrect_ramp else a0,
                   0.0 if precorrect_ramp else b0]
    if quadratic_ramp:
        theta0 += [0.0 if precorrect_ramp else qx0,
                   0.0 if precorrect_ramp else qy0,
                   0.0 if precorrect_ramp else qxy0]
    if fit_amp_scale:
        theta0 += [1.0]

    opt = minimize(cost, theta0, method="Powell")

    # unpack best params
    th = opt.x; i = 0
    dx, dy, phi0 = float(th[i]), float(th[i+1]), float(th[i+2]); i += 3
    if fit_ramp:
        ax, ay = float(th[i]), float(th[i+1]); i += 2
    else:
        ax = ay = 0.0
    if quadratic_ramp:
        qx_, qy_, qxy_ = float(th[i]), float(th[i+1]), float(th[i+2]); i += 3
    else:
        qx_ = qy_ = qxy_ = 0.0
    alpha = float(th[i]) if fit_amp_scale else 1.0

    # build aligned O2
    O2s = _apply_shift(O2_seed, dx, dy, method=shift_method, order=shift_order, mode=shift_mode)
    phase_model = phi0 + ax*X + ay*Y + qx_*(X**2) + qy_*(Y**2) + qxy_*(X*Y)
    O2_aligned = alpha * O2s * np.exp(1j * phase_model)

    return {
        "params": {
            "dx": dx, "dy": dy, "phi0": phi0,
            "ax": ax, "ay": ay, "qx": qx_, "qy": qy_, "qxy": qxy_,
            "alpha": alpha,
            "shift_method": shift_method, "seed_mode": seed_with,
            "ramp_seed_model": ramp_info["model"], "precorrect_ramp": bool(precorrect_ramp),
            "use_unwrap": bool(use_unwrap)
        },
        "O2_aligned": O2_aligned,
        "success": bool(opt.success),
        "fun": float(opt.fun),
        "nit": int(opt.nit),
    }

def error(reconstruction, simulation):
    return np.sum(np.abs(reconstruction - simulation) ** 2) / np.sum(np.abs(simulation) ** 2)

def align_objects_basic(Obj1, Obj2):
    # align objects
    # check error metric before shift
    error_init = error(Obj1, Obj2)
    # First check if both images have the same center:
    # check using amplitude
    shift_distance1 = phase_cross_correlation(np.abs(Obj1), np.abs(Obj2), upsample_factor=100)[0]
    print("Shift distance amp: " + str(shift_distance1))
    # check using phase
    shift_distance2 = phase_cross_correlation(np.angle(Obj1), np.angle(Obj2), upsample_factor=100)[0]
    print("Shift distance phase: " + str(shift_distance2))
    # shift_distance1 = -shift_distance1
    # shift_distance2 = -shift_distance2

    temp = nd_shift(np.real(Obj2), shift_distance1, order=5) + 1j * nd_shift(np.imag(Obj2),
                                                                                shift_distance1,
                                                                                order=5)
    temp2 = nd_shift(np.real(Obj2), shift_distance2, order=5) + 1j * nd_shift(np.imag(Obj2),
                                                                               shift_distance2,
                                                                               order=5)

    # check error after shift
    error_amp_shift = error(Obj1, temp)
    error_phase_shift = error(Obj1, temp2)
    print(f'error before shift: {error_init}\n'
          f'error after shift (amp): {error_amp_shift}\n'
          f'error after shift (phase): {error_phase_shift}')
    if error_amp_shift < error_init or error_phase_shift < error_init:
        if error_amp_shift < error_phase_shift:
            return temp, False
        else:
            return temp2, False
    else:
        return Obj2, True
