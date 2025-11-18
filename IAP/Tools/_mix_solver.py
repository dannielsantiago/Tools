import numpy as np
from typing import Dict, Tuple

# -------------------------- helpers --------------------------

def as_n(delta_map, beta_map):
    """Build complex refractive index map from delta,beta maps (H,W)."""
    return 1.0 - delta_map + 1j*beta_map

def brugg_A(eps_j, eps_eff):
    return (eps_j - eps_eff) / (eps_j + 2.0*eps_eff)

def mg_B(eps_i, eps_h):
    return (eps_i - eps_h) / (eps_i + 2.0*eps_h)

def project_simplex_2d(f1, f2):
    """Project (f1,f2) to {f1>=0, f2>=0, f1+f2<=1}."""
    f1 = max(0.0, f1)
    f2 = max(0.0, f2)
    s = f1 + f2
    if s > 1.0:
        f1 /= s; f2 /= s
    return f1, f2

# ------------------ Bruggeman (constrained) core ------------------

def _solve_bruggeman_on_active_set(A_vec, active_idx):
    """
    A_vec: complex array shape (3,) with A_j = (eps_j - eps_eff)/(eps_j + 2 eps_eff)
    active_idx: indices (subset of {0,1,2}) that are active (nonzero)
    Returns (f, res2, ok):
        f     : full length-3 fractions (zeros on inactive), sums to 1 on active
        res2  : residual squared (float)
        ok    : feasibility flag for nonnegativity on active (within tiny tol)
    """
    k = len(active_idx)

    # Corner (k=1)
    if k == 1:
        f = np.zeros(3, float)
        f[active_idx[0]] = 1.0
        S = A_vec[active_idx[0]]
        res2 = (S.real**2 + S.imag**2)
        return f, res2, True

    # Build real 2 x k matrix for active set and solve KKT:
    # min ||M f_act||^2 s.t. 1^T f_act = 1
    A_act = A_vec[np.array(active_idx)]
    M = np.stack([A_act.real, A_act.imag], axis=0)  # (2, k)
    ones = np.ones((k, 1), float)

    H = 2.0 * (M.T @ M)             # (k, k)
    KKT = np.block([[H,    ones],   # (k+1, k+1)
                    [ones.T, np.zeros((1,1))]])
    rhs = np.zeros((k+1,), float)
    rhs[-1] = 1.0

    try:
        sol = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(KKT, rhs, rcond=None)

    f_act = sol[:k]
    if np.any(f_act < -1e-10):
        return None, np.inf, False

    f = np.zeros(3, float)
    f[np.array(active_idx)] = f_act

    S = np.sum(f * A_vec)  # complex sum
    res2 = S.real*S.real + S.imag*S.imag
    return f, res2, True

def solve_bruggeman_constrained_single(eps_eff, eps_list):
    """
    Constrained Bruggeman for one pixel:
      minimize || sum_j f_j A_j ||^2  s.t. f_j >= 0, sum f_j = 1
    where A_j = (eps_j - eps_eff)/(eps_j + 2 eps_eff).
    eps_list: tuple/list of three epsilons corresponding to (n_1, n_2, n_3).
    Returns f (length-3, nonnegative, sums to 1) in that same order.
    """
    A_vec = np.array([(e - eps_eff)/(e + 2.0*eps_eff) for e in eps_list], dtype=complex)

    candidates = []
    idxs = (0, 1, 2)

    # singletons (corners)
    for i in idxs:
        f, r2, ok = _solve_bruggeman_on_active_set(A_vec, (i,))
        if ok: candidates.append((r2, f))

    # pairs (edges)
    for i in idxs:
        for j in idxs:
            if j <= i: continue
            f, r2, ok = _solve_bruggeman_on_active_set(A_vec, (i, j))
            if ok: candidates.append((r2, f))

    # full set (interior)
    f, r2, ok = _solve_bruggeman_on_active_set(A_vec, (0, 1, 2))
    if ok: candidates.append((r2, f))

    if not candidates:
        # extremely unlikely; safe fallback
        return np.array([1.0, 0.0, 0.0])

    candidates.sort(key=lambda t: t[0])
    f_best = candidates[0][1]

    # Numerical polish
    f_best = np.clip(f_best, 0.0, None)
    s = f_best.sum()
    if s <= 0:
        f_best = np.array([1.0, 0.0, 0.0])
    else:
        f_best /= s
    return f_best

# ---------------------- Public map solvers -----------------------

def linear_mix_n_map(delta_map, beta_map, n_1, n_2, n_3):
    """
    Linear mixing in n (heuristic), parameterized.
    We anchor at n_1 and solve:
        n_eff - n_1 = f2*(n_2 - n_1) + f3*(n_3 - n_1),  f1 = 1 - f2 - f3
    Returns (f1_map, f2_map, f3_map) in the SAME order as (n_1, n_2, n_3).
    """
    n_eff = as_n(delta_map, beta_map)

    A = np.array([
        [(n_2 - n_1).real, (n_3 - n_1).real],
        [(n_2 - n_1).imag, (n_3 - n_1).imag],
    ], dtype=float)

    # Guard: ill-conditioned design
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)

    b = np.stack([(n_eff - n_1).real, (n_eff - n_1).imag], axis=0)  # (2, H, W)

    fg = A_inv @ b.reshape(2, -1)  # (2, H*W)
    f2 = fg[0].reshape(delta_map.shape)
    f3 = fg[1].reshape(delta_map.shape)
    f1 = 1.0 - f2 - f3
    return f1, f2, f3

def bruggeman_map_constrained(delta_map, beta_map, n_1, n_2, n_3):
    """
    Constrained Bruggeman EMA, parameterized.
    Returns (f1_map, f2_map, f3_map) corresponding to (n_1, n_2, n_3),
    with f >= 0 and f1+f2+f3 = 1 per pixel.
    """
    eps_1, eps_2, eps_3 = n_1**2, n_2**2, n_3**2
    eps_list = (eps_1, eps_2, eps_3)

    n_eff = as_n(delta_map, beta_map)
    eps_eff = n_eff**2

    H, W = delta_map.shape
    f1 = np.empty((H, W), float)
    f2 = np.empty((H, W), float)
    f3 = np.empty((H, W), float)

    for i in range(H):
        for j in range(W):
            f = solve_bruggeman_constrained_single(eps_eff[i, j], eps_list)
            f1[i, j], f2[i, j], f3[i, j] = f
    return f1, f2, f3

def mg_map_constrained(delta_map, beta_map, n_host, n_inc1, n_inc2):
    """
    Constrained Maxwell–Garnett with arbitrary host.
    Solves B_inc1*f_inc1 + B_inc2*f_inc2 ≈ S, with f>=0, f_inc1+f_inc2<=1,
    then f_host = 1 - (f_inc1 + f_inc2).
    Returns (f_host_map, f_inc1_map, f_inc2_map) in that SAME order.
    """
    # constants
    eps_h   = n_host**2
    eps_i1  = n_inc1**2
    eps_i2  = n_inc2**2
    B_i1    = mg_B(eps_i1, eps_h)
    B_i2    = mg_B(eps_i2, eps_h)

    def mg_constrained_pixel(B1, B2, S):
        A = np.array([[B1.real, B2.real],
                      [B1.imag, B2.imag]], float)
        b = np.array([S.real, S.imag], float)

        cands = []
        # unconstrained LS
        try:
            fU = np.linalg.lstsq(A, b, rcond=None)[0]
            cands.append(fU)
        except np.linalg.LinAlgError:
            pass
        # edges
        if abs(B2) > 0:
            f2 = (b @ np.array([B2.real, B2.imag])) / (B2.real**2 + B2.imag**2)
            cands.append(np.array([0.0, f2]))
        if abs(B1) > 0:
            f1 = (b @ np.array([B1.real, B1.imag])) / (B1.real**2 + B1.imag**2)
            cands.append(np.array([f1, 0.0]))
        # f1+f2=1 edge
        v   = A[:, 0] - A[:, 1]
        rhs = b - A[:, 1]
        denom = v @ v
        if denom > 0:
            f1 = (v @ rhs) / denom
            f2 = 1.0 - f1
            cands.append(np.array([f1, f2]))
        # corners
        cands += [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]

        def resid(fe):
            r = A @ fe - b
            return float(r @ r)

        best_val, best = None, None
        for fe in cands:
            f1p, f2p = project_simplex_2d(fe[0], fe[1])
            rp = resid(np.array([f1p, f2p]))
            if best is None or rp < best_val:
                best_val, best = rp, (f1p, f2p)
        return best  # (f1, f2)

    n_eff = as_n(delta_map, beta_map)
    eps_eff = n_eff**2

    H, W = delta_map.shape
    f_host = np.empty((H, W), float)
    f_i1   = np.empty((H, W), float)
    f_i2   = np.empty((H, W), float)

    for i in range(H):
        for j in range(W):
            ee = eps_eff[i, j]
            S  = (ee - eps_h) / (ee + 2.0*eps_h)
            g1, g2 = mg_constrained_pixel(B_i1, B_i2, S)
            fh = 1.0 - (g1 + g2)
            # clip tiny numerical negatives
            f_i1[i, j]   = max(0.0, g1)
            f_i2[i, j]   = max(0.0, g2)
            f_host[i, j] = max(0.0, fh) if fh >= 0 else 0.0
    return f_host, f_i1, f_i2

def material_mole_fraction_maps(
    f_maps: Dict[str, np.ndarray],         # material -> (H,W) volume fractions
    rho: Dict[str, float],                 # material -> density [g/cm^3]
    M: Dict[str, float],                   # material -> molar mass [g/mol]
    eps: float = 1e-30,
) -> Dict[str, np.ndarray]:
    """
    Convert per-material volume-fraction maps to per-material mole-fraction maps.
    x_j ∝ f_j * rho_j / M_j, normalized so sum_j x_j = 1 per pixel.
    """
    mats = list(f_maps.keys())
    shape = next(iter(f_maps.values())).shape
    num = np.zeros(shape, dtype=float)
    w = {}
    for m in mats:
        w[m] = np.clip(f_maps[m], 0, None) * (rho[m] / M[m])  # nonneg guard
        num += w[m]
    out = {}
    den = np.maximum(num, eps)
    for m in mats:
        out[m] = w[m] / den
    return out

def elemental_atomic_fraction_maps(
    f_maps: Dict[str, np.ndarray],         # material -> (H,W) volume fractions
    rho: Dict[str, float],                 # material -> density [g/cm^3]
    M: Dict[str, float],                   # material -> molar mass [g/mol]
    stoich: Dict[str, Dict[str, int]],     # material -> {element: atom_count}, e.g. "Ga2O3": {"Ga":2,"O":3}
    eps: float = 1e-30,
) -> Dict[str, np.ndarray]:
    """
    Convert material volume-fraction maps into elemental atomic-fraction maps X_e.
    X_e ∝ sum_j f_j * rho_j / M_j * ν_{e,j}, normalized over elements at each pixel.
    Returns a dict element -> (H,W) map, summing to 1 over elements per pixel.
    """
    # collect all elements present
    elements = sorted({e for m in stoich for e in stoich[m].keys()})
    shape = next(iter(f_maps.values())).shape

    # accumulate per-element numerators
    num_e = {e: np.zeros(shape, dtype=float) for e in elements}
    total_atoms = np.zeros(shape, dtype=float)

    for m, f_map in f_maps.items():
        w_m = np.clip(f_map, 0, None) * (rho[m] / M[m])  # moles per pixel up to constant
        # total atoms contributed by material m: sum_e ν_{e,m} * w_m
        n_tot_m = float(sum(stoich[m].values()))
        total_atoms += n_tot_m * w_m
        for e, nu in stoich[m].items():
            num_e[e] += nu * w_m

    den = np.maximum(total_atoms, eps)
    X = {e: num_e[e] / den for e in elements}
    return X


# -------------------------- example usage --------------------------

if __name__ == "__main__":
    # materials (at one photon energy)
    n_SiN   = 1 - 0.3025672187248983 + 0.13742198359720179j
    n_SiO2  = 1 - 0.19236051493629355 + 0.14953703577050978j
    n_Ga2O3 = 1 - 0.22563306          + 0.24603965j

    # Example: single pixel as 1x1 map
    n_found = 1 - 0.2557917406581781 + 0.2025050062341784j
    delta = np.array([[1 - n_found.real]])
    beta  = np.array([[n_found.imag]])

    # Linear (anchor n_1)
    f1_lin, f2_lin, f3_lin = linear_mix_n_map(delta, beta, n_SiN, n_SiO2, n_Ga2O3)
    print("Linear (n1=SiN, n2=SiO2, n3=Ga2O3):", f1_lin[0,0], f2_lin[0,0], f3_lin[0,0])

    # Bruggeman (constrained, nonnegative, sums to 1)
    f1_brg, f2_brg, f3_brg = bruggeman_map_constrained(delta, beta, n_SiN, n_SiO2, n_Ga2O3)
    print("Bruggeman (f1,f2,f3) -> (SiN,SiO2,Ga2O3):", f1_brg[0,0], f2_brg[0,0], f3_brg[0,0])

    # Maxwell–Garnett (choose host explicitly)
    f_host, f_inc1, f_inc2 = mg_map_constrained(delta, beta, n_SiN, n_SiO2, n_Ga2O3)
    print("MG host=SiN -> (host, inc1, inc2) = (SiN, SiO2, Ga2O3):",
          f_host[0,0], f_inc1[0,0], f_inc2[0,0])
