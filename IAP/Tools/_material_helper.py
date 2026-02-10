import re
import numpy as np
from scipy.constants import physical_constants, h, c, e
import xraydb


NA = physical_constants['Avogadro constant'][0]              # mol^-1
r_e = physical_constants['classical electron radius'][0]     # meters

# --- tiny formula parser (no parentheses; e.g. "Si3N4", "Al2O3", "C") ---
_token = re.compile(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)")
def parse_formula(formula: str):
    parts = _token.findall(formula)
    if not parts:
        raise ValueError(f"Could not parse formula: {formula}")
    comp = {}
    for sym, num in parts:
        n = float(num) if num else 1.0
        comp[sym] = comp.get(sym, 0.0) + n
    return comp  # dict: element -> stoichiometric coefficient

def delta_beta_from_formula(
    formula: str,
    energies_eV, *,
    density_g_cm3: float | None,
    dataset: str = "chantler",
    include_Z_in_f1: bool = True,   # set True to match CXRO (Henke) style
    return_n: bool = False           # if True, also return complex n
):
    """
    energies_eV: float or 1D array
    density_g_cm3: if None, try xraydb.get_material(formula) to fetch one
    dataset: 'chantler' (supported by xraydb here)
    include_Z_in_f1: use f1_eff = Z + f1 (CXRO convention)
    """
    # energies -> ndarray
    E = np.atleast_1d(np.asarray(energies_eV, dtype=float))
    # wavelength in meters
    lam = (h * c) / (E * e)  # m
    factor = (r_e * lam**2) / (2.0 * np.pi)  # broadcast over E

    # get density if not provided
    if density_g_cm3 is None and formula is not 'O2':
        mat, rho = xraydb.get_material(formula)
        print(f'density {mat}: {rho}')
        if rho is None:
            raise ValueError(f"No density for {formula}; pass density_g_cm3 explicitly.")
        density_g_cm3 = float(rho)
        # print(f'{formula}: rho: {rho}')

    # parse composition and molar mass of one formula unit
    comp = parse_formula(formula)
    A = {el: xraydb.atomic_mass(el) for el in comp}     # g/mol
    Z = {el: xraydb.atomic_number(el) for el in comp}

    M_FU = sum(A[el] * comp[el] for el in comp)         # g/mol

    # number density of formula units, then per-element atom densities (m^-3)
    n_FU_m3 = (density_g_cm3 * NA / M_FU) * 1e6         # cm^-3 -> m^-3
    n_atoms = {el: comp[el] * n_FU_m3 for el in comp}   # m^-3

    # get f1, f2 per element at each energy
    sum_nf1 = np.zeros_like(E, dtype=float)
    sum_nf2 = np.zeros_like(E, dtype=float)

    for el, n_j in n_atoms.items():
        if dataset == "chantler":
            f1 = np.array([xraydb.f1_chantler(el, energy=float(eV)) for eV in E], dtype=float)
            f2 = np.array([xraydb.f2_chantler(el, energy=float(eV)) for eV in E], dtype=float)
            print(f'f1 {el}: {f1}, n_j: {n_j}')
            print(f'f2 {el}: {f2}, n_j: {n_j}')
        else:
            raise ValueError("Only 'chantler' dataset is wired here. (Easy to extend.)")

        if include_Z_in_f1:
            f1 = f1 + Z[el]  # CXRO/Henke-style f1_total

        sum_nf1 += n_j * f1
        sum_nf2 += n_j * f2

    delta = factor * sum_nf1
    beta  = factor * sum_nf2

    if np.isscalar(energies_eV):
        delta = float(delta[0])
        beta  = float(beta[0])

    if return_n:
        n_complex = 1.0 - delta + 1j * beta
        return delta, beta, n_complex
    return delta +1j*beta
