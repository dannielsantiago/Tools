import re
import numpy as np
from scipy.constants import physical_constants, h, c, e
import xraydb

try:
    import periodictable
    from periodictable.xsf import index_of_refraction as periodictable_index_of_refraction
except ImportError:
    periodictable = None
    periodictable_index_of_refraction = None


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
    density_g_cm3: float | None = None,
    source: str = "periodictable",
    dataset: str = "chantler",
    include_Z_in_f1: bool = True,   # set True to match CXRO (Henke) style
    return_n: bool = False,          # if True, also return complex n
    low_energy_xraydb_cutoff_eV: float | None = 30.0,
):
    """
    Estimate optical constants from a chemical formula.

    energies_eV: float or 1D array
    density_g_cm3: if None, try xraydb.get_material(formula) as a density fallback
    source: "periodictable" for CXRO/Henke-like tables, or "xraydb" for
        xraydb Chantler f1/f2 summation
    dataset: "chantler" for the xraydb source
    include_Z_in_f1: use f1_eff = Z + f1 for the xraydb source
    return_n: if True, return (delta, beta, n_complex)
    low_energy_xraydb_cutoff_eV: for periodictable/CXRO mode, energies below
        this cutoff are evaluated with xraydb because periodictable can return
        invalid f1 sentinel values at very low energies. Set None to disable.
    """
    E = np.atleast_1d(np.asarray(energies_eV, dtype=float))
    source = source.lower()

    # Preserve old call behavior: if no density is supplied, try the xraydb
    # material table first, then use that density with the selected constants source.
    if density_g_cm3 is None and formula != "O2":
        try:
            mat, rho = xraydb.get_material(formula)
        except Exception:
            rho = None
        if rho is not None:
            density_g_cm3 = float(rho)

    if density_g_cm3 is None:
        raise ValueError(f"No density for {formula}; pass density_g_cm3 explicitly.")

    try:
        density_g_cm3 = float(density_g_cm3)
    except (TypeError, ValueError):
        raise ValueError(f"Density for {formula} must be a numeric value in g/cm^3.") from None
    if not np.isfinite(density_g_cm3) or density_g_cm3 <= 0:
        raise ValueError(f"Density for {formula} must be finite and > 0 g/cm^3.")

    def _xraydb_delta_beta(energies):
        energies = np.atleast_1d(np.asarray(energies, dtype=float))
        lam = (h * c) / (energies * e)  # m
        factor = (r_e * lam**2) / (2.0 * np.pi)

        comp = parse_formula(formula)
        A = {el: xraydb.atomic_mass(el) for el in comp}
        Z = {el: xraydb.atomic_number(el) for el in comp}
        M_FU = sum(A[el] * comp[el] for el in comp)

        n_FU_m3 = (density_g_cm3 * NA / M_FU) * 1e6
        n_atoms = {el: comp[el] * n_FU_m3 for el in comp}

        sum_nf1 = np.zeros_like(energies, dtype=float)
        sum_nf2 = np.zeros_like(energies, dtype=float)

        for el, n_j in n_atoms.items():
            if dataset == "chantler":
                f1 = np.array(
                    [xraydb.f1_chantler(el, energy=float(eV)) for eV in energies],
                    dtype=float,
                )
                f2 = np.array(
                    [xraydb.f2_chantler(el, energy=float(eV)) for eV in energies],
                    dtype=float,
                )
            else:
                raise ValueError("Only 'chantler' dataset is wired here. (Easy to extend.)")

            if include_Z_in_f1:
                f1 = f1 + Z[el]

            sum_nf1 += n_j * f1
            sum_nf2 += n_j * f2

        return factor * sum_nf1, factor * sum_nf2

    def _periodictable_delta_beta(energies):
        if periodictable is None or periodictable_index_of_refraction is None:
            raise RuntimeError("Install periodictable to use source='periodictable'.")

        energies_keV = np.atleast_1d(np.asarray(energies, dtype=float)) / 1000.0
        try:
            n_complex = periodictable_index_of_refraction(
                formula,
                density=density_g_cm3,
                energy=energies_keV,
            )
        except Exception:
            comp = parse_formula(formula)
            compound = {
                getattr(periodictable, element): count
                for element, count in comp.items()
            }
            n_complex = periodictable_index_of_refraction(
                compound,
                density=density_g_cm3,
                energy=energies_keV,
            )

        n_complex = np.asarray(n_complex, dtype=complex)
        return 1.0 - np.real(n_complex), np.abs(np.imag(n_complex))

    if source in {"cxro", "crxo", "periodictable"}:
        delta = np.empty_like(E, dtype=float)
        beta = np.empty_like(E, dtype=float)

        if low_energy_xraydb_cutoff_eV is None:
            use_xraydb = np.zeros_like(E, dtype=bool)
        else:
            use_xraydb = E < float(low_energy_xraydb_cutoff_eV)

        if np.any(use_xraydb):
            delta[use_xraydb], beta[use_xraydb] = _xraydb_delta_beta(E[use_xraydb])
        if np.any(~use_xraydb):
            delta[~use_xraydb], beta[~use_xraydb] = _periodictable_delta_beta(E[~use_xraydb])

    elif source == "xraydb":
        delta, beta = _xraydb_delta_beta(E)

    else:
        raise ValueError("source must be 'periodictable', 'cxro', 'crxo', or 'xraydb'.")

    if np.isscalar(energies_eV):
        delta = float(delta[0])
        beta = float(beta[0])

    if return_n:
        n_complex = 1.0 - delta + 1j * beta
        return delta, beta, n_complex
    return delta + 1j * beta