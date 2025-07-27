from ctypes import c_float, c_int, POINTER, c_char_p, c_ubyte, c_double
import numpy as np
import os
from ctypes import CDLL
from skimage.restoration import unwrap_phase as scikit_unwrap
# import pyunwrap
# from unwrap import unwrap2D

# Path to your compiled DLL
_dll_path = os.path.join(os.path.dirname(__file__), "unwrapping.dll")
lib = CDLL(_dll_path)

def openFile(file, size, dt=np.float32):
    with open(file, 'r') as ff:
        return np.reshape(np.fromfile(ff, dtype=dt), size)

def run_gold(wrapped_phase, infile='phase.float', outfile='uphase.float', format='float',
             maskfile=None, cutlen=0, debug_flag=0, dipole_flag=0):
    """
    Wrapper function to call the C DLL function for Goldstein's branch cut unwrapping algorithm.

    Parameters:
        wrapped_phase (np.ndarray): 2D NumPy array of wrapped phase data (dtype must be float32-compatible).
        infile (str): Path to temporary input file to write the wrapped phase (default: 'phase.float').
        outfile (str): Path to output file for unwrapped phase result (default: 'uphase.float').
        format (str): Input file format keyword ('float', 'byte', 'complex4', or 'complex8').
        maskfile (str or None): Optional path to a binary mask file; use 'none' if not provided.
        cutlen (int): Maximum length of allowed branch cuts (0 uses default heuristic).
        debug_flag (int): Whether to save intermediate debug files (0 = no, 1 = yes).
        dipole_flag (int): Whether to eliminate dipole residues before unwrapping (0 = no, 1 = yes).

    Returns:
        np.ndarray: 2D NumPy array with the unwrapped phase result.
    """
    wrapped_phase = wrapped_phase.astype(np.float32)
    wrapped_phase.tofile(infile)

    xsize = wrapped_phase.shape[-1]
    ysize = wrapped_phase.shape[-2]

    if maskfile is None:
        maskfile = 'none'

    # Convert Python strings to C-compatible byte strings
    infile_ctypes = infile.encode('utf-8')
    outfile_ctypes = outfile.encode('utf-8')
    maskfile_ctypes = maskfile.encode('utf-8')
    format_ctypes = format.encode('utf-8')

    # Call the C shared library function
    lib.run_gold(infile_ctypes, outfile_ctypes, maskfile_ctypes, format_ctypes,
                 xsize, ysize, cutlen, debug_flag, dipole_flag)

    # Load the unwrapped phase output and clean up temp files
    uphase = openFile(outfile_ctypes, wrapped_phase.shape)
    os.remove(outfile)
    os.remove(infile)

    return uphase

def run_qual(wrapped_phase, infile='phase.float', outfile='uphase.float', format='float',
             quality_mode='min_grad', maskfile=None, qualfile=None,
             tsize=3, debug_flag=0):
    """
    Wrapper function to call the C DLL function for quality-guided unwrapping.

    Parameters:
        wrapped_phase (np.ndarray): 2D NumPy array of wrapped phase data (dtype must be float32-compatible).
        infile (str): Path to temporary input file to write the wrapped phase (default: 'phase.float').
        outfile (str): Path to output file for unwrapped phase result (default: 'uphase.float').
        format (str): Input file format keyword ('float', 'byte', 'complex4', or 'complex8').
        quality_mode (str): Mode keyword guiding the quality metric ('min_grad', 'min_var', 'max_corr', 'max_pseu').
        maskfile (str or None): Optional path to a binary mask file; use 'none' if not provided.
        qualfile (str or None): Optional quality image file (e.g., correlation); used if quality_mode requires it.
        tsize (int): Averaging template size (used for filtering quality map).
        debug_flag (int): Whether to save intermediate debug files (0 = no, 1 = yes).

    Returns:
        np.ndarray: 2D NumPy array with the unwrapped phase result.
    """
    wrapped_phase = wrapped_phase.astype(np.float32)
    wrapped_phase.tofile(infile)

    xsize = wrapped_phase.shape[-1]
    ysize = wrapped_phase.shape[-2]

    if maskfile is None:
        maskfile = 'none'
    if qualfile is None:
        qualfile = 'none'

    # Convert Python strings to C-compatible byte strings
    infile_ctypes = infile.encode('utf-8')
    outfile_ctypes = outfile.encode('utf-8')
    maskfile_ctypes = maskfile.encode('utf-8')
    qualfile_ctypes = qualfile.encode('utf-8')
    qualmode_ctypes = quality_mode.encode('utf-8')
    format_ctypes = format.encode('utf-8')

    # Call the C shared library function
    lib.run_qual(infile_ctypes, outfile_ctypes, format_ctypes,
                 maskfile_ctypes, qualfile_ctypes,
                 xsize, ysize, tsize,
                 qualmode_ctypes, debug_flag)

    # Load the unwrapped phase output and clean up temp files
    uphase = openFile(outfile_ctypes, wrapped_phase.shape)
    os.remove(outfile)
    os.remove(infile)

    return uphase

def run_mcut(wrapped_phase, infile='phase.float', outfile='uphase.float', format='float',
             quality_mode='min_grad', maskfile=None, qualfile=None,
             tsize=3, debug_flag=0):
    """
    Wrapper function to call the C DLL function for mask-cut-based unwrapping.

    Parameters:
        wrapped_phase (np.ndarray): 2D NumPy array of wrapped phase data (dtype must be float32-compatible).
        infile (str): Path to temporary input file to write the wrapped phase (default: 'phase.float').
        outfile (str): Path to output file for unwrapped phase result (default: 'uphase.float').
        format (str): Input file format keyword ('float', 'byte', 'complex4', or 'complex8').
        quality_mode (str): Mode keyword guiding the quality metric ('min_grad', 'min_var', 'max_corr', 'max_pseu').
        maskfile (str or None): Optional path to a binary mask file; use 'none' if not provided.
        qualfile (str or None): Optional quality image file (e.g., correlation); used if quality_mode requires it.
        tsize (int): Averaging template size (used for filtering quality map).
        debug_flag (int): Whether to save intermediate debug files (0 = no, 1 = yes).

    Returns:
        np.ndarray: 2D NumPy array with the unwrapped phase result.
    """
    wrapped_phase = wrapped_phase.astype(np.float32)
    wrapped_phase.tofile(infile)

    xsize = wrapped_phase.shape[-1]
    ysize = wrapped_phase.shape[-2]

    if maskfile is None:
        maskfile = 'none'
    if qualfile is None:
        qualfile = 'none'

    # Convert strings to bytes for ctypes
    infile_ctypes = infile.encode('utf-8')
    outfile_ctypes = outfile.encode('utf-8')
    maskfile_ctypes = maskfile.encode('utf-8')
    qualfile_ctypes = qualfile.encode('utf-8')
    qualmode_ctypes = quality_mode.encode('utf-8')
    format_ctypes = format.encode('utf-8')

    # Call the C shared library function
    lib.run_mcut(infile_ctypes, outfile_ctypes, format_ctypes,
                 maskfile_ctypes, qualfile_ctypes,
                 xsize, ysize, tsize,
                 qualmode_ctypes, debug_flag)

    # Load result and clean up
    uphase = openFile(outfile_ctypes, wrapped_phase.shape)
    os.remove(outfile)
    os.remove(infile)

    return uphase

def run_flyn(wrapped_phase, infile='phase.float', outfile='uphase.float', format='float',
             quality_mode='min_grad', maskfile=None, qualfile=None,
             tsize=3, debug_flag=0, thresh_flag=0, fatten=2, guess_mode=0):
    """
    Wrapper function to call the C DLL function for Flynn's minimum discontinuity unwrapping.

    Parameters:
        wrapped_phase (np.ndarray): 2D NumPy array of wrapped phase data or intermediate guess.
        infile (str): Path to temporary input file to write the phase data (default: 'phase.float').
        outfile (str): Path to output file for unwrapped phase result (default: 'uphase.float').
        format (str): Input file format keyword ('float', 'byte', 'complex4', or 'complex8').
        quality_mode (str): Quality map mode ('min_grad', 'min_var', 'max_corr', 'max_pseu', or 'none').
        maskfile (str or None): Optional mask file (use 'none' if not provided).
        qualfile (str or None): Optional correlation/quality file.
        tsize (int): Averaging template size.
        debug_flag (int): Enable debug outputs (0 or 1).
        thresh_flag (int): Apply automatic thresholding to quality map (0 or 1).
        fatten (int): Number of pixels to fatten the mask.
        guess_mode (int): Indicates if `wrapped_phase` is an initial guess (0 = no, 1 = yes).

    Returns:
        np.ndarray: 2D NumPy array with the unwrapped phase result.
    """
    wrapped_phase = wrapped_phase.astype(np.float32)
    wrapped_phase.tofile(infile)

    xsize = wrapped_phase.shape[-1]
    ysize = wrapped_phase.shape[-2]

    if maskfile is None:
        maskfile = 'none'
    if qualfile is None:
        qualfile = 'none'

    # Convert to C-compatible strings
    infile_ctypes = infile.encode('utf-8')
    outfile_ctypes = outfile.encode('utf-8')
    format_ctypes = format.encode('utf-8')
    maskfile_ctypes = maskfile.encode('utf-8')
    qualfile_ctypes = qualfile.encode('utf-8')
    qualmode_ctypes = quality_mode.encode('utf-8')

    lib.run_flyn(infile_ctypes, outfile_ctypes, format_ctypes,
                 maskfile_ctypes, qualfile_ctypes,
                 xsize, ysize, tsize,
                 qualmode_ctypes, debug_flag,
                 thresh_flag, fatten, guess_mode)

    uphase = openFile(outfile_ctypes, wrapped_phase.shape)
    os.remove(outfile)
    os.remove(infile)

    return uphase

def run_fmg(wrapped_phase, infile='phase.float', outfile='uphase.float', format='float',
            quality_mode='min_grad', maskfile=None, qualfile=None,
            tsize=3, debug_flag=0, thresh_flag=1, fatten=2,
            num_iter=100, num_cycles=0):
    """
    Wrapper function to call the C DLL function for multigrid-based unwrapping.

    Parameters:
        wrapped_phase (np.ndarray): 2D NumPy array of wrapped phase data.
        infile (str): Temporary input file to write phase (default: 'phase.float').
        outfile (str): Output file with unwrapped phase (default: 'uphase.float').
        format (str): One of 'float', 'byte', 'complex4', 'complex8'.
        quality_mode (str): Quality guidance mode (e.g., 'min_grad').
        maskfile (str or None): Optional mask file.
        qualfile (str or None): Optional correlation/quality file.
        tsize (int): Template size for quality smoothing.
        debug_flag (int): Whether to save debug files.
        thresh_flag (int): Whether to apply automatic thresholding.
        fatten (int): Pixels to thicken threshold mask.
        num_iter (int): Gauss-Seidel iterations per level (default 2).
        num_cycles (int): Multigrid V-cycle count (default 2).

    Returns:
        np.ndarray: Unwrapped phase as a NumPy array.
    """
    wrapped_phase = wrapped_phase.astype(np.float32)
    wrapped_phase.tofile(infile)

    xsize = wrapped_phase.shape[-1]
    ysize = wrapped_phase.shape[-2]

    if maskfile is None:
        maskfile = 'none'
    if qualfile is None:
        qualfile = 'none'

    # Convert to C-compatible strings
    infile_ctypes = infile.encode('utf-8')
    outfile_ctypes = outfile.encode('utf-8')
    format_ctypes = format.encode('utf-8')
    maskfile_ctypes = maskfile.encode('utf-8')
    qualfile_ctypes = qualfile.encode('utf-8')
    mode_ctypes = quality_mode.encode('utf-8')

    # Call C function
    lib.run_fmg(infile_ctypes, outfile_ctypes, format_ctypes,
                maskfile_ctypes, qualfile_ctypes,
                xsize, ysize, tsize,
                mode_ctypes, debug_flag,
                thresh_flag, fatten,
                num_iter, num_cycles)

    uphase = openFile(outfile_ctypes, wrapped_phase.shape)
    os.remove(outfile)
    os.remove(infile)

    return uphase

def run_unmg(wrapped_phase, infile='phase.float', outfile='uphase.float', format='float',
             num_iter=10, num_cycles=0):
    """
    Wrapper function to call the C DLL function for unweighted multigrid phase unwrapping.

    Parameters:
        wrapped_phase (np.ndarray): 2D NumPy array of wrapped phase data.
        infile (str): Temporary input file to write wrapped phase (default: 'phase.float').
        outfile (str): Output file with unwrapped phase (default: 'uphase.float').
        format (str): One of 'float', 'byte', 'complex4', 'complex8'.
        num_iter (int): Number of Gauss-Seidel iterations per level.
        num_cycles (int): Number of V-cycle iterations.

    Returns:
        np.ndarray: Unwrapped phase as a NumPy array.
    """
    wrapped_phase = wrapped_phase.astype(np.float32)
    wrapped_phase.tofile(infile)

    xsize = wrapped_phase.shape[-1]
    ysize = wrapped_phase.shape[-2]

    # Convert to C-compatible byte strings
    infile_ctypes = infile.encode('utf-8')
    outfile_ctypes = outfile.encode('utf-8')
    format_ctypes = format.encode('utf-8')

    # Call the C shared library function
    lib.run_unmg(infile_ctypes, outfile_ctypes, format_ctypes,
                 xsize, ysize, num_iter, num_cycles)

    # Read and return result
    uphase = openFile(outfile_ctypes, wrapped_phase.shape)
    os.remove(outfile)
    os.remove(infile)

    return uphase

def run_pcg(wrapped_phase, infile='phase.float', outfile='uphase.float', format='float',
            quality_mode='min_grad', maskfile=None, qualfile=None,
            tsize=3, debug_flag=0, num_iter=10, tolerance=0.0,
            thresh_flag=1, fatten=0):
    """
    Wrapper for the PCG-based phase unwrapping algorithm.

    Parameters:
        wrapped_phase (np.ndarray): Input 2D array with wrapped phase.
        infile, outfile (str): Temp file paths.
        format (str): 'float', 'complex4', 'complex8', or 'byte'.
        quality_mode (str): Quality guidance ('min_grad', etc).
        maskfile (str or None): Optional mask file.
        qualfile (str or None): Optional quality file.
        tsize (int): Template size for smoothing.
        debug_flag (int): Whether to write .qual debug image.
        num_iter (int): Maximum PCG iterations.
        tolerance (float): Stopping condition for convergence.
        thresh_flag (int): Whether to threshold the quality map.
        fatten (int): Border thickening (in pixels).

    Returns:
        np.ndarray: Unwrapped phase.
    """
    wrapped_phase = wrapped_phase.astype(np.float32)
    wrapped_phase.tofile(infile)

    xsize = wrapped_phase.shape[-1]
    ysize = wrapped_phase.shape[-2]

    maskfile = maskfile or 'none'
    qualfile = qualfile or 'none'

    # Encode for ctypes
    args = [
        infile.encode(), outfile.encode(), format.encode(),
        maskfile.encode(), qualfile.encode(),
        xsize, ysize, tsize,
        quality_mode.encode(), debug_flag,
        num_iter, c_double(tolerance),
        thresh_flag, fatten
    ]

    lib.run_pcg(*args)

    uphase = openFile(outfile.encode(), wrapped_phase.shape)
    os.remove(outfile)
    os.remove(infile)
    return uphase

def run_lpno(wrapped_phase, infile='phase.float', outfile='uphase.float', format='float',
             quality_mode='min_grad', maskfile=None, qualfile=None,
             tsize=3, debug_flag=0, num_iter=10, pcg_iter=20, e0=0.001,
             thresh_flag=0, fatten=0):
    """
    Run minimum Lp norm phase unwrapping via C library.

    Parameters:
        wrapped_phase (np.ndarray): Wrapped phase input.
        format (str): Input format ('float', 'byte', 'complex4', etc.)
        quality_mode (str): Guidance mode ('min_grad', 'none', etc.)
        maskfile (str or None): Optional mask file.
        qualfile (str or None): Optional correlation/quality file.
        tsize (int): Template size.
        debug_flag (int): Save intermediate .qual image.
        num_iter (int): Max outer iterations.
        pcg_iter (int): PCG steps per iteration.
        e0 (float): Normalization parameter.
        thresh_flag (int): Use quality thresholding.
        fatten (int): Fatten quality mask in pixels.

    Returns:
        np.ndarray: Unwrapped phase.
    """
    wrapped_phase = wrapped_phase.astype(np.float32)
    wrapped_phase.tofile(infile)

    xsize = wrapped_phase.shape[-1]
    ysize = wrapped_phase.shape[-2]

    maskfile = maskfile or 'none'
    qualfile = qualfile or 'none'

    args = [
        infile.encode(), outfile.encode(), format.encode(),
        maskfile.encode(), qualfile.encode(),
        xsize, ysize, tsize,
        quality_mode.encode(), debug_flag,
        num_iter, pcg_iter, float(e0),
        thresh_flag, fatten
    ]

    lib.run_lpno(*args)
    uphase = openFile(outfile.encode(), wrapped_phase.shape)
    os.remove(outfile)
    os.remove(infile)
    return uphase

def run_unwt(wrapped_phase, infile='phase.float', outfile='uphase.float', format='float'):
    """
    Wrapper for unweighted least-squares phase unwrapping using DCT.

    Parameters:
        wrapped_phase (np.ndarray): 2D NumPy array of wrapped phase.
        infile (str): Input binary file path.
        outfile (str): Output unwrapped result file path.
        format (str): One of 'float', 'byte', 'complex4', or 'complex8'.

    Returns:
        np.ndarray: Unwrapped phase array.
    """
    wrapped_phase = wrapped_phase.astype(np.float32)
    wrapped_phase.tofile(infile)

    xsize = wrapped_phase.shape[-1]
    ysize = wrapped_phase.shape[-2]

    # Check that dimensions are 2^n + 1
    def is_dct_shape(n): return (n - 1) & (n - 2) == 0
    if not is_dct_shape(xsize) or not is_dct_shape(ysize):
        raise ValueError("xsize and ysize must be 2^n + 1 for DCT (e.g., 257, 513)")

    infile_ctypes = infile.encode('utf-8')
    outfile_ctypes = outfile.encode('utf-8')
    format_ctypes = format.encode('utf-8')

    lib.run_unwt(infile_ctypes, outfile_ctypes, format_ctypes, xsize, ysize)

    uphase = openFile(outfile_ctypes, wrapped_phase.shape)
    os.remove(outfile)
    os.remove(infile)

    return uphase

def unwrap_phase_c(method, *args, **kwargs):
    """
    Select and run the specified 2D phase unwrapping algorithm.

    This function wraps multiple C-based unwrapping implementations
    from the book:
        "Two-Dimensional Phase Unwrapping: Theory, Algorithms, and Software"
        by Dennis C. Ghiglia and Mark D. Pritt (Wiley, 1998).

    Each method corresponds to a different algorithm:
        - 'gold'  : Goldstein’s branch-cut algorithm
        - 'qual'  : Quality-guided unwrapping
        - 'mcut'  : Mask-cut method (guided by quality/residues)
        - 'flyn'  : Flynn’s minimum discontinuity network flow
        - 'fmg'   : Multigrid (weighted least-squares)
        - 'unmg'  : Multigrid (unweighted least-squares)
        - 'pcg'   : Preconditioned conjugate gradient solver
        - 'lpno'  : Minimum Lp norm solver (iterative with PCG)
        - 'unwt'  : Unweighted least-squares via cosine transform (DCT)

    Parameters:
        method (str): Algorithm name key, one of the above.
        *args, **kwargs: Arguments forwarded to the corresponding run_*() function.

    Returns:
        np.ndarray: 2D unwrapped phase array.

    Raises:
        ValueError: If an unsupported method is passed.
    """
    if method == 'gold':
        return run_gold(*args, **kwargs)
    elif method == 'qual':
        return run_qual(*args, **kwargs)
    elif method == 'mcut':
        return run_mcut(*args, **kwargs)
    elif method == 'flyn':
        return run_flyn(*args, **kwargs)
    elif method == 'fmg':
        return run_fmg(*args, **kwargs)
    elif method == 'unmg':
        return run_unmg(*args, **kwargs)
    elif method == 'pcg':
        return run_pcg(*args, **kwargs)
    elif method == 'lpno':
        return run_lpno(*args, **kwargs)
    elif method == 'unwt':
        return run_unwt(*args, **kwargs)
    else:
        raise ValueError(f"Method '{method}' not supported.")


def unwrap_phase(wrapped_phase, method):
    """
    Unwrap a 2D wrapped phase array using the specified method.

    Parameters
    ----------
    wrapped_phase : numpy.ndarray
        A 2D array containing the wrapped phase data.
    method : str
        The unwrapping method to use. Supported methods are:
            - 'scipy' or 'scikit': Uses scikit-image's unwrap_phase.
            - 'pyunwrap': Uses the pyunwrap library (new API; generates a dummy quality array).
            - 'unwrap2d': Uses the unwrap2D library.

    Returns
    -------
    numpy.ndarray
        The unwrapped phase data as a 2D array.

    Raises
    ------
    ValueError
        If an unsupported method is provided.
    """
    method_lower = method.lower()

    if method_lower in ['scipy', 'scikit']:
        # Use scikit-image's unwrap_phase function
        return scikit_unwrap(wrapped_phase)

    elif method_lower == 'pyunwrap':
        # For pyunwrap, create a dummy quality array (all ones) with float32 type.
        quality = np.ones_like(wrapped_phase, dtype=np.float32)
        return pyunwrap.unwrap2D(wrapped_phase, quality, miguel=False)

    elif method_lower == 'unwrap2d':
        # For unwrap2D, create a mask array and an output array for the unwrapped phase.
        mask = np.ones_like(wrapped_phase, dtype=np.float32)
        unwrapped = np.ones_like(wrapped_phase)
        unwrap2D.unwrap2D(wrapped_phase, unwrapped_array=unwrapped, mask=mask,
                          wrap_around_x=False, wrap_around_y=False)
        return unwrapped

    else:
        raise ValueError(
            "Unsupported unwrap method. Supported methods are: 'scipy'/'scikit', 'pyunwrap', 'unwrap2d'."
        )
