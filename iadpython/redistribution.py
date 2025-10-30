r"""Module for calculating the redistribution function.

The single scattering phase function ..math:`p(\nu)` for a tissue determines the
amount of light scattered at an angle nu=cos(theta) from the direction of
incidence.  The subtended angle nu is the dot product incident and exiting
of the unit vectors.

The redistribution function ..math:`h[i,j]` determines the fraction of light
scattered from an incidence cone with angle `\nu_i` into a cone with angle
..math:`\nu_j`.  The redistribution function is calculated by averaging the phase
function over all possible azimuthal angles for fixed angles ..math:`\nu_i` and
..math:`nu_j`,

Note that the angles ..math:`\nu_i` and ..math:`\nu_j` may also be negative (light
travelling in the opposite direction).

When the cosine of the angle of incidence or exitance is unity (..math:`\nu_i=1` or
..math:`\nu_j=1`), then the redistribution function is equivalent to the phase
function ..math:`p(\nu_j)`.
"""

import scipy.special
import scipy.integrate
import warnings
import numpy as np
from numpy.polynomial.legendre import legvander, leggauss
from scipy.special import eval_legendre
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline

__all__ = ('hg_elliptic',
           'phase_legendre',
           'legendre_coeffs_from_df')

def legendre_coeffs_from_df(
        df, *, quad_pts=8, n_mom=None, spline_bc='not-a-knot'):
    """
    Exact Legendre moments a_l (no ½(2l+1) factor) for a phase function
    tabulated on μ = cosθ.

    Parameters
    ----------
    df        : pandas.DataFrame
                index = μ (strictly monotonic), columns = spectra
    quad_pts  : kept only for API symmetry (default 8)
    n_mom     : highest Legendre order (default 2*quad_pts)
    spline_bc : boundary condition passed to CubicSpline

    Returns
    -------
    a_raw : ndarray, shape (n_mom, nλ)
            Legendre moments  a_l = ∫_{-1}^1 p(μ) P_l(μ) dμ ,  l=0…n_mom-1
            *normalised* so that a_0 = 1 for every spectrum.
    """
    # ---------- sanity -------------------------------------------------
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    
    mu = df.index.to_numpy()
    if not (np.all(np.diff(mu) > 0) or np.all(np.diff(mu) < 0)):
        raise ValueError("DataFrame index (μ) must be strictly monotonic.")

    df = df.sort_index()                  # ascending μ for CubicSpline
    mu = df.index.to_numpy()              # ascending version
    p  = df.to_numpy().T                 # shape (nλ, Nμ)

    if n_mom is None:
        n_mom = 2 * quad_pts + 1          # Wiscombe default

    nλ = p.shape[0]
    a_raw = np.zeros((n_mom, nλ))

    g_x, g_w = leggauss(128)                    # 128-pt global GL rule

    # affine-map from [-1,1] to tabulation domain [-1,1] (already same here)
    # evaluate spline and P_l once:
    P_all = np.array([eval_legendre(l, g_x) for l in range(n_mom)])   # (n_mom,128)

    for col, y in enumerate(p):                # spline for this spectrum
        f = CubicSpline(mu, y, bc_type=spline_bc)
        vals = f(g_x)                          # p(μ) on Gauss nodes
        a_raw[:, col] = (P_all * vals).dot(g_w)      # ∑ w_i p_i P_l(μ_i)
        a_raw[:, col] /= a_raw[0, col]               # normalise

    return a_raw

# ----------------------------------------------------------------------
# unified HG / tabulated redistribution via Legendre polynomials
# ----------------------------------------------------------------------
def phase_legendre(sample, *, deltam=True):
    """
    Build (hp, hm) single-scatter redistribution blocks.

    Modes
    -----
    * pf_type == 'HG'          : analytic Henyey–Greenstein (fast)
    * pf_type == 'TABULATED'   : DataFrame in sample.pf_data
                                 → Legendre coeffs via helper above
    Parameters
    ----------
    sample : iadpython.Sample   (must have quad_pts, nu, etc.)
    deltam : bool               apply δ-M forward-spike split

    Returns
    -------
    hp, hm : ndarray (N × N)
    """

    if sample.nu is None:
        sample.update_quadrature()

    mu  = sample.nu
    n   = sample.quad_pts
    P   = legvander(mu, 2*n).T                  # (2N+1, N)
    w   = sample.twonuw / (2*mu)

    pf_type = getattr(sample, "pf_type", "HG").upper()

    # ---------------------------------------------------- HG fast path
    if pf_type == "HG":
        g   = float(sample.g)
        if g == 0.0:
            hp = hm = np.ones((n, n))
            return hp, hm
        a_raw = g ** np.arange(2*n+1)           # analytic a_ℓ = g^ℓ

    # -------------------------------------------- Custom tabulated path
    elif pf_type == "TABULATED":
        df = sample.pf_data
        a_raw = legendre_coeffs_from_df(df,
                                        quad_pts=n,
                                        n_mom=2*n+1)  # (2N+1 , nλ=1)
        a_raw = a_raw.squeeze()                  # 1-D

    else:
        raise ValueError(f"Unknown pf_type '{pf_type}'")

    # -------------------------------------------- optional δ-M trunc
    if deltam:
        f_spike = a_raw[2*n]
        if not (0.0 <= f_spike < 1.0):
            # you can raise, clip, or fall back to isotropic
            print(f"Warning: invalid forward spike fraction: f_spike={f_spike:.4f}. Clipping to [0,1).")
            f_spike = np.clip(f_spike, 0.0, 1.0)

        den = 1.0 - f_spike
        if den < 1e-10:
            print(f"Warning: f_spike too large, δ–M ill-conditioned")
            den = 1e-10
        a_use = (a_raw[:n] - f_spike) / den

        Lmax    = n
        P_use   = P[:Lmax]
    else:
        a_use   = a_raw[:2*n+1]
        Lmax    = 2*n+1
        P_use   = P[:Lmax]

    chi = (2*np.arange(Lmax) + 1) * a_use       # χ_ℓ weights

    # -------------------------------------------- assemble blocks
    hp = np.ones((n, n))
    hm = np.ones((n, n))

    for k in range(1, Lmax):
        pk  = P_use[k]
        add = chi[k] * np.outer(pk, pk)
        hp += add
        hm += (-1)**k * add
    return hp, hm


def hg_elliptic(sample):
    """Calculate redistribution function using elliptic integrals.

    This is the result of a direct integration of the Henyey-
    Greenstein phase function.

    It is not terribly useful because we cannot use the
    delta-M method to more accurate model highly anisotropic
    phase functions.
    """
    if sample.nu is None:
        sample.update_quadrature()

    n = sample.quad_pts
    g = sample.g**n
    if g == 0:
        h = np.ones([n, n])
        return h, h

    hp = np.zeros([n, n])
    hm = np.zeros([n, n])
    for i in range(n):
        for j in range(i + 1):
            ni = sample.nu[i]
            nj = sample.nu[j]
            gamma = 2 * g * np.sqrt(1 - ni**2) * np.sqrt(1 - nj**2)

            alpha = 1 + g**2 - 2 * g * ni * nj
            const = 2 / np.pi * (1 - g**2) / (alpha - gamma) / np.sqrt(alpha + gamma)
            arg = np.sqrt(2 * gamma / (alpha + gamma))
            hp[i, j] = const * scipy.special.ellipe(arg)

            alpha = 1 + g**2 + 2 * g * ni * nj
            const = 2 / np.pi * (1 - g**2) / (alpha - gamma) / np.sqrt(alpha + gamma)
            arg = np.sqrt(2 * gamma / (alpha + gamma))
            hm[i, j] = const * scipy.special.ellipe(arg)

    # fill symmetric entries
    for i in range(n):
        for j in range(i + 1, n):
            hp[i, j] = hp[j, i]
            hm[i, j] = hm[j, i]

    return hp, hm

def legendre_moments(g=None, custom_phase_function=None, max_order=16):
    """
    Compute the Legendre expansion coefficients (chi_k) of a phase function.

    If custom_phase_function is provided, compute chi_k via integration over [-1, 1].
    Otherwise, use Henyey-Greenstein with parameter g.
    If both are None, assume isotropic scattering (g=0) and issue a warning.

    Arguments:
        g: Anisotropy factor for Henyey-Greenstein (float).
        custom_phase_function: Function of mu (float in [-1, 1]) returning P(mu).
        max_order: Number of Legendre moments to compute (int).

    Returns:
        chi: array of chi_k coefficients.
    """
    chi = np.zeros(max_order)

    if custom_phase_function is not None:
        if g is not None:
            warnings.warn("custom_phase_function is set; g will be ignored.")

        for k in range(max_order):
            integrand = lambda mu: custom_phase_function(mu) * scipy.special.legendre(k)(mu)
            chi[k], _ = scipy.integrate.quad(integrand, -1, 1)
            chi[k] *= (2 * k + 1) / 2

    elif isinstance(g, (float, int)):
        for k in range(max_order):
            chi[k] = (2 * k + 1) * (g ** k - g ** max_order) / (1 - g ** max_order)

    elif g is not None:
        raise TypeError("g must be a float or int if provided.")
    
    else:
        warnings.warn("Neither g nor custom_phase_function provided. Proceeding with isotropic Henyey-Greenstein (g=0).")
        g = 0
        for k in range(max_order):
            chi[k] = (2 * k + 1) * (g ** k - g ** max_order) / (1 - g ** max_order) if g != 1 else 0

    return chi