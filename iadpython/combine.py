"""Module for adding layers together.

Two types of starting methods are possible.

Example::

    >>> import iadpython as iad

    >>> # Isotropic finite layer with mismatched slides the hard way.
    >>> s = iad.Sample(a=0.5, b=1, g=0.0, n=1.4, n_above=1.5, n_below=1.6)
    >>> s.quad_pts = 4
    >>> R01, R10, T01, T10 = iad.boundary_matrices(s, top=True)
    >>> R23, R32, T23, T32 = iad.boundary_matrices(s, top=False)
    >>> R12, T12 = iad.simple_layer_matrices(s)
    >>> R02, R20, T02, T20 = iad.add_layers(s, R01, R10, T01, T10, R12, R12, T12, T12)
    >>> rr03, rr30, tt03, tt30 = iad.add_layers(s, R02, R20, T02, T20, R23, R32, T23, T32)

"""

import copy
import scipy
import numpy as np
import iadpython.constants
import iadpython.start
import iadpython.fresnel

__all__ = (
    "add_layers",
    "add_layers_basic",
    "simple_layer_matrices",
    "add_slide_above",
    "add_slide_below",
    "add_same_slides",
    "mismatched_layer_matrices",
)


def add_layers_basic(sample, R10, T01, R12, R21, T12, T21):
    """Add two layers together.

    The basic equations for the adding-doubling sample (neglecting sources) are

    .. math:: T_{02}  = T_{12} (E - R_{10} R_{12})^{-1} T_{01}

    .. math:: R_{20}  = T_{12} (E - R_{10} R_{12})^{-1} R_{10} T_{21} +R_{21}

    .. math:: T_{20}  = T_{10} (E - R_{12} R_{10})^{-1} T_{21}

    .. math:: R_{02}  = T_{10} (E - R_{12} R_{10})^{-1} R_{12} T_{01} +R_{01}

    Upon examination it is clear that the two sets of equations have
    the same form.  These equations assume some of the multiplications are
    star multiplications. Explicitly,

    .. math:: T_{02}  = T_{12} (E - R_{10} C R_{12} )^{-1} T_{01}

    .. math:: R_{20}  = T_{12} (E - R_{10} C R_{12} )^{-1} R_{10} C T_{21} +R_{21}

    where the diagonal matrices C and E are

    .. math:: E_{ij}= 1/(2𝜈_i w_i) 𝛿_{ij}

    .. math:: C_{ij}= 2𝜈_i w_i 𝛿_{ij}

    Args:
        sample: Sample object
        R10: reflection matrix for light moving upwards 1->0
        T01: transmission array for light moving downwards 1->2
        R12: reflection matrix for light moving downwards 1->2
        R21: reflection matrix for light moving upwards 2->1
        T12: transmission matrix for light moving downwards 1->2
        T21: transmission matrix for light moving upwards 2->1

    Returns:
        R02, T20
    """
    C = np.diagflat(sample.twonuw)
    E = np.diagflat(1 / sample.twonuw)

    A = E - R10 @ C @ R12
    B = np.linalg.solve(A.T, T12.T).T
    R20 = B @ R10 @ C @ T21 + R21
    T02 = B @ T01
    return R20, T02


def add_layers(sample, R01, R10, T01, T10, R12, R21, T12, T21):
    """Add two layers together.

    Use this when the combined system is asymmetric R02!=R20 and T02!=T20.
    """
    R20, T02 = add_layers_basic(sample, R10, T01, R12, R21, T12, T21)
    R02, T20 = add_layers_basic(sample, R12, T21, R10, R01, T10, T01)
    return R02, R20, T02, T20


def double_until(sample, r_start, t_start, b_start, b_end):
    """Double until proper thickness is reached."""
    r = r_start
    t = t_start
    if b_end == 0 or b_end <= b_start:
        return r, t

    if b_end > iadpython.AD_MAX_THICKNESS:
        old_utu = 100
        utu = 10
        while abs(utu - old_utu) > 1e-6:
            old_utu = utu
            r, t = add_layers_basic(sample, r, t, r, r, t, t)
            _, _, _, utu = sample.UX1_and_UXU(r, t)
        return r, t

    while abs(b_end - b_start) > 0.00001 and b_end > b_start:
        r, t = add_layers_basic(sample, r, t, r, r, t, t)
        b_start *= 2
    return r, t


def simple_single_layer_matrices(sample):
    """Create R and T matrices for single layer without boundaries."""
    # avoid b=0 calculation which leads to singular matrices
    if sample.b <= 0:
        sample.b = 1e-9
    r_start, t_start = iadpython.start.thinnest_layer(sample)
    b_start = sample.b_thinnest
    b_end = sample.b_delta_M()
    r, t = double_until(sample, r_start, t_start, b_start, b_end)
    return r, t


def simple_layer_matrices(sample):
    """Create R and T matrices for a sample without boundaries.

    Scalar ``a``/``b``/``g`` solve a single homogeneous layer.  Array values
    are only accepted from the internal layered convention (a materialized
    ``Sample(layers=[...])``, marked by ``sample._layer_pf``); the old public
    form where flat arrays meant layers has been removed.
    """
    if np.isscalar(sample.a) and np.isscalar(sample.b) and np.isscalar(sample.g):
        return simple_single_layer_matrices(sample)

    if getattr(sample, "_layer_pf", None) is None:
        raise RuntimeError(
            "array-valued a/b/g now mean wavelengths, not layers; build a "
            "multilayer sample with iadpython.Sample(layers="
            "[iadpython.Layer(...), ...])"
        )

    return _stack_layer_matrices(sample)


def _stack_layer_matrices(sample):
    """R and T matrices for a matched-index multilayer stack (internal).

    ``sample`` follows the internal layered convention: ``a``/``b``/``g`` are
    per-layer vectors and ``sample._layer_pf`` holds each layer's phase
    function as ``(pf_type, one_column_pf_data_or_None)``.  Every layer may
    therefore use its own Henyey-Greenstein anisotropy or tabulated phase
    function.
    """
    n_layers = len(sample._layer_pf)
    a, b, g = _broadcast_layer_params(sample, n_layers)

    s = copy.deepcopy(sample)
    s._layer_pf = None  # each sub-sample is one homogeneous layer

    r, t = None, None
    for i in range(n_layers):
        s.a = float(a[i])
        s.b = float(b[i])
        s.g = float(g[i])
        s.pf_type, s.pf_data = sample._layer_pf[i]

        # Invalidate unconditionally: the g/pf_data setters skip invalidation
        # when a value is unchanged, so an HG layer following a TABULATED
        # layer with the same g could otherwise reuse a stale hp/hm.
        s.hp = None
        s.hm = None
        s.b_thinnest = None

        ri, ti = simple_single_layer_matrices(s)
        if i == 0:
            r, t = ri, ti
        else:
            r, t = add_layers_basic(s, r, t, ri, ri, ti, ti)

    return r, t


def _add_boundary_config_a(sample, R12, R21, T12, T21, R10, T01):
    """Find two matrices when slide is added to top of slab.

    Compute the resulting 'R20' and 'T02' matrices for a glass slide
    on top of an inhomogeneous layer characterized by 'R12', 'R21', 'T12',
    'T21' using:

    .. math:: T_{02}=T_{12} (E-R_{10}R_{12})^{-1} T_{01}

    .. math:: R_{20}=T_{12} (E-R_{10}R_{12})^{-1} R_{10} T_{21} + R_{21}

    Args:
        sample: Sample object
        R12: reflection matrix for light moving downwards 1->2
        R21: reflection matrix for light moving upwards 2->1
        T12: transmission matrix for light moving downwards 1->2
        T21: transmission matrix for light moving upwards 2->1
        R10: reflection array for light moving upwards 0->1
        T01: transmission array for light moving downwards 1->2

    Returns:
        R20, T02: resulting matrices for combined layers
    """
    n = sample.quad_pts
    X = (np.identity(n) - R10 * R12.T).T
    temp = np.linalg.solve(X.T, T12.T).T
    T02 = temp * T01
    R20 = (temp * R10) @ T21 + R21

    return R20, T02


def _add_boundary_config_b(sample, R12, T21, R01, R10, T01, T10):
    """Find two other matrices when slide is added to top of slab.

    Compute the resulting 'R02' and 'T20' matrices for a glass slide
    on top of an inhomogeneous layer characterized by 'R12', 'R21', 'T12',
    'T21' using:

    .. math:: T_{20}=T_{10} (E-R_{12}R_{10})^{-1} T_{21}

    .. math:: R_{02}=T_{10} (E-R_{12}R_{10})^{-1} R_{12} T_{01} + R_{01}

    Args:
        sample: Sample object
        R12: reflection matrix for light moving downwards 1->2
        T21: transmission matrix for light moving upwards 2->1
        R01: reflection matrix for light moving downwards 0->1
        R10: reflection matrix for light moving upwards 1->0
        T01: transmission array for light moving downwards 1->2
        T10: transmission array for light moving upwards 0->1

    Returns:
        R02, T20
    """
    n = sample.quad_pts
    X = np.identity(n) - R12 * R10
    temp = np.linalg.solve(X.T, np.diagflat(T10)).T
    T20 = temp @ T21
    R02 = (temp @ R12) * T01
    R02 += np.diagflat(R01 / sample.twonuw**2)

    return R02, T20


def add_slide_above(sample, R01, R10, T01, T10, R12, R21, T12, T21):
    """Calculate matrices for a slab with a boundary placed above.

    This routine should be used before the slide is been added below!

    Here 0 is the air/top-of-slide, 1 is the bottom-of-slide/top-of-slab boundary,
    and 2 is the is the bottom-of-slab boundary.

    Args:
        sample: Sample object
        R01: reflection arrays for slide 0->1
        R10: reflection arrays for slide 1->0
        T01: transmission arrays for slide 0->1
        T10: transmission arrays for slide 1->0
        R12: reflection matrices for slab 1->2
        R21: reflection matrices for slab 2->1
        T12: transmission matrices for slab 1->2
        T21: transmission matrices for slab 2->1

    Returns:
        R02, R20, T02, T20: matrices for slide+slab combination
    """
    R20, T02 = _add_boundary_config_a(sample, R12, R21, T12, T21, R10, T01)
    R02, T20 = _add_boundary_config_b(sample, R12, T21, R01, R10, T01, T10)
    return R02, R20, T02, T20


def add_slide_below(sample, R01, R10, T01, T10, R12, R21, T12, T21):
    """Calculate matrices for a slab with a boundary placed below.

    This routine should be used after the slide has been added to the top.

    Here 0 is the top of slab, 1 is the bottom-of-slab/top-of-slide boundary,
    and 2 is the is the bottom-of-slide/air boundary.

    Args:
        sample: Sample object
        R01: reflection arrays for slide 0->1
        R10: reflection arrays for slide 1->0
        T01: transmission arrays for slide 0->1
        T10: transmission arrays for slide 1->0
        R12: reflection matrices for slab 1->2
        R21: reflection matrices for slab 2->1
        T12: transmission matrices for slab 1->2
        T21: transmission matrices for slab 2->1

    Returns:
        R02, R20, T02, T20: matrices for slab+slide combination
    """
    R02, T20 = _add_boundary_config_a(sample, R10, R01, T10, T01, R12, T21)
    R20, T02 = _add_boundary_config_b(sample, R10, T01, R21, R12, T21, T12)
    return R02, R20, T02, T20


def add_same_slides(sample, R01, R10, T01, T10, R, T):
    """Find matrix when slab is sandwiched between identical slides.

    This routine is optimized for a slab with equal boundaries on each side.
    It is assumed that the slab is homogeneous and therefore the 'R' and 'T'
    matrices are identical for upward or downward light directions.

    If equal boundary conditions exist on both sides of the slab then, by
    symmetry, the transmission and reflection operator for light travelling
    from the top to the bottom are equal to those for light propagating from
    the bottom to the top. Consequently only one set need be calculated.
    This leads to a faster method for calculating the reflection and
    transmission for a slab with equal boundary conditions on each side.
    Let the top boundary be layer 01, the medium layer 12, and the bottom
    layer 23.  The boundary conditions on each side are equal:  R_{01}=R_{32},
    R_{10}=R_{23}, T_{01}=T_{32}, and T_{10}=T_{23}.

    For example the light reflected from layer 01 (travelling from boundary
    0 to boundary 1) will equal the amount of light reflected from layer 32,
    since there is no physical difference between the two cases.  The switch
    in the numbering arises from the fact that light passes from the medium
    to the outside at the top surface by going from 1 to 0, and from 2 to 3
    on the bottom surface.  The reflection and transmission for the slab
    with boundary conditions are R_{30} and  T_{03} respectively.  These are
    given by

    .. math:: A_{XX} = T_{12}(E-R_{10}R_{12})^{-1}

    .. math:: R_{20} = A_{XX} R_{10}T_{21} + R_{21}

    .. math:: B_{XX} = T_{10}(E-R_{20}R_{10})^{-1}

    .. math:: T_{03} = B_{XX} A_{XX} T_{01}

    .. math:: R_{30} = B_{XX} R_{20} T_{01} + R_{01}/(2 𝜈 w)^2

    Args:
        sample: Sample object
        R01: R for slide assuming 0=air and 1=slab
        R10: R for slide assuming 0=air and 1=slab
        T10: T for slide assuming 0=air and 1=slab
        T01: T for slide assuming 0=air and 1=slab
        R: R12=R21 for homogeneous slab
        T: T12=T21 for homogeneous slab

    Returns:
        T30, T03: R, T for all 3 with top = bottom boundary
    """
    n = sample.quad_pts
    X = np.identity(n) - R10 * R
    AXX = np.linalg.solve(X, T.T).T
    R20 = (AXX * R10) @ T + R

    X = np.identity(n) - R20 * R10
    BXX = scipy.linalg.solve(X.T, np.diagflat(T10)).T
    T03 = BXX @ AXX * T01
    R30 = BXX @ R20 * T01
    R30 += np.diagflat(R01 / sample.twonuw**2)

    return R30, T03


# ======================================================================
# Refractive-index-mismatched multilayer stack
# ======================================================================
#
# When adjacent layers have different refractive indices there is a real
# Fresnel interface between them.  The strategy (see the module design notes)
# is to place *every* layer on one Snell-invariant master grid built by
# ``Sample._build_master_grid``.  On that grid a physical channel ``j`` keeps
# a fixed invariant ``eta_j`` across all interfaces, so an interface is simply
# a *diagonal* operator in ``j`` — the same structure the air/slide boundaries
# already use.  Each homogeneous layer is doubled on the sub-grid of the
# channels that actually propagate in it (the rest are totally internally
# reflected at its boundaries) and then embedded, with zeros, into the full
# master dimension.
#
# Because the stack is no longer up/down symmetric, all four matrices
# ``(R_down, R_up, T_down, T_up)`` are carried through the accumulation.  Each
# interface and each layer is appended with the existing, tested ``add_layers``
# routine, driven by the twonuw of whichever grid the two blocks actually share
# (the interface's transmission carries the n^2-law factor between grids).  The
# outer air/slide boundaries are added afterwards by ``Sample.rt_matrices``
# using the ordinary ``add_slide_above`` / ``add_slide_below`` on the first and
# last layer grids.


def _grid_sample(sample, n_k, grid, n_full):
    """Clone ``sample`` configured to a single layer's master-grid slice.

    The returned object carries the layer's scalar index and the full-length
    (master-dimension) ``nu``/``twonuw`` so that it can drive the existing
    boundary/adding routines, which key off ``sample.twonuw`` and
    ``sample.quad_pts``.

    Args:
        sample: the multilayer sample being solved.
        n_k: scalar refractive index of this layer/medium.
        grid: ``(nu, twonuw, active)`` from ``Sample.layer_grid``.
        n_full: number of master channels (``quad_pts``).
    """
    nu_k, twonuw_k, _ = grid
    s = copy.deepcopy(sample)
    nk = float(np.real(n_k))
    s._n = nk
    s.n_sample_boundary = nk
    s._quad_pts = n_full
    s.nu = nu_k
    s.twonuw = twonuw_k
    s.hp = None
    s.hm = None
    s.b_thinnest = None
    s._layer_pf = None
    return s


def _single_layer_embedded(sample, n_k, a_k, b_k, g_k, grid, n_full, pf=None):
    """Doubled R/T of one homogeneous layer, embedded in the master dimension.

    The layer is doubled only on the channels that propagate in it (``active``
    mask), then scattered back into a full ``n_full`` x ``n_full`` matrix with
    zeros on the inactive rows/columns.  Inactive channels receive no flux
    (the bounding interfaces transmit nothing into them), so leaving them zero
    is exact.

    Args:
        pf: optional ``(pf_type, one_column_pf_data_or_None)`` tuple giving
            this layer its own phase function; when omitted the layer scatters
            Henyey-Greenstein with anisotropy ``g_k``.

    Returns:
        ``(R, T)`` full master-dimension matrices (symmetric for a single
        homogeneous layer, so up == down).
    """
    import iadpython.ad as _ad

    nu_k, twonuw_k, active = grid
    idx = np.where(active)[0]

    s = copy.deepcopy(sample)
    s._n = float(np.real(n_k))
    s.n_sample_boundary = float(np.real(n_k))
    s.a = float(a_k)
    s.b = float(b_k)
    s._g = _ad.sanitize_anisotropy(float(g_k))
    s._quad_pts = len(idx)
    s.nu = nu_k[idx]
    s.twonuw = twonuw_k[idx]
    if pf is not None:
        s.pf_type, s.pf_data = pf
    s.hp = None
    s.hm = None
    s.b_thinnest = None
    s._layer_pf = None

    r_sub, t_sub = simple_single_layer_matrices(s)

    R = np.zeros((n_full, n_full))
    T = np.zeros((n_full, n_full))
    R[np.ix_(idx, idx)] = r_sub
    T[np.ix_(idx, idx)] = t_sub
    return R, T


def _internal_interface_matrices(n_upper, grid_upper, n_lower, grid_lower, cache=None):
    r"""Full diagonal R/T operators for the interface between two media.

    The interface is a zero-thickness layer combined with the general
    ``add_layers``.  Written in the adding-doubling "layer" convention (where a
    transparent layer is ``diag(1/twonuw)``), the diagonal operators are

    * reflection: ``R_power / twonuw`` on the *incidence* grid,
    * transmission: ``T_power / twonuw`` on the *incidence* grid.

    Dividing the transmission by the incidence-side ``twonuw`` — rather than the
    exit side — is exactly what encodes the :math:`n^2`-law of radiance across
    the index step; it was verified against energy conservation.  Total internal
    reflection is automatic because ``interface_rt`` returns ``(R, T) = (1, 0)``
    for any channel beyond the critical angle.

    Args:
        n_upper, n_lower: refractive indices of the two media.
        grid_upper, grid_lower: ``(nu, twonuw, active)`` tuples for each side.

    Returns:
        ``(R12, R21, T12, T21)`` full diagonal matrices, where ``1`` labels the
        upper medium and ``2`` the lower medium (``12`` = downward).
    """
    nu_u, tw_u, active_u = grid_upper
    nu_l, tw_l, active_l = grid_lower

    # Within one stack build, an index pair may recur (e.g. A|B|A|B ...).  The
    # per-build ``cache`` dict avoids recomputing those interfaces.  It is not
    # shared across builds, so it cannot go stale when the master grid changes.
    key = (float(np.real(n_upper)), float(np.real(n_lower)))
    if cache is not None and key in cache:
        return cache[key]

    r_ab, t_ab = iadpython.fresnel.interface_rt(n_upper, nu_u, n_lower)
    r_ba, t_ba = iadpython.fresnel.interface_rt(n_lower, nu_l, n_upper)

    # A channel that does not propagate in a medium must stay fully decoupled
    # there, otherwise its placeholder weight injects spurious flux into the
    # star products and the recursion diverges.  Reflection is kept on the
    # medium it returns to (this is the physical total-internal-reflection of
    # trapped light); transmission survives only where *both* sides propagate.
    both = active_u & active_l
    r_ab = np.where(active_u, np.asarray(r_ab, dtype=float), 0.0)
    r_ba = np.where(active_l, np.asarray(r_ba, dtype=float), 0.0)
    t_ab = np.where(both, np.asarray(t_ab, dtype=float), 0.0)
    t_ba = np.where(both, np.asarray(t_ba, dtype=float), 0.0)

    R12 = np.diagflat(r_ab / tw_u)
    R21 = np.diagflat(r_ba / tw_l)
    T12 = np.diagflat(t_ab / tw_u)
    T21 = np.diagflat(t_ba / tw_l)

    result = (R12, R21, T12, T21)
    if cache is not None:
        cache[key] = result
    return result


def _broadcast_layer_params(sample, n_layers):
    """Return (a, b, g) each as length-``n_layers`` arrays."""
    a = np.atleast_1d(sample.a)
    b = np.atleast_1d(sample.b)
    g = np.atleast_1d(sample.g)

    def fit(x, name):
        if x.size == 1:
            return np.full(n_layers, x.flat[0])
        if x.size != n_layers:
            raise RuntimeError(
                "mismatched_layer_matrices: %s has %d entries but n has %d layers"
                % (name, x.size, n_layers)
            )
        return x

    return fit(a, "a"), fit(b, "b"), fit(g, "g")


def mismatched_layer_matrices(sample):
    """R/T matrices for a stack whose layers have different refractive indices.

    Every layer is expressed on the shared Snell-invariant master grid; the
    Fresnel interface between neighbouring layers is inserted as a diagonal
    layer.  Handles arbitrarily large index steps (via per-channel total
    internal reflection) and reduces exactly to the single-index doubling when
    all indices agree.

    Args:
        sample: a :class:`~iadpython.ad.Sample` with an array-valued ``n``.

    Returns:
        ``(R_down, R_up, T_down, T_up)`` full master-dimension matrices for the
        bare stack (no external air/slide boundaries — those are added by
        ``Sample.rt_matrices``).  The top face is the first layer's medium and
        the bottom face is the last layer's medium.
    """
    layer_pf = getattr(sample, "_layer_pf", None)
    if layer_pf is None and str(getattr(sample, "pf_type", "HG")).upper() != "HG":
        raise NotImplementedError(
            "mismatched-index multilayer needs per-layer phase functions; "
            "build the stack with iadpython.Sample(layers=[iadpython.Layer("
            "pf_type='TABULATED', ...), ...]) or pf_type='MOMENTS'"
        )

    sample._build_master_grid()
    n_full = sample._master_dim
    ns = np.atleast_1d(sample.n)
    n_layers = ns.size
    a, b, g = _broadcast_layer_params(sample, n_layers)

    def pf_of(k):
        """Per-layer phase-function spec, or None for plain HG."""
        if layer_pf is None:
            return None
        return layer_pf[k]

    grids = [sample.layer_grid(ns[k]) for k in range(n_layers)]
    interface_cache = {}

    # First layer seeds the running stack (a homogeneous layer is symmetric).
    R0, T0 = _single_layer_embedded(
        sample, ns[0], a[0], b[0], g[0], grids[0], n_full, pf=pf_of(0)
    )
    r_down, r_up, t_down, t_up = R0, R0, T0, T0

    for k in range(1, n_layers):
        # (1) Add the Fresnel interface between layer k-1 and layer k below the
        #     running stack.  The two blocks share the (k-1) grid, so the star
        #     product uses its twonuw.  Afterwards the stack's bottom face lives
        #     on the k grid.
        R12, R21, T12, T21 = _internal_interface_matrices(
            ns[k - 1], grids[k - 1], ns[k], grids[k], cache=interface_cache
        )
        s_upper = _grid_sample(sample, ns[k - 1], grids[k - 1], n_full)
        r_down, r_up, t_down, t_up = add_layers(
            s_upper, r_down, r_up, t_down, t_up, R12, R21, T12, T21
        )

        # (2) Add layer k below the stack.  They share the k grid.
        Lk_R, Lk_T = _single_layer_embedded(
            sample, ns[k], a[k], b[k], g[k], grids[k], n_full, pf=pf_of(k)
        )
        s_lower = _grid_sample(sample, ns[k], grids[k], n_full)
        r_down, r_up, t_down, t_up = add_layers(
            s_lower, r_down, r_up, t_down, t_up, Lk_R, Lk_R, Lk_T, Lk_T
        )

    return r_down, r_up, t_down, t_up
