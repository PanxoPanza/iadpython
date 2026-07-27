"""Class for doing adding-doubling calculations for a sample.

Example::

    >>> import iadpython as iad
    >>> n=4
    >>> sample = iad.Sample(a=0.9, b=10, g=0.9, n=1.5, quad_pts=4)
    >>> ur1, ut1, uru, utu = sample.rt()
    >>> print(ur1)
    >>> print(ut1)
"""

import copy
import numpy as np
import pandas as pd
import iadpython.fresnel
import iadpython.quadrature
import iadpython.start
import iadpython.combine
import iadpython.layer
import iadpython.redistribution
from scipy.interpolate import CubicSpline as _CubicSpline
from scipy.integrate import quad as _quad

G_MINUS_ONE_SUBSTITUTE = -0.9999
_OUTER_INTERFACE_GL_ORDER = 64
_OUTER_INTERFACE_GL_X, _OUTER_INTERFACE_GL_W = np.polynomial.legendre.leggauss(_OUTER_INTERFACE_GL_ORDER)


def stringify(form, x):
    """
    Create a string from x.

    Args:
        form: format for conversion
        x: scalar or array

    Returns:
        reasonable string
    """
    if x is None:
        s = "None"
    elif np.isscalar(x) or np.ndim(x) == 0:
        s = form % x
    else:
        mn = min(x)
        mx = max(x)
        s = form % mn
        s += " to "
        s += form % mx
    return s


def sanitize_anisotropy(g):
    """Map singular anisotropy endpoint g=-1 to a nearby finite value."""
    if g is None:
        return None

    if np.isscalar(g):
        if np.isclose(g, -1.0):
            return G_MINUS_ONE_SUBSTITUTE
        return g

    arr = np.asarray(g)
    return np.where(np.isclose(arr, -1.0), G_MINUS_ONE_SUBSTITUTE, arr)

def _indices_match(a, b, *, tol=1e-12):
    """Compare refractive indices with a small complex tolerance."""
    return bool(np.isclose(np.real(a), np.real(b), atol=tol, rtol=0.0) and
                np.isclose(np.imag(a), np.imag(b), atol=tol, rtol=0.0))


class Sample:
    """
    Most things can be changed after creation by assigning to an element.

    The angle of incidence is assumed to be perpendicular to the
    surface.  This is stored as the cosine of the angle and therefore to
    change it to 60° from the normal, one does

    Example::

        sample.nu_0 = 0.5

    To avoid needing to calculate the quadrature angles each time a
    calculation is done, the `Sample` object stores the quadrature angles as
    well as the redistribution function.  A bit of trouble is taken
    to ensure that these values get updated when something changes
    e.g., the anisotropy, the angle of incidence, or the number of
    quadrature points.

    Attributes:
        - a: albedo
        - b: optical thickness
        - d: physical thickness [mm]
        - g: scattering anisotropy [-]
        - n: index of refraction of sample
        - n_above: index of refraction of slide above
        - n_below: index of refraction of slide below
        - quad_pts: number of quadrature points

    Multilayer samples.
        Build a stack from explicit layers: ``Sample(layers=[Layer(...),
        Layer(...)])``.  Each :class:`iadpython.Layer` carries its own ``a``,
        ``b``, ``g``, ``n``, ``d`` and phase function (Henyey-Greenstein or
        TABULATED, freely mixed between layers).  The layer axis is the
        position in the list; the *wavelength* axis is arrays inside each
        layer attribute (scalars broadcast across the sweep).  Call ``rt()``
        to drive the wavelength sweep, or ``rt_matrices`` +  ``UX1_and_UXU``
        for a single-wavelength stack.

        Flat arrays assigned to ``a``/``b``/``g`` of a non-layered sample
        always mean wavelengths (they are swept by ``rt``); the historical
        form where such arrays meant layers was removed.

        When the per-layer indices differ, the real Fresnel interfaces
        between the layers are modelled on a shared Snell-invariant grid that
        handles total internal reflection, so arbitrarily large index steps
        are allowed.  Equal indices run the original single-index code
        unchanged.  The mismatched solver splits the angular grid into one
        panel per distinct index, each with ``quad_pts`` nodes, so the working
        dimension is ``quad_pts`` times the number of distinct indices.  Keep
        ``quad_pts`` moderate (roughly 8-16); as with the classic solver,
        very large values eventually become ill-conditioned.  Strongly
        scattering, weakly absorbing stacks with large index steps converge
        more slowly and benefit from a somewhat higher ``quad_pts``.

    """

    def __init__(
        self,
        a=0,
        b=1,
        g=0,
        d=1,
        n=1,
        n_above=1,
        n_below=1,
        n_outer_above=1,
        n_outer_below=1,
        n_sample_boundary=None,
        quad_pts=4,
        pf_type="HG",
        pf_data=None,
        layers=None,
    ):
        """Object initialization.

        ``pf_data`` may be a pandas ``DataFrame`` with one column per
        wavelength when ``pf_type`` is ``'TABULATED'``, or a plain ndarray of
        Legendre moments (1-D, or 2-D shape ``(n_mom, n_wavelength)``) when
        ``pf_type`` is ``'MOMENTS'`` -- moments supplied directly bypass the
        spline+quadrature ``'TABULATED'`` needs, for callers that can
        evaluate and integrate their own phase function exactly. The number
        of columns must match other wavelength-dependent parameters.

        ``layers`` may be a list of :class:`iadpython.Layer` objects to build
        a multilayer sample.  When given, the flat ``a``/``b``/``g``/``d``/
        ``n``/``pf_type``/``pf_data`` arguments are ignored -- each layer
        carries its own values (scalar or wavelength-dependent).  Boundary
        properties (``n_above`` etc.) always live on the Sample.

        Returns:
            object with all details needed to do a radiative calculation
        """
        self.a = a
        self.b = b
        self._g = sanitize_anisotropy(g)
        self.d = d  # thickness of sample in mm
        self._n = n
        self.n_above = n_above
        self.n_below = n_below
        self.n_outer_above = n_outer_above
        self.n_outer_below = n_outer_below
        self.n_sample_boundary = n if n_sample_boundary is None else n_sample_boundary
        self.d_above = 1  # thickness of top slide in mm
        self.d_below = 1  # thickness of bot slide in mm
        self._nu_0 = 1.0
        self.b_above = 0
        self.b_below = 0
        self._quad_pts = quad_pts
        # ----- phase-function control ----------------------------------
        self.pf_type = str(pf_type).upper()  # 'HG' or 'TABULATED'
        self._pf_data = None
        self.pf_data = pf_data  # None or pandas DataFrame
        # ---------------------------------------------------------------
        self.b_thinnest = None
        self.nu = None
        self.twonuw = None
        self.hp = None
        self.hm = None
        # ----- multilayer structure (Layer API) ------------------------
        # layers: public list of iadpython.Layer objects (or None for the
        # classic flat single-layer form).  _layer_pf: internal marker set by
        # _materialize_layers -- a per-layer [(pf_type, one_column_df)] list
        # signalling that a/b/g/n currently hold the arrays-over-layers
        # convention for one wavelength of a layered solve.
        self.layers = list(layers) if layers is not None else None
        self._layer_pf = None

    @property
    def n(self):
        """Getter property for index of refraction."""
        return self._n

    @n.setter
    def n(self, value):
        """When index is changed quadrature becomes invalid.

        ``value`` may be a scalar (single index shared by every layer) or an
        array with one entry per layer.  A per-layer array enables the
        refractive-index-mismatched multilayer solver (see ``rt_matrices``).
        """
        same = np.array_equal(np.atleast_1d(value), np.atleast_1d(self._n))
        if not same:
            self.nu = None
            self.twonuw = None
            self.hp = None
            self.hm = None
            self._n = value

    @property
    def nu_0(self):
        """Getter property for index of refraction."""
        return self._nu_0

    @nu_0.setter
    def nu_0(self, value):
        """When index is changed quadrature becomes invalid."""
        if value != self._nu_0:
            self.nu = None
            self.twonuw = None
            self.hp = None
            self.hm = None
            self._nu_0 = value

    @property
    def g(self):
        """Getter property for anisotropy."""
        return self._g

    @g.setter
    def g(self, value):
        """When anisotropy is changed phi is invalid."""
        value = sanitize_anisotropy(value)
        if np.isscalar(value) and np.isscalar(self._g) and value == self._g:
            return

        self.hp = None
        self.hm = None
        self._g = value

    @property
    def quad_pts(self):
        """Getter property for number of quadrature points."""
        return self._quad_pts

    @quad_pts.setter
    def quad_pts(self, value):
        """When quadrature points are changed everything is invalid."""
        if value != self._quad_pts:
            self.nu = None
            self.twonuw = None
            self.hp = None
            self.hm = None
            self._quad_pts = value

    @property
    def pf_data(self):
        """Phase function data as DataFrame."""
        return self._pf_data

    @pf_data.setter
    def pf_data(self, value):
        """Set phase data and invalidate cached hp/hm."""
        self._pf_data = value
        self.hp = None
        self.hm = None

    def mu_a(self):
        """Absorption coefficient for the sample.

        Works for scalar or per-layer array ``b`` (elementwise ``np.isinf``
        check, so a mix of finite and semi-infinite layers is handled too).
        """
        if self.a is None or self.b is None or self.d is None:
            return None
        a = np.asarray(self.a, dtype=float)
        b = np.asarray(self.b, dtype=float)
        d = np.asarray(self.d, dtype=float)
        with np.errstate(invalid="ignore"):
            finite = (1 - a) * b / d
        out = np.where(np.isinf(b), 1 - a, finite)
        return float(out) if out.ndim == 0 else out

    def mu_s(self):
        """Scattering coefficient for the sample.

        Works for scalar or per-layer array ``b``; see ``mu_a``.
        """
        if self.a is None or self.b is None or self.d is None:
            return None
        a = np.asarray(self.a, dtype=float)
        b = np.asarray(self.b, dtype=float)
        d = np.asarray(self.d, dtype=float)
        with np.errstate(invalid="ignore"):
            finite = a * b / d
        out = np.where(np.isinf(b), a, finite)
        return float(out) if out.ndim == 0 else out

    def mu_sp(self):
        """Reduced scattering coefficient for the sample.

        Works for scalar or per-layer array ``b``; see ``mu_a``.
        """
        if self.a is None or self.b is None or self.d is None or self.g is None:
            return None
        a = np.asarray(self.a, dtype=float)
        b = np.asarray(self.b, dtype=float)
        d = np.asarray(self.d, dtype=float)
        g = np.asarray(self.g, dtype=float)
        with np.errstate(invalid="ignore"):
            finite = (1 - g) * a * b / d
        out = np.where(np.isinf(b), a * (1 - g), finite)
        return float(out) if out.ndim == 0 else out

    # ------------------------------------------------------------------
    # Per-layer refractive index support
    # ------------------------------------------------------------------
    #
    # ``Sample.n`` may be a scalar (all layers share one index, the classic
    # case) or a 1-D array with one entry per layer.  When the per-layer
    # indices differ, ``rt_matrices`` switches to a Snell-invariant "master
    # grid" solver that handles the Fresnel interfaces *between* layers,
    # including total internal reflection for large index steps.  The helpers
    # below are the gate (``is_index_matched``) that keeps the classic,
    # single-index code path byte-for-byte unchanged, plus the grid
    # construction used only by the mismatched solver.

    def index_array(self):
        """Return the per-layer refractive indices as a 1-D array."""
        return np.atleast_1d(self.n)

    def is_index_matched(self, tol=1e-9):
        """True when every layer shares the same refractive index.

        This is the performance gate: when it returns True the sample is a
        single homogeneous index and all the original (fast) code paths run
        untouched.  Only a genuine per-layer index mismatch activates the
        master-grid machinery.
        """
        n = np.atleast_1d(self.n)
        if n.size == 1:
            return True
        return bool(np.allclose(n, n.flat[0], atol=tol, rtol=0.0))

    def n_max(self):
        """Largest real part among the per-layer indices (the densest layer)."""
        return float(np.max(np.real(np.atleast_1d(self.n))))

    def _master_breakpoints_u(self, n_hi):
        r"""Invariant values :math:`u=\eta^2` where a medium's critical angle sits.

        A ray with invariant :math:`\eta` is totally internally reflected when
        it meets a medium of index :math:`n<\eta`, i.e. once :math:`u>n^2`.
        Splitting the master grid at :math:`u=n_k^2` for every distinct medium
        index (layers and escape media) makes each layer's propagating range a
        union of whole panels, so its flux integral converges, and it puts the
        physically measured air cone (``u`` up to the outer index squared) in
        its own fully resolved panel.
        """
        indices = list(np.real(np.atleast_1d(self.n)))
        for exit_index in (self._exit_index(top=True), self._exit_index(top=False)):
            if exit_index is not None:
                indices.append(float(exit_index))

        n_hi2 = n_hi**2
        breaks = set()
        for n_k in indices:
            u_k = float(n_k) ** 2
            if 1e-12 < u_k < n_hi2 - 1e-12:
                breaks.add(u_k)
        return sorted(breaks)

    def _build_master_grid(self):
        r"""Build the Snell-invariant master quadrature shared by all layers.

        The invariant along any ray crossing planar interfaces is
        :math:`\eta=n\sin\theta` (Snell's law).  We discretise the squared
        invariant :math:`u=\eta^2` on :math:`[0, n_{max}^2]`, split into panels
        at every medium's :math:`u=n_k^2` (see ``_master_breakpoints_u``).  Each
        panel carries ``quad_pts`` nodes, so both the measured air cone and the
        internally trapped range are resolved.  Building directly in ``u`` keeps
        the flux weights well conditioned (unlike a densest-layer cosine grid,
        whose air cone is a narrow, weight-starved sliver).

        Each layer's cosine grid then follows in closed form,

        .. math:: \nu_{k,j}=\sqrt{1-u_j/n_k^2},

        with flux weights :math:`\Omega_j/n_k^2` — the :math:`1/n_k^2` being
        exactly the :math:`n^2`-law of radiance.  A channel with
        :math:`u_j>n_k^2` does not propagate in layer ``k`` (it is beyond the
        critical angle) and is flagged inactive; at an interface it is totally
        internally reflected, which is how arbitrarily large index steps stay
        exact.

        Stores ``_u_master`` (the :math:`u_j`), ``_omega_master`` (the weights
        :math:`\Omega_j` integrating :math:`\int du`) and ``_n_hi``.
        """
        n_hi = self.n_max()
        edges = [0.0] + self._master_breakpoints_u(n_hi) + [n_hi**2]
        n_panels = len(edges) - 1
        m = self.quad_pts

        u_parts = []
        w_parts = []
        for i in range(n_panels):
            lo, hi = edges[i], edges[i + 1]
            if i == 0:
                # First panel: Radau reflected so that u = 0 (normal incidence,
                # nu = 1 in every layer) is a node -- needed for the collimated
                # beam.  radau() includes its upper endpoint, so build on
                # [0, hi] and reflect u -> hi - u to land the node on u = 0.
                x, w = iadpython.quadrature.radau(m, a=0.0, b=hi)
                x = hi - x
            else:
                x, w = iadpython.quadrature.gauss(m, a=lo, b=hi)
            u_parts.append(x)
            w_parts.append(w)

        u = np.concatenate(u_parts)
        omega = np.concatenate(w_parts)
        # Sort by decreasing u so that the per-layer cosines nu = sqrt(1-u/n^2)
        # come out increasing (grazing first, normal incidence u=0 last), the
        # same channel ordering the classic single-index grid uses.
        order = np.argsort(u)[::-1]
        self._u_master = u[order]
        self._omega_master = omega[order]
        self._n_hi = n_hi
        self._master_dim = len(u)
        return self._u_master, self._omega_master

    def layer_grid(self, n_k):
        r"""Cosine grid and flux weights of one layer on the master grid.

        Args:
            n_k: refractive index of the layer.

        Returns:
            ``(nu_k, twonuw_k, active)`` each of length ``quad_pts``.
            ``nu_k`` are the direction cosines in the layer (0 on inactive
            channels), ``twonuw_k`` the adding-doubling flux weights
            :math:`\Omega_j/n_k^2` (a harmless placeholder of 1 on inactive
            channels so that ``1/twonuw`` never divides by zero), and
            ``active`` a boolean mask of the channels that propagate in the
            layer.
        """
        if getattr(self, "_u_master", None) is None:
            self._build_master_grid()

        nk = float(np.real(n_k))
        nk2 = nk * nk
        u = self._u_master
        active = u <= nk2 * (1.0 + 1e-9)

        nu_k = np.zeros_like(u)
        nu_k[active] = np.sqrt(np.clip(1.0 - u[active] / nk2, 0.0, 1.0))
        twonuw_k = np.where(active, self._omega_master / nk2, 1.0)
        return nu_k, twonuw_k, active

    @staticmethod
    def _slice_start_grid(nu, nu_c):
        """First channel of ``nu`` whose cosine exceeds the escape cutoff ``nu_c``."""
        idx = np.where(nu > nu_c)[0]
        if idx.size == 0:
            return len(nu) - 1
        return int(idx[0])

    def _exit_index(self, *, top):
        """Effective real index that limits escape on one side.

        Only ``slide``/``outer`` need to be real here -- they are the two
        values this function actually reduces to a critical-angle index.
        ``n_sample_boundary`` feeds the *Fresnel amplitude* calculation in
        ``rt_matrices`` and is unrelated to this geometric cutoff; gating on
        it made every absorbing sample (any nonzero Im(n_sample_boundary),
        however small) fall back to ``nu_c = 0`` -- i.e. "no critical angle,
        full hemisphere escapes" -- which silently breaks the n**2 diffuse
        (Lambertian) normalization in ``UX1_and_UXU`` for any index-mismatched
        absorbing sample (URU could exceed 1).
        """
        slide = self.n_above if top else self.n_below
        outer = self.n_outer_above if top else self.n_outer_below
        b_slide = self.b_above if top else self.b_below

        if (iadpython.fresnel._is_complex_value(slide) or
                iadpython.fresnel._is_complex_value(outer)):
            return None

        slide_real = iadpython.fresnel._transport_index(slide)
        outer_real = iadpython.fresnel._transport_index(outer)

        # A slide index of 1.0 with zero optical thickness represents "no slide".
        if np.isclose(b_slide, 0.0) and np.isclose(slide_real, 1.0):
            return outer_real

        return min(slide_real, outer_real)

    def nu_c_above(self):
        """Cosine of the escape critical angle on the top side."""
        n_exit = self._exit_index(top=True)
        if n_exit is None:
            return 0.0
        return iadpython.fresnel.cos_critical(self.n, n_exit)

    def nu_c_below(self):
        """Cosine of the escape critical angle on the bottom side."""
        n_exit = self._exit_index(top=False)
        if n_exit is None:
            return 0.0
        return iadpython.fresnel.cos_critical(self.n, n_exit)

    def _slice_start(self, nu_c):
        """Index of the first quadrature angle that can escape."""
        idx = np.where(self.nu > nu_c)[0]
        if idx.size == 0:
            return len(self.nu) - 1
        return int(idx[0])

    def _collapse_to_outer_interface(self):
        """True when a zero-thickness sample reduces to one direct interface.

        Only a scalar ``d`` can legitimately trigger this (a single
        zero-thickness layer).  A per-layer ``d`` array -- used for the
        ``mu_a``/``mu_s``/``mu_sp`` display quantities in a multilayer stack,
        not consumed by the RT solver itself -- never collapses the sample.
        """
        if not np.isscalar(self.d) and np.ndim(self.d) > 0:
            return False
        return (
            np.isclose(self.d, 0.0)
            and np.isclose(self.b_above, 0.0)
            and np.isclose(self.b_below, 0.0)
            and _indices_match(self.n_above, 1.0)
            and _indices_match(self.n_below, 1.0)
        )

    def _outer_interface_rt(self, nu):
        """Direct outer-medium interface RT for the collapsed zero-thickness case."""
        return iadpython.fresnel.interface_rt(self.n_outer_above, nu, self.n_outer_below)

    def _outer_interface_average_rt(self, nu_min=0.0, nu_max=1.0, *, weighted):
        """Average direct-interface RT over an angular interval."""
        nu_min = max(float(nu_min), 0.0)
        nu_max = min(float(nu_max), 1.0)
        if nu_max <= nu_min:
            return 0.0, 0.0

        jac = 0.5 * (nu_max - nu_min)
        nu = jac * _OUTER_INTERFACE_GL_X + 0.5 * (nu_max + nu_min)
        r, t = self._outer_interface_rt(nu)
        if weighted:
            denom = nu_max**2 - nu_min**2
            r_avg = jac * np.dot(_OUTER_INTERFACE_GL_W, 2.0 * nu * r) / denom
            t_avg = jac * np.dot(_OUTER_INTERFACE_GL_W, 2.0 * nu * t) / denom
        else:
            r_avg = 0.5 * np.dot(_OUTER_INTERFACE_GL_W, r)
            t_avg = 0.5 * np.dot(_OUTER_INTERFACE_GL_W, t)
        ur = float(np.real(r_avg))
        ut = float(np.real(t_avg))
        return float(ur), float(ut)

    def nu_c(self):
        """Smallest positive escape critical-angle cosine across both boundaries."""
        criticals = [nu_c for nu_c in (self.nu_c_above(), self.nu_c_below()) if nu_c > 0]
        if not criticals:
            return 0.0
        return min(criticals)

    def _deltaM_forward_fraction(self):
        """delta-M forward-scattering fraction f (order-quad_pts Legendre moment).

        HG: g**quad_pts.  TABULATED: the actual order-quad_pts moment, so the
        albedo/optical-depth reduction below stays consistent with the
        redistribution-matrix truncation in phase_legendre().  MOMENTS: read
        directly off the supplied array via the same validation/normalization
        helper phase_legendre() uses, instead of re-deriving anything.
        """
        pf_type = str(getattr(self, "pf_type", "HG")).upper()
        if pf_type == "TABULATED" and self.pf_data is not None:
            a_raw = iadpython.legendre_coeffs_from_df(
                self.pf_data, quad_pts=self.quad_pts, n_mom=2 * self.quad_pts + 1)
            return float(np.clip(np.ravel(np.asarray(a_raw)[self.quad_pts])[0], 0.0, 1.0 - 1e-10))
        if pf_type == "MOMENTS" and self.pf_data is not None:
            a_raw = iadpython.redistribution._prepare_moments(self.pf_data, self.quad_pts + 1)
            return float(np.clip(a_raw[self.quad_pts], 0.0, 1.0 - 1e-10))
        return self.g**self.quad_pts

    def a_delta_M(self):
        """Reduced albedo in delta-M approximation."""
        af = self.a * self._deltaM_forward_fraction()
        num = np.asarray(self.a - af, dtype=float)
        den = np.asarray(1 - af, dtype=float)
        out = np.zeros_like(num, dtype=float)
        np.divide(num, den, out=out, where=~np.isclose(den, 0.0))
        if out.ndim == 0:
            return float(out)
        return out

    def b_delta_M(self):
        """Reduced optical thickness in delta-M approximation."""
        af = self.a * self._deltaM_forward_fraction()
        return self.b * (1 - af)

    def as_array(self):
        """Return details as an array."""
        return [self.a, self.b, self.g, self.d, self.n]

    def init_from_array(self, a):
        """Initialize basic details as an array."""
        self.a = a[0]
        self.b = a[1]
        self.g = a[2]
        self.d = a[3]
        self.n = a[4]

    def __str__(self):
        """Return basic details as a string for printing."""
        s = "Intrinsic Properties\n"
        s += "   albedo              = %s\n" % stringify("%.3f", self.a)
        s += "   optical thickness   = %s\n" % stringify("%.3f", self.b)
        s += "   anisotropy          = %s\n" % stringify("%.3f", self.g)
        s += "   thickness           = %s mm\n" % stringify("%.3f", self.d)
        s += "   sample index        = %s\n" % stringify("%.3f", self.n)
        s += "   top slide index     = %s\n" % stringify("%.3f", self.n_above)
        s += "   top outer index     = %s\n" % stringify("%.3f", self.n_outer_above)
        if self.b_above != 0:
            s += "   top slide OD        = %s\n" % stringify("%.3f", self.b_above)
        s += "   bottom slide index  = %s\n" % stringify("%.3f", self.n_below)
        s += "   bottom outer index  = %s\n" % stringify("%.3f", self.n_outer_below)
        if self.b_below != 0:
            s += "   bottom slide OD     = %s\n" % stringify("%.3f", self.b_below)
        s += "   cos(theta incident) = %s\n" % stringify("%.3f", self.nu_0)
        s += "   quadrature points   = %d\n" % self.quad_pts

        s += "\n"
        s += "Derived quantities\n"
        s += "   mu_a                = %s 1/mm\n" % stringify("%.3f", self.mu_a())
        s += "   mu_s                = %s 1/mm\n" % stringify("%.3f", self.mu_s())
        s += "   mu_s*(1-g)          = %s 1/mm\n" % stringify("%.3f", self.mu_sp())
        s += "       theta incident  = %.1f°\n" % np.degrees(np.arccos(self.nu_0))
        s += "   cos(theta critical) = %.4f\n" % self.nu_c()
        s += "       theta critical  = %.1f°\n" % np.degrees(np.arccos(self.nu_c()))
        return s

    def wrmatrix(self, a, title=None):
        """Print matrix and sums."""
        n = self.quad_pts

        # header line
        if title is not None:
            print(title)
        print("cos_theta |", end="")
        for i in range(n):
            print("%9.5f" % self.nu[i], end="")
        print(" |     flux")
        print("----------+", end="")
        for i in range(n):
            print("---------", end="")
        print("-+---------")

        # contents + row fluxes
        for i in range(n):
            print("%9.5f |" % self.nu[i], end="")
            for j in range(n):
                if a[i, j] < -100 or a[i, j] > 100:
                    print("    *****", end="")
                else:
                    print("%9.5f" % a[i, j], end="")
            flux = 0.0
            for j in range(n):
                flux += a[i, j] * self.twonuw[j]
            print(" |%9.5f" % flux)

        # identify index of first quadrature angle greater than the critical angle
        nu_c = self.nu_c()
        k = self._slice_start(nu_c)
        UXx = np.dot(self.twonuw[k:], a[k:, k:])
        tflux = np.dot(self.twonuw[k:], UXx) * self.n**2

        # column fluxes
        print("----------+", end="")
        for i in range(n):
            print("---------", end="")
        print("-+---------")
        print("%9s |" % "flux   ", end="")
        for i in range(n):
            flux = 0.0
            for j in range(n):
                flux += a[j, i] * self.twonuw[j]
            print("%9.5f" % flux, end="")
        print(" |%9.5f\n" % tflux)

    def wrarray(self, a, title=None):
        """Print diagonal array as matric with sums."""
        b = np.diag(a)
        self.wrmatrix(b, title)

    def prmatrix(self, a, title=None):
        """Print matrix and sums."""
        if title is not None:
            print(title)
        n = self.quad_pts

        # first row
        print("[[", end="")
        for j in range(n - 1):
            print("%9.5f," % a[0, j], end="")
        print("%9.5f]," % a[0, -1])

        for i in range(1, n - 1):
            print(" [", end="")
            for j in range(n - 1):
                print("%9.5f," % a[i, j], end="")
            print("%9.5f]," % a[i, -1])

        # last row
        print(" [", end="")
        for j in range(n - 1):
            print("%9.5f," % a[-1, j], end="")
        print("%9.5f]]" % a[-1, -1])

    def update_quadrature(self):
        """Calculate the correct set of quadrature points.

        This returns the quadrature angles using Radau quadrature over the
        interval 0 to 1 if there is no critical angle for total internal reflection
        in the self.  If there is a critical angle whose cosine is 'nu_c' then
        Radau quadrature points are chosen from 0 to 'nu_c' and Radau
        quadrature points over the interval 'nu_c' to 1.

        Now we need to include three angles, the critical angle, the cone
        angle, and perpendicular.  Now the important angles are the ones in
        the self.  So we calculate the cosine of the critical angle in the
        sample and cosine of the cone angle in the self.

        The critical angle will always be greater than the cone angle in the
        sample and therefore the cosine of the critical angle will always be
        less than the cosine of the cone angle.  Thus we will integrate from
        zero to the cosine of the critical angle (using Gaussian quadrature
        to avoid either endpoint) then from the critical angle to the cone
        angle (using Radau quadrature so that the cosine angle will be
        included) and finally from the cone angle to 1 (again using Radau
        quadrature so that 1 will be included).
        """
        nby2 = self.quad_pts // 2

        if self.nu_0 == 1:
            # case 1.  Normal incidence, no critical angle
            nu_c = self.nu_c()
            if self._n == 1 or nu_c <= 0:
                a1 = []
                w1 = []
                a2, w2 = iadpython.quadrature.radau(self.quad_pts, a=0, b=1)

            # case 2.  Normal incidence, with critical angle
            else:
                a1, w1 = iadpython.quadrature.gauss(nby2, a=0, b=nu_c)
                a2, w2 = iadpython.quadrature.radau(nby2, a=nu_c, b=1)
        else:
            # case 3.  Conical incidence.  Include nu_0
            nu_c = self.nu_c()
            if self._n == 1.0 or nu_c <= 0:
                a1, w1 = iadpython.quadrature.radau(nby2, a=0, b=self.nu_0)
                a2, w2 = iadpython.quadrature.radau(nby2, a=self.nu_0, b=1)

            # case 4.  Conical incidence.  Include nu_c, nu_00, and 1
            else:
                nby3 = int(self.quad_pts / 3)

                # cosine of nu_0 in sample
                nu_00 = iadpython.fresnel.cos_snell(
                    iadpython.fresnel._transport_index(self.n_outer_above),
                    self.nu_0,
                    self.n,
                )
                a00, w00 = iadpython.quadrature.gauss(nby3, a=0, b=nu_c)
                a01, w01 = iadpython.quadrature.radau(nby3, a=nu_c, b=nu_00)
                a1 = np.append(a00, a01)
                w1 = np.append(w00, w01)
                a2, w2 = iadpython.quadrature.radau(nby3, a=nu_00, b=1)

        self.nu = np.append(a1, a2)
        self.twonuw = 2 * self.nu * np.append(w1, w2)

    def rt_matrices(self):
        """Total reflection and transmission.

        This is the top level routine for accessing the adding-doubling
        algorithm. By passing the optical paramters characteristic of the sample,
        this routine will do what it must to return the total reflection and
        transmission for collimated and diffuse irradiance.

        This routine has three different components based on if zero, one, or
        two boundary layers must be included.  If the index of refraction of the
        sample and the top and bottom slides are all one, then no boundaries need
        to be included.  If the top and bottom slides are identical, then some
        simplifications can be made and some time saved as a consequence. If the
        top and bottom slides are different, then the full red carpet treatment
        is required.

        Since the calculation time increases for each of these cases we test for
        matched boundaries first.  If the boundaries are matched then don't
        bother with boundaries for the top and bottom.  Just calculate the
        integrated reflection and transmission.   Similarly, if the top and
        bottom slides are similar, then quickly calculate these.
        """
        # Layered sample: convert to the internal flat convention in place
        # (mirrors the pre-existing equal-index collapse, which also mutates
        # self).  Multi-wavelength stacks must go through rt().
        if self.layers is not None:
            n_wave = iadpython.layer.validate_layers(self.layers)
            if n_wave > 1:
                raise RuntimeError(
                    "rt_matrices: this layered sample spans %d wavelengths; "
                    "use rt() for the wavelength sweep" % n_wave
                )
            self._materialize_layers(0)

        # Hard break: on a flat sample, arrays no longer mean layers.
        self._guard_flat_arrays("rt_matrices")
        if self._layer_pf is None:
            for name in ("a", "b", "g"):
                value = getattr(self, name)
                if not np.isscalar(value) and np.ndim(value) > 0:
                    raise RuntimeError(
                        "rt_matrices: array-valued %s now means wavelengths "
                        "(use rt()); build a multilayer sample with "
                        "iadpython.Sample(layers=[iadpython.Layer(...), ...])"
                        % name
                    )

        # cone not implemented yet
        if self.nu_0 != 1.0:
            #            RT_Cone(n,sample,OBLIQUE,UR1,UT1,URU,UTU);
            r, t = iadpython.start.zero_layer(self.quad_pts)
            return r, r, t, t

        # An equal-valued index *array* is physically single-index: collapse it
        # to a scalar so the classic (scalar-n) path runs unchanged.  Rebuild
        # the quadrature so a subsequent UX1_and_UXU has a valid ``nu``.
        if not np.isscalar(self.n) and np.ndim(self.n) > 0 and self.is_index_matched():
            self.n = float(np.real(np.atleast_1d(self.n)[0]))
            if not np.isscalar(self.n_sample_boundary) and np.ndim(self.n_sample_boundary) > 0:
                self.n_sample_boundary = float(np.real(np.atleast_1d(self.n_sample_boundary)[0]))
            self.update_quadrature()

        # Per-layer refractive-index mismatch needs the master-grid solver,
        # which builds the internal Fresnel interfaces explicitly.  The classic
        # single-index path below is left completely untouched.
        if not self.is_index_matched():
            return self._rt_matrices_mismatched()

        R12, T12 = iadpython.simple_layer_matrices(self)

        # all done if boundaries are not an issue
        if (_indices_match(self.n_sample_boundary, self.n_outer_above) and
                _indices_match(self.n_sample_boundary, self.n_outer_below) and
                np.isclose(self.b_above, 0.0) and np.isclose(self.b_below, 0.0) and
                np.isclose(self.n_above, 1.0) and np.isclose(self.n_below, 1.0)):
            return R12, R12, T12, T12

        # reflection/transmission arrays for top boundary
        R01, R10, T01, T10 = iadpython.start.boundary_layer(self, top=True)

        # same slide above and below.
        if (self.n_above == self.n_below and
                self.b_above == self.b_below and
                self.n_outer_above == self.n_outer_below):
            R03, T03 = iadpython.add_same_slides(self, R01, R10, T01, T10, R12, T12)
            return R03, R03, T03, T03

        # reflection/transmission arrays for bottom boundary
        R23, R32, T23, T32 = iadpython.start.boundary_layer(self, top=False)

       # different boundaries on top and bottom
        R02, R20, T02, T20 = iadpython.add_slide_above(self, R01, R10, T01, T10, R12, R12, T12, T12)
        R03, R30, T03, T30 = iadpython.add_slide_below(self, R02, R20, T02, T20, R23, R32, T23, T32)

        return R03, R30, T03, T30

    def _rt_matrices_mismatched(self):
        """Assemble R/T for an index-mismatched stack, adding air/slide boundaries.

        The bare stack comes from ``mismatched_layer_matrices`` on the master
        grid.  The two outer boundaries are then added exactly as in the classic
        path, but each is evaluated on the grid of the layer it touches: the top
        boundary on the first layer's grid, the bottom boundary on the last
        layer's grid.
        """
        r_down, r_up, t_down, t_up = iadpython.combine.mismatched_layer_matrices(self)

        ns = np.atleast_1d(self.n)
        n_full = self._master_dim

        # Top boundary: air/slide -> first layer.
        s_top = iadpython.combine._grid_sample(
            self, ns[0], self.layer_grid(ns[0]), n_full
        )
        R01, R10, T01, T10 = iadpython.start.boundary_layer(s_top, top=True)
        R02, R20, T02, T20 = iadpython.add_slide_above(
            s_top, R01, R10, T01, T10, r_down, r_up, t_down, t_up
        )

        # Bottom boundary: last layer -> slide/air.
        s_bot = iadpython.combine._grid_sample(
            self, ns[-1], self.layer_grid(ns[-1]), n_full
        )
        R23, R32, T23, T32 = iadpython.start.boundary_layer(s_bot, top=False)
        R03, R30, T03, T30 = iadpython.add_slide_below(
            s_bot, R02, R20, T02, T20, R23, R32, T23, T32
        )

        return R03, R30, T03, T30

    def _UX1_and_UXU_mismatched(self, R, T):
        """Escape-flux integration for an index-mismatched stack.

        Both reflectance and transmittance are referenced to the incident
        (top) medium, so all sums use the first layer's grid and the top index
        ``n[0]``.  This is what makes the collimated result satisfy
        ``ur1 + ut1 = 1`` for a lossless stack and the diffuse transmittance
        obey reciprocity ``T(0->L) == T(L->0)`` exactly (both verified
        numerically; the top-grid form is algebraically identical to summing
        the exit rows on the bottom grid with an ``n[-1]**2`` factor).

        The reflected and transmitted beams escape through different critical
        angles, so the trapped-channel slice differs: reflection is cut at the
        top escape angle (in ``n[0]``), transmission at the bottom escape angle
        (in ``n[-1]``).  Channel indices are shared across grids, so the cut is
        applied as a shared channel offset.
        """
        ns = np.atleast_1d(self.n)
        n0 = float(np.real(ns[0]))
        nL = float(np.real(ns[-1]))

        nu0, tw0, _ = self.layer_grid(ns[0])
        nuL, _twL, _ = self.layer_grid(ns[-1])

        nu_c_r = iadpython.fresnel.cos_critical(n0, self._exit_index(top=True))
        nu_c_t = iadpython.fresnel.cos_critical(nL, self._exit_index(top=False))
        k_r = self._slice_start_grid(nu0, nu_c_r)
        k_t = self._slice_start_grid(nuL, nu_c_t)

        URx = np.dot(tw0[k_r:], R[k_r:, k_r:])
        UTx = np.dot(tw0[k_t:], T[k_t:, k_t:])
        URU = np.dot(tw0[k_r:], URx) * n0**2
        UTU = np.dot(tw0[k_t:], UTx) * n0**2

        return URx[-1], UTx[-1], URU, UTU

    def UX1_and_UXU(self, R, T):
        """Calculate total reflected and transmitted fluxes from matrices.

        The trick here is that the integration must be done over the fluxes
        that leave the sample.  This is not much of an issue for the transmitted
        fluxes because they are zero.  However, the internal reflected fluxes will
        not be zero and should be excluded from the sums.
        """
        if self._collapse_to_outer_interface():
            ur1, ut1 = self._outer_interface_rt(self.nu_0)
            uru, utu = self._outer_interface_average_rt(weighted=True)
            return float(ur1), float(ut1), uru, utu

        if not self.is_index_matched():
            return self._UX1_and_UXU_mismatched(R, T)

        k_r = self._slice_start(self.nu_c_above())
        k_t = self._slice_start(self.nu_c_below())
        URx = np.dot(self.twonuw[k_r:], R[k_r:, k_r:])       # Reflectance collimated incident flux
        UTx = np.dot(self.twonuw[k_t:], T[k_t:, k_t:])       # Transmittance collimated incident flux
        URU = np.dot(self.twonuw[k_r:], URx) * self.n**2     # Reflected diffuse (Lambertian) flux
        UTU = np.dot(self.twonuw[k_t:], UTx) * self.n**2     # Transmitted diffuse (Lambertian) flux

        return URx[-1], UTx[-1], URU, UTU

    def rt_diffuse_cone(self, R, T, nu_min=0, nu_max=1):
        """Find average reflection and transmission for collimated incident flux over a cone.
        The approximation is usefull to simulate coarse roughness (compared to the wavelength) 
        on the surface of the sample.

        Parameters:
            R: reflection matrix
            T: transmission matrix
            nu_min: cosine of minimum angle in cone
            nu_max: cosine of maximum angle in cone
        Returns:
            reflected and transmitted fluxes over the cone
        """
        if self._collapse_to_outer_interface():
            return self._outer_interface_average_rt(nu_min, nu_max, weighted=False)

        nu_c_r = self.nu_c_above()
        nu_c_t = self.nu_c_below()
        nu_min = max(nu_min, nu_c_r)
        
        k_r = self._slice_start(nu_c_r)
        k_t = self._slice_start(nu_c_t)
        URx = np.dot(self.twonuw[k_r:], R[k_r:, k_r:])
        UTx = np.dot(self.twonuw[k_t:], T[k_t:, k_t:])
        
        # creaate interpolation functions for integration
        UR_nu = _CubicSpline(self.nu[k_r:], URx)
        UT_nu = _CubicSpline(self.nu[k_t:], UTx)

        # Get average reflectance and transmittance over the cone
        if nu_max <= nu_min:
            return 0.0, 0.0
        UR_cone = _quad(UR_nu, nu_min, nu_max)[0] / (nu_max - nu_min)
        UT_cone = _quad(UT_nu, max(nu_min, nu_c_t), nu_max)[0] / (nu_max - nu_min)
        return UR_cone, UT_cone
    
    @staticmethod
    def _wavelength_slice(value, i):
        """Pick wavelength ``i`` from a scalar-or-array layer attribute.

        Scalars are returned unchanged (held constant across the sweep);
        length-1 arrays broadcast; longer arrays are indexed.
        """
        if value is None or np.isscalar(value) or np.ndim(value) == 0:
            return value
        if len(value) == 1:
            return value[0]
        return value[i]

    def _materialize_layers(self, i):
        """Convert this layered sample into the internal flat convention.

        Writes wavelength ``i`` of every layer onto the flat attributes:
        ``a``/``b``/``g``/``n``/``d`` become length-n_layers vectors (plain
        scalars for a single layer, so the classic code path runs unchanged)
        and ``_layer_pf`` becomes the per-layer ``(pf_type, one_column_df)``
        list that marks the arrays-over-layers convention as active for the
        solver internals.  ``layers`` is cleared -- after materialization the
        sample is a single-wavelength snapshot of the stack.

        Callers that need to preserve the layered sample work on a
        ``copy.deepcopy`` (see ``rt``); ``rt_matrices`` materializes in place,
        mirroring the pre-existing equal-index collapse behavior.
        """
        layers = self.layers

        a_i = [self._wavelength_slice(layer.a, i) for layer in layers]
        b_i = [self._wavelength_slice(layer.b, i) for layer in layers]
        g_i = [self._wavelength_slice(layer.g, i) for layer in layers]
        n_i = [self._wavelength_slice(layer.n, i) for layer in layers]
        d_i = [layer.d for layer in layers]

        pf_list = []
        for layer in layers:
            if layer.pf_type == "TABULATED":
                col = i if layer.pf_data.shape[1] > 1 else 0
                pf_list.append(("TABULATED", layer.pf_data.iloc[:, [col]]))
            elif layer.pf_type == "MOMENTS":
                if layer.pf_data.ndim == 2 and layer.pf_data.shape[1] > 1:
                    # Defense in depth: validate_layers() should already have
                    # checked shape[1] against the stack's wavelength count,
                    # but a mismatched i here must raise clearly rather than
                    # degrade into a bare IndexError deep in materialization.
                    if i >= layer.pf_data.shape[1]:
                        raise ValueError(
                            "layer MOMENTS pf_data has %d columns but "
                            "wavelength index %d was requested"
                            % (layer.pf_data.shape[1], i)
                        )
                    col = i
                elif layer.pf_data.ndim == 2:
                    col = 0
                else:
                    col = None
                sliced = layer.pf_data[:, col] if col is not None else layer.pf_data
                pf_list.append(("MOMENTS", sliced))
            else:
                pf_list.append(("HG", None))

        if len(layers) == 1:
            # single layer: plain flat sample, classic path byte-identical
            self.a = float(a_i[0])
            self.b = float(b_i[0])
            self.g = float(g_i[0])
            self.n = n_i[0]
            self.d = float(d_i[0])
            self.pf_type, self.pf_data = pf_list[0]
            self._layer_pf = None
        else:
            self.a = np.asarray(a_i, dtype=float)
            self.b = np.asarray(b_i, dtype=float)
            self.g = np.asarray(g_i, dtype=float)
            self.n = np.asarray(n_i)
            self.d = np.asarray(d_i, dtype=float)
            # flat pf attributes are unused on this path; per-layer pf rules
            self.pf_type = "HG"
            self.pf_data = None
            self._layer_pf = pf_list

        self.n_sample_boundary = self.n
        self.layers = None

        # setters invalidate most caches, but be explicit so a stale
        # quadrature or redistribution matrix can never leak across layers
        self.nu = None
        self.twonuw = None
        self.hp = None
        self.hm = None
        self.b_thinnest = None

    def _guard_flat_arrays(self, caller):
        """Hard break: flat samples may not carry array-valued n (or layers)."""
        if self._layer_pf is not None:
            return
        if not np.isscalar(self.n) and np.ndim(self.n) > 0:
            raise RuntimeError(
                "%s: array-valued n on a flat Sample is no longer supported; "
                "build a multilayer sample with iadpython.Sample(layers="
                "[iadpython.Layer(...), ...])" % caller
            )

    def rt(self):
        """Find total reflection and transmission.

        For a layered sample (``Sample(layers=[...])``) this sweeps the
        wavelength axis of the layers: each layer attribute may be a scalar
        (held constant) or an array over wavelengths, and tabulated phase
        functions contribute one ``pf_data`` column per wavelength.  Scalars
        are returned for a single-wavelength stack, arrays otherwise.

        For the classic flat sample, ``pf_data`` may contain one column per
        wavelength when ``pf_type`` is ``"TABULATED"``.  Otherwise the ``g``
        array provides wavelength variation.  Whenever any wavelength-dependent
        input is an array, the lengths of ``a``, ``b``, and the phase-function
        data must match.
        """
        if self.layers is not None:
            n_wave = iadpython.layer.validate_layers(self.layers)
            if n_wave == 1:
                x = copy.deepcopy(self)
                x._materialize_layers(0)
                R, _, T, _ = x.rt_matrices()
                return x.UX1_and_UXU(R, T)

            ur1 = np.empty(n_wave)
            ut1 = np.empty(n_wave)
            uru = np.empty(n_wave)
            utu = np.empty(n_wave)
            for i in range(n_wave):
                x = copy.deepcopy(self)
                x._materialize_layers(i)
                R, _, T, _ = x.rt_matrices()
                ur1[i], ut1[i], uru[i], utu[i] = x.UX1_and_UXU(R, T)
            return ur1, ut1, uru, utu

        self._guard_flat_arrays("rt")

        len_a = 0
        len_b = 0
        len_pf = 0

        if not np.isscalar(self.a):
            len_a = len(self.a)

        if not np.isscalar(self.b):
            len_b = len(self.b)

        if self.pf_type == "TABULATED" and isinstance(self.pf_data, pd.DataFrame):
            len_pf = self.pf_data.shape[1]
        elif self.pf_type == "MOMENTS" and isinstance(self.pf_data, np.ndarray) and self.pf_data.ndim == 2:
            len_pf = self.pf_data.shape[1]
        elif not np.isscalar(self.g):
            len_pf = len(self.g)

        thelen = max(len_a, len_b, len_pf)

        if thelen == 0:
            R, _, T, _ = self.rt_matrices()
            return self.UX1_and_UXU(R, T)

        if len_a and len_b and len_a != len_b:
            raise RuntimeError("rt: a and b arrays must be same length")

        if len_pf and len_a and len_pf != len_a:
            raise RuntimeError("rt: pf_data and a arrays must be same length")

        if len_pf and len_b and len_pf != len_b:
            raise RuntimeError("rt: pf_data and b arrays must be same length")

        ur1 = np.empty(thelen)
        ut1 = np.empty(thelen)
        uru = np.empty(thelen)
        utu = np.empty(thelen)

        if self.nu is None:
            self.update_quadrature()

        sample = copy.deepcopy(self)
        for i in range(thelen):
            if len_a > 0:
                sample.a = self.a[i]

            if len_b > 0:
                sample.b = self.b[i]

            if self.pf_type == "TABULATED":
                if len_pf > 0:
                    sample.pf_data = self.pf_data.iloc[:, [i]]
            elif self.pf_type == "MOMENTS":
                if len_pf > 0:
                    sample.pf_data = self.pf_data[:, i]
            elif len_pf > 0:
                sample.g = self.g[i]

            R, _, T, _ = sample.rt_matrices()
            ur1[i], ut1[i], uru[i], utu[i] = sample.UX1_and_UXU(R, T)

        return ur1, ut1, uru, utu

    def unscattered_scalar_rt(self):
        """Find unscattered r and t for diagonal matrices (scalar of array version)."""
        if self._collapse_to_outer_interface():
            r, t = self._outer_interface_rt(self.nu_0)
            return float(r), float(t)

        n_top = self.n_above
        n_slab = self.n_sample_boundary
        n_bot = self.n_below
        b_slab = self.b
        return iadpython.fresnel.specular_rt(
            n_top,
            n_slab,
            n_bot,
            b_slab,
            self.nu_0,
            n_outer_top=self.n_outer_above,
            n_outer_bot=self.n_outer_below,
            n_slab_transport=self.n,
        )

    def _unscattered_slabs(self):
        """Ordered ``(index, optical thickness)`` slabs the ballistic beam crosses.

        The sequence is: top slide (if any), every layer, bottom slide (if any).
        Following the same convention as ``_exit_index``, a slide of index 1.0
        with zero optical thickness means "no slide" and is omitted -- adding it
        would insert a spurious extra interface when the outer medium is not air.
        """
        slabs = []

        if not (np.isclose(self.b_above, 0.0) and _indices_match(self.n_above, 1.0)):
            slabs.append((self.n_above, self.b_above))

        # Broadcast n and b to a common layer count: after the equal-index
        # collapse a materialized stack can carry a scalar n next to a
        # per-layer b (or vice versa) -- zip must never truncate layers.
        ns = np.atleast_1d(self.n)
        bs = np.atleast_1d(self.b)
        n_layers = max(ns.size, bs.size)
        if ns.size == 1:
            ns = np.full(n_layers, ns.flat[0])
        if bs.size == 1:
            bs = np.full(n_layers, bs.flat[0])
        for n_k, b_k in zip(ns, bs):
            slabs.append((n_k, b_k))

        if not (np.isclose(self.b_below, 0.0) and _indices_match(self.n_below, 1.0)):
            slabs.append((self.n_below, self.b_below))

        return slabs

    def _unscattered_mismatched(self):
        r"""Ballistic (unscattered) R/T through a stack of mismatched layers.

        This is the N-slab generalisation of ``fresnel.specular_rt``: an
        incoherent bottom-up recursion that accumulates Fresnel reflection at
        every interface and Beer-Lambert attenuation within every slab,
        including all multiple internal reflections.  For slab ``k`` bounded
        above by medium ``a``, with ``(R, T)`` already known at its lower face,

        .. math:: R' = r_1 + t_1 t_1' R e^{-2b_k/\nu_k} / D

        .. math:: T' = t_1 T e^{-b_k/\nu_k} / D

        where :math:`D = 1 - r_1' R e^{-2b_k/\nu_k}` sums the round trips.

        The result is exact (no quadrature involved), so unlike the scattered
        solution it does not depend on ``quad_pts``.  Angles follow the Snell
        invariant from the top outer medium; a totally internally reflected
        channel simply gets ``t_1 = 0``, which zeroes the transmission.

        Returns:
            ``(r, t)`` scalars for the whole stack.
        """
        slabs = self._unscattered_slabs()
        n_out_top = iadpython.fresnel._transport_index(self.n_outer_above)
        n_out_bot = iadpython.fresnel._transport_index(self.n_outer_below)

        def cos_in(n_medium):
            """Cosine inside a medium for the beam entering at nu_0 from the top."""
            return iadpython.fresnel.cos_snell(
                n_out_top, self.nu_0, iadpython.fresnel._transport_index(n_medium)
            )

        # Start at the bottom face of the last slab, looking into the exit medium.
        n_last = slabs[-1][0]
        r_below, t_below = iadpython.fresnel.interface_rt(
            n_last, cos_in(n_last), n_out_bot
        )

        for k in range(len(slabs) - 1, -1, -1):
            n_k, b_k = slabs[k]
            nu_k = cos_in(n_k)
            n_a = n_out_top if k == 0 else slabs[k - 1][0]
            nu_a = cos_in(n_a)

            r1, t1 = iadpython.fresnel.interface_rt(n_a, nu_a, n_k)
            r1p, t1p = iadpython.fresnel.interface_rt(n_k, nu_k, n_a)

            expo = np.exp(-b_k / iadpython.fresnel._sanitize_path_cos(nu_k))
            denom = 1.0 - r1p * r_below * expo**2
            r_new = r1 + t1 * t1p * r_below * expo**2 / denom
            t_new = t1 * t_below * expo / denom
            r_below, t_below = r_new, t_new

        return float(np.real(r_below)), float(np.real(t_below))

    def unscattered_rt(self):
        """Find unscattered r and t.

        A layered sample (``Sample(layers=[...])``) sweeps the wavelength axis
        with the exact analytic N-slab recursion, returning scalars for a
        single wavelength and arrays otherwise.  For the classic flat sample
        the historical behaviour is kept: a scalar ``b`` gives one pair, and
        an array ``b`` is treated as a *wavelength* sweep (matching ``rt``).
        """
        if self.layers is not None:
            n_wave = iadpython.layer.validate_layers(self.layers)
            if n_wave == 1:
                x = copy.deepcopy(self)
                x._materialize_layers(0)
                return x._unscattered_mismatched()

            r = np.empty(n_wave)
            t = np.empty(n_wave)
            for i in range(n_wave):
                x = copy.deepcopy(self)
                x._materialize_layers(i)
                r[i], t[i] = x._unscattered_mismatched()
            return r, t

        self._guard_flat_arrays("unscattered_rt")

        # A materialized layered sample (arrays = layers, _layer_pf set) and
        # any index-mismatched stack both use the exact N-slab recursion --
        # its per-layer b must not be mistaken for a wavelength sweep.
        if self._layer_pf is not None or not self.is_index_matched():
            return self._unscattered_mismatched()

        if np.isscalar(self.b):
            return self.unscattered_scalar_rt()

        r = np.empty_like(self.b, dtype=type(self.nu_0))
        t = np.empty_like(self.b, dtype=type(self.nu_0))
        x = copy.deepcopy(self)
        for i, b in enumerate(self.b):
            x.b = b
            r[i], t[i] = x.unscattered_scalar_rt()

        return r, t
