"""One layer of a multilayer sample.

A :class:`Layer` bundles the optical properties of one slab in a stack.  The
two independent axes of a simulation are kept structurally distinct:

* **layer axis** -- the position of the ``Layer`` in the ``layers`` list passed
  to :class:`iadpython.Sample`;
* **wavelength axis** -- arrays *inside* each ``Layer`` attribute (``a``,
  ``b``, ``g``, ``n`` may each be a scalar or a 1-D array over wavelengths;
  a ``TABULATED`` phase function carries one ``pf_data`` column per
  wavelength).

A scalar attribute is automatically held constant (broadcast) across all
wavelengths of the sweep, so layers with constant properties can be freely
mixed with wavelength-dependent ones.

Example::

    >>> import iadpython as iad
    >>> l0 = iad.Layer(a=0.95, b=0.84, g=0.20, n=1.35, d=0.10)
    >>> l1 = iad.Layer(a=0.81, b=0.31, g=0.85, n=2.00, d=0.05)
    >>> s = iad.Sample(layers=[l0, l1], quad_pts=16)
    >>> ur1, ut1, uru, utu = s.rt()
"""

import numpy as np

__all__ = ("Layer",)

_PF_TYPES = ("HG", "TABULATED", "MOMENTS")


def _as_scalar_or_1d(value, name):
    """Return ``value`` unchanged if scalar, else as a 1-D float array.

    Lists/tuples are converted to numpy arrays so downstream code can rely on
    numpy semantics.  Anything with more than one dimension is rejected --
    the wavelength axis is the only in-attribute axis a ``Layer`` carries.
    """
    if value is None or np.isscalar(value):
        return value
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    if arr.ndim > 1:
        raise ValueError(
            "Layer.%s must be a scalar or 1-D wavelength array, got shape %s"
            % (name, arr.shape)
        )
    return arr


def _fmt(value):
    """Compact scalar-or-range formatting for repr (no iadpython imports)."""
    if value is None:
        return "None"
    if np.isscalar(value) or np.ndim(value) == 0:
        return "%.3f" % np.real(value)
    arr = np.asarray(value)
    return "%.3f..%.3f[%d]" % (np.real(arr.min()), np.real(arr.max()), arr.size)


class Layer:
    """Optical properties of one slab in a multilayer sample.

    Attributes:
        a: single-scattering albedo (scalar or wavelength array)
        b: optical thickness (scalar or wavelength array)
        g: scattering anisotropy, used when ``pf_type == 'HG'``
           (scalar or wavelength array)
        d: physical thickness [mm] (scalar; informational -- the radiative
           transfer solve consumes ``b``, not ``d``)
        n: refractive index (scalar or wavelength array)
        pf_type: ``'HG'`` (Henyey-Greenstein, from ``g``), ``'TABULATED'``
           (from ``pf_data``), or ``'MOMENTS'`` (from ``pf_data``)
        pf_data: for ``'TABULATED'``, a pandas ``DataFrame`` indexed by
           mu = cos(theta) with one column per wavelength (a single column
           is broadcast across the sweep). For ``'MOMENTS'``, a plain
           ndarray of Legendre moments a_l -- 1-D ``(n_mom,)`` (broadcast
           across the sweep) or 2-D ``(n_mom, n_wavelength)`` -- supplied
           directly instead of a tabulated phase function, bypassing the
           spline+quadrature step ``'TABULATED'`` requires. Required when
           ``pf_type`` is ``'TABULATED'`` or ``'MOMENTS'``.

    Cross-layer consistency (matching wavelength counts, physical ranges) is
    checked at solve time by ``validate_layers`` so partially built layers can
    be edited freely before use.

    Note on ``pf_type='MOMENTS'`` in a mismatched-refractive-index stack:
    unlike ``'TABULATED'`` (a continuous curve that can be re-integrated to
    any order on demand), a MOMENTS array is a fixed, finite set of
    precomputed numbers. The mismatched-index solver's internal master grid
    may need a *larger* effective quadrature order for some sub-layers than
    the Sample's own ``quad_pts`` -- so a MOMENTS array sized only for the
    nominal ``quad_pts`` can raise "needs at least N moments" partway
    through a solve. Supply generously more moments than ``2*quad_pts+1``
    when index-mismatched layers are involved.
    """

    def __init__(self, a=0, b=1, g=0, d=1, n=1, pf_type="HG", pf_data=None):
        """Create one layer; see class docstring for attribute meanings."""
        self.a = _as_scalar_or_1d(a, "a")
        self.b = _as_scalar_or_1d(b, "b")
        self.g = _as_scalar_or_1d(g, "g")
        self.n = _as_scalar_or_1d(n, "n")
        self.d = d

        self.pf_type = str(pf_type).upper()
        if self.pf_type not in _PF_TYPES:
            raise ValueError(
                "Layer pf_type must be one of %s, got %r" % (_PF_TYPES, pf_type)
            )
        if self.pf_type == "TABULATED":
            # duck-typed DataFrame check to keep this module pandas-free
            if pf_data is None or not hasattr(pf_data, "iloc"):
                raise TypeError(
                    "pf_type='TABULATED' requires pf_data to be a pandas "
                    "DataFrame (index = cos(theta), one column per wavelength)"
                )
        elif self.pf_type == "MOMENTS":
            # Legendre moments a_l, bypassing TABULATED's spline+quadrature.
            # Explicitly reject DataFrame-like input -- np.asarray(df) would
            # otherwise silently coerce it instead of failing loudly.
            if pf_data is None or hasattr(pf_data, "iloc"):
                raise TypeError(
                    "pf_type='MOMENTS' requires pf_data to be a plain array "
                    "of Legendre moments (1-D), or a 2-D array (n_mom, "
                    "n_wavelength) for a wavelength sweep -- not a DataFrame"
                )
            pf_data = np.asarray(pf_data, dtype=float)
            if pf_data.ndim not in (1, 2):
                raise ValueError(
                    "MOMENTS pf_data must be 1-D (n_mom,) or 2-D (n_mom, "
                    "n_wavelength), got shape %s" % (pf_data.shape,)
                )
        self.pf_data = pf_data

    def n_wavelengths(self):
        """Number of wavelengths this layer's own data spans (1 if scalar)."""
        counts = [1]
        for value in (self.a, self.b, self.g, self.n):
            if not (value is None or np.isscalar(value) or np.ndim(value) == 0):
                counts.append(len(value))
        if self.pf_type == "TABULATED" and self.pf_data is not None:
            counts.append(self.pf_data.shape[1])
        if self.pf_type == "MOMENTS" and self.pf_data is not None and self.pf_data.ndim == 2:
            counts.append(self.pf_data.shape[1])
        return max(counts)

    def __repr__(self):
        """Compact one-line description."""
        s = "Layer(a=%s, b=%s, g=%s, n=%s, d=%s, pf=%s" % (
            _fmt(self.a),
            _fmt(self.b),
            _fmt(self.g),
            _fmt(self.n),
            _fmt(self.d),
            self.pf_type,
        )
        if self.pf_type == "TABULATED" and self.pf_data is not None:
            s += "[%d col]" % self.pf_data.shape[1]
        if self.pf_type == "MOMENTS" and self.pf_data is not None:
            if self.pf_data.ndim == 2:
                s += "[%d mom x %d col]" % self.pf_data.shape
            else:
                s += "[%d mom]" % self.pf_data.shape[0]
        return s + ")"


def validate_layers(layers):
    """Check a list of layers for cross-layer consistency.

    Determines the wavelength count of the sweep as the maximum over every
    layer's array lengths and ``pf_data`` column counts, then verifies that
    each per-layer axis is compatible with it (length 1 broadcasts).

    Args:
        layers: sequence of :class:`Layer` objects (at least one)

    Returns:
        n_wavelengths: the number of wavelengths of the sweep (>= 1)

    Raises:
        RuntimeError: on empty input, non-Layer entries, conflicting
            wavelength counts, or out-of-range ``a``/``b`` values.
    """
    if layers is None or len(layers) == 0:
        raise RuntimeError("Sample(layers=...) needs at least one Layer")

    for k, layer in enumerate(layers):
        if not isinstance(layer, Layer):
            raise RuntimeError(
                "layers[%d] is %s, expected iadpython.Layer" % (k, type(layer).__name__)
            )

    n_wave = max(layer.n_wavelengths() for layer in layers)

    for k, layer in enumerate(layers):
        for name in ("a", "b", "g", "n"):
            value = getattr(layer, name)
            if value is None or np.isscalar(value) or np.ndim(value) == 0:
                continue
            if len(value) not in (1, n_wave):
                raise RuntimeError(
                    "layers[%d].%s has %d wavelengths but the stack sweep "
                    "has %d (lengths must be 1 or match)"
                    % (k, name, len(value), n_wave)
                )

        if not (np.isscalar(layer.d) or np.ndim(layer.d) == 0):
            raise RuntimeError(
                "layers[%d].d must be a scalar physical thickness "
                "(wavelength-dependent d is not meaningful)" % k
            )

        if layer.pf_type == "TABULATED":
            ncols = layer.pf_data.shape[1]
            if ncols not in (1, n_wave):
                raise RuntimeError(
                    "layers[%d].pf_data has %d columns but the stack sweep "
                    "has %d wavelengths (columns must be 1 or match)"
                    % (k, ncols, n_wave)
                )

        if layer.pf_type == "MOMENTS" and layer.pf_data.ndim == 2:
            ncols = layer.pf_data.shape[1]
            if ncols not in (1, n_wave):
                raise RuntimeError(
                    "layers[%d].pf_data has %d columns but the stack sweep "
                    "has %d wavelengths (columns must be 1 or match)"
                    % (k, ncols, n_wave)
                )

        a_arr = np.atleast_1d(layer.a).astype(float)
        b_arr = np.atleast_1d(layer.b).astype(float)
        if np.any(a_arr < 0) or np.any(a_arr > 1):
            raise RuntimeError("layers[%d].a must lie in [0, 1]" % k)
        if np.any(b_arr < 0):
            raise RuntimeError("layers[%d].b must be non-negative" % k)

    return n_wave
