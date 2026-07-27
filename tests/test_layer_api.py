# pylint: disable=invalid-name

"""
Tests for the Layer API: explicit multilayer structure with per-layer
wavelength dependence and per-layer phase functions.

Axis convention under test:
  * layer axis      = position in the ``layers`` list
  * wavelength axis = arrays inside each Layer attribute / pf_data columns

Also covered: the hard break that removed the old flat-arrays-as-layers form.
"""

import unittest
import numpy as np
import pandas as pd
import iadpython


def _hg_table(g, n_mu=401):
    """Henyey-Greenstein phase function tabulated on mu = cos(theta).

    Used to check that a TABULATED layer reproduces the analytic HG result.
    """
    mu = np.linspace(-1, 1, n_mu)
    p = (1 - g**2) / (1 + g**2 - 2 * g * mu) ** 1.5 / 2.0
    return pd.DataFrame({"pf": p}, index=mu)


class LayerConstruction(unittest.TestCase):
    """Layer object basics."""

    def test_defaults(self):
        """Default layer is clear, isotropic, index-matched."""
        layer = iadpython.Layer()
        self.assertEqual(layer.a, 0)
        self.assertEqual(layer.b, 1)
        self.assertEqual(layer.g, 0)
        self.assertEqual(layer.n, 1)
        self.assertEqual(layer.d, 1)
        self.assertEqual(layer.pf_type, "HG")

    def test_lists_become_arrays(self):
        """Lists/tuples are normalized to numpy arrays."""
        layer = iadpython.Layer(a=[0.5, 0.6], n=(1.3, 1.4))
        self.assertIsInstance(layer.a, np.ndarray)
        self.assertIsInstance(layer.n, np.ndarray)
        self.assertEqual(layer.n_wavelengths(), 2)

    def test_bad_pf_type(self):
        """Unknown phase function types fail immediately."""
        with self.assertRaises(ValueError):
            iadpython.Layer(pf_type="MIE")

    def test_tabulated_needs_dataframe(self):
        """TABULATED without a DataFrame fails immediately."""
        with self.assertRaises(TypeError):
            iadpython.Layer(pf_type="TABULATED")

    def test_2d_attribute_rejected(self):
        """Only scalar or 1-D wavelength arrays are allowed."""
        with self.assertRaises(ValueError):
            iadpython.Layer(a=np.zeros((2, 2)))


class FlatEquivalence(unittest.TestCase):
    """A single-Layer sample is exactly the classic flat sample."""

    def test_rt(self):
        """rt() matches the flat form."""
        flat = iadpython.Sample(a=0.9, b=1.0, g=0.3, n=1.5, n_above=1.5, quad_pts=16)
        layered = iadpython.Sample(
            layers=[iadpython.Layer(a=0.9, b=1.0, g=0.3, n=1.5)],
            n_above=1.5,
            quad_pts=16,
        )
        np.testing.assert_allclose(layered.rt(), flat.rt(), atol=1e-12)

    def test_rt_matrices(self):
        """rt_matrices() matches the flat form."""
        flat = iadpython.Sample(a=0.9, b=1.0, g=0.3, n=1.5, quad_pts=8)
        layered = iadpython.Sample(
            layers=[iadpython.Layer(a=0.9, b=1.0, g=0.3, n=1.5)], quad_pts=8
        )
        for m_flat, m_lay in zip(flat.rt_matrices(), layered.rt_matrices()):
            np.testing.assert_allclose(m_lay, m_flat, atol=1e-12)

    def test_unscattered_rt(self):
        """unscattered_rt() matches the flat form (no slides)."""
        flat = iadpython.Sample(a=0.9, b=1.0, g=0.3, n=1.5, quad_pts=8)
        layered = iadpython.Sample(
            layers=[iadpython.Layer(a=0.9, b=1.0, g=0.3, n=1.5)], quad_pts=8
        )
        np.testing.assert_allclose(
            layered.unscattered_rt(), flat.unscattered_rt(), atol=1e-12
        )


class WavelengthBroadcast(unittest.TestCase):
    """Scalar layers extend across the sweep; conflicts raise."""

    def test_sweep_shape_and_values(self):
        """A 3-wavelength layer next to a scalar layer gives length-3 output."""
        varying = iadpython.Layer(a=np.array([0.9, 0.8, 0.7]), b=0.5, g=0.2, n=1.4)
        constant = iadpython.Layer(a=0.6, b=0.5, g=0.0, n=1.4)
        s = iadpython.Sample(layers=[varying, constant], quad_pts=8)
        ur1, ut1, uru, utu = s.rt()
        self.assertEqual(ur1.shape, (3,))

        # element 1 of the sweep == a manual single-wavelength solve
        manual = iadpython.Sample(
            layers=[iadpython.Layer(a=0.8, b=0.5, g=0.2, n=1.4), constant],
            quad_pts=8,
        )
        np.testing.assert_allclose(
            [ur1[1], ut1[1], uru[1], utu[1]], manual.rt(), atol=1e-12
        )

    def test_wavelength_dependent_index(self):
        """Per-layer n may be wavelength dependent (dispersion)."""
        disp = iadpython.Layer(a=0.5, b=0.5, n=np.array([1.3, 1.5]))
        other = iadpython.Layer(a=0.5, b=0.5, n=2.0)
        s = iadpython.Sample(layers=[disp, other], quad_pts=8)
        ur1, _, uru, utu = s.rt()
        self.assertEqual(ur1.shape, (2,))
        self.assertTrue(np.all(uru + utu <= 1.0 + 1e-9))

    def test_conflicting_lengths_raise(self):
        """Wavelength arrays of different lengths (>1) are rejected."""
        l0 = iadpython.Layer(a=np.array([0.9, 0.8, 0.7]))
        l1 = iadpython.Layer(b=np.array([0.5, 0.6]))
        s = iadpython.Sample(layers=[l0, l1], quad_pts=4)
        with self.assertRaises(RuntimeError):
            s.rt()

    def test_conflicting_pf_columns_raise(self):
        """pf_data column count must be 1 or match the sweep."""
        table = _hg_table(0.5)
        two_col = pd.concat([table, table], axis=1)
        l0 = iadpython.Layer(a=np.array([0.9, 0.8, 0.7]))
        l1 = iadpython.Layer(pf_type="TABULATED", pf_data=two_col)
        s = iadpython.Sample(layers=[l0, l1], quad_pts=4)
        with self.assertRaises(RuntimeError):
            s.rt()

    def test_unscattered_sweep(self):
        """Layered unscattered_rt() sweeps wavelengths too."""
        varying = iadpython.Layer(a=np.array([0.9, 0.8, 0.7]), b=0.5, n=1.4)
        s = iadpython.Sample(
            layers=[varying, iadpython.Layer(a=0.6, b=0.5, n=2.0)], quad_pts=8
        )
        ru, tu = s.unscattered_rt()
        self.assertEqual(ru.shape, (3,))
        # ballistic beam does not depend on albedo -> constant across sweep
        np.testing.assert_allclose(ru, ru[0], atol=1e-14)
        np.testing.assert_allclose(tu, tu[0], atol=1e-14)


class MixedPhaseFunctions(unittest.TestCase):
    """HG and TABULATED layers can be combined freely in one stack."""

    def test_tabulated_matches_hg_matched_index(self):
        """A tabulated HG table next to an analytic HG layer == all-HG stack."""
        table = _hg_table(0.6)
        tab_layer = iadpython.Layer(a=0.9, b=0.6, n=1.4, pf_type="TABULATED", pf_data=table)
        hg_layer = iadpython.Layer(a=0.7, b=0.4, g=0.3, n=1.4)

        mixed = iadpython.Sample(layers=[tab_layer, hg_layer], quad_pts=8)
        allhg = iadpython.Sample(
            layers=[iadpython.Layer(a=0.9, b=0.6, g=0.6, n=1.4), hg_layer], quad_pts=8
        )
        np.testing.assert_allclose(mixed.rt(), allhg.rt(), atol=1e-6)

    def test_tabulated_matches_hg_mismatched_index(self):
        """Same equivalence across a refractive-index mismatch."""
        table = _hg_table(0.6)
        tab_layer = iadpython.Layer(a=0.9, b=0.6, n=1.35, pf_type="TABULATED", pf_data=table)
        hg_layer = iadpython.Layer(a=0.7, b=0.4, g=0.3, n=2.0)

        mixed = iadpython.Sample(layers=[tab_layer, hg_layer], quad_pts=8)
        allhg = iadpython.Sample(
            layers=[iadpython.Layer(a=0.9, b=0.6, g=0.6, n=1.35), hg_layer], quad_pts=8
        )
        np.testing.assert_allclose(mixed.rt(), allhg.rt(), atol=1e-6)

    def test_tabulated_wavelength_sweep(self):
        """Two pf_data columns sweep two wavelengths."""
        mu = np.linspace(-1, 1, 401)

        def hg(g):
            return (1 - g**2) / (1 + g**2 - 2 * g * mu) ** 1.5 / 2.0

        table = pd.DataFrame({"w0": hg(0.2), "w1": hg(0.7)}, index=mu)
        tab_layer = iadpython.Layer(a=0.9, b=0.6, n=1.4, pf_type="TABULATED", pf_data=table)
        other = iadpython.Layer(a=0.6, b=0.5, n=1.4)
        s = iadpython.Sample(layers=[tab_layer, other], quad_pts=8)
        out = s.rt()
        self.assertEqual(out[0].shape, (2,))

        # wavelength 0 must match the analytic-HG(0.2) stack
        ref = iadpython.Sample(
            layers=[iadpython.Layer(a=0.9, b=0.6, g=0.2, n=1.4), other], quad_pts=8
        ).rt()
        np.testing.assert_allclose([o[0] for o in out], ref, atol=1e-6)


def _hg_moments(g, n_mom):
    """Analytic Legendre moments of the Henyey-Greenstein phase function.

    Used to check that a MOMENTS layer reproduces the analytic HG result.
    """
    return g ** np.arange(n_mom)


class MomentsPhaseFunction(unittest.TestCase):
    """HG and MOMENTS layers can be combined freely in one stack.

    Mirrors MixedPhaseFunctions above, one-for-one, for the MOMENTS path.
    """

    def test_moments_matches_hg_matched_index(self):
        """Analytic HG moments fed as MOMENTS == an equivalent all-HG stack."""
        a_l = _hg_moments(0.6, 17)  # 2*quad_pts+1 for quad_pts=8
        mom_layer = iadpython.Layer(a=0.9, b=0.6, n=1.4, pf_type="MOMENTS", pf_data=a_l)
        hg_layer = iadpython.Layer(a=0.7, b=0.4, g=0.3, n=1.4)

        mixed = iadpython.Sample(layers=[mom_layer, hg_layer], quad_pts=8)
        allhg = iadpython.Sample(
            layers=[iadpython.Layer(a=0.9, b=0.6, g=0.6, n=1.4), hg_layer], quad_pts=8
        )
        np.testing.assert_allclose(mixed.rt(), allhg.rt(), atol=1e-6)

    def test_moments_matches_hg_mismatched_index(self):
        """Same equivalence across a refractive-index mismatch.

        The mismatched-index solver's internal master grid may need a
        larger effective quadrature order than the Sample's own quad_pts
        for some sub-layers, so the moments array must be sized generously
        (not just 2*quad_pts+1) -- see the note in Layer's docstring.
        """
        a_l = _hg_moments(0.6, 65)
        mom_layer = iadpython.Layer(a=0.9, b=0.6, n=1.35, pf_type="MOMENTS", pf_data=a_l)
        hg_layer = iadpython.Layer(a=0.7, b=0.4, g=0.3, n=2.0)

        mixed = iadpython.Sample(layers=[mom_layer, hg_layer], quad_pts=8)
        allhg = iadpython.Sample(
            layers=[iadpython.Layer(a=0.9, b=0.6, g=0.6, n=1.35), hg_layer], quad_pts=8
        )
        np.testing.assert_allclose(mixed.rt(), allhg.rt(), atol=1e-6)

    def test_moments_wavelength_sweep(self):
        """Two pf_data columns sweep two wavelengths."""
        table = np.column_stack([_hg_moments(0.2, 17), _hg_moments(0.7, 17)])
        mom_layer = iadpython.Layer(a=0.9, b=0.6, n=1.4, pf_type="MOMENTS", pf_data=table)
        other = iadpython.Layer(a=0.6, b=0.5, n=1.4)
        s = iadpython.Sample(layers=[mom_layer, other], quad_pts=8)
        out = s.rt()
        self.assertEqual(out[0].shape, (2,))

        # wavelength 0 must match the analytic-HG(0.2) stack
        ref = iadpython.Sample(
            layers=[iadpython.Layer(a=0.9, b=0.6, g=0.2, n=1.4), other], quad_pts=8
        ).rt()
        np.testing.assert_allclose([o[0] for o in out], ref, atol=1e-6)

    def test_moments_requires_array(self):
        """MOMENTS without pf_data, or with a DataFrame, fails immediately."""
        with self.assertRaises(TypeError):
            iadpython.Layer(pf_type="MOMENTS")
        with self.assertRaises(TypeError):
            iadpython.Layer(pf_type="MOMENTS", pf_data=_hg_table(0.5))

    def test_moments_too_few_raises(self):
        """Fewer than 2*quad_pts+1 moments raises a clear ValueError."""
        s = iadpython.Sample(
            a=0.9, b=1.0, n=1.4, quad_pts=8, pf_type="MOMENTS", pf_data=_hg_moments(0.5, 5)
        )
        with self.assertRaises(ValueError):
            s.rt_matrices()

    def test_moments_column_mismatch_raises(self):
        """pf_data column count must be 1 or match the sweep."""
        two_col = np.column_stack([_hg_moments(0.5, 17), _hg_moments(0.5, 17)])
        l0 = iadpython.Layer(a=np.array([0.9, 0.8, 0.7]))
        l1 = iadpython.Layer(pf_type="MOMENTS", pf_data=two_col)
        s = iadpython.Sample(layers=[l0, l1], quad_pts=4)
        with self.assertRaises(RuntimeError):
            s.rt()


class HardBreak(unittest.TestCase):
    """Flat arrays no longer mean layers; errors must mention Layer."""

    def test_simple_layer_matrices_rejects_arrays(self):
        """Old flat-arrays-as-layers form raises with a pointer to Layer."""
        s = iadpython.Sample(quad_pts=4)
        s.a = np.array([0.5, 0.5])
        s.b = np.array([0.5, 0.5])
        s.g = np.array([0.0, 0.0])
        with self.assertRaises(RuntimeError) as ctx:
            iadpython.simple_layer_matrices(s)
        self.assertIn("Layer", str(ctx.exception))

    def test_flat_array_n_rejected(self):
        """Array n on a flat sample raises in rt, rt_matrices, unscattered_rt."""
        for method in ("rt", "rt_matrices", "unscattered_rt"):
            s = iadpython.Sample(n=np.array([1.3, 1.6]), quad_pts=4)
            with self.assertRaises(RuntimeError) as ctx:
                getattr(s, method)()
            self.assertIn("Layer", str(ctx.exception))

    def test_multiwavelength_rt_matrices_rejected(self):
        """rt_matrices on a multi-wavelength layered sample points to rt()."""
        varying = iadpython.Layer(a=np.array([0.9, 0.8]))
        s = iadpython.Sample(layers=[varying, iadpython.Layer()], quad_pts=4)
        with self.assertRaises(RuntimeError) as ctx:
            s.rt_matrices()
        self.assertIn("rt()", str(ctx.exception))

    def test_flat_wavelength_arrays_still_work(self):
        """Flat a/b/g arrays keep their wavelength meaning in rt()."""
        s = iadpython.Sample(n=1.4, quad_pts=8)
        s.a = np.array([0.9, 0.8])
        s.b = np.array([1.0, 1.0])
        s.g = np.array([0.5, 0.5])
        ur1, ut1, uru, utu = s.rt()
        self.assertEqual(ur1.shape, (2,))
        one = iadpython.Sample(a=0.8, b=1.0, g=0.5, n=1.4, quad_pts=8).rt()
        np.testing.assert_allclose([ur1[1], ut1[1], uru[1], utu[1]], one, atol=1e-12)


class ValidateLayers(unittest.TestCase):
    """Cross-layer validation errors are specific."""

    def test_empty(self):
        """No layers is an error."""
        with self.assertRaises(RuntimeError):
            iadpython.layer.validate_layers([])

    def test_non_layer_entry(self):
        """Non-Layer entries are named by index."""
        with self.assertRaises(RuntimeError) as ctx:
            iadpython.layer.validate_layers([iadpython.Layer(), "oops"])
        self.assertIn("layers[1]", str(ctx.exception))

    def test_bad_albedo(self):
        """Albedo outside [0, 1] is rejected."""
        with self.assertRaises(RuntimeError):
            iadpython.layer.validate_layers([iadpython.Layer(a=1.5)])

    def test_returns_wavelength_count(self):
        """The sweep length is the max across layers."""
        layers = [
            iadpython.Layer(a=np.array([0.5, 0.6, 0.7])),
            iadpython.Layer(b=np.array([0.5])),
            iadpython.Layer(),
        ]
        self.assertEqual(iadpython.layer.validate_layers(layers), 3)


if __name__ == "__main__":
    unittest.main()
