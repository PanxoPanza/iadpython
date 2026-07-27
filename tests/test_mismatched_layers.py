# pylint: disable=invalid-name
"""
Tests for multilayer samples whose layers have *different* refractive indices.

These exercise the Snell-invariant master-grid solver added to handle a real
Fresnel interface (including total internal reflection) between neighbouring
layers.  The classic single-index path is untouched, so the tests here focus on
the properties that uniquely validate the new machinery:

1.  reduction   -- an equal-index array reproduces the classic single slab,
2.  interfaces  -- a lossless mismatched stack conserves energy exactly for
                   both collimated (ur1 + ut1 = 1) and diffuse (uru + utu = 1)
                   illumination,
3.  reciprocity -- diffuse transmittance is the same top->bottom and
                   bottom->top,
4.  energy      -- a non-absorbing *scattering* stack conserves energy as the
                   resolution grows,
5.  routing     -- ``is_index_matched`` keeps matched samples on the fast path.
"""

import unittest
import numpy as np
import iadpython


def _stack(nvec, avec, bvec, gvec, quad_pts):
    """Build a layered sample from per-layer property lists."""
    layers = [
        iadpython.Layer(a=a, b=b, g=g, n=n)
        for a, b, g, n in zip(avec, bvec, gvec, nvec)
    ]
    return iadpython.Sample(layers=layers, quad_pts=quad_pts)


def _scalars(nvec, avec, bvec, gvec, quad_pts):
    """Return (ur1, ut1, uru, utu) for a per-layer-index stack."""
    s = _stack(nvec, avec, bvec, gvec, quad_pts)
    R, _, T, _ = s.rt_matrices()
    return s.UX1_and_UXU(R, T)


class IndexGate(unittest.TestCase):
    """The matched/mismatched routing predicate (internal, post-materialization)."""

    def test_scalar_is_matched(self):
        """A scalar index is always matched."""
        self.assertTrue(iadpython.Sample(n=1.4).is_index_matched())

    def test_equal_layer_indices_are_matched(self):
        """Equal per-layer indices route to the classic matched solver."""
        s = _stack([1.4, 1.4, 1.4], [0.5] * 3, [0.5] * 3, [0.0] * 3, 4)
        s._materialize_layers(0)
        self.assertTrue(s.is_index_matched())

    def test_unequal_layer_indices_are_mismatched(self):
        """Differing per-layer indices route to the mismatched solver."""
        s = _stack([1.4, 2.0, 1.4], [0.5] * 3, [0.5] * 3, [0.0] * 3, 4)
        s._materialize_layers(0)
        self.assertFalse(s.is_index_matched())


class Reduction(unittest.TestCase):
    """Equal indices must reproduce the classic single-index answer."""

    def test_two_equal_layers_match_single_slab(self):
        """[n, n] with split thickness equals one slab of the summed thickness."""
        m = _scalars([1.5, 1.5], [0.9, 0.9], [0.5, 0.5], [0.3, 0.3], quad_pts=16)

        c = iadpython.Sample(a=0.9, b=1.0, g=0.3, n=1.5, quad_pts=16).rt()
        np.testing.assert_allclose(m, c, atol=1e-6)

    def test_three_equal_layers_match_single_slab(self):
        """Three equal-index layers reduce to a single slab."""
        m = _scalars([1.4, 1.4, 1.4], [0.7, 0.7, 0.7],
                     [0.5, 0.2, 0.3], [0.0, 0.0, 0.0], quad_pts=16)

        c = iadpython.Sample(a=0.7, b=1.0, g=0.0, n=1.4, quad_pts=16).rt()
        np.testing.assert_allclose(m, c, atol=1e-6)


class LosslessInterfaces(unittest.TestCase):
    """Pure Fresnel interfaces (a=0, b=0) conserve energy exactly."""

    def test_collimated_energy(self):
        """ur1 + ut1 == 1 for a stack of bare interfaces."""
        for nvec in ([1.5, 2.0], [1.3, 2.5], [2.0, 1.4], [1.0, 4.0, 1.0]):
            k = len(nvec)
            ur1, ut1, _, _ = _scalars(nvec, [0.0] * k, [0.0] * k, [0.0] * k, 16)
            self.assertAlmostEqual(ur1 + ut1, 1.0, places=6, msg=str(nvec))

    def test_diffuse_energy(self):
        """uru + utu == 1 for a stack of bare interfaces."""
        for nvec in ([1.5, 2.0], [1.3, 2.5], [2.0, 1.4], [1.0, 4.0, 1.0]):
            k = len(nvec)
            _, _, uru, utu = _scalars(nvec, [0.0] * k, [0.0] * k, [0.0] * k, 16)
            self.assertAlmostEqual(uru + utu, 1.0, places=6, msg=str(nvec))


class Reciprocity(unittest.TestCase):
    """Diffuse transmittance is direction independent."""

    def test_diffuse_transmittance_reciprocal(self):
        """utu(0->L) == utu(L->0) for absorbing, scattering stacks."""
        for nvec in ([1.3, 2.0, 1.7], [1.2, 1.8], [1.0, 1.5, 2.0]):
            k = len(nvec)
            fwd = _scalars(nvec, [0.8] * k, [0.5] * k, [0.2] * k, 16)
            bwd = _scalars(nvec[::-1], [0.8] * k, [0.5] * k, [0.2] * k, 16)
            self.assertAlmostEqual(fwd[3], bwd[3], places=9, msg=str(nvec))


class EnergyConservation(unittest.TestCase):
    """A non-absorbing scattering stack conserves energy as resolution grows."""

    def test_converges_to_unity(self):
        """uru + utu approaches 1 monotonically for a=1 as quad_pts grows."""
        nvec = [1.0, 1.5, 1.0]
        k = len(nvec)
        sums = [sum(_scalars(nvec, [1.0] * k, [0.5] * k, [0.0] * k, N)[2:])
                for N in (8, 16, 32)]
        # monotone increasing toward 1 and never above it
        self.assertTrue(sums[0] < sums[1] < sums[2] <= 1.0 + 1e-9)
        self.assertGreater(sums[2], 0.94)


class Unscattered(unittest.TestCase):
    """The ballistic (unscattered) beam through a mismatched stack."""

    def test_matches_specular_rt_for_single_index(self):
        """The N-slab recursion reproduces the exact single-slab specular_rt."""
        cases = [
            dict(a=0.5, b=1.0, g=0.0, n=1.4),
            dict(a=0.5, b=0.7, g=0.0, n=1.4, n_above=1.5, n_below=1.6),
            dict(a=0.5, b=2.0, g=0.0, n=2.0),
            dict(a=0.5, b=0.7, g=0.0, n=1.4, n_outer_above=1.33, n_outer_below=1.33),
        ]
        for kw in cases:
            s = iadpython.Sample(quad_pts=8, **kw)
            np.testing.assert_allclose(
                s._unscattered_mismatched(), s.unscattered_scalar_rt(), atol=1e-12
            )

    def test_matches_specular_rt_at_oblique_incidence(self):
        """The recursion is correct away from normal incidence."""
        s = iadpython.Sample(a=0.5, b=0.7, g=0.0, n=1.4,
                             n_above=1.5, n_below=1.5, quad_pts=8)
        s.nu_0 = 0.6
        np.testing.assert_allclose(
            s._unscattered_mismatched(), s.unscattered_scalar_rt(), atol=1e-12
        )

    def test_multilayer_does_not_raise(self):
        """A mismatched stack returns a single scalar (r, t) pair."""
        s = _stack([1.35, 2.0, 1.45], [0.9, 0.7, 0.95],
                   [0.35, 0.28, 0.42], [0.2, 0.85, 0.6], 16)
        ru, tu = s.unscattered_rt()
        self.assertIsInstance(ru, float)
        self.assertIsInstance(tu, float)
        self.assertTrue(0.0 <= ru <= 1.0 and 0.0 <= tu <= 1.0)

    def test_equals_total_when_not_scattering(self):
        """With a=0 the total collimated result *is* the unscattered beam."""
        s = _stack([1.35, 2.0, 1.45], [0.0, 0.0, 0.0],
                   [0.35, 0.28, 0.42], [0.0, 0.0, 0.0], 32)
        ru, tu = s.unscattered_rt()
        R, _, T, _ = s.rt_matrices()
        ur1, ut1, _, _ = s.UX1_and_UXU(R, T)
        np.testing.assert_allclose([ur1, ut1], [ru, tu], atol=1e-4)

    def test_independent_of_quad_pts(self):
        """The ballistic beam is analytic, so quad_pts must not change it."""
        results = []
        for N in (4, 16, 32):
            s = _stack([1.35, 2.0, 1.45], [0.9, 0.7, 0.95],
                       [0.35, 0.28, 0.42], [0.2, 0.85, 0.6], N)
            results.append(s.unscattered_rt())
        np.testing.assert_allclose(results[0], results[1], atol=1e-14)
        np.testing.assert_allclose(results[1], results[2], atol=1e-14)

    def test_unscattered_below_total(self):
        """The ballistic beam is a strict subset of the total flux."""
        s = _stack([1.35, 2.0, 1.45], [0.9, 0.7, 0.95],
                   [0.35, 0.28, 0.42], [0.2, 0.85, 0.6], 16)
        ru, tu = s.unscattered_rt()
        R, _, T, _ = s.rt_matrices()
        ur1, ut1, _, _ = s.UX1_and_UXU(R, T)
        self.assertLess(ru, ur1)
        self.assertLess(tu, ut1)


class NonScattering(unittest.TestCase):
    """a=0 leaves only the specular (collimated) beam."""

    def test_absorbing_nonscattering_between_zero_and_one(self):
        """With absorption (a=0, b>0) the collimated fluxes stay physical."""
        ur1, ut1, uru, utu = _scalars([2.0, 1.4], [0.0, 0.0],
                                      [0.2, 0.3], [0.0, 0.0], 16)
        for value in (ur1, ut1, uru, utu):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        # some light is absorbed, so the collimated total is strictly below 1
        self.assertLess(ur1 + ut1, 1.0)


if __name__ == "__main__":
    unittest.main()
