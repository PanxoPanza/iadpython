# pylint: disable=invalid-name

"""Tests for outer-media boundary semantics."""

import unittest
import numpy as np
import iadpython


def _as_scalar(value):
    """Return a scalar from iadpython's scalar-or-array outputs."""
    return float(np.asarray(value).reshape(-1)[0])


class OuterMediaTest(unittest.TestCase):
    """Outer-medium support for zero-thickness and boundary controls."""

    def test_01_zero_thickness_arbitrary_outer_media(self):
        """Zero-thickness, no-slide samples collapse to the outer interface."""
        s = iadpython.Sample(
            a=0.0,
            b=0.0,
            g=0.0,
            d=0.0,
            n=1.5,
            n_sample_boundary=1.5 + 1e-5j,
            n_above=1.0,
            n_below=1.0,
            n_outer_above=1.0,
            n_outer_below=1.5 + 1e-5j,
            quad_pts=16,
        )

        r_ref, t_ref = iadpython.fresnel.interface_rt(1.0, 1.0, 1.5 + 1e-5j)
        ur1, ut1, _, _ = s.rt()
        r_spec, t_spec = s.unscattered_rt()

        np.testing.assert_allclose([_as_scalar(ur1), _as_scalar(ut1)], [r_ref, t_ref], atol=1e-6)
        np.testing.assert_allclose([_as_scalar(r_spec), _as_scalar(t_spec)], [r_ref, t_ref], atol=1e-6)

    def test_02_zero_thickness_no_interface_control(self):
        """Zero-thickness, matched outer media have no residual interface."""
        s = iadpython.Sample(
            a=0.0,
            b=0.0,
            g=0.0,
            d=0.0,
            n=1.5,
            n_sample_boundary=1.5,
            n_above=1.0,
            n_below=1.0,
            n_outer_above=1.0,
            n_outer_below=1.0,
            quad_pts=16,
        )

        ur1, ut1, _, _ = s.rt()
        np.testing.assert_allclose([_as_scalar(ur1), _as_scalar(ut1)], [0.0, 1.0], atol=1e-12)

    def test_03_top_interface_only_control(self):
        """Finite zero-scattering slabs honor the outer-medium top interface only."""
        s = iadpython.Sample(
            a=0.0,
            b=0.0,
            g=0.0,
            d=5.0,
            n=1.5,
            n_sample_boundary=1.5,
            n_above=1.0,
            n_below=1.0,
            n_outer_above=1.0,
            n_outer_below=1.5,
            quad_pts=16,
        )

        r_ref, t_ref = iadpython.fresnel.interface_rt(1.0, 1.0, 1.5)
        ur1, ut1, _, _ = s.rt()
        r_spec, t_spec = s.unscattered_rt()

        np.testing.assert_allclose([_as_scalar(ur1), _as_scalar(ut1)], [r_ref, t_ref], atol=1e-6)
        np.testing.assert_allclose([_as_scalar(r_spec), _as_scalar(t_spec)], [r_ref, t_ref], atol=1e-6)

    def test_04_default_outer_media_backward_compatible(self):
        """Leaving outer media at default air preserves prior slide semantics."""
        s_default = iadpython.Sample(n=1.3, n_above=1.5, n_below=1.6, quad_pts=4)
        s_explicit = iadpython.Sample(
            n=1.3,
            n_above=1.5,
            n_below=1.6,
            n_outer_above=1.0,
            n_outer_below=1.0,
            quad_pts=4,
        )

        for top in (True, False):
            with self.subTest(top=top):
                default = iadpython.boundary_layer(s_default, top=top)
                explicit = iadpython.boundary_layer(s_explicit, top=top)
                for arr_default, arr_explicit in zip(default, explicit):
                    np.testing.assert_allclose(arr_default, arr_explicit, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
