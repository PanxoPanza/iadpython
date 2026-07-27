# pylint: disable=invalid-name

"""
Tests for multi-layer samples built with the Layer API.

Multilayer structure is declared as ``Sample(layers=[Layer(...), ...])``;
the old form (flat arrays on a/b/g meaning layers) was removed in this fork.
The expected numerical values below are unchanged from the original tests.
"""

import unittest
import numpy as np
import iadpython


class LayeredSample(unittest.TestCase):
    """Layered Sample calculations."""

    def test_00_layers(self):
        """Two identical non-scattering layers without boundaries."""
        s = iadpython.Sample(
            layers=[
                iadpython.Layer(a=0.0, b=0.5, g=0.0),
                iadpython.Layer(a=0.0, b=0.5, g=0.0),
            ],
            quad_pts=4,
        )
        rr, _, tt, _ = s.rt_matrices()

        single = iadpython.Sample(a=0.0, b=1.0, g=0.0, quad_pts=4)
        R, _, T, _ = single.rt_matrices()

        np.testing.assert_allclose(R, rr, atol=1e-5)
        np.testing.assert_allclose(T, tt, atol=1e-5)

    def test_01_layers(self):
        """Three identical non-scattering layers with boundaries."""
        s = iadpython.Sample(
            layers=[
                iadpython.Layer(a=0.0, b=0.5, g=0.9, n=1.4),
                iadpython.Layer(a=0.0, b=0.2, g=0.9, n=1.4),
                iadpython.Layer(a=0.0, b=0.3, g=0.9, n=1.4),
            ],
            n_above=1.5,
            n_below=1.5,
            quad_pts=16,
        )
        rr, _, tt, _ = s.rt_matrices()

        single = iadpython.Sample(
            a=0.0, b=1.0, g=0.0, n=1.4, n_above=1.5, n_below=1.5, quad_pts=16
        )
        R, _, T, _ = single.rt_matrices()

        np.testing.assert_allclose(R, rr, atol=1e-4)
        np.testing.assert_allclose(T, tt, atol=1e-4)

    def test_02_layers(self):
        """Two identical isotropic scattering layers without boundaries."""
        s = iadpython.Sample(
            layers=[
                iadpython.Layer(a=0.5, b=0.5, g=0.0),
                iadpython.Layer(a=0.5, b=0.5, g=0.0),
            ],
            quad_pts=4,
        )
        rr, _, tt, _ = s.rt_matrices()

        R = np.array(
            [
                [0.80010, 0.31085, 0.18343, 0.14931],
                [0.31085, 0.20307, 0.14031, 0.11912],
                [0.18343, 0.14031, 0.10265, 0.08848],
                [0.14931, 0.11912, 0.08848, 0.07658],
            ]
        )

        T = np.array(
            [
                [0.01792, 0.06195, 0.07718, 0.07528],
                [0.06195, 0.37419, 0.09621, 0.08830],
                [0.07718, 0.09621, 0.62536, 0.07502],
                [0.07528, 0.08830, 0.07502, 3.00924],
            ]
        )

        np.testing.assert_allclose(R, rr, atol=1e-5)
        np.testing.assert_allclose(T, tt, atol=1e-5)

    def test_03_layers(self):
        """Three identical isotropic scattering layers without boundaries."""
        s = iadpython.Sample(
            layers=[
                iadpython.Layer(a=0.5, b=0.5, g=0.0),
                iadpython.Layer(a=0.5, b=0.2, g=0.0),
                iadpython.Layer(a=0.5, b=0.3, g=0.0),
            ],
            quad_pts=4,
        )
        rr, _, tt, _ = s.rt_matrices()

        R = np.array(
            [
                [0.80010, 0.31085, 0.18343, 0.14931],
                [0.31085, 0.20307, 0.14031, 0.11912],
                [0.18343, 0.14031, 0.10265, 0.08848],
                [0.14931, 0.11912, 0.08848, 0.07658],
            ]
        )

        T = np.array(
            [
                [0.01792, 0.06195, 0.07718, 0.07528],
                [0.06195, 0.37419, 0.09621, 0.08830],
                [0.07718, 0.09621, 0.62536, 0.07502],
                [0.07528, 0.08830, 0.07502, 3.00924],
            ]
        )

        np.testing.assert_allclose(R, rr, atol=1e-4)
        np.testing.assert_allclose(T, tt, atol=1e-4)

    def test_04_layers(self):
        """Three identical layers without boundaries."""
        s = iadpython.Sample(
            layers=[
                iadpython.Layer(a=0.5, b=0.5, g=0.9),
                iadpython.Layer(a=0.5, b=0.2, g=0.9),
                iadpython.Layer(a=0.5, b=0.3, g=0.9),
            ],
            quad_pts=16,
        )
        rr, _, tt, _ = s.rt_matrices()

        single = iadpython.Sample(a=0.5, b=1.0, g=0.9, quad_pts=16)
        R, _, T, _ = single.rt_matrices()

        np.testing.assert_allclose(R, rr, atol=1e-5)
        np.testing.assert_allclose(T, tt, atol=1e-5)

    def test_05_layers(self):
        """Three identical layers with boundaries."""
        s = iadpython.Sample(
            layers=[
                iadpython.Layer(a=0.5, b=0.5, g=0.9, n=1.4),
                iadpython.Layer(a=0.5, b=0.2, g=0.9, n=1.4),
                iadpython.Layer(a=0.5, b=0.3, g=0.9, n=1.4),
            ],
            n_above=1.5,
            n_below=1.5,
            quad_pts=16,
        )
        rr, _, tt, _ = s.rt_matrices()

        single = iadpython.Sample(
            a=0.5, b=1.0, g=0.9, n=1.4, n_above=1.5, n_below=1.5, quad_pts=16
        )
        R, _, T, _ = single.rt_matrices()

        np.testing.assert_allclose(R, rr, atol=6e-5)
        np.testing.assert_allclose(T, tt, atol=6e-5)


if __name__ == "__main__":
    unittest.main()
