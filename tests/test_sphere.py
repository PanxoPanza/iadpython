# pylint: disable=invalid-name
# pylint: disable=no-self-use
# pylint: disable=too-many-locals
# pylint: disable=protected-access

"""Tests for sphere object."""

import unittest
import numpy as np
import iadpython


class A_nothing_sandwich(unittest.TestCase):
    """Empty layer in air."""

    def test_01_object_creation(self):
        """Simple sphere creation."""
        s = iadpython.sphere.Sphere(200, 20)
        np.testing.assert_allclose(s.d_sphere, 200, atol=1e-5)
        np.testing.assert_allclose(s.d_sample, 20, atol=1e-5)
        np.testing.assert_allclose(s.d_entrance, 0, atol=1e-5)
        np.testing.assert_allclose(s.d_detector, 0, atol=1e-5)

    def test_02_portsize(self):
        """Test setting ."""
        s = iadpython.sphere.Sphere(200, 20)
        s.d_entrance = 10
        s.d_detector = 5
        s.d_sample = 18
        np.testing.assert_allclose(s._d_sphere, 200, atol=1e-5)
        np.testing.assert_allclose(s._d_sample, 18, atol=1e-5)
        np.testing.assert_allclose(s._d_entrance, 10, atol=1e-5)
        np.testing.assert_allclose(s._d_detector, 5, atol=1e-5)
        np.testing.assert_allclose(s.d_sphere, 200, atol=1e-5)
        np.testing.assert_allclose(s.d_sample, 18, atol=1e-5)
        np.testing.assert_allclose(s.d_entrance, 10, atol=1e-5)
        np.testing.assert_allclose(s.d_detector, 5, atol=1e-5)

    def test_03_cap(self):
        """Spherical cap calculations."""
        R = 100
        s = iadpython.sphere.Sphere(2 * R, 5)
        r = 10
        acap = s.cap_area(2 * r) / (4 * np.pi * R**2)
        a_cap = s.relative_cap_area(2 * r)
        np.testing.assert_allclose(acap, a_cap, atol=1e-5)

        acap1 = np.pi * r**2 / (4 * np.pi * R**2)
        a_cap1 = s.approx_relative_cap_area(20)
        np.testing.assert_allclose(acap1, a_cap1, atol=1e-5)

    def test_04_gain(self):
        """Gain calculations, r_wall=0."""
        s = iadpython.sphere.Sphere(200, 25, d_entrance=5)
        s.r_wall = 0
        g = s.gain(0)
        np.testing.assert_allclose(g, 1, atol=1e-5)

    def test_05_gain(self):
        """Gain calculations."""
        s = iadpython.sphere.Sphere(200, 25, d_entrance=5)
        s.r_wall = 1
        g = s.gain(0)
        gg = 1/(s.a_detector+s.a_entrance+s.a_sample)
        np.testing.assert_allclose(g, gg, atol=1e-5)

    def test_05_multiplier(self):
        """Multiplier calculations."""
        R = 100
        s = iadpython.sphere.Sphere(2 * R, 5)
        s.a_wall = 0.98
        s.r_wall = 0.8 / s.a_wall
        M = s.multiplier(UR1=1, URU=0)
        np.testing.assert_allclose(5, M, atol=1e-5)

        M = s.multiplier(UR1=0.8, URU=0)
        np.testing.assert_allclose(0.8 * 5, M, atol=1e-5)

        M = s.multiplier(UR1=1, URU=1)
        M1 = 1 / (1 - s.a_wall * s.r_wall - s.a_sample)
        np.testing.assert_allclose(M, M1, atol=1e-5)

    def test_06_multiplier(self):
        """Array of r_wall values."""
        R = 100
        s = iadpython.sphere.Sphere(2 * R, 25, d_entrance=5, r_detector=0.1)
        s.r_wall = np.linspace(0,1,4)
        M = s.multiplier(UR1=1, URU=0)
        mm = [1.0, 1.496948, 2.975731, 245.224041]
        np.testing.assert_allclose(mm, M, atol=1e-5)

        M = s.multiplier(UR1=0.8, URU=0)
        mm = [0.8, 1.197558, 2.380584, 196.179233]
        np.testing.assert_allclose(mm, M, atol=1e-5)

        M = s.multiplier(UR1=1, URU=1)
        M1 = 1 / (1 - s.a_wall * s.r_wall - s.a_sample)
        np.testing.assert_allclose(M, M1, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
