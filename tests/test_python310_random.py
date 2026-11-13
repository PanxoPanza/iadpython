"""Sphere sampling remains compatible with Python 3.10's random API."""

import random

import numpy as np

import iadpython as iad


def test_uniform_standard_normal_arguments_and_seeded_sequence(monkeypatch):
    """Require explicit distribution parameters without changing seeded draws."""
    actual_generator = random.Random(318)
    expected_generator = random.Random(318)

    # Python 3.10 requires these two arguments; the spy has the same contract.
    def gauss(mu, sigma):
        assert (mu, sigma) == (0.0, 1.0)
        return actual_generator.gauss(mu, sigma)

    monkeypatch.setattr(random, "gauss", gauss)
    sphere = iad.Sphere(100, 30, d_detector=10)
    for _ in range(12):
        vector = np.array([expected_generator.gauss(0.0, 1.0) for _ in range(3)])
        expected = vector / np.linalg.norm(vector) * 50.0
        actual = sphere.uniform()
        np.testing.assert_allclose(actual, expected, rtol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(actual), 50.0, rtol=1e-14)
    assert actual_generator.getstate() == expected_generator.getstate()
