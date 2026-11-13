"""Analytic interface limits and nearby physical contrast regressions."""

import numpy as np
import pytest

from iadpython.fresnel import interface_rt


@pytest.mark.parametrize("index", [1.0, 1.5, 1.5 + 0.2j, 1.5 - 0.2j])
@pytest.mark.parametrize("cosine", [0.0, 1e-16, 1e-10, 0.5, 1.0, np.array(0.0)])
def test_identical_media_scalar(index, cosine):
    """A boundary between identical media cannot scatter or absorb power."""
    reflection, transmission = interface_rt(index, cosine, index)
    assert isinstance(reflection, float)
    assert isinstance(transmission, float)
    assert reflection == 0.0
    assert transmission == 1.0


@pytest.mark.parametrize("index", [1.0, 1.5 + 0.2j])
def test_identical_media_array_shape_and_energy(index):
    cosine = np.array([[0.0, 1e-16, 1e-10], [0.3, 0.8, 1.0]])
    reflection, transmission = interface_rt(index, cosine, index)
    assert reflection.shape == cosine.shape
    assert transmission.shape == cosine.shape
    np.testing.assert_array_equal(reflection, np.zeros_like(cosine))
    np.testing.assert_array_equal(transmission, np.ones_like(cosine))
    np.testing.assert_array_equal(reflection + transmission, np.ones_like(cosine))


def test_identical_media_complex_cosines():
    """The matched-medium limit also supports complex propagation angles."""
    cosine = np.array([0.0j, 1e-16 + 1e-16j, 0.5 + 0.1j])
    reflection, transmission = interface_rt(1.5 + 0.2j, cosine, 1.5 + 0.2j)
    np.testing.assert_array_equal(reflection, np.zeros(cosine.shape))
    np.testing.assert_array_equal(transmission, np.ones(cosine.shape))
    assert interface_rt(1.5 + 0.2j, 0.5 + 0.1j, 1.5 + 0.2j) == (0.0, 1.0)


@pytest.mark.parametrize("n_i,n_t", [(1.0, 1.5), (1.0, 1.0 + 1e-12)])
def test_mismatched_grazing_reflects(n_i, n_t):
    """An arbitrarily small nonzero contrast still reflects at exact grazing."""
    reflection, transmission = interface_rt(n_i, 0.0, n_t)
    assert reflection == 1.0
    assert transmission == 0.0


def test_total_internal_reflection():
    reflection, transmission = interface_rt(1.5, np.array([0.0, 0.2, 0.5]), 1.0)
    np.testing.assert_allclose(reflection, 1.0)
    np.testing.assert_array_equal(transmission, 0.0)


@pytest.mark.parametrize("n_t", [1.5, 1.0 + 1e-12, 1.5 + 0.2j])
def test_normal_incidence_analytic_power(n_t):
    """Normal-incidence Fresnel power is known without Snell's law."""
    n_i = 1.0
    reflection, transmission = interface_rt(n_i, 1.0, n_t)
    expected_r = abs((n_i - n_t) / (n_i + n_t)) ** 2
    expected_t = n_t.real / n_i * abs(2 * n_i / (n_i + n_t)) ** 2
    assert reflection == pytest.approx(expected_r, rel=1e-12, abs=0.0)
    assert transmission == pytest.approx(expected_t, rel=1e-12)
