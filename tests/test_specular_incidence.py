"""External incident-angle contract checked with independent real-index optics."""

import numpy as np
import pytest

import iadpython as iad


def _cosine(index, incident_index, incident_cosine):
    """Snell invariant, for the propagating real-index cases used here."""
    return np.sqrt(1 - (incident_index / index) ** 2 * (1 - incident_cosine**2))


def _reflection(n1, c1, n2, c2):
    rs = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
    rp = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
    return (rs**2 + rp**2) / 2


def _glass_reflection(outer, glass, slab, cosine):
    cg = _cosine(glass, outer, cosine)
    cs = _cosine(slab, outer, cosine)
    r1 = _reflection(outer, cosine, glass, cg)
    r2 = _reflection(glass, cg, slab, cs)
    return r1 + (1 - r1) ** 2 * r2 / (1 - r1 * r2)


def _stack_rt(indices, cosine, slab_thickness):
    """Sum incoherent round trips across each real, nonabsorbing interface.

    Only the central slab attenuates.  This oracle uses no library angle,
    interface, glass, or specular helpers.
    """
    cosines = [_cosine(n, indices[0], cosine) for n in indices]
    reflectance, transmittance = 0.0, 1.0
    for j in range(len(indices) - 2, -1, -1):
        r = _reflection(indices[j], cosines[j], indices[j + 1], cosines[j + 1])
        path = slab_thickness / cosines[j + 1] if j + 1 == 2 else 0.0
        attenuation = np.exp(-path)
        denominator = 1 - r * reflectance * attenuation**2
        transmittance = (1 - r) * attenuation * transmittance / denominator
        reflectance = r + (1 - r) ** 2 * attenuation**2 * reflectance / denominator
    return reflectance, transmittance


@pytest.mark.parametrize("outer_top,outer_bot", [(1.0, 1.0), (1.1, 1.25)])
@pytest.mark.parametrize("flip", [False, True])
def test_opaque_stack_external_cosine(outer_top, outer_bot, flip):
    top, slab, bottom = 1.6, 1.5, 1.7
    cosine = np.array([0.4, 0.65, 0.9])
    outer, glass = (outer_bot, bottom) if flip else (outer_top, top)
    expected = _glass_reflection(outer, glass, slab, cosine)
    r, t = iad.specular_rt(
        top, slab, bottom, np.inf, cosine, flip=flip,
        n_outer_top=outer_top, n_outer_bot=outer_bot,
    )
    np.testing.assert_allclose(r, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(t, 0.0)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("thickness", [0.0, 0.7, np.inf])
def test_sample_and_direct_specular_external_incidence(reverse, thickness):
    indices = [1.1, 1.6, 1.5, 1.7, 1.25]
    if reverse:
        indices.reverse()
    outer_top, top, slab, bottom, outer_bot = indices
    cosine = 0.65
    sample = iad.Sample(
        b=thickness, n=slab, n_above=top, n_below=bottom,
        n_outer_above=outer_top, n_outer_below=outer_bot,
    )
    sample.nu_0 = cosine
    expected = _stack_rt(indices, cosine, thickness)
    direct = iad.specular_rt(
        top, slab, bottom, thickness, sample.nu_0,
        n_outer_top=outer_top, n_outer_bot=outer_bot,
    )
    np.testing.assert_allclose(direct, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(sample.unscattered_rt(), expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("flip", [False, True])
def test_nonabsorbing_stack_conserves_power(flip):
    r, t = iad.specular_rt(
        1.6, 1.5, 1.7, 0.0, np.array([0.4, 0.65, 0.9]), flip=flip,
        n_outer_top=1.1, n_outer_bot=1.25,
    )
    np.testing.assert_allclose(r + t, 1.0, rtol=1e-12, atol=1e-14)
