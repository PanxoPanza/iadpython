"""Inverse fits must escape conservative-albedo L1 plateaus."""

from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import Bounds

import iadpython


@pytest.mark.parametrize("g", [-0.9, 0.0, 0.9])
def test_glass_slide_unscattered_round_trip(g):
    """Recover measured fluxes when the warm start is conservative."""
    boundaries = dict(n=1.4, n_above=1.5, n_below=1.5, quad_pts=16)
    forward = iadpython.Sample(a=0.99, b=1.0, g=g, **boundaries)
    reflected, transmitted, _, _ = forward.rt()
    _, unscattered = forward.unscattered_rt()
    experiment = iadpython.Experiment(
        r=float(reflected), t=float(transmitted), u=float(unscattered),
        sample=iadpython.Sample(**boundaries),
    )
    a_fit, b_fit, g_fit = experiment.invert_rt()
    recovered = iadpython.Sample(a=a_fit, b=b_fit, g=g_fit, **boundaries)
    r_fit, t_fit, _, _ = recovered.rt()
    _, u_fit = recovered.unscattered_rt()
    np.testing.assert_allclose(
        [r_fit, t_fit, u_fit], [reflected, transmitted, unscattered],
        rtol=0, atol=1e-4,
    )


def test_refinement_restores_better_original_fit():
    """A worse trial and its side effects must not replace the selected fit."""
    experiment = iadpython.Experiment(r=0.4, t=0.4)
    experiment.measured_rt = lambda: (experiment.sample.a, experiment.sample.g)
    bounds = Bounds([0, -0.999999], [1, 0.999999])

    def worse_trial(residual, values, **kwargs):
        residual([0.9, -0.9])
        return SimpleNamespace(x=np.array([0.9, -0.9]))

    with patch("iadpython.iad.scipy.optimize.least_squares", side_effect=worse_trial):
        experiment._refine_two_parameter_fit([0.5, 0.5], ("a", "g"), bounds)
    assert experiment.sample.a == 0.5
    assert experiment.sample.g == 0.5


def test_well_fitted_solution_needs_no_refinement():
    """Preserve successful Nelder-Mead fits without another optimization."""
    experiment = iadpython.Experiment(r=0.4, t=0.4)
    experiment.measured_rt = lambda: (experiment.sample.a, experiment.sample.g)
    bounds = Bounds([0, -0.999999], [1, 0.999999])
    with patch("iadpython.iad.scipy.optimize.least_squares") as refine:
        experiment._refine_two_parameter_fit([0.4, 0.4], ("a", "g"), bounds)
    refine.assert_not_called()
