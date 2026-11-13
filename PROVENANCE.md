# RadPhen adding-doubling backend

Distribution: **radphen-iadpython 0.7.0**. Import: `iadpython`.
CLI: `iadp`. This is a maintained fork, not the upstream distribution.

Exact audited source: https://github.com/PanxoPanza/iadpython/tree/687a1b803594075d7f4f8feba9e1f7468827e7f9
(commit `687a1b803594075d7f4f8feba9e1f7468827e7f9`). That fork derives from
Scott Prahl's https://github.com/scottprahl/iadpython .
The original MIT copyright and license remain verbatim in LICENSE.txt and
are included in wheels and source distributions. Package author and numerical
code attribution remain Scott Prahl; RadPhen maintains this distribution.

This release changes distribution metadata and release tooling and includes
three narrowly reviewed source corrections, each audited explicitly against
the source commit by tools/compare_backend.py:

- `interface_rt` returns analytic R=0, T=1 for identical finite positive-real-part
  scalar media, including exact grazing incidence.
- `Sphere.uniform` passes explicit `(0.0, 1.0)` to three `random.gauss` calls,
  retaining the standard-normal sampling while supporting Python 3.10.
- Three two-parameter inverse searches retain their Nelder-Mead objectives,
  then recompute the selected point's L1 residual. Only when it exceeds the
  existing tolerance does a bounded least-squares residual-vector fit run.
  A candidate is accepted only for finite parameters and a finite, strictly
  smaller L1 residual. Selected parameters are explicitly restored after trials.
  This escapes flat L1 plateaus near conservative albedo; scalar searches,
  initialization, measurement physics and acceptance tolerance are unchanged.

The remaining 18 executable module ASTs match the source commit. Function
docstrings clarify the external incident-cosine convention. HG, TABULATED,
MOMENTS and complex-boundary forward fixtures remain unchanged.
See tools/README.md for reproducible source and fixture checks.

The original import and CLI are retained. Do not install upstream `iadpython`
and `radphen-iadpython` together: both own the same `iadpython` package and
`iadp` command. Use separate clean environments for baseline comparisons.

## Release procedure

1. Recheck the proposed name radphen-iadpython on PyPI immediately before
   publishing; the repository does not reserve it.
2. Configure a PyPI Trusted Publisher for the actual GitHub owner/repository,
   workflow pypi.yaml, environment pypi. Protect the environment with owner
   approval. No API token is stored in this repository.
3. Review the source diff and validation, push the fork, and create reviewed
   tag v0.7.0. Only owner-initiated manual dispatch on that tag can publish.
   If transferring to an organization, replace the actor ownership condition
   with the explicitly authorized release owner.
4. Dispatch the publish workflow. It builds and checks both artifacts before
   the protected job requests a short-lived PyPI OIDC credential.
5. Verify installed wheel and iadp --version from PyPI before
   releasing empylib 0.2.0.

Local build/test commands do not publish packages.

## Resolution of the seven initial baseline failures

The initial audited baseline had seven failing tests. Review separated one
physical edge-case defect from stale test calls and inputs:

- Three matched-interface Fresnel/glass failures shared the identical-media
  precision/grazing defect. The new exact matched-media guard fixes that limit.
  Nearby unequal indices retain the original Fresnel path.
- Two specular tests converted the incident cosine into the slab, then passed
  it to an API that takes the cosine in the illuminated external medium.
  Their inputs now use the original external cosine; expectations are retained.
  The production specular algorithm is unchanged and its docstrings clarified.
- Two redistribution tests called the removed `hg_legendre` name. They now
  call the existing `phase_legendre` API with the same expected matrices.

The reviewed Fresnel/redistribution and new interface/incidence regression
selection passes 71 tests. No test exclusions or xfails were added. Numerical
fixtures remain exactly equal for all six cases and 816 recorded values.
The full inherited suite is a separate required CI gate; the earlier bounded,
incomplete run is not a passing result. Publication requires that gate to pass.

## Additional findings from the first full GitHub CI

Full CI for initial packaging commit 78d24 found five failing tests on Python
3.10 (four sphere tests reached zero-argument random.gauss and one inverse
roundtrip stalled), and one inverse-roundtrip failure on Python 3.14. These
were additional inherited issues beyond the initial seven targeted failures.
They motivated the explicit Gaussian-default compatibility correction and
bounded inverse-fit refinement above, rather than relaxed tolerances or skips.
The sphere selection passes 13 tests on both Python 3.10 and 3.14. Five new
inverse boundary/control regressions pass, checking real measured transport at
g=-0.9, 0 and 0.9, rejection/restoration of a worse candidate, and no fallback
for an already adequate fit. The unchanged full zero-sphere inverse-roundtrip file also passes all six tests.
Full CI after these corrections remains required;
the first failing CI is not evidence of a complete passing release.
