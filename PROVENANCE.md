# RadPhen adding-doubling backend

Distribution: **radphen-iadpython 0.7.0**. Import: `iadpython`.
CLI: `iadp`. This is a maintained fork, not the upstream distribution.

Exact audited source: https://github.com/PanxoPanza/iadpython/tree/687a1b803594075d7f4f8feba9e1f7468827e7f9
(commit `687a1b803594075d7f4f8feba9e1f7468827e7f9`). That fork derives from
Scott Prahl's https://github.com/scottprahl/iadpython .
The original MIT copyright and license remain verbatim in LICENSE.txt and
are included in wheels and source distributions. Package author and numerical
code attribution remain Scott Prahl; RadPhen maintains this distribution.

This release changes distribution metadata and release tooling, and includes
one reviewed numerical correction: `interface_rt` returns the analytic R=0,
T=1 limit for identical finite positive-real-part scalar media, including
exact grazing incidence. The exact guard is audited in tools/compare_backend.py;
the remainder of all 21 module executable ASTs matches the source commit.
Function docstrings clarify the external incident-cosine convention. HG,
TABULATED and MOMENTS phase functions and complex boundaries are preserved.
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
