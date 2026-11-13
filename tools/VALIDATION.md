# Backend release validation

Audited source: PanxoPanza/iadpython commit
687a1b803594075d7f4f8feba9e1f7468827e7f9. Original checkout remains untouched;
builds use clean temporary copies, excluding build/egg-info artifacts.
Distribution: radphen-iadpython 0.7.0. Import: iadpython. CLI: iadp.

## Reviewed source and prior preflight wheel

- All 21 modules are audited against the source SHA: 18 executable ASTs are
  unchanged; exact expected patches cover the Fresnel matched-media guard,
  three explicit standard-normal sphere arguments, and three two-parameter
  inverse result/refinement calls plus the exact approved helper. Complete
  module ASTs must match after those explicit transformations. Only function
  docstrings and source URL attribution are otherwise normalized. Resources
  and MIT license remain byte-identical to the baseline.
- The seven initial failures are resolved as documented in PROVENANCE.md:
  three matched-interface defect cases, two stale specular input cases, two
  stale redistribution API calls. The reviewed targeted selection passes 71
  tests, including new interface/incidence regression coverage.
- Before the additional sphere/inverse CI corrections, a clean preflight wheel was installed into a private Python 3.14.6 virtual
  environment. Under python -I outside both source checkouts it loaded from
  site-packages, exposed radphen-iadpython==0.7.0 metadata, and had no upstream
  iadpython distribution or radphen_iadpython import alias. pip check and
  iadp --version passed. Both solver distributions were uninstalled from this
  private environment before installing the new wheel; no global env changed.
- That installed-wheel fixture matched the preserved audited original fixture:
  six cases, 816 values, rtol=1e-10, atol=1e-12, maximum absolute difference 0,
  maximum relative difference 0. NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.1.

## Frozen artifact checks

The final source is copied cleanly after this report is prepared. Frozen wheel
and sdist hashes are recorded externally in SHA256SUMS.txt. Release validation
checks twine metadata, package paths, the iadp entry point, NIST CSV, original
MIT license, provenance and both fixtures. It installs the frozen wheel and an
independently rebuilt sdist wheel, verifies metadata/resources/CLI and compares
all rebuilt wheel member bytes. Logs and final results accompany the release
review; this source report distinguishes completed preflight checks from the
subsequent frozen artifact checks.

The full inherited suite remains a separate mandatory publication CI gate.
The earlier bounded full run was incomplete and is not a passing result.
No tests are excluded or xfailed beyond the inherited notebook exclusion.
All local numerical validation uses one BLAS/OpenMP thread.

## Full CI findings and reviewed follow-up

Initial full GitHub CI at commit 78d24 failed five tests on Python 3.10
(four sphere gauss-default compatibility failures and one inverse roundtrip)
and one inverse roundtrip on Python 3.14. These follow-up changes do not relax
residual tolerances: the sphere selection passes 13 tests on each interpreter;
five inverse boundary/control tests pass, including high positive/negative
anisotropy measured-flux closure, a worse fallback rejection/restoration case,
and an already-converged fit that must not invoke the fallback.
The unchanged full zero-sphere inverse-roundtrip file passes all six tests
(117.69 seconds). Full GitHub CI after the follow-up source corrections remains
pending.
