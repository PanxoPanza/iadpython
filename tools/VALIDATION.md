# Backend release validation

Audited source: PanxoPanza/iadpython commit
687a1b803594075d7f4f8feba9e1f7468827e7f9. Original checkout remains untouched;
builds use clean temporary copies, excluding build/egg-info artifacts.
Distribution: radphen-iadpython 0.7.0. Import: iadpython. CLI: iadp.

## Reviewed source and preflight wheel

- All 21 modules were audited against the source SHA: 20 executable ASTs are
  unchanged; fresnel.py matches the exact approved interface_rt matched-media
  guard plus its baseline executable AST. Only function docstrings and source
  URL attribution are otherwise normalized. Resources and MIT license match
  the baseline byte for byte. The algorithm change is explicit, not parity.
- The seven initial failures are resolved as documented in PROVENANCE.md:
  three matched-interface defect cases, two stale specular input cases, two
  stale redistribution API calls. The reviewed targeted selection passes 71
  tests, including new interface/incidence regression coverage.
- A clean preflight wheel was installed into a private Python 3.14.6 virtual
  environment. Under python -I outside both source checkouts it loaded from
  site-packages, exposed radphen-iadpython==0.7.0 metadata, and had no upstream
  iadpython distribution or radphen_iadpython import alias. pip check and
  iadp --version passed. Both solver distributions were uninstalled from this
  private environment before installing the new wheel; no global env changed.
- A new installed-wheel fixture matched the preserved audited original fixture:
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
