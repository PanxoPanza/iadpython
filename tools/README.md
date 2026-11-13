# Reproducing distribution parity

Publication setup and the default-branch workflow registration procedure are
documented in [PUBLISHING.md](PUBLISHING.md).

Create separate virtual environments using the same Python and NumPy/SciPy/
pandas versions. Install the original source at exact SHA
`687a1b803594075d7f4f8feba9e1f7468827e7f9` in the first, and the fork
wheel in the second. Build the original from a clean exported copy so the
working repository and its untracked work are untouched. Run outside either
source directory (use absolute script and output paths):

```text
<original-python> -I compare_backend.py record iadpython original.json
<fork-python> -I compare_backend.py record iadpython renamed.json
<fork-python> -I compare_backend.py compare original.json renamed.json
<fork-python> -I compare_backend.py source <original-git-repo> <fork-repo>
```

The six cases cover HG, TABULATED, MOMENTS, real and complex sample/exterior
boundaries, normal and Lambertian transport, unscattered transport, diffuse
cone transport, and full reflection/transmission matrices. Comparison uses
rtol=1e-10 and atol=1e-12, prints maximum absolute and relative differences
(relative denominator floored at 1e-12), and currently reports exact equality.
The source check audits every module against the exact SHA. It applies only
three explicit expected AST patches to the baseline: the matched-interface
guard; three standard-normal random.gauss argument pairs; and the three
inverse result/refinement calls plus the literal approved fallback helper.
It then requires each complete module AST to match. Only function docstrings
and source URL attribution are normalized; no function/file is exempted.
The other 18 executable module ASTs match their baseline unchanged. Runtime
resources and the MIT license are checked byte for byte. The original fixture
is preserved; renamed.json is the newly recorded fork fixture.

Both distributions import `iadpython`. Never co-install them; baseline and fork
fixtures must be recorded in separate environments. Artifact filenames retain
the normalized distribution prefix `radphen_iadpython-0.7.0`.
