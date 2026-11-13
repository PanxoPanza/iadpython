"""Record parity fixtures in separate environments and compare audited ASTs."""

import argparse
import ast
import importlib
import json
from pathlib import Path
import subprocess

SOURCE_SHA = "687a1b803594075d7f4f8feba9e1f7468827e7f9"


def record(module, output):
    import numpy as np
    import pandas as pd

    iad = importlib.import_module(module)
    mu = np.linspace(-1, 1, 181)
    g = 0.55
    phase = (1 - g**2) / (1 + g**2 - 2 * g * mu)**1.5
    values = {}
    for kind in ("HG", "TABULATED", "MOMENTS"):
        for complex_boundary in (False, True):
            kwargs = dict(a=0.8, b=1.2, g=g, n=1.4, quad_pts=8)
            if complex_boundary:
                kwargs.update(n_sample_boundary=1.4+0.015j,
                              n_outer_above=1.0+0.002j, n_outer_below=1.2+0.01j)
            if kind == "TABULATED":
                kwargs.update(pf_type=kind, pf_data=pd.DataFrame({550: phase}, index=mu))
            elif kind == "MOMENTS":
                kwargs.update(pf_type=kind, pf_data=g**np.arange(17))
            sample = iad.Sample(**kwargs)
            r, _, t, _ = sample.rt_matrices()
            result = np.concatenate([
                np.ravel(sample.UX1_and_UXU(r, t)),
                np.ravel(sample.unscattered_rt()),
                np.ravel(sample.rt_diffuse_cone(r, t, 0.4, 0.95)),
                np.ravel(r), np.ravel(t),
            ])
            assert np.all(np.isfinite(result))
            values[f"{kind}-complex={complex_boundary}"] = result.tolist()
    Path(output).write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(values)} cases from {iad.__file__} in {output}")


def compare(before, after):
    import numpy as np

    left = json.loads(Path(before).read_text(encoding="utf-8"))
    right = json.loads(Path(after).read_text(encoding="utf-8"))
    assert left.keys() == right.keys()
    max_abs = max_rel = 0.0
    for name in left:
        a, b = np.asarray(left[name]), np.asarray(right[name])
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12, err_msg=name)
        delta = np.abs(a - b)
        max_abs = max(max_abs, float(np.max(delta)))
        max_rel = max(max_rel, float(np.max(delta / np.maximum(np.abs(a), 1e-12))))
    print(f"Numerical parity: {len(left)} cases, "
          f"{sum(len(v) for v in left.values())} values; "
          f"rtol=1e-10 atol=1e-12; max_abs={max_abs:.17g}; "
          f"max_rel={max_rel:.17g} (denominator floor 1e-12)")


# The only approved executable delta from SOURCE_SHA. Compare the exact added
# guard plus every original statement; never exempt the whole function or file.
MATCHED_INTERFACE_GUARD = """
if (
    np.ndim(n_i) == 0
    and np.ndim(n_t) == 0
    and n_i == n_t
    and np.isfinite(n_i)
    and np.real(n_i) > 0
):
    if np.ndim(nu_i) == 0:
        return 0.0, 1.0
    return np.zeros(np.shape(nu_i)), np.ones(np.shape(nu_i))
"""


class RemoveFunctionDocstrings(ast.NodeTransformer):
    """Allow documentation corrections, preserving all executable statements."""

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body.pop(0)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef


def check_source(original_repo, fork_root):
    names = subprocess.check_output([
        "git", "-C", original_repo, "ls-tree", "-r", "--name-only", SOURCE_SHA, "iadpython"
    ], text=True).splitlines()
    count = approved = 0
    for name in names:
        original = subprocess.check_output([
            "git", "-C", original_repo, "show", f"{SOURCE_SHA}:{name}"
        ])
        target = Path(fork_root) / name
        if name.endswith(".py"):
            actual = target.read_text(encoding="utf-8")
            actual = actual.replace("https://github.com/PanxoPanza/iadpython", "https://github.com/scottprahl/iadpython")
            expected_tree = RemoveFunctionDocstrings().visit(ast.parse(original.decode("utf-8")))
            actual_tree = RemoveFunctionDocstrings().visit(ast.parse(actual))
            if name == "iadpython/fresnel.py":
                functions = [n for n in expected_tree.body
                             if isinstance(n, ast.FunctionDef) and n.name == "interface_rt"]
                assert len(functions) == 1
                functions[0].body.insert(0, ast.parse(MATCHED_INTERFACE_GUARD).body[0])
                approved += 1
            assert ast.dump(expected_tree) == ast.dump(actual_tree), name
            count += 1
        else:
            assert target.read_bytes() == original, name
    assert count == 21 and approved == 1
    license_bytes = subprocess.check_output([
        "git", "-C", original_repo, "show", f"{SOURCE_SHA}:LICENSE.txt"
    ])
    assert (Path(fork_root) / "LICENSE.txt").read_bytes() == license_bytes
    print(f"Audited {count} modules: 20 unchanged executable ASTs; fresnel.py matches "
          "the exact approved interface_rt matched-media guard plus baseline AST. "
          "Only function docstrings/source URL normalized; resources and MIT license byte-identical")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("record")
    p.add_argument("module", choices=["iadpython"])
    p.add_argument("output")
    p = commands.add_parser("compare")
    p.add_argument("before")
    p.add_argument("after")
    p = commands.add_parser("source")
    p.add_argument("original_repo")
    p.add_argument("fork_root")
    args = parser.parse_args()
    if args.command == "record":
        record(args.module, args.output)
    elif args.command == "compare":
        compare(args.before, args.after)
    else:
        check_source(args.original_repo, args.fork_root)
