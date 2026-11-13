# Publication setup

The `master` branch remains the original `iadpython` development branch. The
`codex/radphen-package` branch contains the separately named distribution
`radphen-iadpython==0.7.0`, importing `iadpython` and providing `iadp`.

Before using GitHub Releases, replace **only** the default branch's
`.github/workflows/pypi.yaml` with a manual-only discovery stub that fails with
instructions to dispatch `v0.7.0`. It must contain no upload action, OIDC
permission, or reusable-workflow reference. The legacy default-branch workflow publishes on `release: published`
and builds the original distribution. Do not merge the packaging branch into
`master` to register the workflow. The default-branch stub deliberately fails
on `master`. The packaging branch's full workflow can publish only when dispatched at
`refs/tags/v0.7.0` with confirmation `0.7.0`.

GitHub requires the dispatch workflow to exist on the default branch. Dispatch
the registered filename with `--ref v0.7.0` so its checkout and reusable CI use
the release tag, including that tag's `test.yaml`. The tag must include both
workflows and all final fixes. First run backend CI on the packaging branch and
inspect every matrix result; then dispatch publication at the validated tag.
Publication reruns the complete required backend CI matrix before building and
uploading its artifacts.

```powershell
gh workflow run test.yaml --repo PanxoPanza/iadpython --ref codex/radphen-package
gh workflow run pypi.yaml --repo PanxoPanza/iadpython --ref v0.7.0 -f confirm-version=0.7.0
```

The pre-existing `citation.yaml` also reacts to published GitHub Releases and
attempts to commit to the release's target branch. Creating a GitHub Release is
not needed for Trusted Publishing; use manual tag dispatch and review this
separate automation before creating one.

Configure the GitHub environment `pypi` with an allowed tag `v0.7.0`. Configure
the pending PyPI publisher at <https://pypi.org/manage/account/publishing/>:

| Project | GitHub owner | Repository | Workflow filename | Environment |
| --- | --- | --- | --- | --- |
| radphen-iadpython | PanxoPanza | iadpython | pypi.yaml | pypi |
| empylib | radphen-lab | empylib | publish.yml | pypi |

For the existing empylib project, use its owner-only configuration page:
<https://pypi.org/manage/project/empylib/settings/publishing/>. Do not configure
this backend workflow as a publisher for the original `iadpython` project.
GitHub repository admin access configures environments and tag policies; PyPI
project/account access separately registers each Trusted Publisher.

The matrix tests the installed wheel on Python 3.10–3.14, Linux and Windows.
Each job caps BLAS/OpenMP at one thread, has a 90-minute bound, logs individual
test progress and durations, and saves JUnit result artifacts
even after test failure. Only the pre-existing notebook exclusion is retained;
all collected numerical and CLI tests must pass. A timeout or interrupted run
is incomplete and never qualifies as a passing publication gate.

References: [manual workflow dispatch](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow),
[PyPI Trusted Publisher registration](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
