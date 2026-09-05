# Releasing

The version lives in one place, `project(mlrisk VERSION x.y.z)` in
`CMakeLists.txt`. The C reads it through `version.h`, the Python package reads
it through `pyproject.toml`, and the release workflow refuses to publish if a
tag disagrees with it.

## One-time PyPI setup

Publishing uses [trusted publishing](https://docs.pypi.org/trusted-publishers/):
PyPI accepts a short-lived token that GitHub mints for the workflow run, so
there is no API token in this repository and nothing to rotate or leak.

Under the project on PyPI, Settings, Publishing, add a GitHub publisher:

| Field | Value |
|---|---|
| Owner | `haeganm` |
| Repository name | `walkforward` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

The workflow name is the file name in `.github/workflows/`, letter for letter.
`release.yaml` does not match `release.yml`, and the failure PyPI reports for
that (`invalid-publisher`) does not say which field is wrong. The GitHub
environment `pypi` already exists and gates the publish job; add a required
reviewer to it if you want a manual approval before anything ships.

## Cutting a release

1. Set the version in `CMakeLists.txt` and add the section to `CHANGELOG.md`.
2. Commit, push to `main`, and wait for CI to go green.
3. Tag and push it:

   ```bash
   git tag v3.4.1
   git push origin v3.4.1
   ```

4. The `Release` workflow builds a source distribution and five wheels
   (Linux x86-64 and aarch64, macOS Intel and Apple silicon, Windows x64),
   runs the test suite against each wheel on the platform it was built for,
   and installs the sdist from scratch in a clean directory. A check job
   then runs `twine check --strict`, requires exactly five wheels named
   for the source version and each carrying `walkforward_native`, and
   requires a `## <version>` section in `CHANGELOG.md`. Only then does it
   publish. A tag pushed with the section still headed `Unreleased` fails
   at that check, after the wheels are built; delete the tag, add the
   section, tag again. Publishing skips files PyPI already has, so a
   re-run after a partial upload finishes the job.
5. Write the GitHub release notes from the changelog section:

   ```bash
   gh release create v3.4.1 --title "walkforward 3.4.1" --notes-file notes.md --latest
   ```

## Checking the build without publishing

Run the `Release` workflow by hand from the Actions tab. The publish job only
runs for a tag, so a manual run builds and tests every wheel and leaves them as
downloadable artifacts. Do this after any change to the build, the packaging or
the supported platforms. The changelog check applies to manual runs too: after
a version bump, the section has to exist before the run goes green.

## Why the wheels look the way they do

The binding is ctypes and never touches the Python C API, so each wheel is
tagged `py3-none-<platform>` and works on every Python 3 that meets
`requires-python`. That is why `cibuildwheel` builds a single CPython per
platform instead of one wheel per interpreter version: the others would be
byte-identical. The ordinary CI workflow still tests the package from source
on the oldest and newest supported Python.
