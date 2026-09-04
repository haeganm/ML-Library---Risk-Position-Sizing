# Releasing

The version lives in one place, `project(mlrisk VERSION x.y.z)` in
`CMakeLists.txt`. The C reads it through `version.h`, the Python package reads
it through `pyproject.toml`, and the release workflow refuses to publish if a
tag disagrees with it.

## One-time PyPI setup

Publishing uses [trusted publishing](https://docs.pypi.org/trusted-publishers/),
so there is no API token in this repository and nothing to rotate. On PyPI, add
a publisher under the project's settings (or under "pending publishers" before
the first release, which also reserves the name):

| Field | Value |
|---|---|
| PyPI project name | `walkforward` |
| Owner | `haeganm` |
| Repository name | `walkforward` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Then create a GitHub environment called `pypi` under Settings, Environments.
Add a required reviewer if you want a manual gate before anything is published.

## Cutting a release

1. Set the version in `CMakeLists.txt` and add the section to `CHANGELOG.md`.
2. Commit, push to `main`, and wait for CI to go green.
3. Tag and push it:

   ```bash
   git tag v3.4.0
   git push origin v3.4.0
   ```

4. The `Release` workflow builds a source distribution and five wheels
   (Linux x86-64 and aarch64, macOS Intel and Apple silicon, Windows x64),
   runs the test suite against each wheel on the platform it was built for,
   installs the sdist from scratch in a clean directory, and publishes
   everything to PyPI.
5. Write the GitHub release notes from the changelog section:

   ```bash
   gh release create v3.4.0 --title "walkforward 3.4.0" --notes-file notes.md --latest
   ```

## Checking the build without publishing

Run the `Release` workflow by hand from the Actions tab. The publish job only
runs for a tag, so a manual run builds and tests every wheel and leaves them as
downloadable artifacts. Do this after any change to the build, the packaging or
the supported platforms.

## Why the wheels look the way they do

The binding is ctypes and never touches the Python C API, so each wheel is
tagged `py3-none-<platform>` and works on every Python 3 that meets
`requires-python`. That is why `cibuildwheel` builds a single CPython per
platform instead of one wheel per interpreter version: the others would be
byte-identical. The ordinary CI workflow still tests the package from source
on the oldest and newest supported Python.
