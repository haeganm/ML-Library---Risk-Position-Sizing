# Releasing

The version lives in one place, `project(mlrisk VERSION x.y.z)` in
`CMakeLists.txt`. The C reads it through `version.h`, the Python package reads
it through `pyproject.toml`, and the release workflow refuses to publish if a
tag disagrees with it.

## One-time PyPI setup

Publishing uses a PyPI API token held as a repository secret.

1. On PyPI, under Account settings, create an API token. Before the project
   exists the token has to be account-scoped; after the first release, replace
   it with one scoped to the `walkforward` project alone.
2. In this repository, under Settings, Secrets and variables, Actions, add it
   as `PYPI_API_TOKEN`.

The GitHub environment `pypi` already exists and gates the publish job. Add a
required reviewer to it if you want a manual approval before anything ships.

### Why not trusted publishing

Trusted publishing would be better: no token, nothing to rotate or leak. It
does not work for this repository today. GitHub issues it an OIDC subject
claim of the form

```
repo:haeganm@220532114/walkforward@1134726644:environment:pypi
```

with the numeric owner and repository identifiers embedded, while PyPI matches
publishers against the older

```
repo:haeganm/walkforward:environment:pypi
```

Every other claim PyPI checks (`repository`, `repository_owner`,
`workflow_ref`, `environment`) matches a correctly configured publisher, and
the exchange is still refused with `invalid-publisher`. Overriding the subject
template through the repository OIDC customization API is accepted but does
not change the claim GitHub actually emits. Worth revisiting once PyPI accepts
the identifier-bearing subject, at which point the token and the `password:`
line in `release.yml` can both go away.

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
