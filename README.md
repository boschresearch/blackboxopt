# Blackbox Optimization

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CI/CD](https://github.com/boschresearch/blackboxopt/workflows/ci-cd-pipeline/badge.svg)](https://github.com/boschresearch/blackboxopt/actions?query=workflow%3Aci-cd-pipeline+branch%3Amain)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/blackboxopt)](https://pypi.org/project/blackboxopt/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/blackboxopt)](https://pypi.org/project/blackboxopt/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Various blackbox optimization algorithms with a common interface along with useful
helpers like parallel optimization loops, analysis and visualization scripts.

Random search is provided as an example optimizer along with tests for the interface.

New optimizers can require `blackboxopt` as a dependency, which is just the light-weight
interface definition.
If you want all optimizer implementations that come with this package, install
`blackboxopt[all]`
Alternatively, you can get individual optimizers with e.g. `blackboxopt[bohb]`

This software is a research prototype.
The software is not ready for production use.
It has neither been developed nor tested for a specific use case.
However, the license conditions of the applicable Open Source licenses allow you to
adapt the software to your needs.
Before using it in a safety relevant setting, make sure that the software fulfills your
requirements and adjust it according to any applicable safety standards
(e.g. ISO 26262).

## Documentation

**Visit [boschresearch.github.io/blackboxopt](https://boschresearch.github.io/blackboxopt/)**

## Development

Install poetry >= 1.5.0

```
pip install --upgrade poetry
```

Install the `blackboxopt` package from source by running the following from the root
directory of _this_ repository

```
poetry install
```

(Optional) Install [pre-commit](https://pre-commit.com) hooks to check code standards
before committing changes:

```
poetry run pre-commit install
```

## Test

Make sure to install all extras before running tests

```
poetry install -E testing
poetry run pytest tests/
```

For HTML test coverage reports run

```
poetry run pytest tests/ --cov --cov-report html:htmlcov
```

## Building Documentation

Make sure to install _all_ necessary dependencies:

```
poetry install --extras=all
```

The documentation can be built from the repository root as follows:

```
poetry run mkdocs build --clean --no-directory-urls
```

For serving it locally while working on the documentation run:

```
poetry run mkdocs serve
```

## Architectural Decision Records

### Create evaluation result from specification

In the context of initializing an evaluation result from a specification, facing the
concern that having a constructor with a specification argument while the specification
attributes end up as toplevel attributes and not summarized under a specification
attribute we decided for unpacking the evaluation specification like a dictionary into
the result constructor to prevent the said cognitive dissonance, accepting that the
unpacking operator can feel unintuitive and that users might tend to matching the
attributes explictly to the init arguments.

### Report multiple evaluations

In the context of many optimizers just sequentally reporting the individual evaluations
when multiple evaluations are reported at once and thus not leveraging any batch
reporting benefits, facing the concern that representing that common behaviour in the
optimizer base class requires the definition of an abstract report single and an
abstract report multi method for which the report single does not need to be implemented
if the report multi is, we decided to refactor the arising redundancy into a function
`call_functions_with_evaluations_and_collect_errors`, accepting that this increases the
cognitive load when reading the code.

## License

`blackboxopt` is open-sourced under the Apache-2.0 license. See the [LICENSE](LICENSE)
file for details.

For a list of other open source components included in `blackboxopt`, see the file
[3rd-party-licenses.txt](3rd-party-licenses.txt).
