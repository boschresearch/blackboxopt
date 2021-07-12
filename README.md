# Blackbox Optimization

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

## Development

Install poetry

```
pip install poetry
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

### Custom Optimizers

When you develop an optimizer based on the interface defined as part of
`blackboxopt.base`, you can use `blackboxopt.testing` to directly test whether your
implementation follows the specification by adding a test like this to your test suite.

```python
from blackboxopt.testing import ALL_REFERENCE_TESTS

@pytest.mark.parametrize("reference_test", ALL_REFERENCE_TESTS)
def test_all_reference_tests(reference_test):
    reference_test(CustomOptimizer, custom_optimizer_init_kwargs)
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

## License

`blackboxopt` is open-sourced under the Apache-2.0 license. See the [LICENSE](LICENSE)
file for details.

For a list of other open source components included in `blackboxopt`, see the file
[3rd-party-licenses.txt](3rd-party-licenses.txt).