# Reference Tests

When you develop an optimizer based on the interface defined as part of
`blackboxopt.base`, you can use `blackboxopt.testing` to directly test whether your
implementation follows the specification by adding a test like this to your test suite:

```python
import pytest
from blackboxopt.testing import ALL_REFERENCE_TESTS

@pytest.mark.parametrize("reference_test", ALL_REFERENCE_TESTS)
def test_all_reference_tests(reference_test):
    reference_test(CustomOptimizer, optional_optimizer_init_kwargs)
```
