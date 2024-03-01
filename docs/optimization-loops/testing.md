# Reference Tests

To test an optimization loop implementation across various reference scenarios, follow:

```python
import pytest
from blackboxopt.optimization_loops.testing import ALL_REFERENCE_TESTS

@pytest.mark.parametrize("reference_test", testing.ALL_REFERENCE_TESTS)
def test_all_reference_tests(reference_test):
    reference_test(custom_optimization_loop, {"opt_loop_specific_kwarg": 123})
```

where you can include custom keyword arguments that are passed to the optimization loop
calls in the reference tests.
