### Context
While our suite of unit tests should run in a stable and deterministic manner,
running the tests under stochastic conditions allows us to detect instabilities
that would otherwise remain undetected. Especially for the set of reference tests,
we would like to ascertain that they run stable even with random seeds.
In addition, we would like to mitigate the issues that arise
from some of the optimizers setting a global (torch) seed, which is affecting
subsequent tests.

### Decision
Instead of using a constant (fixed) seed, we will seed all reference tests with a
random seed from a test fixture. That ensures that all tests will be fully reproducible,
since the seed will be displayed in the test output.

### Consequences
We accept that our test suite will not be fully deterministic any more, and that it
is possible that instabilities will surface in the existing tests.
