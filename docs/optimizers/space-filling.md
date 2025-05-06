# Space Filling Optimizer

The `SpaceFilling` optimizer is a
[Sobol sequence](https://en.wikipedia.org/wiki/Sobol_sequence) based optimizer that
covers the search space based on a quasi-random low-discrepancy sequence.
This strategy requires a larger budget for evaluations but can be a good initial
approach to get to know the optimization problem at hand.
While this implementation follows the overall interface including the specification and
reporting of objectives and their values, the actual objective values are
inconsequential for the underlying Sobol sequence and do not guide the optimization.

## Reference

::: blackboxopt.optimizers.space_filling
