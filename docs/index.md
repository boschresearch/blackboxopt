# Blackbox Optimization

The `blackboxopt` Python package contains blackbox optimization algorithms with a common
interface, along with useful helpers like parallel optimization loops, analysis and
visualization tools.

## Key Features

### Common Interface

The `blackboxopt` base classes along with the `EvaluationSpecification` and `Evaluation`
data classes specify a unified interface for different blackbox optimization method
implementations.
In addition to these interfaces, a standard pytest compatible testsuite is available
to ensure functional compatibility of an optimizer implementation with the `blackboxopt`
framework.

### Optimizers

Aside from random search, the two main optimizers part of this package are Hyperband and
BOHB. Where BOHB is intended as a cleaner replacement of the former implementation in
[HpBandSter](https://github.com/automl/HpBandSter).

### Optimization Loops

As part of the `blackboxopt.optimization_loops` module compatible implementations for
optimization loops are avilable bot for local, serial execution as well as for
distributed optimization via `dask.distributed`.

### Visualizations

Interactive visualizations like objective value over time or duration for single
objective optimization, as well as an objectives pair plot with a highlighted pareto
front for multi objective optimization is available as part of the
`blackboxopt.visualizations` module.

## Getting Started

The following example outlines how a quadratic function can be optimized with random
search in a distributed manner.

```python
--8<--
blackboxopt/examples/dask_distributed.py
--8<--
```

## License

`blackboxopt` is open-sourced under the Apache-2.0 license. See the [LICENSE](LICENSE)
file for details.

For a list of other open source components included in `blackboxopt`, see the file
[3rd-party-licenses.txt](https://github.com/boschresearch/blackboxopt/blob/main/3rd-party-licenses.txt).
