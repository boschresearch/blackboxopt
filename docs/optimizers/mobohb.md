# MO-BOHB Optimizer

MO-BOHB is the multi-objective variant of [BOHB](bohb.md). It reuses BOHB's Hyperband
racing and its kernel-density-estimator based Bayesian Optimization, but replaces the
single-objective, scalar-loss based decisions with multi-objective ones:

- The **good/bad split** that trains the kernel density estimators ranks configurations by
  nondominated (Pareto) front and breaks ties within the boundary front by crowding
  distance (NSGA-II style), instead of sorting by a single scalar loss.
- The **successive halving promotion** between stages advances configurations by the same
  nondominated ranking.

The MO-BOHB baseline from
[Bag of Baselines for Multi-objective Joint Neural Architecture Search and Hyperparameter
Optimization](https://arxiv.org/abs/2105.01015) is similar but differs in two ways:
Instead of the multivariate kernel density estimators, it uses per-hyperparameter
univariate MO-TPE kernels, and it uses NDS + HSSP instead of ranking configurations by
nondominated front with the NSGA-II crowding distance.

Since MO-BOHB shares BOHB's Hyperband schedule, the fidelity calculator on the
[BOHB page](bohb.md#fidelities) applies here as well.


## Reference

::: blackboxopt.optimizers.mobohb
