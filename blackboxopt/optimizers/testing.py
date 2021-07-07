# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

"""Tests that can be imported and used to test optimizer implementations against this
packages blackbox optimizer interface."""

from typing import List

import parameterspace as ps

from blackboxopt import Objective, ObjectivesError, OptimizationComplete, Optimizer
from blackboxopt.base import MultiObjectiveOptimizer, SingleObjectiveOptimizer


def _initialize_optimizer(
    optimizer_class,
    optimizer_kwargs: dict,
    objective: Objective,
    objectives: List[Objective],
    seed=42,
) -> Optimizer:
    space = ps.ParameterSpace()
    space.add(ps.IntegerParameter("p1", bounds=[1, 32], transformation="log"))
    space.add(ps.ContinuousParameter("p2", [-2, 2]))
    space.add(ps.ContinuousParameter("p3", [0, 1]))
    space.add(ps.CategoricalParameter("p4", [True, False]))

    if issubclass(optimizer_class, MultiObjectiveOptimizer):
        return optimizer_class(space, objectives, seed=seed, **optimizer_kwargs)

    if issubclass(optimizer_class, SingleObjectiveOptimizer):
        return optimizer_class(space, objective, seed=seed, **optimizer_kwargs)

    return optimizer_class(space, seed=seed, **optimizer_kwargs)


def optimize_single_parameter_sequentially_for_n_max_evaluations(
    optimizer_class, optimizer_kwargs: dict, n_max_evaluations: int = 20
) -> bool:
    """[summary]

    Args:
        optimizer_class: [description]
        optimizer_kwargs: [description]
        n_max_evaluations: [description]

    Returns:
        [description]
    """

    def quadratic_function(p1):
        return p1 ** 2

    assert issubclass(optimizer_class, Optimizer), (
        "The default test suite is only applicable for implementations of "
        "blackboxopt.base.Optimizer"
    )

    optimizer = _initialize_optimizer(
        optimizer_class,
        optimizer_kwargs,
        objective=Objective("loss", False),
        objectives=[Objective("loss", False), Objective("score", True)],
    )

    eval_spec = optimizer.get_evaluation_specification()

    if issubclass(optimizer_class, MultiObjectiveOptimizer):
        optimizer.report_evaluation(
            eval_spec.get_evaluation(objectives={"loss": None, "score": None})
        )
    else:
        optimizer.report_evaluation(eval_spec.get_evaluation(objectives={"loss": None}))

    for _ in range(n_max_evaluations):

        try:
            eval_spec = optimizer.get_evaluation_specification()
        except OptimizationComplete:
            break

        loss = quadratic_function(p1=eval_spec.configuration["p1"])
        if issubclass(optimizer_class, MultiObjectiveOptimizer):
            evaluation_result = {"loss": loss, "score": -loss}
        else:
            evaluation_result = {"loss": loss}

        optimizer.report_evaluation(
            eval_spec.get_evaluation(objectives=evaluation_result)
        )

    return True


def is_deterministic_with_fixed_seed(optimizer_class, optimizer_kwargs: dict) -> bool:
    """Check if optimizer is deterministic.

    Repeatedly initialize the optimizer with the same parameter space and a fixed seed,
    get an evaluation specification, report a placeholder result and get another
    evaluation specification. The configuration of all final evaluation specifications
    should be equal.

    Args:
        optimizer_class: Optimizer to test.
        optimizer_kwargs: Expected to contain additional arguments for initializating
            the optimizer. (`search_space` and `objective(s)` are set automatically
            by the test.)

    Returns:
        `True` if the test is passed.
    """
    final_configurations = []

    for _ in range(2):
        opt = _initialize_optimizer(
            optimizer_class,
            optimizer_kwargs,
            objective=Objective("loss", False),
            objectives=[Objective("loss", False)],
        )

        es1 = opt.get_evaluation_specification()
        opt.report_evaluation(es1.get_evaluation(objectives={"loss": 0.42}))
        es2 = opt.get_evaluation_specification()

        final_configurations.append(es2.configuration.copy())

    assert final_configurations[0] == final_configurations[1]
    return True


def raises_objectives_error_when_reporting_unknown_objective(
    optimizer_class, optimizer_kwargs: dict
) -> bool:
    opt = _initialize_optimizer(
        optimizer_class,
        optimizer_kwargs,
        objective=Objective("loss", False),
        objectives=[Objective("loss", False)],
    )
    es = opt.get_evaluation_specification()

    try:
        opt.report_evaluation(es.get_evaluation(objectives={"unknown_objective": 0}))

        raise AssertionError(
            f"Optimizer {optimizer_class} did not raise an ObjectivesError when a "
            + "result including an unknown objective name was reported."
        )

    except ObjectivesError:
        pass

    return True


ALL_REFERENCE_TESTS = [
    optimize_single_parameter_sequentially_for_n_max_evaluations,
    is_deterministic_with_fixed_seed,
    raises_objectives_error_when_reporting_unknown_objective,
]
