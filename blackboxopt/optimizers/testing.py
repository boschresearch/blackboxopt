# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

"""Tests that can be imported and used to test optimizer implementations against this
packages blackbox optimizer interface."""

from typing import List

import parameterspace as ps

from blackboxopt import Objective, ObjectivesError, OptimizationComplete, Optimizer
from blackboxopt.base import (
    EvaluationsError,
    MultiObjectiveOptimizer,
    SingleObjectiveOptimizer,
)


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
        evaluation = eval_spec.create_evaluation(
            objectives={"loss": None, "score": None}
        )
    else:
        evaluation = eval_spec.create_evaluation(objectives={"loss": None})
    optimizer.report(evaluation)

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

        evaluation = eval_spec.create_evaluation(objectives=evaluation_result)
        optimizer.report(evaluation)

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
        evaluation1 = es1.create_evaluation(objectives={"loss": 0.42})
        opt.report(evaluation1)
        es2 = opt.get_evaluation_specification()

        final_configurations.append(es2.configuration.copy())

    assert final_configurations[0] == final_configurations[1]
    return True


def handles_reporting_evaluations_list(optimizer_class, optimizer_kwargs: dict) -> bool:
    """Check if optimizer's report method can process an iterable of evalutions.

    All optimizers should be able to allow reporting batches of evalutions. It's up to
    the optimizer's implementation, if evaluations in a batch are processed
    one by one like if they were reported individually, or if a batch is handled
    differently.

    Args:
        optimizer_class: Optimizer to test.
        optimizer_kwargs: Expected to contain additional arguments for initializating
            the optimizer. (`search_space` and `objective(s)` are set automatically
            by the test.)

    Returns:
        `True` if the test is passed.
    """
    opt = _initialize_optimizer(
        optimizer_class,
        optimizer_kwargs,
        objective=Objective("loss", False),
        objectives=[Objective("loss", False)],
    )
    evaluations = []
    for _ in range(3):
        es = opt.get_evaluation_specification()
        evaluation = es.create_evaluation(objectives={"loss": 0.42})
        evaluations.append(evaluation)

    opt.report(evaluations)
    return True


def raises_evaluation_error_when_reporting_unknown_objective(
    optimizer_class, optimizer_kwargs: dict
) -> bool:
    """Check if optimizer's report method raises exception in case objective is unknown.

    Also make sure that the faulty evaluations (and only those) are included in the
    exception.

    Args:
        optimizer_class: Optimizer to test.
        optimizer_kwargs: Expected to contain additional arguments for initializating
            the optimizer. (`search_space` and `objective(s)` are set automatically
            by the test.)

    Returns:
        `True` if the test is passed.
    """
    opt = _initialize_optimizer(
        optimizer_class,
        optimizer_kwargs,
        objective=Objective("loss", False),
        objectives=[Objective("loss", False)],
    )
    es_1 = opt.get_evaluation_specification()
    es_2 = opt.get_evaluation_specification()
    es_3 = opt.get_evaluation_specification()

    # NOTE: The following is not using pytest.raises because this would add pytest as
    #       a regular dependency to blackboxopt.
    try:
        evaluation_1 = es_1.create_evaluation(objectives={"loss": 1})
        evaluation_2 = es_2.create_evaluation(objectives={"unknown_objective": 2})
        evaluation_3 = es_3.create_evaluation(objectives={"loss": 4})
        opt.report([evaluation_1, evaluation_2, evaluation_3])

        raise AssertionError(
            f"Optimizer {optimizer_class} did not raise an ObjectivesError when a "
            + "result including an unknown objective name was reported."
        )

    except EvaluationsError as exception:
        invalid_evaluations = [e for e, _ in exception.evaluations_with_errors]
        assert len(invalid_evaluations) == 1
        assert evaluation_2 in invalid_evaluations

    return True


ALL_REFERENCE_TESTS = [
    optimize_single_parameter_sequentially_for_n_max_evaluations,
    is_deterministic_with_fixed_seed,
    handles_reporting_evaluations_list,
    raises_evaluation_error_when_reporting_unknown_objective,
]
