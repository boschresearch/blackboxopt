# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

"""Tests that can be imported and used to test optimizer implementations against this
packages blackbox optimizer interface."""

import random
from typing import Dict, List, Optional, Type, Union

import numpy as np
import parameterspace as ps

from blackboxopt import Objective, OptimizationComplete, Optimizer
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
    space: Optional[ps.ParameterSpace] = None,
    seed=42,
) -> Optimizer:
    if space is None:
        space = ps.ParameterSpace()
        space.add(ps.IntegerParameter("p1", bounds=[1, 32], transformation="log"))
        space.add(ps.ContinuousParameter("p2", [-2, 2]))
        space.add(ps.ContinuousParameter("p3", [0, 1]))
        space.add(ps.CategoricalParameter("p4", [True, False]))
        space.add(ps.OrdinalParameter("p5", ("small", "medium", "large")))

    if issubclass(optimizer_class, MultiObjectiveOptimizer):
        return optimizer_class(space, objectives, seed=seed, **optimizer_kwargs)

    if issubclass(optimizer_class, SingleObjectiveOptimizer):
        return optimizer_class(space, objective, seed=seed, **optimizer_kwargs)

    return optimizer_class(space, seed=seed, **optimizer_kwargs)


def optimize_single_parameter_sequentially_for_n_max_evaluations(
    optimizer_class: Union[
        Type[SingleObjectiveOptimizer], Type[MultiObjectiveOptimizer]
    ],
    optimizer_kwargs: dict,
    n_max_evaluations: int = 20,
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
        return p1**2

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

    eval_spec = optimizer.generate_evaluation_specification()

    if issubclass(optimizer_class, MultiObjectiveOptimizer):
        evaluation = eval_spec.create_evaluation(
            objectives={"loss": None, "score": None}
        )
    else:
        evaluation = eval_spec.create_evaluation(objectives={"loss": None})
    optimizer.report(evaluation)

    for _ in range(n_max_evaluations):

        try:
            eval_spec = optimizer.generate_evaluation_specification()
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


def is_deterministic_with_fixed_seed_and_larger_space(
    optimizer_class: Union[
        Type[SingleObjectiveOptimizer], Type[MultiObjectiveOptimizer]
    ],
    optimizer_kwargs: dict,
) -> bool:
    """Check if optimizer is deterministic.

    Repeatedly initialize the optimizer with the same parameter space and a fixed seed,
    get an evaluation specification, report a placeholder result and get another
    evaluation specification. The configuration of all final evaluation specifications
    should be equal.


    This tests covers multiple parameter types by using a mixed search space.

    Args:
        optimizer_class: Optimizer to test.
        optimizer_kwargs: Expected to contain additional arguments for initializing
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

        es1 = opt.generate_evaluation_specification()
        evaluation1 = es1.create_evaluation(objectives={"loss": 0.42})
        opt.report(evaluation1)
        es2 = opt.generate_evaluation_specification()

        final_configurations.append(es2.configuration.copy())

    assert final_configurations[0] == final_configurations[1]
    return True


def is_deterministic_when_reporting_shuffled_evaluations(
    optimizer_class: Union[
        Type[SingleObjectiveOptimizer], Type[MultiObjectiveOptimizer]
    ],
    optimizer_kwargs: dict,
) -> bool:
    """Check if determinism isn't affected by the order of initially reported data.

    Repeatedly initialize the optimizer with the same parameter space and a fixed seed.
    Report a set of initial evaluations in randomized order as initial data. Start
    optimizing and check if the generated configurations for all optimizers are equal.

    By doing multiple evaluations, this tests covers effects that become visible after
    a while, e.g. only after stages got completed in staged iteration samplers.

    Args:
        optimizer_class: Optimizer to test.
        optimizer_kwargs: Expected to contain additional arguments for initializing
            the optimizer. (`search_space` and `objective(s)` are set automatically
            by the test.)

    Returns:
        `True` if the test is passed.
    """

    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("p1", [0, 1]))

    def _run_experiment_1d(es):
        x = es.configuration["p1"]
        _x = np.copy(np.atleast_2d(x))
        params = np.array([0.75, 0.0, -10.0, 0.0, 0.0])
        y = np.polyval(params, _x)
        return float(np.squeeze(y))

    optimization_runs: Dict[int, list] = {0: [], 1: []}
    for run_idx in optimization_runs:
        opt = _initialize_optimizer(
            optimizer_class,
            optimizer_kwargs,
            objective=Objective("loss", False),
            objectives=[Objective("loss", False)],
            space=space,
        )

        # Report initial data in different order
        eval_specs = [opt.generate_evaluation_specification() for _ in range(8)]
        evaluations = [
            es.create_evaluation(objectives={"loss": _run_experiment_1d(es)})
            for es in eval_specs
        ]
        random.seed(run_idx)
        random.shuffle(evaluations)
        opt.report(evaluations)

        # Start optimizing
        for _ in range(5):
            es = opt.generate_evaluation_specification()
            opt.report(
                es.create_evaluation(objectives={"loss": _run_experiment_1d(es)})
            )
            optimization_runs[run_idx].append(es.configuration.copy())

    assert optimization_runs[0] == optimization_runs[1]
    return True


def handles_reporting_evaluations_list(
    optimizer_class: Union[
        Type[SingleObjectiveOptimizer], Type[MultiObjectiveOptimizer]
    ],
    optimizer_kwargs: dict,
) -> bool:
    """Check if optimizer's report method can process an iterable of evaluations.

    All optimizers should be able to allow reporting batches of evaluations. It's up to
    the optimizer's implementation, if evaluations in a batch are processed
    one by one like if they were reported individually, or if a batch is handled
    differently.

    Args:
        optimizer_class: Optimizer to test.
        optimizer_kwargs: Expected to contain additional arguments for initializing
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
        es = opt.generate_evaluation_specification()
        evaluation = es.create_evaluation(objectives={"loss": 0.42})
        evaluations.append(evaluation)

    opt.report(evaluations)
    return True


def raises_evaluation_error_when_reporting_unknown_objective(
    optimizer_class: Union[
        Type[SingleObjectiveOptimizer], Type[MultiObjectiveOptimizer]
    ],
    optimizer_kwargs: dict,
) -> bool:
    """Check if optimizer's report method raises exception in case objective is unknown.

    Also make sure that the faulty evaluations (and only those) are included in the
    exception.

    Args:
        optimizer_class: Optimizer to test.
        optimizer_kwargs: Expected to contain additional arguments for initializing
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
    es_1 = opt.generate_evaluation_specification()
    es_2 = opt.generate_evaluation_specification()
    es_3 = opt.generate_evaluation_specification()

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


def respects_fixed_parameter(
    optimizer_class: Union[
        Type[SingleObjectiveOptimizer], Type[MultiObjectiveOptimizer]
    ],
    optimizer_kwargs: dict,
) -> bool:
    """Check if optimizer's generated evaluation specifications contain the values
    a parameter in the search space was fixed to.

    Args:
        optimizer_class: Optimizer to test.
        optimizer_kwargs: Expected to contain additional arguments for initializing
            the optimizer. (`search_space` and `objective(s)` are set automatically
            by the test.)

    Returns:
        `True` if the test is passed.
    """
    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("my_fixed_param", (-10.0, 200.0)))
    space.add(ps.ContinuousParameter("x", (-2.0, 2.0)))

    fixed_value = 1.0
    space.fix(my_fixed_param=fixed_value)
    opt = _initialize_optimizer(
        optimizer_class,
        optimizer_kwargs,
        objective=Objective("loss", False),
        objectives=[Objective("loss", False)],
        space=space,
    )
    for _ in range(5):
        es = opt.generate_evaluation_specification()
        assert es.configuration["my_fixed_param"] == fixed_value
        opt.report(
            es.create_evaluation(objectives={"loss": es.configuration["x"] ** 2})
        )

    return True


def handles_conditional_space(
    optimizer_class: Union[
        Type[SingleObjectiveOptimizer], Type[MultiObjectiveOptimizer]
    ],
    optimizer_kwargs: dict,
) -> bool:
    """Check if optimizer handles conditional i.e. hierarchical search spaces.

    Args:
        optimizer_class: Optimizer to test.
        optimizer_kwargs: Expected to contain additional arguments for initializing
            the optimizer. (`search_space` and `objective(s)` are set automatically
            by the test.)

    Returns:
        `True` if the test is passed.
    """
    space = ps.ParameterSpace()
    space.add(ps.CategoricalParameter("optimizer", ("adam", "sgd")))
    space.add(ps.ContinuousParameter("lr", (0.0001, 0.1), transformation="log"))
    space.add(
        ps.ContinuousParameter("momentum", (0.0, 1.0)),
        lambda optimizer: optimizer == "sgd",
    )

    opt = _initialize_optimizer(
        optimizer_class,
        optimizer_kwargs,
        objective=Objective("loss", False),
        objectives=[Objective("loss", False)],
        space=space,
    )

    for _ in range(10):
        es = opt.generate_evaluation_specification()
        dummy_loss = es.configuration.get("momentum", 1.0) * es.configuration["lr"] ** 2
        opt.report(es.create_evaluation({"loss": dummy_loss}))

    return True


ALL_REFERENCE_TESTS = [
    optimize_single_parameter_sequentially_for_n_max_evaluations,
    is_deterministic_with_fixed_seed_and_larger_space,
    is_deterministic_when_reporting_shuffled_evaluations,
    handles_reporting_evaluations_list,
    raises_evaluation_error_when_reporting_unknown_objective,
    respects_fixed_parameter,
    handles_conditional_space,
]
