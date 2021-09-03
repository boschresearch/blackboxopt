# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Callable

import parameterspace as ps

from blackboxopt import Evaluation, EvaluationSpecification, Objective
from blackboxopt.optimizers.random_search import RandomSearch

SPACE = ps.ParameterSpace()
SPACE.add(ps.ContinuousParameter("p1", [-1, 1]))


def _evaluation_function(eval_spec: EvaluationSpecification) -> Evaluation:
    loss = eval_spec.configuration["p1"] ** 2
    return eval_spec.create_evaluation(objectives={"loss": loss})


def limit_with_max_evaluations(run_optimization_loop: Callable, loop_kwargs: dict):
    evaluations = run_optimization_loop(
        RandomSearch(SPACE, [Objective("loss", False)], max_steps=10),
        _evaluation_function,
        max_evaluations=8,
        **loop_kwargs,
    )

    assert len(evaluations) == 8
    assert all([not e.all_objectives_none for e in evaluations])


def limit_with_loop_timeout(run_optimization_loop: Callable, loop_kwargs: dict):
    class SlowRandomSearch(RandomSearch):
        def get_evaluation_specification(self) -> EvaluationSpecification:
            time.sleep(1)
            return super().get_evaluation_specification()

    evaluations = run_optimization_loop(
        SlowRandomSearch(SPACE, [Objective("loss", False)], max_steps=10),
        _evaluation_function,
        timeout_s=3.0,
        **loop_kwargs,
    )
    assert len(evaluations) < 4
    assert all([not e.all_objectives_none for e in evaluations])


def failing_evaluations(run_optimization_loop: Callable, loop_kwargs: dict):
    def __evaluation_function(_):
        raise RuntimeError("Test Exception")

    max_steps = 10
    evaluations = run_optimization_loop(
        RandomSearch(SPACE, [Objective("loss", False)], max_steps=max_steps),
        __evaluation_function,
        **loop_kwargs,
    )

    assert len(evaluations) == max_steps
    assert all([e.all_objectives_none for e in evaluations])
    assert all([e.stacktrace is not None for e in evaluations])


def reporting_user_info(run_optimization_loop: Callable, loop_kwargs: dict):
    def __evaluation_function(eval_spec):
        return eval_spec.create_evaluation(
            objectives={"loss": 1.0}, user_info={"user": "info"}
        )

    max_steps = 10
    evaluations = run_optimization_loop(
        RandomSearch(SPACE, [Objective("loss", False)], max_steps=max_steps),
        __evaluation_function,
        **loop_kwargs,
    )

    assert len(evaluations) == max_steps
    assert all([not e.all_objectives_none for e in evaluations])
    assert all([e.user_info is not None for e in evaluations])


ALL_REFERENCE_TESTS = [
    limit_with_max_evaluations,
    limit_with_loop_timeout,
    failing_evaluations,
    reporting_user_info,
]
