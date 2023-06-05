# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from blackboxopt import Objective
from blackboxopt.evaluation import Evaluation
from blackboxopt.optimization_loops import testing
from blackboxopt.optimization_loops.sequential import run_optimization_loop
from blackboxopt.optimization_loops.utils import EvaluationFunctionError
from blackboxopt.optimizers.random_search import RandomSearch


@pytest.mark.parametrize("reference_test", testing.ALL_REFERENCE_TESTS)
def test_all_reference_tests(reference_test):
    reference_test(run_optimization_loop, {})


def test_failed_evaluations_handled():
    def __evaluation_function(_):
        raise RuntimeError("Test Exception")

    max_steps = 10
    evals = run_optimization_loop(
        RandomSearch(testing.SPACE, [Objective("loss", False)], max_steps=max_steps),
        __evaluation_function,
        catch_exceptions_from_evaluation_function=True,
    )

    assert len(evals) == max_steps
    assert all([e.stacktrace is not None and e.all_objectives_none for e in evals])


def test_failed_evaluation_interrupts_loop_by_default():
    def __evaluation_function(_):
        raise RuntimeError("Test Exception")

    with pytest.raises(EvaluationFunctionError):
        run_optimization_loop(
            RandomSearch(testing.SPACE, [Objective("loss", False)], max_steps=10),
            __evaluation_function,
        )


def test_post_evaluation_callback():
    evaluations_from_callback = []

    def callback(e: Evaluation):
        evaluations_from_callback.append(e)

    evaluations = run_optimization_loop(
        RandomSearch(testing.SPACE, [Objective("loss", False)], max_steps=10),
        lambda e: e.create_evaluation(objectives={"loss": 0.0}),
        post_evaluation_callback=callback,
    )

    assert len(evaluations) == len(evaluations_from_callback)
    assert evaluations == evaluations_from_callback


def test_pre_evaluation_callback():
    eval_specs_from_callback = []

    def callback(e: Evaluation):
        eval_specs_from_callback.append(e)

    evaluations = run_optimization_loop(
        RandomSearch(testing.SPACE, [Objective("loss", False)], max_steps=10),
        lambda e: e.create_evaluation(objectives={"loss": 0.0}),
        pre_evaluation_callback=callback,
    )

    assert len(evaluations) == len(eval_specs_from_callback)
    assert [e.get_specification() for e in evaluations] == eval_specs_from_callback
