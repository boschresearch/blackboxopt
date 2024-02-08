# Copyright (c) 2024 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from functools import partial
from threading import Thread

import pytest

from blackboxopt import Evaluation, EvaluationSpecification
from blackboxopt.optimization_loops import testing
from blackboxopt.optimization_loops.file_based_distributed import (
    evaluate_specifications,
    run_optimization_loop,
)
from blackboxopt.optimizers.space_filling import Objective, SpaceFilling


def test_successful_loop(tmpdir):
    opt = SpaceFilling(
        testing.SPACE, objectives=[Objective("loss", greater_is_better=False)]
    )

    def eval_func(eval_spec: EvaluationSpecification) -> Evaluation:
        return eval_spec.create_evaluation({"loss": eval_spec.configuration["p1"] ** 2})

    max_evaluations = 3

    thread = Thread(
        target=evaluate_specifications,
        kwargs=dict(
            target_directory=tmpdir,
            evaluation_function=eval_func,
            objectives=opt.objectives,
            max_evaluations=max_evaluations,
        ),
    )
    thread.start()

    evaluations = run_optimization_loop(
        optimizer=opt, target_directory=tmpdir, max_evaluations=max_evaluations
    )

    assert len(evaluations) == max_evaluations
    assert (
        len(set([str(e.configuration) for e in evaluations])) == max_evaluations
    ), "Evaluated configurations are not unique"
    thread.join()


def test_failed_evaluations(tmpdir):
    opt = SpaceFilling(
        testing.SPACE, objectives=[Objective("loss", greater_is_better=False)]
    )

    def eval_func(eval_spec: EvaluationSpecification) -> Evaluation:
        raise ValueError("This is a test error to make the evaluation fail.")

    max_evaluations = 3

    thread = Thread(
        target=evaluate_specifications,
        kwargs=dict(
            target_directory=tmpdir,
            evaluation_function=eval_func,
            objectives=opt.objectives,
            max_evaluations=max_evaluations,
            catch_exceptions_from_evaluation_function=True,
        ),
    )
    thread.start()

    evaluations = run_optimization_loop(
        optimizer=opt, target_directory=tmpdir, max_evaluations=max_evaluations
    )

    assert len(evaluations) == max_evaluations
    assert evaluations[0].objectives[opt.objectives[0].name] is None
    assert evaluations[1].objectives[opt.objectives[0].name] is None
    assert evaluations[2].objectives[opt.objectives[0].name] is None
    thread.join()


def test_callbacks(tmpdir):
    from_callback = defaultdict(list)

    def callback(e: Evaluation, callback_name: str):
        from_callback[callback_name].append(e)

    def eval_func(eval_spec: EvaluationSpecification) -> Evaluation:
        return eval_spec.create_evaluation({"loss": eval_spec.configuration["p1"] ** 2})

    max_evaluations = 3
    opt = SpaceFilling(
        testing.SPACE, objectives=[Objective("loss", greater_is_better=False)]
    )
    thread = Thread(
        target=evaluate_specifications,
        kwargs=dict(
            target_directory=tmpdir,
            evaluation_function=eval_func,
            objectives=opt.objectives,
            max_evaluations=max_evaluations,
            pre_evaluation_callback=partial(callback, callback_name="evaluate_pre"),
            post_evaluation_callback=partial(callback, callback_name="evaluate_post"),
        ),
    )
    thread.start()

    evaluations = run_optimization_loop(
        optimizer=opt,
        target_directory=tmpdir,
        max_evaluations=max_evaluations,
        pre_evaluation_callback=partial(callback, callback_name="run_loop_pre"),
        post_evaluation_callback=partial(callback, callback_name="run_loop_post"),
    )

    # NOTE: These are set comparisons instead of list comparisons because the order
    #       of the evaluations is not guaranteed.
    assert len(evaluations) == len(from_callback["evaluate_post"])
    assert set([e.to_json() for e in evaluations]) == set(
        [e.to_json() for e in from_callback["evaluate_post"]]
    )
    assert len(evaluations) == len(from_callback["run_loop_post"])
    assert set([e.to_json() for e in evaluations]) == set(
        [e.to_json() for e in from_callback["run_loop_post"]]
    )
    assert len(evaluations) == len(from_callback["evaluate_pre"])
    assert set([e.get_specification().to_json() for e in evaluations]) == set(
        [es.to_json() for es in from_callback["evaluate_pre"]]
    )
    assert len(evaluations) == len(from_callback["run_loop_pre"])
    assert set([e.get_specification().to_json() for e in evaluations]) == set(
        [es.to_json() for es in from_callback["run_loop_pre"]]
    )
    thread.join()
