# Copyright (c) 2024 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

from threading import Thread

import parameterspace as ps

from blackboxopt import Evaluation, EvaluationSpecification
from blackboxopt.optimization_loops.file_based_distributed import (
    evaluate_specifications,
    run_optimization_loop,
)
from blackboxopt.optimizers.space_filling import Objective, SpaceFilling


def test_successful_loop(tmpdir):
    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("p", (-10, 10)))
    opt = SpaceFilling(space, objectives=[Objective("loss", greater_is_better=False)])

    def eval_func(spec: EvaluationSpecification) -> Evaluation:
        return spec.create_evaluation({"loss": spec.configuration["p"] ** 2})

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
    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("p", (-10, 10)))
    opt = SpaceFilling(space, objectives=[Objective("loss", greater_is_better=False)])

    def eval_func(spec: EvaluationSpecification) -> Evaluation:
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
