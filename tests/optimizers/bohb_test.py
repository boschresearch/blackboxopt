# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import math
import re

import numpy as np
import parameterspace as ps
import pytest

from blackboxopt import OptimizationComplete, OptimizerNotReady
from blackboxopt.base import EvaluationsError, Objective
from blackboxopt.evaluation import Evaluation
from blackboxopt.optimizers.bohb import BOHB
from blackboxopt.optimizers.testing import ALL_REFERENCE_TESTS


@pytest.mark.parametrize("reference_test", ALL_REFERENCE_TESTS)
def test_all_reference_tests(reference_test):
    reference_test(BOHB, dict(min_fidelity=0.2, max_fidelity=1, num_iterations=1))


def test_bohb_sequential():
    paramspace = ps.ParameterSpace()
    paramspace.add(ps.ContinuousParameter("p1", [0, 1]))
    opt = BOHB(
        paramspace,
        Objective("loss", False),
        min_fidelity=0.2,
        max_fidelity=1,
        num_iterations=1,
    )

    for i in range(3):
        es = opt.get_evaluation_specification()
        assert es.optimizer_info["configuration_key"] == (0, 0, i)
        evaluation = es.create_evaluation(objectives={"loss": i})
        opt.report(evaluation)
    es = opt.get_evaluation_specification()
    assert es.optimizer_info["configuration_key"] == (0, 0, 0)

    with pytest.raises(OptimizerNotReady):
        opt.get_evaluation_specification()

    evaluation = es.create_evaluation(objectives={"loss": 0.0})
    opt.report(evaluation)

    with pytest.raises(OptimizationComplete):
        opt.get_evaluation_specification()


def test_bohb_parallel():
    paramspace = ps.ParameterSpace()
    paramspace.add(ps.ContinuousParameter("p1", [0, 1]))
    opt = BOHB(
        paramspace,
        Objective("loss", False),
        min_fidelity=0.2,
        max_fidelity=1,
        num_iterations=2,
    )

    eval_specs = []

    for i in range(3):
        # note, this test doesn't return results immediately, but has 3 concurrently
        # 'pending` evaluations.
        es = opt.get_evaluation_specification()
        assert es.optimizer_info["configuration_key"] == (0, 0, i)
        eval_specs.append(es)

    assert len(opt.pending_configurations) == 3

    for i, eval_spec in enumerate(eval_specs):
        evaluation = eval_spec.create_evaluation(objectives={"loss": i})
        opt.report(evaluation)

    assert len(opt.pending_configurations) == 0

    es = opt.get_evaluation_specification()
    assert es.optimizer_info["configuration_key"] == (0, 0, 0)

    for i in range(2):
        es = opt.get_evaluation_specification()
        assert es.optimizer_info["configuration_key"] == (1, 0, i)


def test_bohb_report_as_batch():
    paramspace = ps.ParameterSpace()
    paramspace.add(ps.ContinuousParameter("p1", [0, 1]))
    opt = BOHB(
        paramspace,
        Objective("loss", False),
        min_fidelity=0.2,
        max_fidelity=1,
        num_iterations=1,
    )

    evaluations = []
    for i in range(3):
        es = opt.get_evaluation_specification()
        evaluation = es.create_evaluation(objectives={"loss": i})
        evaluations.append(evaluation)

    # Add an evaluation _not_ created by the optimizer to see if it get's skipped
    # and triggers a warning on reporting:
    invalid_evaluation = Evaluation(objectives={"loss": 42}, configuration={})
    evaluations.append(invalid_evaluation)

    assert len(opt.pending_configurations) == 3
    with pytest.raises(EvaluationsError) as excinfo:
        opt.report(evaluations)
    assert len(opt.pending_configurations) == 0
    assert excinfo.value.message.startswith("An error with one or more evaluation")
    assert excinfo.value.evaluations_with_errors[0][0] == invalid_evaluation


def test_bohb_number_of_configs_and_fidelities_in_iterations():
    paramspace = ps.ParameterSpace()
    paramspace.add(ps.ContinuousParameter("p1", [0, 1]))
    opt = BOHB(
        paramspace,
        Objective("loss", False),
        min_fidelity=1.0,
        max_fidelity=81.0,
        num_iterations=4,
    )
    # first iteration has powers of 3 as num_configs and fidelities
    expected_num_configs = [[81, 27, 9, 3, 1], [34, 11, 3, 1], [15, 5, 1], [8, 2], [5]]
    expected_fidelities = [
        [1, 3, 9, 27, 81],
        [3, 9, 27, 81],
        [9, 27, 81],
        [27, 81],
        [81],
    ]

    for i in range(10):
        it = opt._create_new_iteration(i)
        assert (
            np.array(it.num_configs, dtype=np.int64)
            == np.array(expected_num_configs[i % 5], dtype=np.int64)
        ).all()
        assert (
            np.array(it.fidelities, dtype=np.int64)
            == np.array(expected_fidelities[i % 5], dtype=np.int64)
        ).all()

    # bohb is designed such that each iteration does not take longer than the first
    # one (assuming that the fidelity scales linearly w.r.t. the compute time).
    max_total_budget = 5 * 81.0
    for i in range(5):
        it = opt._create_new_iteration(i)
        total_budget = np.sum(np.array(it.num_configs) * np.array(it.fidelities))
        assert total_budget <= max_total_budget


def test_bohb_sequential_with_failed_evaluations(n_evaluations=16):
    paramspace = ps.ParameterSpace()
    paramspace.add(ps.ContinuousParameter("p1", [0, 1]))
    opt = BOHB(
        paramspace,
        Objective("loss", False),
        min_fidelity=1,
        max_fidelity=1,
        num_iterations=n_evaluations,
        random_fraction=0.0,
    )

    for i in range(n_evaluations // 2):
        es = opt.get_evaluation_specification()
        assert es.optimizer_info["configuration_key"] == (i, 0, 0)
        if np.random.rand() < 0.8:
            evaluation = es.create_evaluation(
                objectives={"loss": es.configuration["p1"]}
            )
        else:
            evaluation = es.create_evaluation(objectives={"loss": None})
        opt.report(evaluation)

    for i in range(n_evaluations // 2, n_evaluations):
        es = opt.get_evaluation_specification()
        assert es.optimizer_info["configuration_key"] == (i, 0, 0)
        assert es.optimizer_info["model_based_pick"]
        evaluation = es.create_evaluation(objectives={"loss": es.configuration["p1"]})
        opt.report(evaluation)

    with pytest.raises(OptimizationComplete):
        opt.get_evaluation_specification()


def test_bohb_sequential_with_non_finite_losses(n_evaluations=16):
    paramspace = ps.ParameterSpace()
    paramspace.add(ps.ContinuousParameter("p1", [0, 1]))
    opt = BOHB(
        paramspace,
        Objective("loss", False),
        min_fidelity=1,
        max_fidelity=1,
        num_iterations=n_evaluations,
        random_fraction=0.0,
    )
    return_values = [
        "Unused placeholder",
        np.inf,
        np.nan,
        float("inf"),
        float("nan"),
        math.inf,
        math.nan,
        None,
    ]

    for i in range(n_evaluations // 2):
        es = opt.get_evaluation_specification()
        assert es.optimizer_info["configuration_key"] == (i, 0, 0)

        if i // len(return_values) == 0:
            evaluation = es.create_evaluation(
                objectives={"loss": es.configuration["p1"]}
            )
        else:
            evaluation = es.create_evaluation(
                result={"loss": return_values[i // len(return_values)]}
            )
        opt.report(evaluation)

    for i in range(n_evaluations // 2, n_evaluations):
        es = opt.get_evaluation_specification()
        assert es.optimizer_info["configuration_key"] == (i, 0, 0)
        assert es.optimizer_info["model_based_pick"]
        evaluation = es.create_evaluation(objectives={"loss": es.configuration["p1"]})
        opt.report(evaluation)

    with pytest.raises(OptimizationComplete):
        opt.get_evaluation_specification()


def test_bohb_with_none_min_samples_in_model():
    paramspace = ps.ParameterSpace()
    paramspace.add(ps.ContinuousParameter("p1", [0, 1]))
    opt = BOHB(
        paramspace,
        Objective("loss", False),
        min_fidelity=0.2,
        max_fidelity=1,
        num_iterations=1,
        min_samples_in_model=None,
    )
    assert opt.config_sampler.min_samples_in_model == 3
