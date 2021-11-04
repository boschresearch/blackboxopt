# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import parameterspace as ps
import pytest

from blackboxopt import OptimizationComplete, OptimizerNotReady
from blackboxopt.base import Objective
from blackboxopt.optimizers.hyperband import Hyperband
from blackboxopt.optimizers.testing import ALL_REFERENCE_TESTS


@pytest.mark.parametrize("reference_test", ALL_REFERENCE_TESTS)
def test_all_reference_tests(reference_test):
    reference_test(
        Hyperband,
        dict(min_fidelity=0.2, max_fidelity=1, num_iterations=1),
    )


def test_hyperband_sequential():
    paramspace = ps.ParameterSpace()
    paramspace.add(ps.ContinuousParameter("p1", [0, 1]))
    hb = Hyperband(
        paramspace,
        Objective("loss", False),
        min_fidelity=0.2,
        max_fidelity=1,
        num_iterations=1,
    )

    for i in range(3):
        es = hb.generate_evaluation_specification()
        assert es.optimizer_info["configuration_key"] == (0, 0, i)
        evaluation = es.create_evaluation(objectives={"loss": i})
        hb.report(evaluation)
    es = hb.generate_evaluation_specification()
    assert es.optimizer_info["configuration_key"] == (0, 0, 0)

    with pytest.raises(OptimizerNotReady):
        hb.generate_evaluation_specification()

    evaluation = es.create_evaluation(objectives={"loss": i})
    hb.report(evaluation)

    with pytest.raises(OptimizationComplete):
        hb.generate_evaluation_specification()


def test_hyperband_parallel():
    paramspace = ps.ParameterSpace()
    paramspace.add(ps.ContinuousParameter("p1", [0, 1]))
    hb = Hyperband(
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
        es = hb.generate_evaluation_specification()
        assert es.optimizer_info["configuration_key"] == (0, 0, i)
        eval_specs.append(es)

    assert len(hb.pending_configurations) == 3

    for i, es in enumerate(eval_specs):
        evaluation = es.create_evaluation(objectives={"loss": i})
        hb.report(evaluation)

    assert len(hb.pending_configurations) == 0

    es = hb.generate_evaluation_specification()
    assert es.optimizer_info["configuration_key"] == (0, 0, 0)

    for i in range(2):
        es = hb.generate_evaluation_specification()
        assert es.optimizer_info["configuration_key"] == (1, 0, i)


def test_hyperband_number_of_configs_and_fidelities_in_iterations():
    paramspace = ps.ParameterSpace()
    paramspace.add(ps.ContinuousParameter("p1", [0, 1]))
    hb = Hyperband(
        paramspace,
        Objective("loss", False),
        min_fidelity=1,
        max_fidelity=81,
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
        it = hb._create_new_iteration(i)
        assert (
            np.array(it.num_configs, dtype=np.int64)
            == np.array(expected_num_configs[i % 5], dtype=np.int64)
        ).all()
        assert (
            np.array(it.fidelities, dtype=np.int64)
            == np.array(expected_fidelities[i % 5], dtype=np.int64)
        ).all()

    # hyperband is designed such that each iteration does not take longer than the first
    # one (assuming that the fidelity scales linearly w.r.t. the compute time).
    max_total_budget = 5 * 81.0
    for i in range(5):
        it = hb._create_new_iteration(i)
        total_budget = np.sum(np.array(it.num_configs) * np.array(it.fidelities))
        assert total_budget <= max_total_budget
