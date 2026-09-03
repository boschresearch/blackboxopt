# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import parameterspace as ps
import pytest

from blackboxopt import OptimizationComplete, OptimizerNotReady
from blackboxopt.base import Objective
from blackboxopt.optimizers.mobohb import MOBOHB
from blackboxopt.optimizers.staged.mobohb import Sampler
from blackboxopt.optimizers.testing import ALL_REFERENCE_TESTS
from blackboxopt.utils import filter_pareto_efficient

OBJECTIVES = [Objective("loss_a", False), Objective("loss_b", False)]


@pytest.mark.parametrize("reference_test", ALL_REFERENCE_TESTS)
def test_all_reference_tests(reference_test, seed):
    reference_test(
        MOBOHB, dict(min_fidelity=0.2, max_fidelity=1, num_iterations=5), seed=seed
    )


def _run_until_complete(opt, evaluate, max_evaluations=500):
    evaluations = []
    for _ in range(max_evaluations):
        try:
            es = opt.generate_evaluation_specification()
        except OptimizationComplete:
            break
        except OptimizerNotReady:
            raise AssertionError("Sequential loop should never be not ready")
        evaluation = es.create_evaluation(objectives=evaluate(es.configuration))
        opt.report(evaluation)
        evaluations.append(evaluation)
    return evaluations


def test_sequential_multi_objective():
    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("p1", (0.0, 1.0)))
    opt = MOBOHB(space, OBJECTIVES, min_fidelity=0.2, max_fidelity=1, num_iterations=1)

    for i in range(3):
        es = opt.generate_evaluation_specification()
        assert es.optimizer_info["configuration_key"] == (0, 0, i)
        evaluation = es.create_evaluation(
            objectives={"loss_a": float(i), "loss_b": float(-i)}
        )
        opt.report(evaluation)

    es = opt.generate_evaluation_specification()
    assert es.optimizer_info["configuration_key"] == (0, 0, 0)
    with pytest.raises(OptimizerNotReady):
        opt.generate_evaluation_specification()

    opt.report(es.create_evaluation(objectives={"loss_a": 0.0, "loss_b": 0.0}))
    with pytest.raises(OptimizationComplete):
        opt.generate_evaluation_specification()


def test_parallel_multi_objective():
    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("p1", (0.0, 1.0)))
    opt = MOBOHB(space, OBJECTIVES, min_fidelity=0.2, max_fidelity=1, num_iterations=2)

    eval_specs = [opt.generate_evaluation_specification() for _ in range(3)]
    assert len(opt.pending_configurations) == 3

    for i, es in enumerate(eval_specs):
        opt.report(
            es.create_evaluation(objectives={"loss_a": float(i), "loss_b": float(-i)})
        )
    assert len(opt.pending_configurations) == 0


def test_with_none_min_samples_in_model():
    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("p1", (0.0, 1.0)))
    opt = MOBOHB(
        space,
        OBJECTIVES,
        min_fidelity=0.2,
        max_fidelity=1,
        num_iterations=1,
        min_samples_in_model=None,
    )
    assert opt.config_sampler.min_samples_in_model == 3


def test_good_bad_split_excludes_non_finite_rows_from_good():
    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("p1", (0.0, 1.0)))
    sampler = Sampler(
        search_space=space,
        objectives=OBJECTIVES,
        min_samples_in_model=2,
        top_n_percent=50,
        num_samples=8,
        random_fraction=0.0,
        bandwidth_factor=3.0,
        min_bandwidth=1e-3,
    )

    # Row 1 is a partial failure that is nondominated w.r.t. the finite rows and
    # would otherwise be ranked into the good set by the nondominated sort.
    losses = np.array(
        [
            [1.0, 1.0],
            [0.0, np.inf],
            [2.0, 0.5],
            [np.inf, np.inf],
        ]
    )

    idx_good, idx_bad = sampler._good_bad_split(losses, n_good=2, n_bad=2)

    assert np.all(np.isfinite(losses[idx_good])), "non-finite rows must stay out of good"
    # Every invalid row must be routed to the bad candidates instead.
    assert 1 in idx_bad and 3 in idx_bad


def test_finds_pareto_front():
    # Two conflicting objectives; the Pareto-optimal set is p1 in [0, 1].
    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("p1", (-2.0, 2.0)))
    opt = MOBOHB(
        space,
        OBJECTIVES,
        min_fidelity=1.0,
        max_fidelity=9.0,
        num_iterations=4,
        seed=7,
    )

    def evaluate(config):
        return {"loss_a": config["p1"] ** 2, "loss_b": (config["p1"] - 1) ** 2}

    evaluations = _run_until_complete(opt, evaluate)
    max_fidelity_evals = [e for e in evaluations if e.settings["fidelity"] == 9.0]

    pareto = filter_pareto_efficient(max_fidelity_evals, OBJECTIVES)
    # A proper trade-off front has more than a single point
    assert len(pareto) > 1
    # and it should approach the optimum of each individual objective.
    assert min(e.objectives["loss_a"] for e in max_fidelity_evals) < 0.1
    assert min(e.objectives["loss_b"] for e in max_fidelity_evals) < 0.1
