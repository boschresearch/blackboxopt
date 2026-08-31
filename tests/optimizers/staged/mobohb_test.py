# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import parameterspace as ps
import pytest

from blackboxopt import Evaluation, Objective
from blackboxopt.optimizers.mobohb import _nondominated_promotion
from blackboxopt.optimizers.staged.iteration import Datum
from blackboxopt.optimizers.staged.kde_sampler import KDESampler
from blackboxopt.optimizers.staged.mobohb import Sampler, argsort_nondominated


def _sampler():
    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("p1", (0.0, 1.0)))
    space.add(ps.ContinuousParameter("p2", (0.0, 1.0)))
    return Sampler(
        search_space=space,
        objectives=[Objective("loss_a", False), Objective("loss_b", False)],
        min_samples_in_model=3,
        top_n_percent=15,
        num_samples=8,
        random_fraction=1 / 3,
        bandwidth_factor=3.0,
        min_bandwidth=1e-3,
        seed=42,
    )


def test_evaluation_to_loss_returns_sign_corrected_vector():
    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("p1", (0.0, 1.0)))
    sampler = Sampler(
        search_space=space,
        objectives=[Objective("loss", False), Objective("score", True)],
        min_samples_in_model=2,
        top_n_percent=15,
        num_samples=8,
        random_fraction=1 / 3,
        bandwidth_factor=3.0,
        min_bandwidth=1e-3,
        seed=1,
    )

    evaluation = Evaluation(
        configuration={"p1": 0.5}, objectives={"loss": 2.0, "score": 3.0}
    )
    np.testing.assert_array_equal(
        sampler._evaluation_to_loss(evaluation), np.array([2.0, -3.0])
    )

    crashed = Evaluation(
        configuration={"p1": 0.5}, objectives={"loss": None, "score": None}
    )
    losses = sampler._evaluation_to_loss(crashed)
    assert np.all(np.isinf(losses))


def test_count_finite_losses_requires_all_objectives_finite():
    sampler = _sampler()
    losses = [
        np.array([0.0, 1.0]),
        np.array([np.inf, 1.0]),  # partial failure -> not finite
        np.array([2.0, 3.0]),
    ]
    assert sampler._count_finite_losses(losses) == 2


def test_good_bad_split_selects_nondominated_as_good():
    sampler = _sampler()
    # Points 0 and 1 form the first Pareto front, 2 and 3 are dominated.
    losses = np.array(
        [
            [0.0, 1.0],  # front 0
            [1.0, 0.0],  # front 0
            [2.0, 2.0],  # dominated
            [3.0, 3.0],  # dominated
        ]
    )
    idx_good, idx_bad = sampler._good_bad_split(losses, n_good=2, n_bad=2)
    assert set(idx_good.tolist()) == {0, 1}
    assert set(idx_bad.tolist()) == {2, 3}


def test_nondominated_promotion_advances_by_rank():
    data = [
        Datum((0, 0, 0), "FINISHED", losses=np.array([0.0, 1.0])),
        Datum((0, 0, 1), "FINISHED", losses=np.array([1.0, 0.0])),
        Datum((0, 0, 2), "FINISHED", losses=np.array([2.0, 2.0])),
        Datum((0, 0, 3), "FINISHED", losses=np.array([3.0, 3.0])),
    ]
    promoted = _nondominated_promotion(data, num_configs=2)
    assert set(promoted) == {(0, 0, 0), (0, 0, 1)}


def test_nondominated_promotion_handles_empty():
    assert _nondominated_promotion([], num_configs=2) == []


def test_kde_sampler_is_abstract():
    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("p1", (0.0, 1.0)))
    # KDESampler leaves the good/bad split seams abstract and cannot be instantiated.
    with pytest.raises(TypeError):
        KDESampler(
            search_space=space,
            min_samples_in_model=3,
            top_n_percent=15,
            num_samples=8,
            random_fraction=1 / 3,
            bandwidth_factor=3.0,
            min_bandwidth=1e-3,
            seed=42,
        )


def test_argsort_nondominated_orders_by_rank_then_crowding():
    losses = np.array(
        [
            [0.0, 2.0],  # front 0, boundary -> inf crowding
            [1.0, 1.0],  # front 0, interior -> finite crowding
            [2.0, 0.0],  # front 0, boundary -> inf crowding
            [3.0, 3.0],  # front 1
        ]
    )
    order = argsort_nondominated(losses)
    # Front 0 comes first; within it the interior point (index 1) comes last
    assert set(order[:3].tolist()) == {0, 1, 2}
    assert order[2] == 1
    assert order[3] == 3
