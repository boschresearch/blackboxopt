# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import random

import numpy as np
import pytest

from blackboxopt import Evaluation, Objective
from blackboxopt.utils import (
    filter_pareto_efficient,
    get_loss_vector,
    mask_pareto_efficient,
    sort_evaluations,
)


@pytest.mark.parametrize(
    "known,reported,expected",
    [
        (
            [Objective("score", True), Objective("loss", False)],
            {"loss": 2.0, "score": 1.0},
            [-1.0, 2.0],
        ),
        (
            [Objective("score", True), Objective("loss", False)],
            {"loss": 2.0, "score": None},
            [np.nan, 2.0],
        ),
    ],
)
def test_get_loss_vector(known, reported, expected):
    loss_vector = get_loss_vector(known_objectives=known, reported_objectives=reported)
    np.testing.assert_array_equal(loss_vector, np.array(expected))


def test_get_loss_vector_with_custom_none_replacement():
    known = [Objective("score", True), Objective("loss", False)]

    reported = {"loss": 2.0, "score": None}
    expected = [np.inf, 2.0]

    loss_vector = get_loss_vector(
        known_objectives=known,
        reported_objectives=reported,
        none_replacement=float("Inf"),
    )
    np.testing.assert_array_equal(loss_vector, np.array(expected))


def test_mask_pareto_efficient():
    evals = np.array(
        [
            [0.0, 1.0],
            [1.1, 0.1],
            [0.0, 1.0],
            [1.0, 0.0],
            [3.1, 1.1],
            [0.1, 1.0],
            [0.0, 1.1],
            [-1.0, 2.0],
        ]
    )
    pareto_efficient = mask_pareto_efficient(evals)
    assert len(pareto_efficient) == evals.shape[0]
    assert pareto_efficient[0]
    assert not pareto_efficient[1]
    assert pareto_efficient[2]
    assert pareto_efficient[3]
    assert not pareto_efficient[4]
    assert not pareto_efficient[5]
    assert not pareto_efficient[6]
    assert pareto_efficient[7]


def test_filter_pareto_efficient():
    evals = [
        Evaluation(configuration={"i": 0}, objectives={"loss": 0.0, "score": -1.0}),
        Evaluation(configuration={"i": 1}, objectives={"loss": 1.1, "score": -0.1}),
        Evaluation(configuration={"i": 2}, objectives={"loss": 0.0, "score": -1.0}),
        Evaluation(configuration={"i": 3}, objectives={"loss": 1.0, "score": 0.0}),
        Evaluation(configuration={"i": 4}, objectives={"loss": 3.1, "score": -1.1}),
        Evaluation(configuration={"i": 5}, objectives={"loss": 0.1, "score": -1.0}),
        Evaluation(configuration={"i": 6}, objectives={"loss": 0.0, "score": -1.1}),
        Evaluation(configuration={"i": 7}, objectives={"loss": -1.0, "score": -2.0}),
    ]
    pareto_efficient = filter_pareto_efficient(
        evals, [Objective("loss", False), Objective("score", True)]
    )
    assert len(pareto_efficient) == 4
    assert set([0, 2, 3, 7]) == set([e.configuration["i"] for e in pareto_efficient])


@pytest.mark.parametrize(
    "evals",
    (
        [  # Different parameter values
            Evaluation(configuration={"i": 3, "c": "A"}, objectives={"l": 0.0}),
            Evaluation(configuration={"i": 2, "c": "B"}, objectives={"l": 0.0}),
            Evaluation(configuration={"i": 1, "c": "C"}, objectives={"l": 0.0}),
        ],
        [  # Different objective values
            Evaluation(configuration={"i": 1}, objectives={"l": 0.0}),
            Evaluation(configuration={"i": 1}, objectives={"l": 0.3}),
            Evaluation(configuration={"i": 1}, objectives={"l": 0.2}),
        ],
        [  # Different parameters
            Evaluation(configuration={"i": 1}, objectives={"l": 0.0}),
            Evaluation(configuration={"i": 1, "j": 2}, objectives={"l": 0.0}),
            Evaluation(configuration={"k": 1}, objectives={"l": 0.0}),
        ],
        [  # Different multi objective values
            Evaluation(configuration={"i": 1}, objectives={"l": 0.0, "m": 0.1}),
            Evaluation(configuration={"i": 1}, objectives={"l": 0.0, "m": 0.2}),
            Evaluation(configuration={"i": 1}, objectives={"l": 0.0, "m": 0.3}),
        ],
        [  # Different context
            Evaluation(configuration={"i": 1}, objectives={"l": 0}, context={"c": 1}),
            Evaluation(configuration={"i": 1}, objectives={"l": 0}, context={"c": 2}),
            Evaluation(configuration={"i": 1}, objectives={"l": 0}, context={"c": 3}),
        ],
        [  # Different settings
            Evaluation(configuration={"i": 1}, objectives={"l": 0}, settings={"s": 1}),
            Evaluation(configuration={"i": 1}, objectives={"l": 0}, settings={"s": 2}),
            Evaluation(configuration={"i": 1}, objectives={"l": 0}, settings={"s": 3}),
        ],
        [  # Different constraints
            Evaluation(
                configuration={"i": 1}, objectives={"l": 0}, constraints={"c": 1}
            ),
            Evaluation(
                configuration={"i": 1}, objectives={"l": 0}, constraints={"c": 2}
            ),
            Evaluation(
                configuration={"i": 1}, objectives={"l": 0}, constraints={"c": 3}
            ),
        ],
    ),
)
def test_sort_evaluations(evals):
    # Create n shuffled versions of the evals and cache the sorted results
    all_sorted_evals = []
    for _ in range(10):
        shuffled_evals = evals.copy()
        random.shuffle(shuffled_evals)
        evals_sorted = sort_evaluations(shuffled_evals)
        all_sorted_evals.append(evals_sorted)

    # Test if the evaluations of all cached results ended up in the same order by
    # checking if the evaluations with the same index are all equal
    for evals in zip(*all_sorted_evals):
        assert len(set([str(e.configuration) for e in evals])) == 1
        assert len(set([str(e.objectives) for e in evals])) == 1
        assert len(set([str(e.settings) for e in evals])) == 1
        assert len(set([str(e.context) for e in evals])) == 1
        assert len(set([str(e.constraints) for e in evals])) == 1


def test_sort_evaluations_with_different_parameter_order():
    evals_a = [
        Evaluation(configuration={"i": 1, "c": "C"}, objectives={"l": 0.0}),
        Evaluation(configuration={"i": 2, "c": "B"}, objectives={"l": 0.0}),
        Evaluation(configuration={"i": 3, "c": "A"}, objectives={"l": 0.0}),
    ]
    evals_b = [
        Evaluation(configuration={"c": "C", "i": 1}, objectives={"l": 0.0}),
        Evaluation(configuration={"c": "A", "i": 3}, objectives={"l": 0.0}),
        Evaluation(configuration={"c": "B", "i": 2}, objectives={"l": 0.0}),
    ]

    evals_a_sorted = sort_evaluations(evals_a)
    evals_b_sorted = sort_evaluations(evals_b)

    for a, b in zip(evals_a_sorted, evals_b_sorted):
        assert a.configuration == b.configuration
        assert a.objectives == b.objectives
