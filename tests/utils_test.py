# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import random

import numpy as np
import parameterspace as ps
import pytest

from blackboxopt import Evaluation, Objective
from blackboxopt.utils import (
    filter_pareto_efficient,
    get_loss_vector,
    load_study_from_json,
    load_study_from_pickle,
    mask_pareto_efficient,
    save_study_as_json,
    save_study_as_pickle,
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
            Evaluation(configuration={"p1": 3, "p2": "A"}, objectives={"o": 0.0}),
            Evaluation(configuration={"p1": 2, "p2": "B"}, objectives={"o": 0.0}),
            Evaluation(configuration={"p1": 1, "p2": "C"}, objectives={"o": 0.0}),
        ],
        [  # Different objective values
            Evaluation(configuration={"p": 1}, objectives={"o": 0.0}),
            Evaluation(configuration={"p": 1}, objectives={"o": 0.3}),
            Evaluation(configuration={"p": 1}, objectives={"o": 0.2}),
        ],
        [  # Different parameters
            Evaluation(configuration={"p1": 1}, objectives={"o": 0.0}),
            Evaluation(configuration={"p1": 1, "p2": 2}, objectives={"o": 0.0}),
            Evaluation(configuration={"p3": 1}, objectives={"o": 0.0}),
        ],
        [  # Different multi objective values
            Evaluation(configuration={"p": 1}, objectives={"o1": 0.0, "o2": 0.1}),
            Evaluation(configuration={"p": 1}, objectives={"o1": 0.0, "o2": 0.2}),
            Evaluation(configuration={"p": 1}, objectives={"o1": 0.0, "o2": 0.3}),
        ],
        [  # Different context
            Evaluation(configuration={"p": 1}, objectives={"o": 0}, context={"c": 1}),
            Evaluation(configuration={"p": 1}, objectives={"o": 0}, context={"c": 2}),
            Evaluation(configuration={"p": 1}, objectives={"o": 0}, context={"c": 3}),
        ],
        [  # Different settings
            Evaluation(configuration={"p": 1}, objectives={"o": 0}, settings={"s": 1}),
            Evaluation(configuration={"p": 1}, objectives={"o": 0}, settings={"s": 2}),
            Evaluation(configuration={"p": 1}, objectives={"o": 0}, settings={"s": 3}),
        ],
        [  # Different constraints
            Evaluation(
                configuration={"p": 1}, objectives={"o": 0}, constraints={"c": 1}
            ),
            Evaluation(
                configuration={"p": 1}, objectives={"o": 0}, constraints={"c": 2}
            ),
            Evaluation(
                configuration={"p": 1}, objectives={"o": 0}, constraints={"c": 3}
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


def test_save_and_load_study_pickle(tmp_path):
    tmp_file = tmp_path / "out.json"

    search_space = ps.ParameterSpace()
    search_space.add(ps.IntegerParameter("p1", (-10, 10)))
    objectives = [Objective("loss", False), Objective("score", True)]
    evaluations = [
        Evaluation(configuration={"p1": 1.0}, objectives={"loss": 1.0, "score": 3.0}),
        Evaluation(configuration={"p1": 2.0}, objectives={"loss": 0.1, "score": 2.0}),
        Evaluation(configuration={"p1": 3.0}, objectives={"loss": 0.0, "score": 1.0}),
    ]
    save_study_as_pickle(search_space, objectives, evaluations, tmp_file)

    # Check that default overwrite=False causes IOError on existing file
    with pytest.raises(IOError, match=str(tmp_file)):
        save_study_as_pickle(search_space, objectives, evaluations, tmp_file)

    loaded_study = load_study_from_pickle(tmp_file)

    assert loaded_study[1] == objectives
    assert loaded_study[2] == evaluations
    for _ in range(128):
        assert search_space.sample() == loaded_study[0].sample()


def test_save_and_load_study_pickle_fails_on_missing_output_directory():
    pickle_file_path = "/this/directory/does/not/exist/pickles/out.pkl"
    with pytest.raises(IOError, match=pickle_file_path):
        save_study_as_pickle(
            search_space=ps.ParameterSpace(),
            objectives=[],
            evaluations=[],
            pickle_file_path=pickle_file_path,
        )


def test_save_and_load_study_json(tmp_path):
    tmp_file = tmp_path / "out.json"

    search_space = ps.ParameterSpace()
    search_space.add(ps.IntegerParameter("p1", (-10.0, 10.0)))
    objectives = [Objective("loss", False), Objective("score", True)]
    evaluations = [
        Evaluation(configuration={"p1": 1.0}, objectives={"loss": 1.0, "score": 3.0}),
        Evaluation(configuration={"p1": 2.0}, objectives={"loss": 0.1, "score": 2.0}),
        Evaluation(configuration={"p1": 3.0}, objectives={"loss": 0.0, "score": 1.0}),
    ]
    save_study_as_json(search_space, objectives, evaluations, tmp_file)

    # Check that default overwrite=False causes ValueError on existing file
    with pytest.raises(IOError):
        save_study_as_json(search_space, objectives, evaluations, tmp_file)

    loaded_study = load_study_from_json(tmp_file)

    assert loaded_study[1] == objectives
    assert loaded_study[2] == evaluations
    for _ in range(128):
        assert search_space.sample() == loaded_study[0].sample()


def test_save_and_load_study_json_fails_on_missing_output_directory():
    json_file_path = "/this/directory/does/not/exist/jsons/out.json"
    with pytest.raises(IOError, match=json_file_path):
        save_study_as_json(
            search_space=ps.ParameterSpace(),
            objectives=[],
            evaluations=[],
            json_file_path=json_file_path,
        )


def test_save_and_load_study_json_fails_with_complex_type_in_evaluation(tmp_path):
    tmp_file = tmp_path / "out.json"

    search_space = ps.ParameterSpace()
    objectives = []
    evaluations = [
        Evaluation(
            configuration={},
            objectives={},
            user_info={"complex typed value": ps.ParameterSpace()},
        ),
    ]

    with pytest.raises(TypeError):
        save_study_as_json(search_space, objectives, evaluations, tmp_file)
