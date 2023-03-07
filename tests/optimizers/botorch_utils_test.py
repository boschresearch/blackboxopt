# Copyright (c) 2023 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from blackboxopt import ConstraintsError, Evaluation, Objective
from blackboxopt.optimizers.botorch_base import (
    filter_y_nans,
    impute_nans_with_constant,
    to_numerical,
)

from .conftest import constraint_name_1, constraint_name_2, objective_name


def test_impute_nans_with_constant():
    x1_no_nans = torch.tensor([[0.1, 0.1], [0.7, 0.2], [1.0, 0.3]])
    x1_some_nans = torch.tensor([[0.1, 0.1], [0.7, float("nan")], [1.0, 0.3]])
    x1_all_nans = torch.tensor(
        [[0.1, float("nan")], [0.7, float("nan")], [1.0, float("nan")]]
    )

    x1_no_nans_i = impute_nans_with_constant(x1_no_nans)
    x1_some_nans_i = impute_nans_with_constant(x1_some_nans)
    x1_all_nans_i = impute_nans_with_constant(x1_all_nans)

    assert torch.equal(x1_no_nans_i, x1_no_nans)
    assert torch.equal(
        x1_some_nans_i, torch.tensor([[0.1, 0.1], [0.7, -1.0], [1.0, 0.3]])
    )
    assert torch.equal(
        x1_all_nans_i, torch.tensor([[0.1, -1.0], [0.7, -1.0], [1.0, -1.0]])
    )

    # test batched representation
    x1_some_nans_batched = x1_some_nans.reshape(torch.Size((1,)) + x1_some_nans.shape)
    x1_some_nans_batched_i = impute_nans_with_constant(x1_some_nans_batched)
    assert torch.equal(
        x1_some_nans_batched_i,
        torch.tensor([[0.1, 0.1], [0.7, -1.0], [1.0, 0.3]]).reshape(
            x1_some_nans_batched.shape
        ),
    )


@pytest.mark.parametrize("greater_is_better", [False, True])
def test_to_numerical(evaluations, search_space, greater_is_better):
    # Remove None from list it is up to user/specific optimizer what to do with None
    del evaluations[3]

    objective = Objective(objective_name, greater_is_better)

    X, Y = to_numerical(evaluations, search_space, [objective])

    assert X.dtype == torch.float32
    assert Y.dtype == torch.float32
    assert X.size() == (len(evaluations), len(search_space))
    assert Y.size() == (len(evaluations), 1)

    # check that numerical representation on inputs is in hypercube
    assert torch.all(X[X >= 0]) and torch.all(X[X <= 1])

    # check sign of objective values
    objectives_original = torch.Tensor(
        [evaluations[i]["objectives"]["objective"] for i in range(len(evaluations))]
    ).reshape(Y.shape)
    assert (
        torch.equal(Y, -1 * objectives_original)
        if greater_is_better
        else torch.equal(Y, objectives_original)
    )


def test_to_numerical_with_batch(evaluations, search_space):
    # Remove None from list it is up to user/specific optimizer what to do with None
    del evaluations[3]

    objective = Objective(objective_name, False)

    batch_shape = torch.Size((1,))
    X, Y = to_numerical(evaluations, search_space, [objective], batch_shape=batch_shape)

    assert X.dtype == torch.float32
    assert Y.dtype == torch.float32
    assert X.size() == (batch_shape[0], len(evaluations), len(search_space))
    assert Y.size() == (batch_shape[0], len(evaluations), 1)

    # check that numerical representation on inputs is in hypercube
    assert torch.all(X[X >= 0]) and torch.all(X[X <= 1])


def test_to_numerical_raises_errors(search_space):
    objective = Objective(objective_name, False)

    # configuration has parameter not part of search space
    eval_err = Evaluation(
        configuration={"x0": 0.57, "x1": False, "x2": "small", "x3": 0.5, "fp": 0.5},
        objectives={objective_name: 0.64},
    )
    with pytest.raises(ValueError, match="Mismatch"):
        to_numerical([eval_err], search_space, [objective])

    # parameter not within defined bounds -> invalid configuration
    eval_err = Evaluation(
        configuration={"x0": 0.1, "x1": False, "x2": "small", "fp": 0.5},
        objectives={objective_name: 0.64},
    )
    with pytest.raises(ValueError, match="not valid"):
        to_numerical([eval_err], search_space, [objective])

    # conditional parameter should be active, but is inactive -> invalid configuration
    eval_err = Evaluation(
        configuration={"x0": 0.57, "x1": True, "x2": "small", "fp": 0.5},
        objectives={objective_name: 0.64},
    )
    with pytest.raises(ValueError, match="not valid"):
        to_numerical([eval_err], search_space, [objective])

    # conditional parameter should be inactive, but is active -> invalid configuration
    eval_err = Evaluation(
        configuration={"x0": 0.57, "x1": False, "x2": "small", "cp": 0.3, "fp": 0.5},
        objectives={objective_name: 0.64},
    )
    with pytest.raises(ValueError, match="not valid"):
        to_numerical([eval_err], search_space, [objective])


@pytest.mark.parametrize(
    "constraints",
    [
        [constraint_name_1, constraint_name_2],
        [constraint_name_2, constraint_name_1],
        [constraint_name_1],
    ],
)
def test_to_numerical_with_constraints(
    evaluations_with_constraints, search_space, constraints
):
    objective = Objective(objective_name, greater_is_better=False)
    num_eval = len(evaluations_with_constraints)
    num_constrains = len(constraints)

    _, Y = to_numerical(
        evaluations_with_constraints,
        search_space,
        [objective],
        constraints,
    )

    assert Y.dtype == torch.float32
    assert Y.size() == (num_eval, 1 + num_constrains)

    # check order of values in the output tensor: the first is always an objective value,
    # the order of constraints depends on order in the list of
    for i in range(num_eval):
        assert Y[i, 0] == evaluations_with_constraints[i].objectives[objective_name]
        for c_i, c in enumerate(constraints):
            assert Y[i, 1 + c_i] == evaluations_with_constraints[i].constraints[c]


def test_to_numerical_raises_error_on_wrong_constraints(
    search_space, evaluations_with_constraints
):
    # If wrong constraint name is requested raises an error.
    objective = Objective(objective_name, False)

    with pytest.raises(ConstraintsError, match="Constraint name"):
        to_numerical(
            evaluations_with_constraints,
            search_space,
            [objective],
            constraint_names=["WRONG_NAME"],
        )

    # If evaluation does not contain constraints at all
    objective = Objective(objective_name, False)

    evaluations = [
        Evaluation(
            configuration={"x0": 0.57, "x1": True, "x2": "small", "cp": 0.3, "fp": 0.5},
            objectives={objective_name: 0.64},
        )
    ]

    with pytest.raises(ConstraintsError, match="Constraint name"):
        to_numerical(
            evaluations,
            search_space,
            [objective],
            constraint_names=[constraint_name_1],
        )


def test_to_numerical_multiple_objectives(search_space):
    objectives = [
        Objective("score", greater_is_better=True),
        Objective("loss", greater_is_better=False),
    ]

    evaluations = [
        Evaluation(
            objectives={"score": 0.1, "loss": 0.1}, configuration=search_space.sample()
        )
    ]

    X, Y = to_numerical(evaluations, search_space, objectives)

    assert X.dtype == torch.float32
    assert Y.dtype == torch.float32
    assert X.size() == (len(evaluations), len(search_space))
    assert Y.size() == (len(evaluations), len(objectives))
    assert torch.equal(Y, torch.Tensor([[-0.1, 0.1]]))


def test_filter_y_nans():
    x1 = torch.tensor([[0.1], [0.7], [1.0]])
    y1 = torch.tensor([[0.8], [0.3], [0.5]])
    x1_f, y1_f = filter_y_nans(x1, y1)
    assert x1_f.size() == torch.Size([3, 1])
    assert y1_f.size() == torch.Size([3, 1])
    assert torch.equal(x1, x1_f)
    assert torch.equal(y1, y1_f)

    x2 = torch.tensor([[0.1], [0.7], [1.0]])
    y2 = torch.tensor([[0.8], [np.nan], [0.5]])
    x2_f, y2_f = filter_y_nans(x2, y2)
    assert x2_f.size() == torch.Size([2, 1])
    assert y2_f.size() == torch.Size([2, 1])

    x3 = torch.tensor([[0.1], [0.7], [1.0]])
    y3 = torch.tensor([[np.nan], [np.nan], [np.nan]])
    x3_f, y3_f = filter_y_nans(x3, y3)
    assert x3_f.size() == torch.Size([0, 1])
    assert y3_f.size() == torch.Size([0, 1])

    # test batched representation
    x2_batched = x2.reshape(torch.Size((1,)) + x2.shape)
    y2_batched = y2.reshape(torch.Size((1,)) + y2.shape)
    x2_batched_f, y2_batched_f = filter_y_nans(x2_batched, y2_batched)
    assert x2_batched_f.size() == torch.Size([1, 2, 1])
    assert y2_batched_f.size() == torch.Size([1, 2, 1])
    assert torch.equal(x2_batched_f, x2_f.reshape(torch.Size((1,)) + x2_f.shape))
    assert torch.equal(y2_batched_f, y2_f.reshape(torch.Size((1,)) + y2_f.shape))

    x_multi_batch = torch.tensor([[[0.1], [1.0]], [[0.2], [2.0]]])
    y_multi_batch = torch.tensor([[[0.4], [0.4]], [[0.8], [np.nan]]])
    with pytest.raises(ValueError, match="Multiple batches"):
        filter_y_nans(x_multi_batch, y_multi_batch)
