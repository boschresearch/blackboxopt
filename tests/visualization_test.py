# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

import numpy as np
import pytest

from blackboxopt import Evaluation
from blackboxopt.visualizations.utils import mask_pareto_efficient
from blackboxopt.visualizations.visualizer import (
    NoSuccessfulEvaluationsError,
    Visualizer,
    create_hover_information,
    multi_objective_visualization,
)


def test_visualizer_calls():
    evaluations = [
        Evaluation(
            objectives={"loss": 0.5 * i},
            configuration={
                "mlp_shape": 0.14054333845130684,
                "optimizer": "Adam",
                "batch_size": 160,
            },
            optimizer_info={"rung": -1},
            user_info={},
            settings={"fidelity": float(i)},
        )
        for i in range(5)
    ] + [
        Evaluation(
            objectives={"loss": 0.1 * i ** 2},
            configuration={
                "mlp_shape": 0.14054333845130684,
                "optimizer": "Adam",
                "batch_size": 161,
            },
            optimizer_info={"rung": -1},
            user_info={},
            settings={"fidelity": float(i)},
        )
        for i in range(5)
    ]

    viz = Visualizer(evaluations)

    viz.loss_over_time()
    viz.loss_over_duration()
    viz.cdf_losses()
    viz.cdf_durations()


def test_unsuccessful_visualizer_calls():
    with pytest.raises(NoSuccessfulEvaluationsError):
        Visualizer([])

    evaluations_without_any_result_with_all_objectives_evaluated = [
        Evaluation(
            objectives={"loss": None},
            configuration={
                "mlp_shape": 0.14054333845130684,
                "optimizer": "Adam",
                "batch_size": 160,
            },
            optimizer_info={"rung": -1},
            user_info={},
            settings={"fidelity": 2.5},
        )
    ] * 5
    with pytest.raises(NoSuccessfulEvaluationsError):
        Visualizer(evaluations_without_any_result_with_all_objectives_evaluated)


def test_int_none_float_loss_mix_does_not_break_viz():
    evaluations = [
        Evaluation(
            objectives={"loss": 815948261060003800000000},  # int
            configuration={
                "mlp_shape": 0.14054333845130684,
                "optimizer": "Adam",
                "batch_size": 161,
            },
            optimizer_info={"rung": -1},
            user_info={},
            settings={"fidelity": 0.1},
        ),
        Evaluation(
            objectives={"loss": None},  # None
            configuration={
                "mlp_shape": 0.14054333845130684,
                "optimizer": "Adam",
                "batch_size": 161,
            },
            optimizer_info={"rung": -1},
            user_info={},
            settings={"fidelity": 0.1},
        ),
        Evaluation(
            objectives={"loss": 123.123},  # float
            configuration={
                "mlp_shape": 0.14054333845130684,
                "optimizer": "Adam",
                "batch_size": 161,
            },
            optimizer_info={"rung": -1},
            user_info={},
            settings={"fidelity": 0.1},
        ),
    ]

    viz = Visualizer(evaluations)
    viz.loss_over_time()
    viz.loss_over_duration()
    viz.cdf_losses()
    viz.cdf_durations()


def test_multi_objective_visualization():
    evaluations = [
        Evaluation(
            objectives={"loss_1": 0.5 * i, "loss_2": -0.5 * i},
            configuration={
                "mlp_shape": 0.14054333845130684,
                "optimizer": "Adam",
                "batch_size": 160,
            },
            optimizer_info={"rung": -1},
            user_info={},
            settings={"fidelity": 2.5},
        )
        for i in range(5)
    ]

    multi_objective_visualization(evaluations)


def test_multi_objective_visualization_no_eval_specs():
    with pytest.raises(NoSuccessfulEvaluationsError):
        multi_objective_visualization([])


def test_multi_objective_visualization_no_successful_evaluations():
    evaluations_without_any_result_with_all_objectives_evaluated = [
        Evaluation(
            objectives={"loss_1": None, "loss_2": -0.5 * i},
            configuration={
                "mlp_shape": 0.14054333845130684,
                "optimizer": "Adam",
                "batch_size": 160,
            },
            optimizer_info={"rung": -1},
            user_info={},
            settings={"fidelity": 2.5},
        )
        for i in range(5)
    ]

    with pytest.raises(NoSuccessfulEvaluationsError):
        multi_objective_visualization(
            evaluations_without_any_result_with_all_objectives_evaluated
        )


def test_mask_pareto_efficient():

    evals = np.array(
        [
            [0.0, 1.0],
            [1.1, 0.1],
            [0.0, 1.0],
            [1.0, 0.0],
            [3.1, 1.1],
        ]
    )
    pareto_efficient = mask_pareto_efficient(evals)
    assert len(pareto_efficient) == 5
    assert pareto_efficient[0]
    assert not pareto_efficient[1]
    assert pareto_efficient[2]
    assert pareto_efficient[3]
    assert not pareto_efficient[4]
