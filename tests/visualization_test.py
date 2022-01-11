# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import plotly.io._base_renderers
import pytest

from blackboxopt import Evaluation, EvaluationSpecification, Objective
from blackboxopt.visualizations.utils import (
    mask_pareto_efficient,
    get_incumbent_objective_over_time_single_fidelity,
)
from blackboxopt.visualizations.visualizer import (
    NoSuccessfulEvaluationsError,
    Visualizer,
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

    viz = Visualizer(evaluations, Objective("loss", greater_is_better=False))

    viz.objective_over_time()
    viz.objective_over_duration()
    viz.cdf_objective_values()
    viz.cdf_durations()


def test_unsuccessful_visualizer_calls():
    with pytest.raises(NoSuccessfulEvaluationsError):
        Visualizer([], Objective("loss", greater_is_better=False))

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
        Visualizer(
            evaluations_without_any_result_with_all_objectives_evaluated,
            Objective("loss", greater_is_better=False),
        )


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

    viz = Visualizer(evaluations, Objective("loss", greater_is_better=False))
    viz.objective_over_time()
    viz.objective_over_duration()
    viz.cdf_objective_values()
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


def test_multi_objective_visualization_all_none_evaluations():
    evaluations_with_all_objectives_none = [
        Evaluation(
            objectives={"loss_1": None, "loss_2": None},
            configuration={
                "mlp_shape": 0.14054333845130684,
                "optimizer": "Adam",
                "batch_size": 160,
            },
            optimizer_info={"rung": -1},
            user_info={},
            settings={"fidelity": 2.5},
        )
        for _ in range(5)
    ]
    with pytest.raises(NoSuccessfulEvaluationsError):
        multi_objective_visualization(evaluations_with_all_objectives_none)


def test_multi_objective_visualization_without_fidelities():
    evaluations = [
        EvaluationSpecification(configuration={"p1": 1.23}),
        Evaluation(
            configuration={"p1": 1.3},
            objectives={"loss": 1.4, "score": 9001},
        ),
        Evaluation(
            configuration={"p1": 2.3},
            objectives={"loss": 0.4, "score": 9200},
        ),
    ]
    multi_objective_visualization(evaluations)


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


def test_patching_of_plotly_html(tmpdir, monkeypatch):
    # Create example figure
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
    fig = multi_objective_visualization(evaluations)

    # Test displaying in browser
    def open_html_in_browser_mocked(html, *args, **kwargs):
        print(html)
        assert "persistentHoverLayer" in html

    monkeypatch.setattr(
        plotly.io._base_renderers, "open_html_in_browser", open_html_in_browser_mocked
    )
    fig.show()

    # Test writing to file
    html_file = tmpdir.join("test_output.html")
    fig.write_html(html_file)
    with open(html_file, "r") as fh:
        html = fh.read()
        assert "persistentHoverLayer" in html

    # Test ouutput html string
    html = fig.to_html(html_file)
    assert "persistentHoverLayer" in html


def test_get_incumbent_objective_over_time_single_fidelity():
    times, objective_values = get_incumbent_objective_over_time_single_fidelity(
        objective=Objective("loss", greater_is_better=False),
        objective_values=np.array([0.0, 2.0, 1.0, 0.0, 1.0, 0.0]),
        times=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        fidelities=np.array([1.0, 2.0, 1.0, 2.0, 2.0, 2.0]),
        target_fidelity=2.0,
    )
    np.testing.assert_array_equal(objective_values, np.array([2.0, 2.0, 0.0, 0.0]))
    np.testing.assert_array_equal(times, np.array([2.0, 4.0, 4.0, 6.0]))

    times, objective_values = get_incumbent_objective_over_time_single_fidelity(
        objective=Objective("score", greater_is_better=True),
        objective_values=np.array([0.0, 0.0, 1.0, 2.0, 1.0, 1.0]),
        times=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        fidelities=np.array([1.0, 2.0, 1.0, 2.0, 2.0, 1.0]),
        target_fidelity=2.0,
    )
    np.testing.assert_array_equal(objective_values, np.array([0.0, 0.0, 2.0, 2.0]))
    np.testing.assert_array_equal(times, np.array([2.0, 4.0, 4.0, 6.0]))
