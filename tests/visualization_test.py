# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import plotly.io._base_renderers
import pytest
from plotly.graph_objs._figure import Figure

from blackboxopt import Evaluation, EvaluationSpecification, Objective
from blackboxopt.visualizations.utils import (
    get_incumbent_objective_over_time_single_fidelity,
    mask_pareto_efficient,
)
from blackboxopt.visualizations.visualizer import (
    NoSuccessfulEvaluationsError,
    Visualizer,
    _prepare_for_multi_objective_visualization,
    evaluations_to_df,
    multi_objective_visualization,
    parallel_coordinate_plot_parameters,
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
            objectives={"loss": 0.1 * i**2},
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

    multi_objective_visualization(
        evaluations=evaluations,
        objectives=(Objective("loss_1", False), Objective("loss_2", False)),
    )


def test_multi_objective_visualization_no_eval_specs():
    with pytest.raises(NoSuccessfulEvaluationsError):
        multi_objective_visualization(evaluations=[], objectives=())


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
            evaluations_without_any_result_with_all_objectives_evaluated,
            (Objective("loss_1", False), Objective("loss_2", False)),
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
        multi_objective_visualization(
            evaluations_with_all_objectives_none,
            (Objective("loss_1", False), Objective("loss_2", False)),
        )


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
    multi_objective_visualization(
        evaluations, (Objective("loss", False), Objective("score", True))
    )


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


def test_prepare_for_multi_objective_visualization_handles_score_objectives():
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
    df, _ = _prepare_for_multi_objective_visualization(
        evaluations_to_df(evaluations),
        (Objective("loss", False), Objective("score", True)),
    )
    pareto_efficient = df["pareto efficient"].values
    np.testing.assert_array_equal(pareto_efficient, np.array([False, True]))


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
    fig = multi_objective_visualization(
        evaluations, (Objective("loss_1", False), Objective("loss_2", False))
    )

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


def test_parallel_coordinate_plot_parameters():
    evaluations = [
        Evaluation(
            configuration={"float": 1.23, "int": 3, "bool": True, "categorical": "A"},
            objectives={"loss": 0.9, "score": 800},
        ),
        Evaluation(
            configuration={"float": 9.23, "int": 15, "bool": False, "categorical": "B"},
            objectives={"loss": 0.3, "score": 900},
        ),
        Evaluation(
            configuration={"float": 4.3, "int": 7, "bool": True, "categorical": "C"},
            objectives={"loss": 0.1, "score": 700},
        ),
        Evaluation(
            configuration={"float": 4.3, "int": 7, "bool": True, "categorical": "C"},
            objectives={"loss": None, "score": 900},
        ),
    ]
    fig = parallel_coordinate_plot_parameters(evaluations, Objective("loss", False))
    assert isinstance(fig, Figure)

    fig = parallel_coordinate_plot_parameters(evaluations)
    assert isinstance(fig, Figure)


def test_parallel_coordinate_plot_parameters_raise_on_no_succesful_evals():
    with pytest.raises(NoSuccessfulEvaluationsError):
        parallel_coordinate_plot_parameters(
            [], Objective("loss", greater_is_better=False)
        )

    evaluations_without_any_result_with_all_objectives_evaluated = [
        Evaluation(objectives={"loss": None}, configuration={"p1": 0.14})
    ] * 5
    with pytest.raises(NoSuccessfulEvaluationsError):
        parallel_coordinate_plot_parameters(
            evaluations_without_any_result_with_all_objectives_evaluated,
            Objective("loss", greater_is_better=False),
        )
