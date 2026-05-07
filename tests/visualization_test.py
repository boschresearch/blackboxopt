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
    get_constraint_satisfaction_mask,
    get_incumbent_objective_over_iterations,
    get_incumbent_objective_over_time_single_fidelity,
)
from blackboxopt.visualizations.visualizer import (
    NoSuccessfulEvaluationsError,
    Visualizer,
    _prepare_for_multi_objective_visualization,
    evaluations_to_df,
    hypervolume_over_iterations,
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


def test_multi_objective_visualization_two_objectives():
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


def test_hypervolume_over_iterations_two_objectives():
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

    evaluations_per_optimizer = {"test_optimizer": [evaluations]}

    hypervolume_over_iterations(
        evaluations_per_optimizer,
        objectives=(Objective("loss_1", False), Objective("loss_2", False)),
        reference_point=[10, 0],
    )


def test_multi_objective_visualization_more_than_two_objectives():
    evaluations = [
        Evaluation(
            objectives={"loss_1": 0.5 * i, "loss_2": -0.5 * i, "score": -(i**2)},
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
        objectives=(
            Objective("loss_1", False),
            Objective("loss_2", False),
            Objective("score", True),
        ),
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


@pytest.mark.parametrize(
    "plot_kwargs",
    (
        dict(),
        dict(color_by="loss"),
        dict(columns=["int", "float", "loss"]),
        dict(columns=["float", "int", "score"], color_by="score"),
    ),
)
def test_parallel_coordinate_plot_parameters(tmp_path, plot_kwargs):
    evaluations = [
        Evaluation(
            configuration={"float": 1.23, "int": 3, "bool": True, "categorical": "A"},
            objectives={"loss": 0.9, "score": 800},
            settings={"fidelity": 0.3},
        ),
        Evaluation(
            configuration={"float": 9.23, "int": 15, "bool": False, "categorical": "B"},
            objectives={"loss": 0.3, "score": 900},
            settings={"fidelity": 0.66},
        ),
        Evaluation(
            configuration={"float": 4.3, "int": 7, "bool": True, "categorical": "C"},
            objectives={"loss": 0.1, "score": 1000},
            settings={"fidelity": 1.0},
        ),
        Evaluation(
            configuration={"float": 4.3, "int": 7, "bool": True, "categorical": "C"},
            objectives={"loss": None, "score": 900},
            settings={"fidelity": 1},
        ),
    ]

    fig = parallel_coordinate_plot_parameters(evaluations, **plot_kwargs)
    assert isinstance(fig, Figure)
    # Rendering the plot, because this can raise exception, even if instantiating worked
    fig.write_image(tmp_path / "figure.png")


def test_parallel_coordinate_plot_parameters_raise_on_no_succesful_evals():
    with pytest.raises(NoSuccessfulEvaluationsError):
        parallel_coordinate_plot_parameters([])

    evaluations_with_all_objectives_none = [
        Evaluation(objectives={"loss": None}, configuration={"p1": 0.14})
    ] * 5
    with pytest.raises(NoSuccessfulEvaluationsError):
        parallel_coordinate_plot_parameters(evaluations_with_all_objectives_none)


def test_parallel_coordinate_plot_parameters_raise_on_ambigious_column_names():
    evaluations = [
        Evaluation(
            configuration={"DUP": 1, "A": 2}, objectives={"B": 1}, settings={"DUP": 3}
        )
    ]
    with pytest.raises(ValueError, match="DUP"):
        parallel_coordinate_plot_parameters(
            parallel_coordinate_plot_parameters(evaluations)
        )

    evaluations = [
        Evaluation(
            configuration={"DUP": 1, "A": 2}, objectives={"DUP": 1}, settings={"B": 3}
        )
    ]
    with pytest.raises(ValueError, match="DUP"):
        parallel_coordinate_plot_parameters(
            parallel_coordinate_plot_parameters(evaluations)
        )

    evaluations = [
        Evaluation(
            configuration={"A": 1, "B": 2}, objectives={"DUP": 1}, settings={"DUP": 3}
        )
    ]
    with pytest.raises(ValueError, match="DUP"):
        parallel_coordinate_plot_parameters(
            parallel_coordinate_plot_parameters(evaluations)
        )

    # Don't raise if duplicate column but not selected for plotting
    fig = parallel_coordinate_plot_parameters(evaluations, columns=["A", "B"])
    assert isinstance(fig, Figure)


def test_parallel_coordinate_plot_parameters_raise_on_color_by_not_in_columns():
    evaluations = [
        Evaluation(objectives={"loss": 0.3}, configuration={"p1": 0.1, "hidden": 0.2})
    ]
    with pytest.raises(ValueError, match="hidden"):
        parallel_coordinate_plot_parameters(
            evaluations,
            columns=["loss", "p1"],
            color_by="hidden",
        )
    with pytest.raises(ValueError, match="not_existing"):
        parallel_coordinate_plot_parameters(evaluations, color_by="not_existing")


def test_parallel_coordinate_plot_parameters_with_unhashable_parameters():
    evaluations = [
        Evaluation(objectives={"loss": 0.3}, configuration={"p1": [1, 2, 3]})
    ]
    fig = parallel_coordinate_plot_parameters(
        evaluations, columns=["loss", "p1"], color_by="p1"
    )
    assert isinstance(fig, Figure)


def test_objective_over_iteration_no_constraints():
    evaluations = [
        Evaluation(
            objectives={"loss": 0.5 * i},
            configuration={"p1": 0.1 * i},
            optimizer_info={"rung": -1},
            user_info={},
            settings={"fidelity": 1.0},
        )
        for i in range(5)
    ]
    viz = Visualizer(evaluations, Objective("loss", greater_is_better=False))
    fig = viz.objective_over_iteration()
    assert isinstance(fig, Figure)
    # All evaluations should be feasible (no constraints), so 2 traces:
    # feasible scatter + incumbent line
    assert len(fig.data) == 2


def test_objective_over_iteration_with_constraints():
    evaluations = [
        Evaluation(
            objectives={"loss": float(i)},
            constraints={"c1": float(i) - 2.0},
            configuration={"p1": 0.1 * i},
            optimizer_info={"rung": -1},
            user_info={},
            settings={"fidelity": 1.0},
        )
        for i in range(5)
    ]
    viz = Visualizer(evaluations, Objective("loss", greater_is_better=False))
    fig = viz.objective_over_iteration(constraint_bounds={"c1": (0.0, None)})
    assert isinstance(fig, Figure)
    # c1 values: -2, -1, 0, 1, 2 -> feasible: i=2,3,4 (c1 >= 0), infeasible: i=0,1
    # 3 traces: feasible scatter, infeasible scatter, incumbent line
    assert len(fig.data) == 3


def test_objective_over_iteration_all_infeasible():
    evaluations = [
        Evaluation(
            objectives={"loss": float(i)},
            constraints={"c1": -1.0},
            configuration={"p1": 0.1 * i},
            optimizer_info={"rung": -1},
            user_info={},
            settings={"fidelity": 1.0},
        )
        for i in range(3)
    ]
    viz = Visualizer(evaluations, Objective("loss", greater_is_better=False))
    fig = viz.objective_over_iteration(constraint_bounds={"c1": (0.0, None)})
    assert isinstance(fig, Figure)
    # All infeasible: only infeasible scatter, no incumbent line
    assert len(fig.data) == 1


def test_get_constraint_satisfaction_mask():
    evaluations = [
        Evaluation(
            objectives={"loss": 1.0},
            constraints={"c1": 0.5, "c2": -0.1},
            configuration={"p1": 1.0},
        ),
        Evaluation(
            objectives={"loss": 2.0},
            constraints={"c1": -0.5, "c2": 0.3},
            configuration={"p1": 2.0},
        ),
        Evaluation(
            objectives={"loss": 3.0},
            constraints=None,
            configuration={"p1": 3.0},
        ),
    ]

    # No bounds: all feasible
    mask = get_constraint_satisfaction_mask(evaluations)
    np.testing.assert_array_equal(mask, [True, True, True])

    # c2 <= 0
    mask = get_constraint_satisfaction_mask(evaluations, {"c2": (None, 0.0)})
    np.testing.assert_array_equal(mask, [True, False, False])

    # c2 >= 0
    mask = get_constraint_satisfaction_mask(evaluations, {"c2": (0.0, None)})
    np.testing.assert_array_equal(mask, [False, True, False])

    # c1 in [-1, 1]
    mask = get_constraint_satisfaction_mask(evaluations, {"c1": (-1.0, 1.0)})
    np.testing.assert_array_equal(mask, [True, True, False])


def test_get_incumbent_objective_over_iterations():
    objective = Objective("loss", greater_is_better=False)
    objective_values = np.array([3.0, 1.0, 2.0, 0.5, 4.0])
    feasible_mask = np.array([True, True, True, True, True])

    iterations, values = get_incumbent_objective_over_iterations(
        objective, objective_values, feasible_mask
    )
    # Incumbent: 3.0 -> 1.0 -> 0.5
    # Step function: iterations [1, 2, 2, 4, 4, 5], values [3, 3, 1, 1, 0.5, 0.5]
    np.testing.assert_array_equal(iterations, [1, 2, 2, 4, 4, 5])
    np.testing.assert_array_equal(values, [3.0, 3.0, 1.0, 1.0, 0.5, 0.5])


def test_get_incumbent_objective_over_iterations_with_infeasible():
    objective = Objective("loss", greater_is_better=False)
    objective_values = np.array([3.0, 0.1, 2.0, 0.5, 4.0])
    # Evaluation at index 1 (best value) is infeasible
    feasible_mask = np.array([True, False, True, True, True])

    iterations, values = get_incumbent_objective_over_iterations(
        objective, objective_values, feasible_mask
    )
    # Feasible: i=1(3.0), i=3(2.0), i=4(0.5), i=5(4.0)
    # Incumbent: 3.0 -> 2.0 -> 0.5
    np.testing.assert_array_equal(iterations, [1, 3, 3, 4, 4, 5])
    np.testing.assert_array_equal(values, [3.0, 3.0, 2.0, 2.0, 0.5, 0.5])


def test_get_incumbent_objective_over_iterations_greater_is_better():
    objective = Objective("score", greater_is_better=True)
    objective_values = np.array([1.0, 3.0, 2.0, 5.0])
    feasible_mask = np.array([True, True, True, True])

    iterations, values = get_incumbent_objective_over_iterations(
        objective, objective_values, feasible_mask
    )
    # Incumbent: 1.0 -> 3.0 -> 5.0
    np.testing.assert_array_equal(iterations, [1, 2, 2, 4, 4, 4])
    np.testing.assert_array_equal(values, [1.0, 1.0, 3.0, 3.0, 5.0, 5.0])


def test_get_incumbent_objective_over_iterations_no_feasible():
    objective = Objective("loss", greater_is_better=False)
    objective_values = np.array([1.0, 2.0, 3.0])
    feasible_mask = np.array([False, False, False])

    iterations, values = get_incumbent_objective_over_iterations(
        objective, objective_values, feasible_mask
    )
    assert len(iterations) == 0
    assert len(values) == 0
