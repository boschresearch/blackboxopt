# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import itertools
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io._html
import scipy.stats as sps

from blackboxopt import Evaluation, Objective
from blackboxopt.utils import get_loss_vector
from blackboxopt.visualizations import utils

QUALITATIVE_COLORS = px.colors.qualitative.G10

plotly.io._html.to_html = utils.patch_plotly_io_to_html(plotly.io._html.to_html)


class NoSuccessfulEvaluationsError(ValueError):
    pass


def evaluations_to_df(evaluations: List[Evaluation]) -> pd.DataFrame:
    """Convert evaluations into multi index dataframe.

    The evaluations will be casted to dictionaries which will be normalized.
    The keys of the dicts will be used as secondary column index. Evaluations
    with one or more missing objective-value will be dropped.

    Example:

    ```
    Evaluation(objectives={'loss_1': 1.0, 'loss_2': -0.0}, stacktrace=None, ...)
    ```

    Will be transformed into:

    |    objectives   | stacktrace | ... |  <- "group" index
    | loss_1 | loss_2 | stacktrace | ... |  <- "field" index
    | ------ | ------ | ---------- | --- |
    |    1.0 |   -0.0 | None       | ... |
    """
    if not evaluations or len(evaluations) == 0:
        raise NoSuccessfulEvaluationsError

    # Filter out e.g. EvaluationSpecifications which might be passed into
    evaluations = [e for e in evaluations if isinstance(e, Evaluation)]

    # Transform to dicts, filter out evaluations with missing objectives
    evaluation_dicts = [e.__dict__ for e in evaluations if not e.any_objective_none]

    if len(evaluation_dicts) == 0:
        raise NoSuccessfulEvaluationsError

    df = pd.DataFrame(evaluation_dicts)

    # Flatten json/dict columns into single multi-index dataframe
    dfs_expanded = []
    for column in df.columns:
        # Normalize json columns keep original column for non-json columns
        try:
            df_temp = pd.json_normalize(df[column], errors="ignore", max_level=0)
        except AttributeError:
            df_temp = df[[column]]

        # Use keys of dicts as second level of column index
        df_temp.columns = pd.MultiIndex.from_product(
            [[column], df_temp.columns], names=["group", "field"]
        )
        # Drop empty columns
        df_temp = df_temp.dropna(axis=1, how="all")

        dfs_expanded.append(df_temp)

    df = pd.concat(dfs_expanded, join="outer", axis=1)

    # Parse datetime columns
    date_columns = [c for c in df.columns if "unixtime" in str(c)]
    df[date_columns] = df[date_columns].apply(pd.to_datetime, unit="s")

    # Calculate duration in seconds
    df["duration", "duration"] = (
        df["finished_unixtime", "finished_unixtime"]
        - df["created_unixtime", "created_unixtime"]
    )

    return df


def create_hover_information(sections: dict) -> Tuple[str, List]:
    """
    Create a [hovertemplate](https://plotly.com/python/reference/pie/#pie-hovertemplate)
    which is used to render hover hints in plotly charts.

    The data for the chart hovertext has to be provided as `custom_data` attribute to
    the chart and can be e.g. a list of column names.

    One oddness is, that in the template the columns can't be referenced by name, but
    only by index. That's why it is important to have the same ordering in the template
    as in the `custom_data` and the reason why this is done together in one function.

    Args:
        sections: Sections to render. The kyeys will show up as the section titles,
            values are expected to be a list of column names to be rendered under
            the section. E.g.: { "info": ["Objective #1", "Objective #2", "fidelity"] }

    Returns:
        (plotly hover template, data column names)
    """
    template = ""
    idx = 0
    for section, columns in sections.items():
        template += f"<br><b>{section.replace('_', ' ').title()}</b><br>"
        for column in columns:
            template += f"{column}: %{{customdata[{idx}]}}<br>"
            idx += 1
    template += "<extra></extra>"

    data_columns: list = sum(sections.values(), [])

    return template, data_columns


def _prepare_for_multi_objective_visualization(
    df: pd.DataFrame, objectives: Sequence[Objective]
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    # Unify potentially different objective directions according to greater_is_better
    loss_matrix = np.array(
        [
            get_loss_vector(known_objectives=objectives, reported_objectives=ovs)
            for ovs in df["objectives"].to_dict("records")
        ]
    )
    df["pareto efficient", "pareto efficient"] = utils.mask_pareto_efficient(
        loss_matrix
    )

    # Map column names (from field index) with "sections" shown in the mouse-over text
    # TODO: Handle case, where e.g. a column in user_info produces naming collision with
    #       a column from another group-index
    hover_sections: Dict[str, list] = {
        "info": [o.name for o in objectives]
        + ["pareto efficient", "created_unixtime", "duration"]
    }
    for section in ["configuration", "user_info", "settings", "optimizer_info"]:
        if section in df.columns:
            hover_sections[section] = list(df[section].columns)

    # Drop group index for visualizing in plotly
    df.columns = df.columns.droplevel("group")

    # Formatting for nicer output
    df["duration"] = df["duration"].round("s").astype("str")
    df["created_unixtime"] = df["created_unixtime"].dt.round("1s")

    return df, hover_sections


def multi_objective_visualization(
    evaluations: List[Evaluation], objectives: Sequence[Objective]
):
    if not evaluations:
        raise NoSuccessfulEvaluationsError

    # Prepare dataframe for visualization
    df = evaluations_to_df(evaluations)
    df, hover_sections = _prepare_for_multi_objective_visualization(df, objectives)

    # Create hover template and list of corresponding dataframe column names
    hover_template, hover_data_columns = create_hover_information(hover_sections)

    fig = px.scatter_matrix(
        df,
        dimensions=[o.name for o in objectives],
        color="pareto efficient",
        color_discrete_sequence={
            False: QUALITATIVE_COLORS[5],
            True: QUALITATIVE_COLORS[2],
        },
        title="[BBO] Scatter matrix of multi objective values",
        custom_data=hover_data_columns,
    )
    fig.update_traces(
        diagonal_visible=False,
        hovertemplate=hover_template,
        marker=dict(line=dict(width=1, color="white")),
    )
    return fig


class Visualizer:
    def __init__(self, evaluations: List[Evaluation], objective: Objective):
        times_finished = np.array(utils.get_times_finished(evaluations))
        if len(times_finished) == 0 or np.isnan(times_finished).all():
            raise NoSuccessfulEvaluationsError()

        self.objective = objective

        self.times = times_finished - utils.get_t0(evaluations)
        sort_idx = np.argsort(self.times)
        self.times = self.times[sort_idx]
        self.evaluations = [evaluations[i] for i in sort_idx]
        self.objective_values = np.array(
            utils.get_objective_values(self.evaluations), dtype=float
        )
        self.durations = np.array(utils.get_durations(self.evaluations))
        self.fidelities = np.array(utils.get_fidelities(self.evaluations))
        self.configs = utils.get_configs(self.evaluations)

        all_keys = [set(c.keys()) for c in self.configs]
        self.all_config_keys: List[str] = sorted(set().union(*all_keys))
        self.all_fidelities: List[float] = sorted(set(self.fidelities))

        tmp_dict: Dict["str", Union[np.ndarray, List[str]]] = {}
        for key in self.all_config_keys:
            tmp_dict[key] = np.array([c.get(key, "N/A") for c in self.configs])
        tmp_dict["fidelity"] = ["%3.2e" % f for f in self.fidelities]

        self.info_dicts = [
            {
                self.objective.name: "%3.2e" % self.objective_values[i],
                "duration": str(datetime.timedelta(seconds=int(self.durations[i])))
                if np.isfinite(self.durations[i])
                else "N/A",
                "fidelity": "%3.2e" % self.fidelities[i],
            }
            for i in range(self.objective_values.shape[0])
        ]

        self.optimizer_info_dicts = [e["optimizer_info"] for e in self.evaluations]

    def objective_over_time(self, x_range=None, y_range=None, log_x=False, log_y=False):
        colors = px.colors.qualitative.G10
        fig = go.Figure()

        for i, f in enumerate(self.all_fidelities):

            mask = self.fidelities == f

            hover_texts = utils.get_hover_texts(
                self.info_dicts, self.optimizer_info_dicts, self.configs, mask
            )

            fig.add_scatter(
                x=self.times[mask],
                y=self.objective_values[mask],
                name="%3.2e" % f,
                mode="markers",
                marker=dict(color=colors[i]),
                legendgroup=str(f),
                hovertemplate="%{text} <extra></extra>",
                text=hover_texts,
            )

            times, objectives = utils.get_incumbent_objective_over_time_single_fidelity(
                self.objective, self.objective_values, self.times, self.fidelities, f
            )
            fig.add_scatter(
                x=times,
                y=objectives,
                mode="lines",
                line=dict(color=colors[i]),
                legendgroup=str(f),
                showlegend=False,
                hoverinfo="none",
            )

        utils.add_plotly_buttons_for_logscale(fig)
        utils.plotly_set_axis(fig, x_range, y_range, log_x, log_y)
        fig.update_layout(
            title=f"[BBO] Reported {self.objective.name} over time",
            legend_title_text="Fidelity",
            xaxis_title="Time [s]",
            yaxis_title=self.objective.name,
        )

        return fig

    def objective_over_duration(
        self, x_range=None, y_range=None, log_x=False, log_y=False
    ):
        colors = px.colors.qualitative.G10
        fig = go.Figure()

        for i, f in enumerate(self.all_fidelities):

            mask = self.fidelities == f

            hover_texts = utils.get_hover_texts(
                self.info_dicts, self.optimizer_info_dicts, self.configs, mask
            )

            fig.add_scatter(
                x=self.durations[mask],
                y=self.objective_values[mask],
                name="%3.2e" % f,
                mode="markers",
                marker=dict(color=colors[i]),
                legendgroup=str(f),
                hovertemplate="%{text} <extra></extra>",
                text=hover_texts,
            )

        utils.add_plotly_buttons_for_logscale(fig)
        utils.plotly_set_axis(fig, x_range, y_range, log_x, log_y)
        fig.update_layout(
            title=f"[BBO] Reported {self.objective.name} over duration",
            legend_title_text="Fidelity",
            xaxis_title="Duration [s]",
            yaxis_title=self.objective.name,
        )

        return fig

    def cdf_objective_values(self, x_range=None, log_x=False):

        colors = px.colors.qualitative.G10
        fig = go.Figure()

        for i, f in enumerate(self.all_fidelities):

            mask = self.fidelities == f
            x, y = utils.get_cdf_x_and_y(self.objective_values[mask])

            fig.add_scatter(
                x=x,
                y=y,
                name="%3.2e" % f,
                mode="lines",
                marker=dict(color=colors[i]),
                legendgroup=str(f),
            )

        utils.add_plotly_buttons_for_logscale(fig)
        utils.plotly_set_axis(fig, x_range, None, log_x)
        fig.update_layout(
            title=f"[BBO] CDF of {self.objective.name} by fidelity",
            legend_title_text="Fidelity",
            xaxis_title=self.objective.name,
            yaxis_title="CDF",
        )

        return fig

    def cdf_durations(self, x_range=None, log_x=False):

        colors = px.colors.qualitative.G10
        fig = go.Figure()

        for i, f in enumerate(self.all_fidelities):

            mask = self.fidelities == f
            x, y = utils.get_cdf_x_and_y(self.durations[mask])

            fig.add_scatter(
                x=x,
                y=y,
                name="%3.2e" % f,
                mode="lines",
                marker=dict(color=colors[i]),
                legendgroup=str(f),
            )

        utils.add_plotly_buttons_for_logscale(fig)
        utils.plotly_set_axis(fig, x_range, None, log_x)
        fig.update_layout(
            title="[BBO] CDF of durations by fidelity",
            legend_title_text="Fidelity",
            xaxis_title="Duration [s]",
            yaxis_title="CDF",
        )

        return fig

    def correlation_coefficients(self):
        matrix = utils.get_objective_values_matrix(self.evaluations)
        fidelities = sorted(list(set(self.fidelities)))

        corr_coeffs = np.full([len(fidelities), len(fidelities)], np.nan)
        p_values = np.full([len(fidelities), len(fidelities)], np.nan)
        num_samples = np.full([len(fidelities), len(fidelities)], 0)
        for i, j in itertools.combinations(range(len(fidelities)), 2):

            row_mask = np.all(np.isfinite(matrix[:, [i, j]]), axis=1)
            v1, v2 = matrix[row_mask][:, [i, j]].T
            res = sps.spearmanr(v1, v2)
            corr_coeffs[i, j] = res.correlation
            p_values[i, j] = res.pvalue
            num_samples[i, j] = len(v1)

        fig = go.Figure()

        hovertext = [
            [
                "<b>Correlation for %.2e and %.2e</b><br />"
                % (fidelities[i], fidelities[j])
                + "ρ:  %.3f<br />" % (corr_coeffs[i, j])
                + "p value: %.2e<br />" % (p_values[i, j])
                + "N samples: %i" % num_samples[i, j]
                for i in range(len(fidelities) - 1)
            ]
            for j in range(1, len(fidelities))
        ]

        fig.add_trace(
            go.Heatmap(
                z=corr_coeffs[:-1, 1:].T,
                y=["%3.2e" % f for f in fidelities[1:]],
                x=["%3.2e" % f for f in fidelities[:-1]],
                text=hovertext,
                hovertemplate="%{text}<extra></extra>",
                colorscale="viridis",
            )
        )
        fig.update_xaxes(type="category")
        fig.update_yaxes(type="category")

        fig.update_layout(
            title=f"[BBO] {self.objective.name} rank correlation for configurations "
            + "across fidelities",
            xaxis_title="Fidelity",
            yaxis_title="Fidelity",
        )

        return fig
