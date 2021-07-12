# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import itertools
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as sps

from blackboxopt import Evaluation
from blackboxopt.visualizations import utils


class NoSuccessfulEvaluationsError(ValueError):
    pass


def create_hover_information(sections: dict) -> Tuple[str, List]:
    """
    Create a [hovertemplate](https://plotly.com/python/reference/pie/#pie-hovertemplate)
    which is used to render hover hints in plotly charts.

    The data for the chart hovertext has to be provided as `custom_data` attribute to
    the chart and can be e.g a list of column names.

    One oddness is, that in the template the columns can't be referenced by name, but
    only by index. That's why it is important to have the same ordering in the template
    as in the `custom_data` and the reason why this is done together in this function.

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


def multi_objective_visualization(evaluations: List[Evaluation]):
    if not evaluations:
        raise NoSuccessfulEvaluationsError

    evaluation_dicts = [e.__dict__ for e in evaluations]

    df = pd.DataFrame(evaluation_dicts)

    # Used to structure the "sections" in the hover text, puts "info" section on top
    hover_sections: Dict[str, list] = {
        "info": [],
        "configuration": [],
        "optimizer_info": [],
    }

    # Expand and join json/dict columns
    df = df.dropna(axis=0, subset=["objectives"]).reset_index(drop=True)
    dfs_expanded = []
    expand_column_names = ["configuration", "settings", "optimizer_info"]
    for column in expand_column_names:
        df_temp = pd.json_normalize(df[column], errors="ignore", max_level=0)
        dfs_expanded.append(df_temp)
        if column in hover_sections:
            hover_sections[column] = sorted(df_temp.columns)
    df = df.drop(expand_column_names, axis=1)
    df = df.join(dfs_expanded, how="outer")

    df_temp = pd.json_normalize(df["objectives"], errors="ignore", max_level=0)
    objective_columns = list(df_temp.columns)
    df = df.join(df_temp, how="outer")
    df = df.drop("objectives", axis=1)

    # Expand and join loss list column
    df = df.dropna(axis=0, subset=objective_columns).reset_index(drop=True)

    if len(df) <= 0:
        raise NoSuccessfulEvaluationsError

    # Calculate pareto info
    df["pareto efficient"] = utils.mask_pareto_efficient(df[objective_columns].values)

    # Cast datetimes (shown in hover text)
    date_columns = ["finished_unixtime", "created_unixtime"]
    df[date_columns] = df[date_columns].apply(pd.to_datetime, unit="s")
    df["duration"] = df["finished_unixtime"] - df["created_unixtime"]
    # Unclutter by rounding to seconds
    df["duration"] = df["duration"].round("s").astype("str")
    df["created_unixtime"] = df["created_unixtime"].dt.round("1s")

    # Add important data to "info" section in hovertext and remove "objectives-section"
    hover_sections["info"] += objective_columns
    if "fidelity" in df.columns:
        hover_sections["info"] += ["fidelity"]
    hover_sections["info"] += ["pareto efficient", "created_unixtime", "duration"]

    # Create hover template and list of corresponding dataframe column names
    hover_template, hover_data_columns = create_hover_information(hover_sections)

    fig = px.scatter_matrix(
        df,
        dimensions=objective_columns,
        color="pareto efficient",
        color_discrete_sequence={
            False: px.colors.qualitative.G10[5],
            True: px.colors.qualitative.G10[2],
        },
        title="Scatter matrix of multi objective losses",
        custom_data=hover_data_columns,
    )
    fig.update_traces(
        diagonal_visible=False,
        hovertemplate=hover_template,
        marker=dict(line=dict(width=1, color="white")),
    )
    return fig


class Visualizer:
    def __init__(self, evaluations: List[Evaluation]):
        times_finished = np.array(utils.get_times_finished(evaluations))
        if len(times_finished) == 0 or np.isnan(times_finished).all():
            raise NoSuccessfulEvaluationsError()

        self.times = times_finished - utils.get_t0(evaluations)
        sort_idx = np.argsort(self.times)
        self.times = self.times[sort_idx]
        self.evaluations = [evaluations[i] for i in sort_idx]
        self.losses = np.array(
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
                "loss": "%3.2e" % self.losses[i],
                "duration": str(datetime.timedelta(seconds=int(self.durations[i])))
                if np.isfinite(self.durations[i])
                else "N/A",
                "fidelity": "%3.2e" % self.fidelities[i],
            }
            for i in range(self.losses.shape[0])
        ]

        self.optimizer_info_dicts = [e["optimizer_info"] for e in self.evaluations]

    def loss_over_time(self, x_range=None, y_range=None, log_x=False, log_y=False):
        colors = px.colors.qualitative.G10
        fig = go.Figure()

        for i, f in enumerate(self.all_fidelities):

            mask = self.fidelities == f

            hover_texts = utils.get_hover_texts(
                self.info_dicts, self.optimizer_info_dicts, self.configs, mask
            )

            fig.add_scatter(
                x=self.times[mask],
                y=self.losses[mask],
                name="%3.2e" % f,
                mode="markers",
                marker=dict(color=colors[i]),
                legendgroup=str(f),
                hovertemplate="%{text} <extra></extra>",
                text=hover_texts,
            )

            times, losses = utils.get_incumbent_loss_over_time_single_fidelity(
                self.losses, self.times, self.fidelities, f
            )
            fig.add_scatter(
                x=times,
                y=losses,
                mode="lines",
                line=dict(color=colors[i]),
                legendgroup=str(f),
                showlegend=False,
                hoverinfo="none",
            )

        utils.add_plotly_buttons_for_logscale(fig)
        utils.plotly_set_axis(fig, x_range, y_range, log_x, log_y)
        fig.update_layout(
            title="Reported loss over time",
            legend_title_text="Fidelity",
            xaxis_title="Time [s]",
            yaxis_title="Loss",
        )

        return fig

    def loss_over_duration(self, x_range=None, y_range=None, log_x=False, log_y=False):
        colors = px.colors.qualitative.G10
        fig = go.Figure()

        for i, f in enumerate(self.all_fidelities):

            mask = self.fidelities == f

            hover_texts = utils.get_hover_texts(
                self.info_dicts, self.optimizer_info_dicts, self.configs, mask
            )

            fig.add_scatter(
                x=self.durations[mask],
                y=self.losses[mask],
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
            title="Reported loss over duration",
            legend_title_text="Fidelity",
            xaxis_title="Duration [s]",
            yaxis_title="Loss",
        )

        return fig

    def cdf_losses(self, x_range=None, log_x=False):

        colors = px.colors.qualitative.G10
        fig = go.Figure()

        for i, f in enumerate(self.all_fidelities):

            mask = self.fidelities == f
            x, y = utils.get_cdf_x_and_y(self.losses[mask])

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
            title="CDF of losses by fidelity",
            legend_title_text="Fidelity",
            xaxis_title="Loss",
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
            title="CDF of durations by fidelity",
            legend_title_text="Fidelity",
            xaxis_title="Duration [s]",
            yaxis_title="CDF",
        )

        return fig

    def correlation_coefficients(self):
        matrix = utils.get_loss_matrix(self.evaluations)
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
            title="Loss rank correlation for configurations across fidelities",
            xaxis_title="Fidelity",
            yaxis_title="Fidelity",
        )

        return fig
