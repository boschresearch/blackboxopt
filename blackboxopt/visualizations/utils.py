# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import numpy as np

from blackboxopt import Evaluation


def get_t0(evaluations: List[Evaluation]):
    return min([e.created_unixtime for e in evaluations])


def get_objective_values(evaluations: List[Evaluation]):
    return [
        list(e.objectives.values())[0] if not e.any_objective_none else float("NaN")
        for e in evaluations
    ]


def get_times_finished(evaluations: List[Evaluation]):
    return [
        e.finished_unixtime if not e.any_objective_none else float("NaN")
        for e in evaluations
    ]


def get_durations(evaluations: List[Evaluation]):
    return [
        e.finished_unixtime - e.created_unixtime
        if not e.any_objective_none
        else float("NaN")
        for e in evaluations
    ]


def get_fidelities(evaluations: List[Evaluation]):
    return [e.settings.get("fidelity", -1.0) for e in evaluations]


def get_bohb_ids(evaluations: List[Evaluation]):
    return [e.optimizer_info["configuration_key"] for e in evaluations]


def get_configs(evaluations: List[Evaluation]):
    return [e.configuration for e in evaluations]


def get_loss_matrix(evaluations: List[Evaluation]):

    # need to turn config dicts into tuples to use set
    config_tuples = {tuple(e.configuration.values()) for e in evaluations}
    configs = list(config_tuples)

    fidelities = sorted(set(get_fidelities(evaluations)))

    matrix = np.full([len(configs), len(fidelities)], np.nan)

    for e in evaluations:
        row = configs.index(tuple(e.configuration.values()))
        column = fidelities.index(e.settings["fidelity"])
        try:
            matrix[row, column] = list(e.objectives.values())[0]
        except Exception:
            continue

    return matrix


def get_incumbent_loss_over_time_single_fidelity(
    losses, times, fidelities, target_fidelity
):
    """Filter for results with given target fidelity and generate incumbent trace."""
    # filter out fidelity and take minimum of losses
    idx = np.logical_and(fidelities == target_fidelity, np.isfinite(losses))
    _times = times[idx]
    _losses = np.minimum.accumulate(losses[idx])
    # get unique loss values and sort their indices (to be in chronological order)
    _, idx = np.unique(_losses, return_index=True)
    idx.sort()
    # find losses
    _losses = _losses[idx]
    _times = _times[idx]
    # add steps where a new incumbent was found
    _times = np.repeat(_times, 2)[1:]
    _losses = np.repeat(_losses, 2)[:-1]
    # append best value for largest time to extend the lines
    _times = np.concatenate([_times, np.nanmax(times, keepdims=True)])
    _losses = np.concatenate([_losses, _losses[-1:]])
    return _times, _losses


def dict_to_hovertext(d):
    strings = [f"{key}: {d[key]}" for key in sorted(d.keys())]
    return "<br />".join(strings)


def get_hover_texts(info_dicts, optimizer_info_dicts, config_dicts, mask):
    info_texts = [dict_to_hovertext(info_dicts[j]) for j, k in enumerate(mask) if k]
    optimizer_texts = [
        dict_to_hovertext(optimizer_info_dicts[j]) for j, k in enumerate(mask) if k
    ]
    config_texts = [dict_to_hovertext(config_dicts[j]) for j, k in enumerate(mask) if k]
    hover_texts = [
        f"<b>Info</b><br />{i}<br /><br />"
        + f"<b>Optimizer Info</b><br />{o}<br /><br />"
        + f"<b>Configuration</b><br />{c}"
        for i, o, c in zip(info_texts, optimizer_texts, config_texts)
    ]
    return hover_texts


def get_cdf_x_and_y(values):
    values = np.sort(values)
    y = np.linspace(0, 1, len(values), endpoint=True)
    return values, y


def plotly_set_axis(fig, x_range=None, y_range=None, log_x=False, log_y=False):
    if x_range is not None:
        fig.update_xaxes(range=x_range)
    if y_range is not None:
        fig.update_yaxes(range=y_range)

    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")


def add_plotly_buttons_for_logscale(fig):
    updatemenus = [
        dict(
            type="buttons",
            direction="down",
            buttons=list(
                [
                    dict(
                        args=[{"yaxis.type": "linear"}],
                        label="Linear y",
                        method="relayout",
                    ),
                    dict(
                        args=[{"yaxis.type": "log"}], label="Log y", method="relayout"
                    ),
                    dict(
                        args=[{"xaxis.type": "linear"}],
                        label="Linear x",
                        method="relayout",
                    ),
                    dict(
                        args=[{"xaxis.type": "log"}], label="Log x", method="relayout"
                    ),
                ]
            ),
        ),
    ]

    fig.update_layout(updatemenus=updatemenus)


def mask_pareto_efficient(costs: np.ndarray):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] <= c, axis=1
            )  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient
