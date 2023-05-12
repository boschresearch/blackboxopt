# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import pickle
from itertools import compress
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import parameterspace as ps

from blackboxopt.base import Objective
from blackboxopt.evaluation import Evaluation


def get_loss_vector(
    known_objectives: Sequence[Objective],
    reported_objectives: Dict[str, Optional[float]],
    none_replacement: float = float("NaN"),
) -> List[float]:
    """Convert reported objectives into a vector of known objectives.

    Args:
        known_objectives: A sequence of objectives with names and directions
            (whether greate is better). The order of the objectives dictates the order
            of the returned loss values.
        reported_objectives: A dictionary with the objective value for each of the known
            objectives' names.
        none_replacement: The value to use for missing objective values that are `None`

    Returns:
        A list of loss values.
    """
    losses = []
    for objective in known_objectives:
        objective_value = reported_objectives[objective.name]
        if objective_value is None:
            losses.append(none_replacement)
        elif objective.greater_is_better:
            losses.append(-1.0 * objective_value)
        else:
            losses.append(objective_value)

    return losses


def mask_pareto_efficient(losses: np.ndarray):
    """For a given array of objective values where lower values are considered better
    and the dimensions are samples x objectives, return a mask that is `True` for all
    pareto efficient values.

    NOTE: The result marks multiple occurrences of the same point all as pareto
    efficient.
    """
    is_efficient = np.ones(losses.shape[0], dtype=bool)
    for i, c in enumerate(losses):
        if not is_efficient[i]:
            continue

        # Keep any point with a lower cost or when they are the same
        efficient = np.any(losses[is_efficient] < c, axis=1)
        duplicates = np.all(losses[is_efficient] == c, axis=1)
        is_efficient[is_efficient] = np.logical_or(efficient, duplicates)

    return is_efficient


def filter_pareto_efficient(
    evaluations: List[Evaluation], objectives: List[Objective]
) -> List[Evaluation]:
    """Filter pareto efficient evaluations with respect to given objectives."""
    losses = np.array(
        [
            get_loss_vector(
                known_objectives=objectives, reported_objectives=e.objectives
            )
            for e in evaluations
        ]
    )

    pareto_efficient_mask = mask_pareto_efficient(losses)

    return list(compress(evaluations, pareto_efficient_mask))


def sort_evaluations(evaluations: Iterable[Evaluation]) -> Iterable[Evaluation]:
    return sorted(
        evaluations,
        key=lambda e: hashlib.md5(
            pickle.dumps(
                [
                    sorted(e.configuration.items()),
                    sorted(e.objectives.items()),
                    sorted(e.settings.items()),
                    sorted(e.constraints.items()) if e.constraints else {},
                    sorted(e.context.items()) if e.context else {},
                ]
            )
        ).hexdigest(),
    )


def save_study_as_json(
    search_space: ps.ParameterSpace,
    objectives: List[Objective],
    evaluations: List[Evaluation],
    json_file_path: os.PathLike,
    overwrite: bool = False,
):
    """Save space, objectives and evaluations as json at `json_file_path`."""
    _file_path = Path(json_file_path)
    if not _file_path.parent.exists():
        raise IOError(
            f"The parent directory for {_file_path} does not exist, please create it."
        )
    if _file_path.exists() and not overwrite:
        raise IOError(f"{_file_path} exists and overwrite is False")

    with open(_file_path, "w", encoding="UTF-8") as fh:
        json.dump(
            {
                "search_space": search_space.to_dict(),
                "objectives": [o.__dict__ for o in objectives],
                "evaluations": [e.__dict__ for e in evaluations],
            },
            fh,
        )


def load_study_from_json(
    json_file_path: os.PathLike,
) -> Tuple[ps.ParameterSpace, List[Objective], List[Evaluation]]:
    """Load space, objectives and evaluations from a given `json_file_path`."""
    with open(json_file_path, "r", encoding="UTF-8") as fh:
        study = json.load(fh)

    search_space = ps.ParameterSpace.from_dict(study["search_space"])
    objectives = [Objective(**o) for o in study["objectives"]]
    evaluations = [Evaluation(**e) for e in study["evaluations"]]

    return search_space, objectives, evaluations


def save_study_as_pickle(
    search_space: ps.ParameterSpace,
    objectives: List[Objective],
    evaluations: List[Evaluation],
    pickle_file_path: os.PathLike,
    overwrite: bool = False,
):
    """Save space, objectives and evaluations as pickle at `pickle_file_path`."""
    _file_path = Path(pickle_file_path)
    if not _file_path.parent.exists():
        raise IOError(
            f"The parent directory for {_file_path} does not exist, please create it."
        )
    if _file_path.exists() and not overwrite:
        raise IOError(f"{_file_path} exists and overwrite is False")

    with open(_file_path, "wb") as fh:
        pickle.dump(
            {
                "search_space": search_space,
                "objectives": objectives,
                "evaluations": evaluations,
            },
            fh,
        )


def load_study_from_pickle(
    pickle_file_path: os.PathLike,
) -> Tuple[ps.ParameterSpace, List[Objective], List[Evaluation]]:
    """Load space, objectives and evaluations from a given `pickle_file_path`."""
    with open(pickle_file_path, "rb") as fh:
        study = pickle.load(fh)

    return study["search_space"], study["objectives"], study["evaluations"]
