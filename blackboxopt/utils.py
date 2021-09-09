# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Iterable, List, Optional, Tuple

from blackboxopt.base import Objective
from blackboxopt.evaluation import Evaluation


def get_loss_vector(
    known_objectives: List[Objective], reported_objectives: Dict[str, Optional[float]]
) -> list:
    """Convert reported objectives into a vector of known objectives.

    Args:
        known_objectives: [description]
        reported_objectives: [description]

    Returns:
        [description]
    """
    losses = []
    for objective in known_objectives:
        objective_value = reported_objectives[objective.name]
        if objective_value is None:
            losses.append(float("NaN"))
        elif objective.greater_is_better:
            losses.append(-1.0 * objective_value)
        else:
            losses.append(objective_value)

    return losses


def filter_valid(
    evaluations: Iterable[Evaluation],
    evaluations_with_errors: List[Tuple[Evaluation, Exception]],
):
    invalid_evaluations = [evaluation for evaluation, _ in evaluations_with_errors]
    return [e for e in evaluations if e not in invalid_evaluations]
