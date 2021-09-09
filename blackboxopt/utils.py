# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Iterable, List, Optional, Tuple

from blackboxopt.base import EvaluationsError, Objective
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


def report_multiple_evaluations_individually(
    report_func: Callable[[Evaluation], None], evaluations: Iterable[Evaluation]
):
    evaluations_with_errors = []
    for evaluation in evaluations:
        try:
            if evaluation.optimizer_info.get("id") is None:
                raise ValueError("Optimizer info is missing id.")
            report_func(evaluation)
        except Exception as e:
            evaluations_with_errors.append((evaluation, e))

    if evaluations_with_errors:
        raise EvaluationsError(evaluations_with_errors)
