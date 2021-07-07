# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import numpy as np

from blackboxopt.base import Objective
from blackboxopt.evaluation import Evaluation
from blackboxopt.optimizers.staged.iteration import Datum


def greedy_promotion(data: List[Datum], num_configs: int) -> list:
    """Promotes the best configurations to the next stage solely relying on the current
    loss.

    Args:
        data: List with all successful evaluations for this stage. All failed
            configurations have already been removed.
        num_configs: Maximum number of configurations to be promoted.

    Returns:
        List of the config_keys to be evaluated on the next higher fidelity. These must
        only include config_keys found in `data` und must also be of at most length
        `num_configs`. If fewer ids are returned, the remaining configurations for the
        next stage will be sampled using the `config_sample_function` of the
        staged_iteration.
    """
    losses = [d.loss for d in data]
    ranks = np.argsort(np.argsort(losses))
    n = min(num_configs, len(data))
    return [datum.config_key for rank, datum in zip(ranks, data) if rank < n]


def best_evaluation_at_highest_fidelity(
    evaluations: List[Evaluation],
    objective: Objective,
) -> Optional[Evaluation]:
    """From given list of evaluations, get the best in terms of minimal loss at the
    highest fidelity.

    Args:
        evaluations: [description]
        objective: [description]

    Returns:
        [description]
    """
    if not evaluations:
        return None

    successes = [
        evaluation
        for evaluation in evaluations
        if evaluation.objectives[objective.name] is not None
    ]
    if not successes:
        return None

    successful_fidelities = [
        evaluation.settings["fidelity"] for evaluation in successes
    ]
    if not successful_fidelities:
        return None

    max_successful_fidelities = max(successful_fidelities)
    successful_max_fidelity_evaluations = [
        evaluation
        for evaluation in successes
        if evaluation.settings["fidelity"] == max_successful_fidelities
    ]

    if not successful_max_fidelity_evaluations:
        return None

    sort_function = max if objective.greater_is_better else min
    best_evaluation = sort_function(
        successful_max_fidelity_evaluations, key=lambda e: e.objectives[objective.name]
    )

    return best_evaluation
