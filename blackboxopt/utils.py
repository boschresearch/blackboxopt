# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Sequence

from blackboxopt.base import Objective


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
