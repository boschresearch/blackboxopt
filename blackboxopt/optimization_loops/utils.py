# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging
import traceback
from typing import Callable, List

from blackboxopt import Evaluation, EvaluationSpecification, Objective
from blackboxopt.base import raise_on_unknown_or_incomplete_objectives


def init_max_evaluations_with_limit_logging(
    timeout_s: float, logger: logging.Logger, max_evaluations: int = None
) -> float:
    """[summary]

    Args:
        timeout_s: [description]
        logger: [description]
        max_evaluations: [description]

    Returns:
        [description]
    """
    if max_evaluations:
        logger.info(
            f"Starting optimization run until complete or {max_evaluations} evaluations"
        )
        return float(max_evaluations)

    if timeout_s == float("inf"):
        logger.info("Starting optimization run until complete")
    else:
        timeout_pretty = datetime.timedelta(seconds=timeout_s)
        logger.info(
            f"Starting optimization run until complete or {timeout_pretty} passed"
        )

    return float("inf")


def evaluation_function_wrapper(
    evaluation_function: Callable[[EvaluationSpecification], Evaluation],
    evaluation_specification: EvaluationSpecification,
    objectives: List[Objective],
    logger: logging.Logger,
) -> Evaluation:
    """Wrapper for evaluation functions. The evaluation result returned by the
    evaluation function is checked to contain all relevant objectives. An empty
    evaluation with a stacktrace is reported to the optiizer in case an unhandled
    Exception occurrs during the evaluation function call.
    """
    try:
        evaluation = evaluation_function(evaluation_specification)
    except Exception:
        stacktrace = traceback.format_exc()

        logger.warning("Report FAILURE due to unhandled error during evaluation")
        logger.debug(stacktrace)

        evaluation = evaluation_specification.get_evaluation(
            stacktrace=stacktrace, objectives={o.name: None for o in objectives}
        )

    raise_on_unknown_or_incomplete_objectives(
        known_objectives=objectives, reported_objectives=evaluation.objectives
    )

    return evaluation
