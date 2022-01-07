# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging
import traceback
from typing import Callable, List

from blackboxopt import Evaluation, EvaluationSpecification, Objective
from blackboxopt.base import ObjectivesError, raise_on_unknown_or_incomplete


class EvaluationFunctionError(ValueError):
    """Raised on errors originating from the user defined evaluation function."""

    def __init__(self, evaluation_specification: EvaluationSpecification):
        self.message = (
            "An error occurred when attempting to call the user specified evaluation "
            "function with the specification below. Please check the cause of this "
            "exception in the output further up for the original stacktrace.\n"
            f"{evaluation_specification}"
        )
        self.evaluation_specification = evaluation_specification


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
    exit_on_unhandled_exception: bool,
    logger: logging.Logger,
) -> Evaluation:
    """Wrapper for evaluation functions. The evaluation result returned by the
    evaluation function is checked to contain all relevant objectives. An empty
    evaluation with a stacktrace is reported to the optimizer in case an unhandled
    Exception occurrs during the evaluation function call when
    `exit_on_unhandled_exception` is set to `False`, otherwise an
    `EvaluationFunctionError` is raised based on the original exception.
    """
    try:
        evaluation = evaluation_function(evaluation_specification)
    except Exception as e:
        if exit_on_unhandled_exception:
            raise EvaluationFunctionError from e

        stacktrace = traceback.format_exc()

        logger.warning("Report FAILURE due to unhandled error during evaluation")
        logger.debug(stacktrace)

        evaluation = evaluation_specification.create_evaluation(
            stacktrace=stacktrace, objectives={o.name: None for o in objectives}
        )

    raise_on_unknown_or_incomplete(
        exception=ObjectivesError,
        known=[o.name for o in objectives],
        reported=evaluation.objectives.keys(),
    )

    return evaluation
