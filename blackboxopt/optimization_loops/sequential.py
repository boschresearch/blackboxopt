# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from typing import Callable, List, Union

from blackboxopt import (
    Evaluation,
    EvaluationSpecification,
    OptimizationComplete,
    OptimizerNotReady,
)
from blackboxopt.base import MultiObjectiveOptimizer, SingleObjectiveOptimizer
from blackboxopt.optimization_loops.utils import (
    evaluation_function_wrapper,
    init_max_evaluations_with_limit_logging,
)


def run_optimization_loop(
    optimizer: Union[SingleObjectiveOptimizer, MultiObjectiveOptimizer],
    evaluation_function: Callable[[EvaluationSpecification], Evaluation],
    timeout_s: float = float("inf"),
    max_evaluations: int = None,
    logger: logging.Logger = None,
) -> List[Evaluation]:
    """Convenience wrapper for an optimization loop that sequentially fetches evaluation
    specifications until a given timeout or maximum number of evaluations is reached.

    This already handles signals from the optimizer in case there is no evaluation
    specification available yet.

    Args:
        optimizer: The blackboxopt optimizer to run.
        evaluation_function: The function that is called with configuration, settings
            and optimizer info dictionaries as arguments like provided by an evaluation
            specification.
            This is the function that encapsulates the actual execution of
            a parametrized experiment (e.g. ML model training) and should return a
            `blackboxopt.Evaluation` as a result.
        timeout_s: If given, the optimization loop will terminate after the first
            optimization step that exceeded the timeout (in seconds). Defaults to inf.
        max_evaluations: If given, the optimization loop will terminate after the given
            number of steps.
        logger: The logger to use for logging progress.

    Returns:
        List of evluation specification and result for all evaluations.
    """
    logger = logging.getLogger("blackboxopt") if logger is None else logger

    objectives = (
        optimizer.objectives
        if isinstance(optimizer, MultiObjectiveOptimizer)
        else [optimizer.objective]
    )
    evaluations: List[Evaluation] = []

    _max_evaluations = init_max_evaluations_with_limit_logging(
        max_evaluations=max_evaluations, timeout_s=timeout_s, logger=logger
    )

    start = time.time()
    num_evaluations = 0
    while time.time() - start < timeout_s and num_evaluations < _max_evaluations:
        num_evaluations += 1

        try:
            evaluation_specification = optimizer.get_evaluation_specification()
            logger.debug(f"Evaluating: {evaluation_specification}")

            evaluation = evaluation_function_wrapper(
                evaluation_function=evaluation_function,
                evaluation_specification=evaluation_specification,
                logger=logger,
                objectives=objectives,
            )
            optimizer.report(evaluation)
            evaluations.append(evaluation)
            logger.debug(f"Result: {evaluation}")

        except OptimizerNotReady:
            logger.info("Optimizer is not ready yet, retrying in two seconds")
            time.sleep(2)
            continue

        except OptimizationComplete:
            logger.info("Optimization is complete")
            return evaluations

    logger.info("Aborting optimization due to specified maximum evaluations or timeout")
    return evaluations
