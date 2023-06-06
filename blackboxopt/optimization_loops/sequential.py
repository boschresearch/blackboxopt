# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import logging
import pprint
import time
from typing import Any, Callable, List, Optional, Union

from blackboxopt import (
    Evaluation,
    EvaluationSpecification,
    OptimizationComplete,
    OptimizerNotReady,
)
from blackboxopt import logger as default_logger
from blackboxopt.base import MultiObjectiveOptimizer, SingleObjectiveOptimizer
from blackboxopt.optimization_loops.utils import (
    evaluation_function_wrapper,
    init_max_evaluations_with_limit_logging,
)


def run_optimization_loop(
    optimizer: Union[SingleObjectiveOptimizer, MultiObjectiveOptimizer],
    evaluation_function: Callable[[EvaluationSpecification], Evaluation],
    timeout_s: float = float("inf"),
    max_evaluations: Optional[int] = None,
    catch_exceptions_from_evaluation_function: bool = False,
    pre_evaluation_callback: Optional[Callable[[EvaluationSpecification], Any]] = None,
    post_evaluation_callback: Optional[Callable[[Evaluation], Any]] = None,
    logger: Optional[logging.Logger] = None,
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
        catch_exceptions_from_evaluation_function: Whether to exit on an unhandled
            exception raised by the evaluation function or instead store their stack
            trace in the evaluation's `stacktrace` attribute. Set to True if there are
            spurious errors due to e.g. numerical instability that should not halt the
            optimization loop. For more details, see the wrapper that is used internally
            `blackboxopt.optimization_loops.utils.evaluation_function_wrapper`
        pre_evaluation_callback: Reference to a callable that is invoked before each
            evaluation and takes a `blackboxopt.EvaluationSpecification` as an argument.
        post_evaluation_callback: Reference to a callable that is invoked after each
            evaluation and takes a `blackboxopt.Evaluation` as an argument.
        logger: The logger to use for logging progress. Default: `blackboxopt.logger`

    Returns:
        List of evaluation specification and result for all evaluations.
    """
    if logger is None:
        logger = default_logger

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
            evaluation_specification = optimizer.generate_evaluation_specification()

            logger.info(
                "The optimizer proposed the following evaluation specification:\n%s",
                pprint.pformat(evaluation_specification.to_dict(), compact=True),
            )
            if pre_evaluation_callback:
                pre_evaluation_callback(evaluation_specification)

            evaluation = evaluation_function_wrapper(
                evaluation_function=evaluation_function,
                evaluation_specification=evaluation_specification,
                logger=logger,
                objectives=objectives,
                catch_exceptions_from_evaluation_function=catch_exceptions_from_evaluation_function,
            )

            logger.info(
                "Reporting the following evaluation result to the optimizer:\n%s",
                pprint.pformat(evaluation.to_dict(), compact=True),
            )
            if post_evaluation_callback:
                post_evaluation_callback(evaluation)

            optimizer.report(evaluation)
            evaluations.append(evaluation)

        except OptimizerNotReady:
            logger.info("Optimizer is not ready yet, retrying in two seconds")
            time.sleep(2)
            continue

        except OptimizationComplete:
            logger.info("Optimization is complete")
            return evaluations

    logger.info("Aborting optimization due to specified maximum evaluations or timeout")
    return evaluations
