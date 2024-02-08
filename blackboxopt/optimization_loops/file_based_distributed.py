# Copyright (c) 2024 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import glob
import json
import logging
import pprint
import random
import time
from os import PathLike
from pathlib import Path
from typing import Any, Callable, List, Optional, Union
from uuid import uuid4

from blackboxopt import (
    Evaluation,
    EvaluationSpecification,
    Objective,
    OptimizationComplete,
    OptimizerNotReady,
)
from blackboxopt import logger as default_logger
from blackboxopt.base import MultiObjectiveOptimizer, SingleObjectiveOptimizer
from blackboxopt.optimization_loops.utils import (
    evaluation_function_wrapper,
    init_max_evaluations_with_limit_logging,
)

EVALUATION_RESULT_FILE_NAME_PREFIX = "eval_result_"
EVALUATION_SPECIFICATION_FILE_NAME_PREFIX = "eval_spec_"


def run_optimization_loop(
    optimizer: Union[SingleObjectiveOptimizer, MultiObjectiveOptimizer],
    target_directory: PathLike,
    timeout_s: float = float("inf"),
    max_evaluations: Optional[int] = None,
    proposal_queue_size: int = 1,
    pre_evaluation_callback: Optional[Callable[[EvaluationSpecification], Any]] = None,
    post_evaluation_callback: Optional[Callable[[Evaluation], Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Evaluation]:
    """
    Args:
        optimizer: The optimizer instance to use for specification generation and
            evaluation ingestion.
        target_directory: The directory where the evaluation specifications and results
            are stored.
        timeout_s: The maximum time in seconds to run the optimization loop.
        max_evaluations: The maximum number of evaluations to perform.
        proposal_queue_size: The number of proposals to keep in the queue, i.e. the size
            of the evaluation batch for parallel/batch evaluation.
        logger: The logger to use for logging. If `None`, the default logger is used.
    """
    if logger is None:
        logger = default_logger

    target_directory = Path(target_directory)
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
    while True:
        evaluations_to_report = glob.glob(
            str(target_directory / f"{EVALUATION_RESULT_FILE_NAME_PREFIX}*.json")
        )
        for eval_result_file_path in evaluations_to_report:
            with open(eval_result_file_path, "r", encoding="utf-8") as fh:
                evaluation = Evaluation(**json.load(fh))
            Path(eval_result_file_path).unlink()
            if not evaluation.objectives and evaluation.stacktrace:
                evaluation.objectives = {o.name: None for o in objectives}

            logger.info(
                "Reporting this evaluation result to the optimizer:\n%s",
                pprint.pformat(evaluation.to_dict(), compact=True),
            )
            if post_evaluation_callback:
                post_evaluation_callback(evaluation)

            optimizer.report(evaluation)
            evaluations.append(evaluation)

        if time.time() - start >= timeout_s or len(evaluations) >= _max_evaluations:
            return evaluations

        current_proposals = glob.glob(
            str(target_directory / f"{EVALUATION_SPECIFICATION_FILE_NAME_PREFIX}*.json")
        )
        while len(current_proposals) < proposal_queue_size:
            eval_spec_id = str(uuid4())
            try:
                eval_spec = optimizer.generate_evaluation_specification()
                eval_spec.optimizer_info["eval_spec_id"] = eval_spec_id

                logger.info(
                    "The optimizer proposed this evaluation specification:\n%s",
                    pprint.pformat(eval_spec.to_dict(), compact=True),
                )
                if pre_evaluation_callback:
                    pre_evaluation_callback(eval_spec)

                with open(
                    target_directory / f"eval_spec_{eval_spec_id}.json",
                    "w",
                    encoding="utf-8",
                ) as fh:
                    json.dump(eval_spec.to_dict(), fh)
                current_proposals = glob.glob(
                    str(target_directory / "eval_spec_*.json")
                )
            except OptimizerNotReady:
                logger.info("Optimizer is not ready yet, retrying in two seconds")
                time.sleep(2.0)
            except OptimizationComplete:
                logger.info("Optimization is complete")
                return evaluations

        time.sleep(0.5)


def evaluate_specifications(
    target_directory: PathLike,
    evaluation_function: Callable[[EvaluationSpecification], Evaluation],
    objectives: List[Objective],
    timeout_s: float = float("inf"),
    max_evaluations: Optional[int] = None,
    catch_exceptions_from_evaluation_function: bool = False,
    pre_evaluation_callback: Optional[Callable[[EvaluationSpecification], Any]] = None,
    post_evaluation_callback: Optional[Callable[[Evaluation], Any]] = None,
    logger: Optional[logging.Logger] = None,
):
    """Evaluate specifications from the target directory until the `timeout_s` or
    `max_evaluations` is reached.

    Args:
        target_directory: The directory where the evaluation specifications and results
            are stored.
        evaluation_function: The function that evaluates the evaluation specification.
        objectives: The objectives reported in the evaluation function. This is used to
            specify None values for objectives in case of an error.
        timeout_s: The maximum time in seconds to run the optimization loop.
        max_evaluations: The maximum number of evaluations to perform on this agent.
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
        logger: The logger to use for logging. If `None`, the default logger is used.
    """
    if logger is None:
        logger = default_logger

    target_directory = Path(target_directory)

    _max_evaluations = init_max_evaluations_with_limit_logging(
        max_evaluations=max_evaluations, timeout_s=timeout_s, logger=logger
    )

    start = time.time()
    num_evaluations = 0
    while time.time() - start < timeout_s and num_evaluations < _max_evaluations:
        current_proposals = glob.glob(
            str(target_directory / f"{EVALUATION_SPECIFICATION_FILE_NAME_PREFIX}*.json")
        )
        if not current_proposals:
            logger.info("No proposals found, retrying in one second.")
            time.sleep(1.0)
            continue

        # Just pick one random proposal instead of iterating over all available ones, to
        # reduce the risk of a race condition when two or more agents are running in
        # parallel and trying to evaluate the same proposals.
        eval_spec_path = random.choice(current_proposals)
        try:
            with open(eval_spec_path, "r", encoding="utf-8") as fh:
                eval_spec = EvaluationSpecification(**json.load(fh))
            # Allow missing, in case the proposal was already evaluated by another agent
        except FileNotFoundError:
            logging.warning(
                f"Could not read evaluation specification from {eval_spec_path}, "
                + "it was likely already evaluated elsewhere."
            )
            continue
        Path(eval_spec_path).unlink(missing_ok=True)

        logger.info(
            "The optimizer proposed this evaluation specification:\n%s",
            pprint.pformat(eval_spec.to_dict(), compact=True),
        )
        if pre_evaluation_callback:
            pre_evaluation_callback(eval_spec)

        evaluation = evaluation_function_wrapper(
            evaluation_function=evaluation_function,
            evaluation_specification=eval_spec,
            logger=logger,
            objectives=objectives,
            catch_exceptions_from_evaluation_function=catch_exceptions_from_evaluation_function,
        )

        logger.info(
            "Reporting this evaluation result to the optimizer:\n%s",
            pprint.pformat(evaluation.to_dict(), compact=True),
        )
        if post_evaluation_callback:
            post_evaluation_callback(evaluation)

        with open(
            target_directory
            / (
                EVALUATION_RESULT_FILE_NAME_PREFIX
                + f"{eval_spec.optimizer_info['eval_spec_id']}.json"
            ),
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(evaluation.to_dict(), fh)
            num_evaluations += 1
