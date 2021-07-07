# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from typing import Callable, List, Set, Union

try:
    import dask.distributed as dd
except ImportError as e:
    raise ImportError(
        "Unable to import Dask Distributed specific dependencies. "
        + "Make sure to install blackboxopt[dask]"
    ) from e

from blackboxopt import (
    Evaluation,
    EvaluationSpecification,
    OptimizationComplete,
    OptimizerNotReady,
)
from blackboxopt.base import (
    MultiObjectiveOptimizer,
    Objective,
    SingleObjectiveOptimizer,
)
from blackboxopt.optimization_loops.utils import (
    evaluation_function_wrapper,
    init_max_evaluations_with_limit_logging,
)


class MinimalDaskScheduler:
    def __init__(
        self,
        dask_client: dd.Client,
        objectives: List[Objective],
        logger: logging.Logger,
    ):
        self.client = dask_client
        self.objectives = objectives
        self.logger = logger
        self._not_done_futures: Set = set()

    def shutdown(self):
        return self.client.shutdown()

    def has_capacity(self):
        idle = [len(task_id) == 0 for task_id in self.client.processing().values()]
        return sum(idle)

    def has_running_jobs(self):
        return len(self._not_done_futures) > 0

    def submit(
        self,
        eval_function: Callable[[EvaluationSpecification], Evaluation],
        eval_spec: EvaluationSpecification,
    ):
        f = self.client.submit(
            evaluation_function_wrapper,
            evaluation_function=eval_function,
            evaluation_specification=eval_spec,
            objectives=self.objectives,
            logger=self.logger,
        )
        f.bbo_eval_spec = eval_spec
        self._not_done_futures.add(f)

    def check_for_results(self, timeout_s: float = 5.0) -> List[Evaluation]:
        try:
            all_futures = dd.wait(
                self._not_done_futures, timeout=timeout_s, return_when="FIRST_COMPLETED"
            )

            return_values: List[Evaluation] = []
            for f in all_futures.done:

                if f.status == "error":
                    return_values.append(
                        Evaluation(
                            objectives={o.name: None for o in self.objectives},
                            stacktrace=str(f.traceback()),
                            **f.bbo_eval_spec
                        )
                    )
                else:
                    return_values.append(f.result())

            self._not_done_futures = all_futures.not_done
        except dd.TimeoutError:
            return_values = []

        return return_values


def run_optimization_loop(
    optimizer: Union[SingleObjectiveOptimizer, MultiObjectiveOptimizer],
    evaluation_function: Callable[[EvaluationSpecification], Evaluation],
    dask_client: dd.Client,
    timeout_s: float = float("inf"),
    max_evaluations: int = None,
    logger: logging.Logger = None,
) -> List[Evaluation]:
    """Convenience wrapper for an optimization loop that uses Dask to parallelize
    optimization until a given timeout or maximum number of evaluations is reached.

    This already handles signals from the optimizer in case there is no evaluation
    specification available yet.

    Args:
        optimizer: The blackboxopt optimizer to run.
        dask_client: A Dask Distributed client that is configured with workers.
        evaluation_function: The function that is called with configuration, settings
            and optimizer info dictionaries as arguments like provided by an evaluation
            specification.
            This is the function that encapsulates the actual execution of
            a parametrized experiment (e.g. ML model training) and should return a
            `blackboxopt.Evaluation` as a result.
        timeout_s: If given, the optimization loop will terminate after the first
            optimization step that exceeded the timeout (in seconds). Defaults to inf.
        max_evaluations: If given, the optimization loop will terminate after the given
            number of steps. Defaults to None.
        logger: The logger to use for logging progress. Defaults to None.

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

    dask_scheduler = MinimalDaskScheduler(
        dask_client=dask_client, objectives=objectives, logger=logger
    )

    _max_evaluations = init_max_evaluations_with_limit_logging(
        max_evaluations=max_evaluations, timeout_s=timeout_s, logger=logger
    )

    n_eval_specs = 0
    start = time.time()
    while time.time() - start < timeout_s and n_eval_specs < _max_evaluations:
        if dask_scheduler.has_capacity():
            try:
                eval_spec = optimizer.get_evaluation_specification()
                dask_scheduler.submit(evaluation_function, eval_spec)
                n_eval_specs += 1
                continue

            except OptimizerNotReady:
                logger.info("Optimizer is not ready yet; will retry after short pause.")

            except OptimizationComplete:
                logger.info("Optimization is complete")
                break

        for evaluation in dask_scheduler.check_for_results(timeout_s=20):
            optimizer.report_evaluation(evaluation)
            evaluations.append(evaluation)

    while dask_scheduler.has_running_jobs():
        for evaluation in dask_scheduler.check_for_results(timeout_s=20):
            optimizer.report_evaluation(evaluation)
            evaluations.append(evaluation)

    return evaluations
