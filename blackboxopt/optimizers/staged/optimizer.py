# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import abc
import logging
from typing import Dict, Iterable, List, Optional, Union

from parameterspace import ParameterSpace

from blackboxopt import (
    Evaluation,
    EvaluationSpecification,
    Objective,
    OptimizationComplete,
    OptimizerNotReady,
)
from blackboxopt.base import (
    MultiObjectiveOptimizer,
    Optimizer,
    SingleObjectiveOptimizer,
    call_functions_with_evaluations_and_collect_errors,
)
from blackboxopt.optimizers.staged.iteration import StagedIteration


def _validate_optimizer_info_id(evaluation: Evaluation):
    if evaluation.optimizer_info.get("id") is None:
        raise ValueError("Optimizer info is missing id.")


class _StagedIterationScheduler(Optimizer):
    """Scheduling logic shared by the single- and multi-objective staged optimizers."""

    logger: logging.Logger
    num_iterations: int
    iterations: List[StagedIteration]
    evaluation_uuid_to_iteration: Dict[str, int]
    pending_configurations: Dict[str, EvaluationSpecification]

    def _init_staged_scheduler(
        self, num_iterations: int, logger: logging.Logger = None
    ) -> None:
        self.logger = logging.getLogger("blackboxopt") if logger is None else logger
        self.num_iterations = num_iterations
        self.iterations = []
        self.evaluation_uuid_to_iteration = {}
        self.pending_configurations = {}

    def report(self, evaluations: Union[Evaluation, Iterable[Evaluation]]) -> None:
        _evals = [evaluations] if isinstance(evaluations, Evaluation) else evaluations

        call_functions_with_evaluations_and_collect_errors(
            [super().report, _validate_optimizer_info_id, self._report],  # type: ignore[safe-super]
            _evals,
        )

    def _report(self, evaluation: Evaluation) -> None:
        evaluation_specification_id = evaluation.optimizer_info.get("id")
        self.pending_configurations.pop(str(evaluation_specification_id))
        idx = self.evaluation_uuid_to_iteration.pop(str(evaluation_specification_id))
        self.iterations[idx].digest_evaluation(evaluation_specification_id, evaluation)

    def generate_evaluation_specification(self) -> EvaluationSpecification:
        """Get next configuration and settings to evaluate.

        Raises:
            OptimizationComplete: When the optimization run is finished, e.g. when the
                budget has been exhausted.
            OptimizerNotReady: When the optimizer is not ready to propose a new
                evaluation specification.
        """
        # check if any of the already active iterations returns a configuration and
        # simply return that
        for idx, iteration in enumerate(self.iterations):
            es = iteration.generate_evaluation_specification()

            if es is not None:
                self.evaluation_uuid_to_iteration[str(es.optimizer_info["id"])] = idx
                self.pending_configurations[str(es.optimizer_info["id"])] = es
                return es

        # if that didn't work, check if there another iteration can be started and then
        # ask it for a configuration
        if len(self.iterations) < self.num_iterations:
            self.iterations.append(self._create_new_iteration(len(self.iterations)))
            es = self.iterations[-1].generate_evaluation_specification()
            self.evaluation_uuid_to_iteration[str(es.optimizer_info["id"])] = (
                len(self.iterations) - 1
            )
            self.pending_configurations[str(es.optimizer_info["id"])] = es
            return es

        # check if the optimization is already complete or whether the optimizer is
        # waiting for evaluation results -> raise corresponding error
        if all([iteration.finished for iteration in self.iterations]):
            raise OptimizationComplete

        raise OptimizerNotReady

    @abc.abstractmethod
    def _create_new_iteration(self, iteration_index):
        """Optimizer specific way to create a new
        `blackboxopt.optimizer.utils.staged_iteration.StagedIteration` object
        """


class SingleObjectiveStagedIterationOptimizer(
    _StagedIterationScheduler, SingleObjectiveOptimizer
):
    def __init__(
        self,
        search_space: ParameterSpace,
        objective: Objective,
        num_iterations: int,
        seed: Optional[int] = None,
        logger: logging.Logger = None,
    ):
        """Base class for single-objective optimizers using iterations that compare
        configurations at different fidelities and race them in stages, like Hyperband
        or BOHB.

        Args:
            search_space: Search space of the optimization problem.
            objective: Objective that is being optimized.
            num_iterations: The number of iterations that the optimizer will run.
            seed: Random seed for reproducibility
            logger: Optional logger.
        """
        super().__init__(search_space=search_space, objective=objective, seed=seed)
        self._init_staged_scheduler(num_iterations, logger)


class MultiObjectiveStagedIterationOptimizer(
    _StagedIterationScheduler, MultiObjectiveOptimizer
):
    def __init__(
        self,
        search_space: ParameterSpace,
        objectives: List[Objective],
        num_iterations: int,
        seed: Optional[int] = None,
        logger: logging.Logger = None,
    ):
        """Base class for multi-objective optimizers using iterations that compare
        configurations at different fidelities and race them in stages, like
        multi-objective BOHB.

        Args:
            search_space: Search space of the optimization problem.
            objectives: Objectives that are being optimized.
            num_iterations: The number of iterations that the optimizer will run.
            seed: Random seed for reproducibility
            logger: Optional logger.
        """
        super().__init__(search_space=search_space, objectives=objectives, seed=seed)
        self._init_staged_scheduler(num_iterations, logger)
