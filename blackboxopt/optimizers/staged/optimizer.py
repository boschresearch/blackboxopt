# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import abc
import logging
from typing import Dict, Iterable, List, Union

from parameterspace import ParameterSpace

from blackboxopt import (
    Evaluation,
    EvaluationSpecification,
    Objective,
    OptimizationComplete,
    OptimizerNotReady,
)
from blackboxopt.base import (
    SingleObjectiveOptimizer,
    call_functions_with_evaluations_and_collect_errors,
)
from blackboxopt.optimizers.staged.iteration import StagedIteration


def _validate_optimizer_info_id(evaluation: Evaluation):
    if evaluation.optimizer_info.get("id") is None:
        raise ValueError("Optimizer info is missing id.")


class StagedIterationOptimizer(SingleObjectiveOptimizer):
    def __init__(
        self,
        search_space: ParameterSpace,
        objective: Objective,
        num_iterations: int,
        seed: int = None,
        logger: logging.Logger = None,
    ):
        """Base class for optimizers using iterations that compare configurations at
        different fidelities and race them in stages, like Hyperband or BOHB.

        Args:
            search_space: [description]
            objective: [description]
            num_iterations: The number of iterations that the optimizer will run.
            seed: [description]
            logger: [description]
        """
        super().__init__(search_space=search_space, objective=objective, seed=seed)
        self.logger = logging.getLogger("blackboxopt") if logger is None else logger
        self.num_iterations = num_iterations
        self.iterations: List[StagedIteration] = []
        self.evaluation_uuid_to_iteration: Dict[str, int] = {}
        self.pending_configurations: Dict[str, EvaluationSpecification] = {}

    def report(self, evaluations: Union[Evaluation, Iterable[Evaluation]]) -> None:
        _evals = [evaluations] if isinstance(evaluations, Evaluation) else evaluations

        call_functions_with_evaluations_and_collect_errors(
            [super().report, _validate_optimizer_info_id, self._report],
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
