# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np

from blackboxopt import Evaluation, EvaluationSpecification, Objective
from blackboxopt.optimizers.staged.configuration_sampler import (
    StagedIterationConfigurationSampler,
)


@dataclass
class Datum:
    """Small container for bookkeeping only."""

    config_key: Tuple[int, int, int]
    status: str
    loss: float = float("NaN")


class StagedIteration:
    def __init__(
        self,
        iteration: int,
        num_configs: List[int],
        fidelities: List[float],
        config_sampler: StagedIterationConfigurationSampler,
        config_promotion_function: Callable,
        objective: Objective,
        logger: logging.Logger = None,
    ):
        """Base class for iterations that compare configurations at different
        fidelities and race them as in SuccessiveHalving or Hyperband.

        Args:
            iteration: Index of this iteration.
            num_configs: Number of configurations in each stage.
            fidelities: The fidelity for each stage. Must have the same length as
                `num_configs'.
            config_sampler: Configuration Sampler object that suggests a new
                configuration for evaluation given a fidelity.
            config_promotion_function: Function that decides which configurations are
                promoted. Check
                `blackboxopt.optimizers.utils.staged_iteration.greedy_promotion` for
                the signature.
            objective: The objective of the optimization.
            logger: A standard logger to which some debug output might be written.
        """
        assert len(fidelities) == len(
            num_configs
        ), "Please specify the number of configuration and the fidelities."
        self.logger = logging.getLogger("blackboxopt") if logger is None else logger
        self.iteration = iteration
        self.fidelities = fidelities
        self.num_configs = num_configs
        self.config_sampler = config_sampler
        self.config_promotion_function = config_promotion_function
        self.objective = objective
        self.current_stage = 0
        self.evaluation_data: List[List[Datum]] = [[]]
        self.eval_specs: Dict[Tuple[int, int, int], EvaluationSpecification] = {}
        self.pending_evaluations: Dict[UUID, int] = {}
        self.finished = False

    def generate_evaluation_specification(self) -> Optional[EvaluationSpecification]:
        """Pick the next evaluation specification with a budget i.e. fidelity to run.

        Returns:
            [description]
        """
        if self.finished:
            return None

        # try to find a queued entry first
        for i, d in enumerate(self.evaluation_data[self.current_stage]):
            if d.status == "QUEUED":
                es = copy.deepcopy(self.eval_specs[d.config_key])
                es.settings["fidelity"] = self.fidelities[self.current_stage]
                d.status = "RUNNING"
                self.pending_evaluations[es.optimizer_info["id"]] = i
                return es

        # sample a new configuration if there are empty slots to be filled
        if (
            len(self.evaluation_data[self.current_stage])
            < self.num_configs[self.current_stage]
        ):
            conf_key = (
                self.iteration,
                self.current_stage,
                len(self.evaluation_data[self.current_stage]),
            )
            conf, opt_info = self.config_sampler.sample_configuration()
            opt_info.update({"configuration_key": conf_key, "id": str(uuid4())})
            self.eval_specs[conf_key] = EvaluationSpecification(
                configuration=conf, settings={}, optimizer_info=opt_info
            )
            self.evaluation_data[self.current_stage].append(Datum(conf_key, "QUEUED"))
            # To understand recursion, you first must understand recursion :)
            return self.generate_evaluation_specification()

        # at this point there are pending evaluations and this iteration has to wait
        return None

    def digest_evaluation(
        self, evaluation_specificiation_id: UUID, evaluation: Evaluation
    ):
        """Registers the result of an evaluation.

        Args:
            evaluation_specificiation_id: [description]
            evaluation: [description]
        """
        self.config_sampler.digest_evaluation(evaluation)
        i = self.pending_evaluations.pop(evaluation_specificiation_id)
        d = self.evaluation_data[self.current_stage][i]
        d.status = "FINISHED" if not evaluation.all_objectives_none else "CRASHED"
        objective_value = evaluation.objectives[self.objective.name]
        if objective_value is not None:
            d.loss = (
                -objective_value
                if self.objective.greater_is_better
                else objective_value
            )

        # quick check if all configurations have finished yet
        if len(self.evaluation_data[self.current_stage]) == self.num_configs[
            self.current_stage
        ] and all(
            [
                e.status in ["FINISHED", "CRASHED"]
                for e in self.evaluation_data[self.current_stage]
            ]
        ):
            self._progress_to_next_stage()

    def _progress_to_next_stage(self):
        """Implements logic to promote configurations to the next stage."""
        # filter out crashed configurations
        data = [
            d for d in self.evaluation_data[self.current_stage] if np.isfinite(d.loss)
        ]
        self.current_stage += 1
        if self.current_stage == len(self.num_configs):
            self.finished = True
            return

        config_keys = self.config_promotion_function(
            data, self.num_configs[self.current_stage]
        )
        self.logger.debug(
            "Iteration %i: Advancing configurations %s to stage %i.",
            self.iteration,
            str(config_keys),
            self.current_stage,
        )
        self.evaluation_data.append(
            [Datum(config_key, "QUEUED") for config_key in config_keys]
        )
