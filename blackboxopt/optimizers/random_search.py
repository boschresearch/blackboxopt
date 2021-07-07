# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from blackboxopt import (
    EvaluationSpecification,
    Objective,
    OptimizationComplete,
    SearchSpace,
)
from blackboxopt.base import MultiObjectiveOptimizer


class RandomSearch(MultiObjectiveOptimizer):
    def __init__(
        self,
        search_space: SearchSpace,
        objectives: List[Objective],
        max_steps: int,
        seed: int = None,
    ) -> None:
        """Randomly sample up to `max_steps` configurations from the given search space.

        Args:
            search_space: Space to search
            objectives: The objectives of the optimization.
            max_steps: Max number of evaluation specifications the optimizer generates
                before raising `OptimizationComplete`
            seed: Optional number to seed the random number generator with.
                Defaults to None.
        """
        super().__init__(search_space=search_space, objectives=objectives, seed=seed)

        self.max_steps: int = max_steps
        self.n_steps: int = 0

    def get_evaluation_specification(self) -> EvaluationSpecification:
        """[summary]

        Raises:
            OptimizationComplete: Raised if the optimizer's `max_steps` are reached.

        Returns:
            [description]
        """
        if self.n_steps >= self.max_steps:
            raise OptimizationComplete()

        eval_spec = EvaluationSpecification(
            configuration=self.search_space.sample(),
            settings={},
            optimizer_info={"step": self.n_steps},
        )
        self.n_steps += 1

        return eval_spec
