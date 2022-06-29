# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

try:
    from scipy.stats.qmc import Sobol
except ImportError as e:
    raise ImportError(
        "Unable to import SpaceFilling optimizer specific dependencies. "
        + "Make sure to install blackboxopt[space-fill]"
    ) from e
from blackboxopt.base import MultiObjectiveOptimizer, Objective, ParameterSpace
from blackboxopt.evaluation import EvaluationSpecification


class SpaceFilling(MultiObjectiveOptimizer):
    """Sobol sequence based, space filling optimizer.

    Args:
        search_space: The search space to optimize
        objectives: The objectives of the optimization
        seed: The sobol sequence is Owen scrambled and can be seeded for reproducibility
    """

    def __init__(
        self,
        search_space: ParameterSpace,
        objectives: List[Objective],
        seed: int = None,
    ) -> None:
        super().__init__(search_space=search_space, objectives=objectives, seed=seed)
        self.sobol = Sobol(d=len(self.search_space), scramble=True, seed=seed)

    def generate_evaluation_specification(self) -> EvaluationSpecification:
        vector = self.sobol.random().flatten()
        configuration = self.search_space.from_numerical(vector)
        return EvaluationSpecification(configuration=configuration)
