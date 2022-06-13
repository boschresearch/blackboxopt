# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

try:
    import numpy as np
    from scipy.stats.qmc import Sobol
except ImportError as e:
    raise ImportError(
        "Unable to import BOHB optimizer specific dependencies. "
        + "Make sure to install blackboxopt[space-fill]"
    ) from e
from blackboxopt.base import MultiObjectiveOptimizer, Objective, SearchSpace
from blackboxopt.evaluation import EvaluationSpecification


class SpaceFilling(MultiObjectiveOptimizer):
    """Initialize the optimizer with an optional seed for reproducibility.
    Args:
        search_space: The search space to optimize
        objectives: The objectives of the optimization
        seed: A seed for the optimizer, which is also used to re-seed the provided
            search space. Defaults to None.
    """

    def __init__(
        self, search_space: SearchSpace, objectives: List[Objective], seed: int = None
    ) -> None:
        super().__init__(search_space=search_space, objectives=objectives, seed=seed)
        self._rng = np.random.default_rng(self.seed)
        self.sobol = Sobol(d=len(self.search_space), scramble=True, seed=self._rng)

    def generate_evaluation_specification(self) -> EvaluationSpecification:
        vector = self.sobol.random().flatten()
        configuration = self.search_space.from_numerical(vector)
        return EvaluationSpecification(configuration=configuration)
