# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Sequence, Tuple, Union

import numpy as np
from parameterspace import ParameterSpace

from blackboxopt import Evaluation, Objective
from blackboxopt.optimizers.staged.kde_sampler import KDESampler


class Sampler(KDESampler):
    def __init__(
        self,
        search_space: ParameterSpace,
        objective: Objective,
        min_samples_in_model: int,
        top_n_percent: int,
        num_samples: int,
        random_fraction: float,
        bandwidth_factor: float,
        min_bandwidth: float,
        seed: Optional[int] = None,
        logger=None,
    ):
        """Single-objective BOHB kernel density estimator sampler.

        Args:
            search_space: ConfigurationSpace/ ParameterSpace object.
            objective: The objective of the optimization.
            min_samples_in_model: Minimum number of datapoints needed to fit a model.
            top_n_percent: Determines the percentile of configurations that will be used
                as training data for the kernel density estimator of the good
                configuration, e.g if set to 10 the best 10% configurations will be
                considered for training.
            num_samples: Number of samples drawn to optimize EI via sampling.
            random_fraction: Fraction of random configurations returned
            bandwidth_factor: Widens the bandwidth for contiuous parameters for
                proposed points to optimize EI
            min_bandwidth: To keep diversity, even when all (good) samples have the
                same value for one of the parameters, a minimum bandwidth
                (reasonable default: 1e-3) is used instead of zero.
            seed: A seed to make the sampler reproducible.
            logger: [description]
        """
        self.objective = objective
        super().__init__(
            search_space=search_space,
            min_samples_in_model=min_samples_in_model,
            top_n_percent=top_n_percent,
            num_samples=num_samples,
            random_fraction=random_fraction,
            bandwidth_factor=bandwidth_factor,
            min_bandwidth=min_bandwidth,
            seed=seed,
            logger=logger,
        )

    def _evaluation_to_loss(self, evaluation: Evaluation) -> Union[float, np.ndarray]:
        """Return the sign-corrected scalar loss, using `np.inf` for missing values."""
        objective_value = evaluation.objectives[self.objective.name]
        if objective_value is None:
            return np.inf
        return -objective_value if self.objective.greater_is_better else objective_value

    def _count_finite_losses(self, losses: Sequence[Union[float, np.ndarray]]) -> int:
        """Count evaluations with a finite loss, i.e. eligible for model building."""
        return int(np.isfinite(np.asarray(losses)).sum())

    def _good_bad_split(
        self, losses: np.ndarray, n_good: int, n_bad: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split configurations by ascending scalar loss."""
        idx = np.argsort(losses)
        return idx[:n_good], idx[n_good : n_good + n_bad]
