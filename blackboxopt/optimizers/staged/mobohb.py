# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from parameterspace import ParameterSpace
from pymoo.operators.survival.rank_and_crowding.metrics import (
    calc_crowding_distance,
)
from pymoo.util.nds.efficient_non_dominated_sort import (
    efficient_non_dominated_sort,
)

from blackboxopt import Evaluation, Objective
from blackboxopt.optimizers.staged.kde_sampler import KDESampler
from blackboxopt.utils import get_loss_vector


def argsort_nondominated(losses: np.ndarray) -> np.ndarray:
    """Order points from best to worst by nondominated rank, breaking ties by crowding.

    Args:
        losses: Array of shape samples x objectives where lower is better.

    Returns:
        An integer array of indices into `losses`, best first.
    """

    losses = np.asarray(losses, dtype=float)
    if losses.shape[0] == 0:
        return np.array([], dtype=int)

    ordered_indices = []
    for front in efficient_non_dominated_sort(losses):
        front = np.asarray(front)
        if len(front) == 1:
            ordered_indices.append(front)
            continue
        distances = calc_crowding_distance(losses[front])
        ordered_indices.append(front[np.argsort(-distances, kind="mergesort")])

    return np.concatenate(ordered_indices)


class Sampler(KDESampler):
    def __init__(
        self,
        search_space: ParameterSpace,
        objectives: List[Objective],
        min_samples_in_model: int,
        top_n_percent: int,
        num_samples: int,
        random_fraction: float,
        bandwidth_factor: float,
        min_bandwidth: float,
        seed: Optional[int] = None,
        logger=None,
    ):
        """Multi-objective variant of the BOHB KDE sampler.

        Args:
            search_space: ConfigurationSpace/ ParameterSpace object.
            objectives: The objectives of the optimization.
            min_samples_in_model: Minimum number of datapoints needed to fit a model.
            top_n_percent: Determines the percentile of configurations that will be used
                as training data for the kernel density estimator of the good
                configurations, e.g. if set to 10 the best 10% configurations (by
                nondominated rank) will be considered for training.
            num_samples: Number of samples drawn to optimize EI via sampling.
            random_fraction: Fraction of random configurations returned.
            bandwidth_factor: Widens the bandwidth for continuous parameters for
                proposed points to optimize EI.
            min_bandwidth: To keep diversity, even when all (good) samples have the
                same value for one of the parameters, a minimum bandwidth
                (reasonable default: 1e-3) is used instead of zero.
            seed: A seed to make the sampler reproducible.
            logger: [description]
        """
        self.objectives = objectives
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
        """Return the sign-corrected loss vector, using `inf` for missing values."""
        return np.asarray(
            get_loss_vector(
                known_objectives=self.objectives,
                reported_objectives=evaluation.objectives,
                none_replacement=float("inf"),
            )
        )

    def _count_finite_losses(self, losses: Sequence[Union[float, np.ndarray]]) -> int:
        """Count evaluations whose full loss vector is finite."""
        return int(np.all(np.isfinite(np.asarray(losses)), axis=1).sum())

    def _good_bad_split(
        self, losses: np.ndarray, n_good: int, n_bad: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split configurations by nondominated rank instead of a scalar loss."""
        order = argsort_nondominated(losses)
        return order[:n_good], order[n_good : n_good + n_bad]
