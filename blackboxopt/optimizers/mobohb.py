# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Optional

import numpy as np
from parameterspace import ParameterSpace

from blackboxopt import Objective

try:
    from blackboxopt.optimizers.staged.hyperband import create_hyperband_iteration
    from blackboxopt.optimizers.staged.iteration import Datum
    from blackboxopt.optimizers.staged.mobohb import Sampler, argsort_nondominated
    from blackboxopt.optimizers.staged.optimizer import (
        MultiObjectiveStagedIterationOptimizer,
    )
except ImportError as e:
    raise ImportError(
        "Unable to import MO-BOHB optimizer specific dependencies. "
        + "Make sure to install blackboxopt[mobohb]"
    ) from e


def _nondominated_promotion(data: List[Datum], num_configs: int) -> list:
    """Promote configurations to the next stage by multi-objective nondominated rank.

    Args:
        data: List with all successful evaluations for this stage. All failed
            configurations have already been removed.
        num_configs: Maximum number of configurations to be promoted.

    Returns:
        List of the config_keys to be evaluated on the next higher fidelity.
    """
    if not data:
        return []

    losses = np.stack([d.losses for d in data])
    n = min(num_configs, len(data))
    selected = argsort_nondominated(losses)[:n]
    return [data[i].config_key for i in selected]


class MOBOHB(MultiObjectiveStagedIterationOptimizer):
    def __init__(
        self,
        search_space: ParameterSpace,
        objectives: List[Objective],
        min_fidelity: float,
        max_fidelity: float,
        num_iterations: int,
        eta: float = 3.0,
        top_n_percent: int = 15,
        min_samples_in_model: Optional[int] = None,
        num_samples: int = 64,
        random_fraction: float = 1 / 3,
        bandwidth_factor: float = 3.0,
        min_bandwidth: float = 1e-3,
        seed: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Multi-objective BOHB Optimizer.

        MO-BOHB extends BOHB to multiple objectives by replacing its single-objective,
        scalar-loss based good/bad split with a multi-objective one: configurations are
        ranked by nondominated (Pareto) front and, within the boundary front, by
        crowding distance (NSGA-II style). The same ranking drives the successive
        halving promotion between stages. The Bayesian optimization component (kernel
        density estimators and expected improvement sampling) is reused from BOHB
        unchanged.

        Args:
            search_space: [description]
            objectives: The objectives of the optimization.
            min_fidelity: The smallest fidelity value that is still meaningful.
                Must be strictly greater than zero!
            max_fidelity: The largest fidelity value used during the optimization.
                Must not be smaller than `min_fidelity`.
            num_iterations: The number of iterations that the optimizer will run.
            eta: Scaling parameter to control the aggressiveness of Hyperband's racing.
            top_n_percent: Determines the percentile of configurations that will be
                used as training data for the kernel density estimator of the good
                configurations, e.g. if set to 10 the best 10% configurations (by
                nondominated rank) will be considered for training.
            min_samples_in_model: Minimum number of datapoints needed to fit a model.
            num_samples: Number of samples drawn to optimize EI via sampling.
            random_fraction: Fraction of random configurations returned.
            bandwidth_factor: Widens the bandwidth for contiuous parameters for
                proposed points to optimize EI
            min_bandwidth: to keep diversity, even when all (good) samples have the
                same value for one of the parameters, a minimum bandwidth
                (reasonable default: 1e-3) is used instead of zero.
            seed: [description]
            logger: [description]
        """
        if min_samples_in_model is None:
            min_samples_in_model = 3 * len(search_space)

        self.min_fidelity = min_fidelity
        self.max_fidelity = max_fidelity
        self.eta = eta

        self.config_sampler = Sampler(
            search_space=search_space,
            objectives=objectives,
            min_samples_in_model=min_samples_in_model,
            top_n_percent=top_n_percent,
            num_samples=num_samples,
            random_fraction=random_fraction,
            bandwidth_factor=bandwidth_factor,
            min_bandwidth=min_bandwidth,
            seed=seed,
        )

        super().__init__(
            search_space=search_space,
            objectives=objectives,
            num_iterations=num_iterations,
            seed=seed,
            logger=logger,
        )

    def _create_new_iteration(self, iteration_index):
        """Optimizer specific way to create a new
        `blackboxopt.optimizer.utils.staged_iteration.StagedIteration` object
        """
        return create_hyperband_iteration(
            iteration_index=iteration_index,
            min_fidelity=self.min_fidelity,
            max_fidelity=self.max_fidelity,
            eta=self.eta,
            config_sampler=self.config_sampler,
            objectives=self.objectives,
            logger=self.logger,
            config_promotion_function=_nondominated_promotion,
        )
