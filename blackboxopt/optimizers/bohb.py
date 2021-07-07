# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0
#
# This source code is derived from HpBandSter 0.7.4
#  https://github.com/automl/HpBandSter
# Copyright (c) 2017-2018, ML4AAD, licensed under the BSD-3 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
# Changes include:
#     - integration into the blackboxopt API
#     - docstrings and type hints

import logging

from parameterspace.base import SearchSpace

from blackboxopt import Objective

try:
    from blackboxopt.optimizers.staged.bohb import Sampler as BOHBSampler
    from blackboxopt.optimizers.staged.hyperband import create_hyperband_iteration
    from blackboxopt.optimizers.staged.optimizer import StagedIterationOptimizer
except ImportError as e:
    raise ImportError(
        "Unable to import BOHB optimizer specific dependencies. "
        + "Make sure to install blackboxopt[bohb]"
    ) from e


class BOHB(StagedIterationOptimizer):
    def __init__(
        self,
        search_space: SearchSpace,
        objective: Objective,
        min_fidelity: float,
        max_fidelity: float,
        num_iterations: int,
        eta: float = 3.0,
        top_n_percent: int = 15,
        min_samples_in_model: int = None,
        num_samples: int = 64,
        random_fraction: float = 1 / 3,
        bandwidth_factor: float = 3.0,
        min_bandwidth: float = 1e-3,
        seed: int = None,
        logger: logging.Logger = None,
    ):
        """BOHB Optimizer.

        BOHB performs robust and efficient hyperparameter optimization
        at scale by combining the speed of Hyperband searches with the
        guidance and guarantees of convergence of Bayesian
        Optimization. Instead of sampling new configurations at random,
        BOHB uses kernel density estimators to select promising candidates.

        For reference:
        ```
        @InProceedings{falkner-icml-18,
            title =     {{BOHB}: Robust and Efficient Hyperparameter Optimization at
                Scale},
            author =    {Falkner, Stefan and Klein, Aaron and Hutter, Frank},
            booktitle = {Proceedings of the 35th International Conference on Machine
                Learning},
            pages =     {1436--1445},
            year =      {2018},
        }
        ```

        Args:
            search_space: [description]
            objective: [description]
            min_fidelity: The smallest fidelity value that is still meaningful.
                Must be strictly greater than zero!
            max_fidelity: The largest fidelity value used during the optimization.
                Must not be smaller than `min_fidelity`.
            num_iterations: The number of iterations that the optimizer will run.
            eta: Scaling parameter to control the aggressiveness of Hyperband's racing.
            top_n_percent: Determines the percentile of configurations that will be
                used as training data for the kernel density estimator of the good
                configuration, e.g if set to 10 the best 10% configurations will be
                considered for training.
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

        self.config_sampler = BOHBSampler(
            search_space=search_space,
            objective=objective,
            min_samples_in_model=min_samples_in_model,
            top_n_percent=top_n_percent,
            num_samples=num_samples,
            random_fraction=random_fraction,
            bandwidth_factor=bandwidth_factor,
            min_bandwidth=min_bandwidth,
        )

        super().__init__(
            search_space=search_space,
            objective=objective,
            num_iterations=num_iterations,
            seed=seed,
            logger=logger,
        )

    def _create_new_iteration(self, iteration_index):
        """Optimizer specific way to create a new
        `blackboxopt.optimizer.utils.staged_iteration.StagedIteration` object
        """
        return create_hyperband_iteration(
            iteration_index,
            self.min_fidelity,
            self.max_fidelity,
            self.eta,
            self.config_sampler,
            self.objective,
            self.logger,
        )
