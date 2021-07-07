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
    from blackboxopt.optimizers.staged.configuration_sampler import RandomSearchSampler
    from blackboxopt.optimizers.staged.hyperband import create_hyperband_iteration
    from blackboxopt.optimizers.staged.iteration import StagedIteration
    from blackboxopt.optimizers.staged.optimizer import StagedIterationOptimizer
except ImportError as e:
    raise ImportError(
        "Unable to import Hyperband optimizer specific dependencies. "
        + "Make sure to install blackboxopt[hyperband]"
    ) from e


class Hyperband(StagedIterationOptimizer):
    def __init__(
        self,
        search_space: SearchSpace,
        objective: Objective,
        min_fidelity: float,
        max_fidelity: float,
        num_iterations: int,
        eta: float = 3.0,
        seed: int = None,
        logger: logging.Logger = None,
    ):
        """Implementation of Hyperband as proposed in

        Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2016).
        Hyperband: A novel bandit-based approach to hyperparameter optimization.
        arXiv preprint arXiv:1603.06560.

        Args:
            search_space: [description]
            objective: [description]
            min_fidelity: The smallest fidelity value that is still meaningful.
                Must be strictly greater than zero!
            max_fidelity: The largest fidelity value used during the optimization.
                Must not be smaller than `min_fidelity`
            num_iterations: [description]
            eta: Scaling parameter to control the aggressiveness of Hyperband's racing.
            seed: [description]
            logger: [description]
        """
        self.config_sampler = RandomSearchSampler(search_space)
        self.min_fidelity = min_fidelity
        self.max_fidelity = max_fidelity
        self.eta = eta

        super().__init__(
            search_space=search_space,
            objective=objective,
            num_iterations=num_iterations,
            seed=seed,
            logger=logger,
        )

    def _create_new_iteration(self, iteration_index: int) -> StagedIteration:
        """Optimizer specific way to create a new
        `blackboxopt.optimizer.staged.iteration.StagedIteration` object
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
