# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import logging
import math

from blackboxopt.base import Objective
from blackboxopt.optimizers.staged.configuration_sampler import (
    StagedIterationConfigurationSampler,
)
from blackboxopt.optimizers.staged.iteration import StagedIteration
from blackboxopt.optimizers.staged.utils import greedy_promotion


# The following function was derived from HpBandSter 0.7.4
#  https://github.com/automl/HpBandSter
# Copyright (c) 2017-2018, ML4AAD, licensed under the BSD-3 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
# The following changes have been made:
# - code was part of a method, which was moved into this function
# - docstring and type hints for the arguments were added
# - integration into the local classes, e.g.
#   blackboxopt.optimizers.staged.iteration.StagetIteration
def create_hyperband_iteration(
    iteration_index: int,
    min_fidelity: float,
    max_fidelity: float,
    eta: float,
    config_sampler: StagedIterationConfigurationSampler,
    objective: Objective,
    logger: logging.Logger,
) -> StagedIteration:
    """Optimizer specific way to create a new
    `blackboxopt.optimizer.staged.iteration.StagedIteration` object
    """
    # 's_max + 1' in the paper
    max_num_stages = 1 + int(math.floor(math.log(max_fidelity / min_fidelity, eta)))
    # 's+1' in the paper
    num_stages = max_num_stages - (iteration_index % (max_num_stages))
    num_configs_first_stage = int(
        math.ceil((max_num_stages / num_stages) * eta ** (num_stages - 1))
    )
    num_configs_per_stage = [
        int(num_configs_first_stage // (eta ** i)) for i in range(num_stages)
    ]
    fidelities_per_stage = [
        max_fidelity / eta ** i for i in range(num_stages - 1, -1, -1)
    ]
    # Hyperband simple draws random configurations, and there is no additional
    # information that needs to be stored
    return StagedIteration(
        iteration_index,
        num_configs_per_stage,
        fidelities_per_stage,
        config_sampler,
        greedy_promotion,
        objective,
        logger=logger,
    )
