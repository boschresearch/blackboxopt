# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import abc
from typing import Tuple

from parameterspace.base import SearchSpace

from blackboxopt import Evaluation


class StagedIterationConfigurationSampler:
    """Base class for sampling new configurations inside a StagedIterationOpitimzer."""

    @abc.abstractmethod
    def sample_configuration(self) -> Tuple[dict, dict]:
        """Pick the next configuration.

        Returns:
            The configuration to be evaluated,
            Additional information that will be added to the `optimizer_info` dict.
        """

    @abc.abstractmethod
    def digest_evaluation(self, evaluation: Evaluation):
        """Register the result of an evaluation."""


class RandomSearchSampler(StagedIterationConfigurationSampler):
    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space

    def sample_configuration(self) -> Tuple[dict, dict]:
        return self.search_space.sample(), {}

    def digest_evaluation(self, evaluation: Evaluation):
        """Random Search is stateless and does nothing with finished evaluations."""
