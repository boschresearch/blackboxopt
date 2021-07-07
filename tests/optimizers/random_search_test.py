# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import parameterspace as ps
import pytest

from blackboxopt import Objective, OptimizationComplete
from blackboxopt.optimizers.random_search import RandomSearch
from blackboxopt.optimizers.testing import ALL_REFERENCE_TESTS

MAX_STEPS = 5

SPACE = ps.ParameterSpace()
SPACE.add(ps.ContinuousParameter("p1", [0, 1]))


@pytest.mark.parametrize("reference_test", ALL_REFERENCE_TESTS)
def test_all_reference_tests(reference_test):
    reference_test(RandomSearch, dict(max_steps=MAX_STEPS))


def test_max_steps_randomsearch():
    opt = RandomSearch(SPACE, [Objective("loss", False)], max_steps=MAX_STEPS)

    for _ in range(MAX_STEPS):
        opt.get_evaluation_specification()

    with pytest.raises(OptimizationComplete):
        opt.get_evaluation_specification()

    with pytest.raises(OptimizationComplete):
        opt.get_evaluation_specification()
