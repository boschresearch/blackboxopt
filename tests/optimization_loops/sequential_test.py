# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from blackboxopt.optimization_loops.sequential import run_optimization_loop
from blackboxopt.optimization_loops.testing import ALL_REFERENCE_TESTS


@pytest.mark.parametrize("reference_test", ALL_REFERENCE_TESTS)
def test_all_reference_tests(reference_test):
    reference_test(run_optimization_loop, {})
