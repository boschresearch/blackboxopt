# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from blackboxopt import Objective
from blackboxopt.utils import get_loss_vector


@pytest.mark.parametrize(
    "known,reported,expected",
    [
        (
            [Objective("score", True), Objective("loss", False)],
            {"loss": 2.0, "score": 1.0},
            [-1.0, 2.0],
        ),
        (
            [Objective("score", True), Objective("loss", False)],
            {"loss": 2.0, "score": None},
            [np.nan, 2.0],
        ),
    ],
)
def test_get_loss_vector(known, reported, expected):
    loss_vector = get_loss_vector(known_objectives=known, reported_objectives=reported)
    np.testing.assert_array_equal(loss_vector, np.array(expected))


def test_get_loss_vector_with_custom_none_replacement():
    known = [Objective("score", True), Objective("loss", False)]

    reported = {"loss": 2.0, "score": None}
    expected = [np.inf, 2.0]

    loss_vector = get_loss_vector(
        known_objectives=known,
        reported_objectives=reported,
        none_replacement=float("Inf"),
    )
    np.testing.assert_array_equal(loss_vector, np.array(expected))
