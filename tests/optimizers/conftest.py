# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import parameterspace as ps
import pytest

from blackboxopt import Evaluation

objective_name = "objective"


@pytest.fixture
def evaluations():
    return [
        Evaluation(
            configuration={"x0": 0.57, "x1": True, "x2": "small", "cp": 0.3, "fp": 0.5},
            objectives={objective_name: 0.64},
        ),
        Evaluation(
            configuration={"x0": 1.84, "x1": False, "x2": "large", "fp": 0.5},
            objectives={objective_name: 0.57},
        ),
        Evaluation(
            configuration={"x0": 2.14, "x1": False, "x2": "medium", "fp": 0.5},
            objectives={objective_name: -0.37},
        ),
        Evaluation(
            configuration={"x0": 1.14, "x1": True, "x2": "large", "cp": 0.8, "fp": 0.5},
            objectives={objective_name: None},
        ),
        Evaluation(
            configuration={"x0": 3.0, "x1": False, "x2": "small", "fp": 0.5},
            objectives={objective_name: 0.0},
        ),
    ]


@pytest.fixture
def search_space():
    space = ps.ParameterSpace()

    # Basic parameters
    space.add(ps.ContinuousParameter("x0", (0.5, 3)))
    space.add(ps.CategoricalParameter("x1", (True, False)))
    space.add(ps.OrdinalParameter("x2", ["small", "medium", "large"]))

    # Parameter with a condition
    space.add(ps.ContinuousParameter("cp", (-1.0, 1.0)), lambda x1: x1)

    # Fixed parameter
    space.add(ps.ContinuousParameter("fp", (0.0, 1.0)))
    space.fix(fp=0.5)

    return space
