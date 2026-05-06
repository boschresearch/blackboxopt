# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

"""Example demonstrating the objective_over_iteration visualization.

Generates 10 random evaluations with one objective and one constraint, then
plots objective value vs. iteration with feasibility markers and an incumbent
line.
"""

import numpy as np

import blackboxopt as bbo
from blackboxopt.visualizations.visualizer import Visualizer


def main():
    rng = np.random.default_rng(42)

    evaluations = [
        bbo.Evaluation(
            objectives={"loss": float(rng.uniform(0.1, 5.0))},
            constraints={"c1": float(rng.uniform(-1.0, 1.0))},
            configuration={"x": float(rng.uniform(-1, 1))},
            optimizer_info={},
            user_info={},
            settings={"fidelity": 1.0},
        )
        for _ in range(10)
    ]

    objective = bbo.Objective("loss", greater_is_better=False)
    viz = Visualizer(evaluations, objective)

    # Constraint c1 must be >= 0
    return viz.objective_over_iteration(constraint_bounds={"c1": (0.0, None)})
    


if __name__ == "__main__":
    main().show()
