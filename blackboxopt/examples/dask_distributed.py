# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import parameterspace as ps

try:
    import dask.distributed as dd
except ImportError:
    raise ImportError(
        "Unable to import Dask Distributed specific dependencies. "
        + "Make sure to install blackboxopt[dask]"
    )

from blackboxopt import Evaluation, EvaluationSpecification, Objective
from blackboxopt.optimization_loops.dask_distributed import (
    run_optimization_loop,
)
from blackboxopt.optimizers.random_search import RandomSearch


def evaluation_function(eval_spec: EvaluationSpecification) -> Evaluation:
    return eval_spec.create_evaluation(
        objectives={"loss": eval_spec.configuration["p1"] ** 2},
        user_info={"weather": "sunny"},
    )


if __name__ == "__main__":
    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("p1", (-1.0, 1.0)))
    optimizer = RandomSearch(
        space,
        [Objective("loss", greater_is_better=False)],
        max_steps=1000,
    )

    evaluations = run_optimization_loop(
        optimizer, evaluation_function, dd.Client(), max_evaluations=100
    )

    n_successes = len([e for e in evaluations if not e.all_objectives_none])
    print(f"Successfully evaluated {n_successes}/{len(evaluations)}")
