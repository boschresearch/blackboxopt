# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import logging
import time

import dask
import dask.distributed as dd
import parameterspace as ps
import pytest
from distributed.utils_test import (
    cleanup,  # noqa: F401
    client,  # noqa: F401
    cluster_fixture,  # noqa: F401
    loop,  # noqa: F401
    loop_in_thread,  # noqa: F401
)

from blackboxopt import Evaluation, EvaluationSpecification, Objective
from blackboxopt.optimization_loops.dask_distributed import (
    MinimalDaskScheduler,
    run_optimization_loop,
)
from blackboxopt.optimization_loops.testing import ALL_REFERENCE_TESTS
from blackboxopt.optimizers.random_search import RandomSearch


@pytest.mark.timeout(60)
@pytest.mark.parametrize("reference_test", ALL_REFERENCE_TESTS)
def test_all_reference_tests(reference_test, client):  # noqa: F811
    reference_test(run_optimization_loop, {"dask_client": client})


def evaluation_function_cause_worker_restart(
    eval_spec: EvaluationSpecification,
) -> Evaluation:
    should_exit = eval_spec.configuration["exit"]
    if should_exit:
        exit(0)
    else:
        return eval_spec.create_evaluation(objectives={"loss": 0.0})


def test_restarting_workers(tmpdir):
    # setup local cluster in processes (the fixture doesn't seem to work)
    dask.config.set({"distributed.scheduler.allowed-failures": 1})
    cluster = dd.LocalCluster(
        n_workers=1, threads_per_worker=1, local_directory=tmpdir, processes=True
    )
    dd_client = dd.Client(cluster)
    objectives = [Objective("loss", False)]
    scheduler = MinimalDaskScheduler(
        dask_client=dd_client,
        objectives=objectives,
        logger=logging.getLogger("blackboxopt"),
    )

    space = ps.ParameterSpace()
    space.add(ps.CategoricalParameter("exit", [True, False]))
    opt = RandomSearch(space, objectives, max_steps=4)

    # create an evaluation spec that causes the worker to unexpectedly stop
    eval_spec = opt.generate_evaluation_specification()
    eval_spec.configuration["exit"] = True
    scheduler.submit(evaluation_function_cause_worker_restart, eval_spec)
    res = scheduler.check_for_results(20)
    assert res[0].all_objectives_none

    # Wait until scheduler available
    while not scheduler.has_capacity():
        time.sleep(0.05)

    # make sure that the worker is still functional
    eval_spec.configuration["exit"] = False
    scheduler.submit(evaluation_function_cause_worker_restart, eval_spec)
    res = scheduler.check_for_results(20)
    assert not res[0].all_objectives_none
    assert res[0].objectives["loss"] == 0

    # shutdown everything to avoid warning because
    # TemporaryDirectory will be cleaned up first
    scheduler.shutdown()
    del dd_client
    del cluster
