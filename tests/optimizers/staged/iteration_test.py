# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import numpy as np
import pytest

from blackboxopt import Evaluation, Objective
from blackboxopt.optimizers.staged.configuration_sampler import (
    StagedIterationConfigurationSampler,
)
from blackboxopt.optimizers.staged.iteration import StagedIteration
from blackboxopt.optimizers.staged.utils import greedy_promotion


class DummyOptimizer:
    def __init__(self, expected_configs_in_promote):
        self.expected_configs_in_promote = expected_configs_in_promote

    def decide_promotion(self, data, num_configs):
        assert len(data) == self.expected_configs_in_promote
        losses = [d.loss for d in data]
        ranks = np.argsort(np.argsort(losses))

        n = min(num_configs, len(data))
        return [datum.config_key for rank, datum in zip(ranks, data) if rank < n]


class DummyConfigurationSampler(StagedIterationConfigurationSampler):
    def __init__(self):
        self.num_configs = 0

    def sample_configuration(self) -> Tuple[dict, dict]:
        self.num_configs += 1
        return {"value": self.num_configs - 1}, {}

    def digest_evaluation(self, evaluation: Evaluation):
        """This dummy doesn't do anything with evaluation results."""
        pass


@pytest.mark.timeout(1)
def test_staged_iteration_get_and_digest_configuration():

    n_eval_specs = 3
    iteration = StagedIteration(
        iteration=0,
        num_configs=[n_eval_specs, 1],
        fidelities=[0.1, 1],
        config_sampler=DummyConfigurationSampler(),
        config_promotion_function=greedy_promotion,
        objective=Objective("loss", False),
    )

    eval_specs = []
    for i in range(n_eval_specs - 1):
        eval_specs.append(iteration.get_evaluation_specification())
        assert eval_specs[-1].configuration["value"] == i
        iteration.digest_evaluation(
            eval_specs[-1].optimizer_info["id"],
            eval_specs[-1].create_evaluation(objectives={"loss": i}),
        )
        # make sure that the stage has not completed yet
        assert iteration.current_stage == 0

    # get the last config in this stage
    i = n_eval_specs - 1
    eval_specs.append(iteration.get_evaluation_specification())
    assert eval_specs[-1].configuration["value"] == i
    assert iteration.current_stage == 0
    # make sure no configuration is returned when all configurations in a stage have
    # been queried, but not all have finished
    assert iteration.get_evaluation_specification() is None

    # digest it and see that the next stage is reached
    iteration.digest_evaluation(
        eval_specs[-1].optimizer_info["id"],
        eval_specs[-1].create_evaluation(objectives={"loss": i}),
    )
    assert iteration.current_stage == 1
    assert not iteration.finished

    final_eval_spec = iteration.get_evaluation_specification()
    assert final_eval_spec.configuration["value"] == 0
    iteration.digest_evaluation(
        final_eval_spec.optimizer_info["id"],
        final_eval_spec.create_evaluation(objectives={"loss": 0.0}),
    )
    assert iteration.finished
    assert iteration.get_evaluation_specification() is None


@pytest.mark.timeout(1)
def test_staged_iteration_get_and_digest_configuration_with_crashes():

    n_eval_specs = 3
    opt = DummyOptimizer(n_eval_specs - 1)
    iteration = StagedIteration(
        iteration=0,
        num_configs=[n_eval_specs, n_eval_specs],
        fidelities=[1, 1],
        config_sampler=DummyConfigurationSampler(),
        config_promotion_function=opt.decide_promotion,
        objective=Objective("loss", False),
    )

    eval_specs = []
    for i in range(n_eval_specs):
        eval_specs.append(iteration.get_evaluation_specification())
        assert eval_specs[-1].configuration["value"] == i

    for i in range(n_eval_specs - 1):
        iteration.digest_evaluation(
            eval_specs[i].optimizer_info["id"],
            eval_specs[i].create_evaluation(objectives={"loss": i}),
        )

    iteration.digest_evaluation(
        eval_specs[n_eval_specs - 1].optimizer_info["id"],
        eval_specs[n_eval_specs - 1].create_evaluation(objectives={"loss": None}),
    )

    assert all([e.status == "FINISHED" for e in iteration.evaluation_data[0][:-1]])
    assert iteration.evaluation_data[0][-1].status == "CRASHED"
    assert iteration.current_stage == 1

    # make sure the crashed config is not promoted and a new one is sampled instead
    eval_specs = []
    for i in range(n_eval_specs):
        eval_specs.append(iteration.get_evaluation_specification())

    assert all([c.configuration["value"] == i for i, c in enumerate(eval_specs[:-1])])
    assert eval_specs[-1].configuration["value"] == n_eval_specs
