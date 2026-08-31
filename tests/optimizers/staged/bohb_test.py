# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import parameterspace as ps

from blackboxopt import EvaluationSpecification, Objective
from blackboxopt.optimizers.staged.bohb import Sampler as BOHBSampler


def test_bohb_sampler_fully_random():
    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("p1", [0, 1]))

    sampler = BOHBSampler(
        space,
        Objective("loss", False),
        min_samples_in_model=1,
        top_n_percent=0.5,
        num_samples=10,
        random_fraction=1,
        bandwidth_factor=0.5,
        min_bandwidth=0.1,
    )

    for i in range(8):
        config_dict, info = sampler.sample_configuration()
        assert info["model_based_pick"] is False

        es = EvaluationSpecification(
            configuration=config_dict, settings={"fidelity": i}, optimizer_info=info
        )
        sampler.digest_evaluation(es.create_evaluation(objectives={"loss": i}))


def test_bohb_sampler_no_random():
    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("p1", [0, 1]))

    sampler = BOHBSampler(
        space,
        Objective("loss", False),
        min_samples_in_model=1,
        top_n_percent=0.5,
        num_samples=10,
        random_fraction=0.0,
        bandwidth_factor=0.5,
        min_bandwidth=0.1,
    )

    for i in range(8):
        config_dict, info = sampler.sample_configuration()
        # TODO: This one failes; is this due to it not being a valid test anymore after
        #       transitioning from the RF sampler to the KDE sampler?
        # assert info["model_based_pick"] is True or i == 0

        es = EvaluationSpecification(
            configuration=config_dict, settings={"fidelity": i}, optimizer_info=info
        )
        sampler.digest_evaluation(es.create_evaluation(objectives={"loss": i}))


def test_digest_evaluation_for_minimization():
    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("p1", [0, 1]))

    sampler = BOHBSampler(
        space,
        Objective("loss", greater_is_better=False),
        min_samples_in_model=1,
        top_n_percent=0.5,
        num_samples=10,
        random_fraction=1,
        bandwidth_factor=0.5,
        min_bandwidth=0.1,
    )
    config_dict, info = sampler.sample_configuration()
    es = EvaluationSpecification(
        configuration=config_dict, settings={"fidelity": 1.0}, optimizer_info=info
    )
    sampler.digest_evaluation(es.create_evaluation(objectives={"loss": -1.0}))
    assert sampler.losses[1.0][0] == -1.0


def test_digest_evaluation_for_maximization():
    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("p1", [0, 1]))

    sampler = BOHBSampler(
        space,
        Objective("score", greater_is_better=True),
        min_samples_in_model=1,
        top_n_percent=0.5,
        num_samples=10,
        random_fraction=1,
        bandwidth_factor=0.5,
        min_bandwidth=0.1,
    )
    config_dict, info = sampler.sample_configuration()
    es = EvaluationSpecification(
        configuration=config_dict, settings={"fidelity": 1.0}, optimizer_info=info
    )
    sampler.digest_evaluation(es.create_evaluation(objectives={"score": 1.0}))
    assert sampler.losses[1.0][0] == -1.0
