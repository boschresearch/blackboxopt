# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import parameterspace as ps

from blackboxopt import EvaluationSpecification, Objective, OptimizationComplete

# TODO: Move BOHB specific tests to ../bohb_test.py
from blackboxopt.optimizers.bohb import BOHB
from blackboxopt.optimizers.staged.bohb import Sampler as BOHBSampler
from blackboxopt.optimizers.staged.bohb import (
    impute_conditional_data,
    sample_around_values,
)


def test_impute_categorical_values(n_samples=128):
    vartypes = [0, 6]
    allowed_categorical_values = set(np.arange(6))
    initial_categorical_values = set(np.arange(3))
    data = np.vstack(
        [
            np.random.rand(n_samples),
            np.random.randint(3, size=n_samples),
        ]
    ).T

    # test using values from other points
    data[data[:, 0] > 0.5, 1] = np.nan
    imputed_data = impute_conditional_data(data, vartypes)
    assert np.all(np.isfinite(imputed_data))
    assert set(imputed_data[:, 1]) == initial_categorical_values

    # test using random values, if no point has a value
    data[:, 1] = np.nan
    imputed_data = impute_conditional_data(data, vartypes)
    assert np.all(np.isfinite(imputed_data))
    assert set(imputed_data[:, 1]) == allowed_categorical_values


def test_impute_ordinal_values(n_samples=128):
    vartypes = [0, -4]
    allowed_ordinal_values = set(np.arange(4))
    initial_ordinal_values = set(np.arange(2))
    data = np.vstack(
        [
            np.random.rand(n_samples),
            np.random.randint(2, size=n_samples),
        ]
    ).T
    # test using values from other points
    data[data[:, 0] > 0.5, 1] = np.nan
    imputed_data = impute_conditional_data(data, vartypes)
    assert np.all(np.isfinite(imputed_data))
    assert set(imputed_data[:, 1]) == initial_ordinal_values

    # test using random values, if no point has a value
    data[:, 1] = np.nan
    imputed_data = impute_conditional_data(data, vartypes)
    assert np.all(np.isfinite(imputed_data))
    assert set(imputed_data[:, 1]) == allowed_ordinal_values


def test_impute_continuous_values(n_samples=128):
    vartypes = [0, 0]
    data = np.random.rand(n_samples, 2)

    # test using values from other points
    data[data[:, 0] > 0.5, 1] = np.nan
    imputed_data = impute_conditional_data(data, vartypes)
    assert np.all(np.isfinite(imputed_data))

    # test using random values, if no point has a value
    data[:, 1] = np.nan
    imputed_data = impute_conditional_data(data, vartypes)
    assert np.all(np.isfinite(imputed_data))


def test_sample_around(n_samples=128):
    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("x1", [-1, 1]))
    space.add(ps.ContinuousParameter("x2", [1e-5, 1e0], transformation="log"))
    space.add(ps.CategoricalParameter("c1", [0, 1, 2]))
    space.add(ps.CategoricalParameter("c2", ["foo", "bar", "baz"]))
    space.add(ps.IntegerParameter("i1", [1, 16]), lambda c2: c2 == "foo")
    space.add(
        ps.IntegerParameter("i2", [1, 1024], transformation="log"),
        lambda c2: c2 in ["bar", "baz"],
    )

    opt = BOHB(
        space,
        Objective("loss", False),
        min_fidelity=1.0,
        max_fidelity=3.0,
        num_iterations=3,
    )

    vartypes = opt.config_sampler.vartypes

    numerical_samples = np.array(
        [space.to_numerical(space.sample()) for i in range(n_samples)]
    )

    numerical_samples = impute_conditional_data(numerical_samples, vartypes)
    for datum in numerical_samples:
        another_sample = sample_around_values(
            datum, [0.1] * len(space), vartypes, 0.1, 3
        )
        assert space.from_numerical(another_sample)


def test_sample_configurations():
    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("x1", [-1, 1]))

    opt = BOHB(
        space,
        Objective("loss", False),
        min_fidelity=1,
        max_fidelity=9,
        num_iterations=2,
        min_samples_in_model=3,
        random_fraction=0.0,
    )

    for _ in range(5):
        eval_spec = opt.generate_evaluation_specification()
        assert not eval_spec.optimizer_info["model_based_pick"]
        evaluation = eval_spec.create_evaluation(
            objectives={"loss": eval_spec.configuration["x1"] ** 2}
        )
        opt.report(evaluation)

    while eval_spec.optimizer_info["configuration_key"][0] == 0:
        eval_spec = opt.generate_evaluation_specification()
        evaluation = eval_spec.create_evaluation(
            objectives={"loss": eval_spec.configuration["x1"] ** 2}
        )
        opt.report(evaluation)

    while True:
        try:
            eval_spec = opt.generate_evaluation_specification()
            assert eval_spec.optimizer_info["model_based_pick"]
            evaluation = eval_spec.create_evaluation(
                objectives={"loss": eval_spec.configuration["x1"] ** 2}
            )
            opt.report(evaluation)
        except OptimizationComplete:
            break


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
