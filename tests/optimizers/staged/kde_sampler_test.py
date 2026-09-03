# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import parameterspace as ps

from blackboxopt import Objective
from blackboxopt.optimizers.bohb import BOHB
from blackboxopt.optimizers.staged.kde_sampler import (
    impute_conditional_data,
    sample_around_values,
)


class _FakeKDE:
    """Minimal stand-in for a statsmodels KDEMultivariate with a fixed pdf value."""

    def __init__(self, data, bw, pdf_value):
        self.data = np.asarray(data, dtype=float)
        self.bw = np.asarray(bw, dtype=float)
        self._pdf_value = pdf_value

    def pdf(self, x):
        return self._pdf_value


def test_non_finite_acquisition_fallback_returns_valid_model_based_pick():
    space = ps.ParameterSpace()
    space.add(ps.CategoricalParameter("c1", ["a", "b", "c"]))
    opt = BOHB(
        space,
        Objective("loss", False),
        min_fidelity=1.0,
        max_fidelity=3.0,
        num_iterations=3,
    )
    sampler = opt.config_sampler
    sampler.random_fraction = 0.0

    # Every sampled candidate yields a non-finite acquisition value (bad pdf = inf),
    # so the normal `val < best` path never fires. The good pdf stays finite, which
    # is exactly the categorical-not-in-KDE situation the fallback is meant to handle.
    good_kde = _FakeKDE(data=[[2.0]], bw=[0.1], pdf_value=1.0)
    bad_kde = _FakeKDE(data=[[2.0]], bw=[0.1], pdf_value=np.inf)
    sampler.kde_models = {3.0: {"good": good_kde, "bad": bad_kde}}

    config, info = sampler.sample_configuration()

    # The fallback must be used (not a random sample) ...
    assert info["model_based_pick"] is True
    # ... and it must yield a valid categorical value in search-space coordinates.
    assert config["c1"] in {"a", "b", "c"}


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

    # TODO: Can we find a way to test this without needing to instantiate full BOHB?
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
