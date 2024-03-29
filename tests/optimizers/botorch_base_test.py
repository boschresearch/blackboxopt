# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0
import itertools
from functools import partial

import parameterspace as ps
import pytest
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf, optimize_acqf_discrete

from blackboxopt.base import Objective
from blackboxopt.optimizers.botorch_base import (
    SingleObjectiveBOTorchOptimizer,
    _acquisition_function_optimizer_factory,
    _get_numerical_points_from_discrete_space,
)
from blackboxopt.optimizers.testing import (
    ALL_REFERENCE_TESTS,
    handles_conditional_space,
    is_deterministic_when_reporting_shuffled_evaluations,
    respects_fixed_parameter,
)


@pytest.mark.parametrize("reference_test", ALL_REFERENCE_TESTS)
def test_all_reference_tests(reference_test, seed):
    if reference_test in (
        respects_fixed_parameter,
        is_deterministic_when_reporting_shuffled_evaluations,
    ):
        n_features = 1
    elif reference_test == handles_conditional_space:
        n_features = 3
    else:
        n_features = 5

    batch_shape = torch.Size()
    reference_test(
        SingleObjectiveBOTorchOptimizer,
        dict(
            model=SingleTaskGP(
                torch.empty((*batch_shape, 0, n_features), dtype=torch.float64),
                torch.empty((*batch_shape, 0, 1), dtype=torch.float64),
            ),
            acquisition_function_factory=partial(
                UpperConfidenceBound, beta=0.1, maximize=False
            ),
            max_pending_evaluations=5,
        ),
        seed=seed,
    )


def test_acquisition_function_optimizer_factory_with_continuous():
    continuous_space = ps.ParameterSpace()
    continuous_space.add(ps.ContinuousParameter("conti1", (0.0, 1.0)))
    continuous_space.add(ps.ContinuousParameter("conti2", (-1.0, 1.0)))

    af_opt = _acquisition_function_optimizer_factory(
        continuous_space, af_opt_kwargs={}, torch_dtype=torch.float64
    )

    assert af_opt.func == optimize_acqf  # pylint: disable=no-member


def test_acquisition_function_optimizer_factory_with_discrete_space():
    discrete_space = ps.ParameterSpace()
    discrete_space.add(ps.IntegerParameter("integ", (-5, 10)))
    discrete_space.add(ps.OrdinalParameter("ordin", ("small", "medium", "large")))
    discrete_space.add(ps.CategoricalParameter("categ", ("woof", "miaow", "moo")))

    af_opt = _acquisition_function_optimizer_factory(
        discrete_space, af_opt_kwargs={}, torch_dtype=torch.float64
    )

    assert af_opt.func == optimize_acqf_discrete  # pylint: disable=no-member

    af_opt = _acquisition_function_optimizer_factory(
        discrete_space,
        af_opt_kwargs={"num_random_choices": 50},
        torch_dtype=torch.float64,
    )

    assert af_opt.func == optimize_acqf_discrete  # pylint: disable=no-member


def test_acquisition_function_optimizer_factory_with_mixed_space():
    mixed_space = ps.ParameterSpace()
    mixed_space.add(ps.OrdinalParameter("ordin", ("small", "medium", "large")))
    mixed_space.add(ps.ContinuousParameter("conti", (0.0, 1.0)))

    af_opt = _acquisition_function_optimizer_factory(
        mixed_space, af_opt_kwargs={}, torch_dtype=torch.float64
    )

    assert af_opt.func == optimize_acqf  # pylint: disable=no-member


def test_acquisition_function_optimizer_factory_force_discrete():
    continous_space = ps.ParameterSpace()
    continous_space.add(ps.ContinuousParameter("conti", (0.0, 1.0)))

    af_opt = _acquisition_function_optimizer_factory(
        continous_space,
        af_opt_kwargs={"num_random_choices": 1_000},
        torch_dtype=torch.float64,
    )

    assert af_opt.func == optimize_acqf_discrete  # pylint: disable=no-member


def test_acquisition_function_optimizer_factory_force_continuous():
    discrete_space = ps.ParameterSpace()
    discrete_space.add(ps.IntegerParameter("integ", (-5, 10)))
    discrete_space.add(ps.OrdinalParameter("ordin", ("small", "medium", "large")))
    discrete_space.add(ps.CategoricalParameter("categ", ("woof", "miaow", "moo")))

    af_opt = _acquisition_function_optimizer_factory(
        discrete_space,
        af_opt_kwargs={"num_restarts": 10},
        torch_dtype=torch.float64,
    )

    assert af_opt.func == optimize_acqf  # pylint: disable=no-member

    af_opt = _acquisition_function_optimizer_factory(
        discrete_space,
        af_opt_kwargs={"raw_samples": 5_000},
        torch_dtype=torch.float64,
    )

    assert af_opt.func == optimize_acqf  # pylint: disable=no-member


def test_find_optimum_in_1d_discrete_space(seed):
    space = ps.ParameterSpace()
    space.add(ps.IntegerParameter("integ", (0, 2)))
    batch_shape = torch.Size()
    opt = SingleObjectiveBOTorchOptimizer(
        search_space=space,
        objective=Objective("loss", greater_is_better=False),
        model=SingleTaskGP(
            torch.empty((*batch_shape, 0, len(space)), dtype=torch.float64),
            torch.empty((*batch_shape, 0, 1), dtype=torch.float64),
        ),
        acquisition_function_factory=partial(
            UpperConfidenceBound, beta=1.0, maximize=False
        ),
        max_pending_evaluations=5,
        seed=seed,
    )

    losses = []
    for _ in range(10):
        es = opt.generate_evaluation_specification()
        loss = es.configuration["integ"] ** 2
        losses.append(loss)
        opt.report(es.create_evaluation(objectives={"loss": loss}))

    assert (
        sum(loss == 0 for loss in losses) > 5
    ), "After figuring out the best of the three points, it should only propose that."

    best = opt.predict_model_based_best()
    assert best.configuration["integ"] == 0
    assert opt.objective.name in best.objectives


def test_propose_random_until_enough_evaluations_without_missing_objective_values(seed):
    space = ps.ParameterSpace()
    space.add(ps.IntegerParameter("integ", (0, 2)))
    batch_shape = torch.Size()

    opt = SingleObjectiveBOTorchOptimizer(
        search_space=space,
        objective=Objective("loss", greater_is_better=False),
        model=SingleTaskGP(
            torch.empty((*batch_shape, 0, len(space)), dtype=torch.float64),
            torch.empty((*batch_shape, 0, 1), dtype=torch.float64),
        ),
        acquisition_function_factory=partial(
            UpperConfidenceBound, beta=1.0, maximize=False
        ),
        num_initial_random_samples=2,
        max_pending_evaluations=1,
        seed=seed,
    )

    es = opt.generate_evaluation_specification()
    assert not es.optimizer_info[
        "model_based_pick"
    ], "No evaluation reported, 0 < 2 initial random samples"
    opt.report(
        es.create_evaluation(objectives={"loss": es.configuration["integ"] ** 2}),
    )

    es = opt.generate_evaluation_specification()
    assert not es.optimizer_info[
        "model_based_pick"
    ], "One evaluation reported, 1 < 2 initial random samples"
    opt.report(
        es.create_evaluation(objectives={"loss": None}),
    )

    es = opt.generate_evaluation_specification()
    assert not es.optimizer_info[
        "model_based_pick"
    ], "One valid evaluation reported, 1 < 2 initial random samples"
    opt.report(
        es.create_evaluation(objectives={"loss": es.configuration["integ"] ** 2}),
    )

    es = opt.generate_evaluation_specification()
    assert es.optimizer_info[
        "model_based_pick"
    ], "Two valid evaluations reported, 2 >= 2 initial random samples"


def test_get_numerical_points_from_discrete_space():
    p0l, p0h = -5, 10
    p1 = ("small", "medium", "large")
    p2 = ("woof", "miaow", "moo")
    discrete_space = ps.ParameterSpace()
    p_integ = ps.IntegerParameter("integ", (p0l, p0h))
    discrete_space.add(p_integ)
    p_ordin = ps.OrdinalParameter("ordin", p1)
    discrete_space.add(p_ordin)
    p_categ = ps.CategoricalParameter("categ", p2)
    discrete_space.add(p_categ)

    points = _get_numerical_points_from_discrete_space(discrete_space)
    assert (
        points.shape[0] == p_integ.num_values * p_ordin.num_values * p_categ.num_values
    )
    assert points.shape[-1] == len(discrete_space)
    for integ, ordin, categ in itertools.product(
        range(p_integ.bounds[0], p_integ.bounds[1] + 1), p_ordin.values, p_categ.values
    ):
        assert (
            (
                points
                == discrete_space.to_numerical(
                    dict(integ=integ, ordin=ordin, categ=categ)
                )
            )
            .all(axis=1)
            .any()
        ), (
            f"Point {integ}, {ordin}, {categ} belongs to the search space but is not "
            + "returned by `_get_numerical_points_from_discrete_space`"
        )
