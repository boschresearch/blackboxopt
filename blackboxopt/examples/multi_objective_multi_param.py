# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import logging
import time

import numpy as np
import parameterspace as ps
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import blackboxopt as bbo
from blackboxopt.optimization_loops.sequential import run_optimization_loop
from blackboxopt.optimizers.space_filling import SpaceFilling

# Load a sklearn sample dataset
X_TRAIN, X_VALIDATE, Y_TRAIN, Y_VALIDATE = train_test_split(
    *load_diabetes(return_X_y=True)
)

# Set up search space with multiple ML model hyperparameters of different types
SPACE = ps.ParameterSpace()
SPACE.add(ps.IntegerParameter("n_estimators", bounds=(256, 2048), transformation="log"))
SPACE.add(ps.IntegerParameter("min_samples_leaf", bounds=(1, 32), transformation="log"))
SPACE.add(ps.ContinuousParameter("max_samples", bounds=(0.1, 1)))
SPACE.add(ps.ContinuousParameter("max_features", bounds=(0.1, 1)))
SPACE.add(ps.IntegerParameter("max_depth", bounds=(1, 128)))
SPACE.add(ps.CategoricalParameter("criterion", values=("squared_error", "poisson")))


def evaluation_function(
    eval_spec: bbo.EvaluationSpecification,
) -> bbo.Evaluation:
    """Train and evaluate a random forest with given parameter configuration."""
    regr = RandomForestRegressor(
        n_estimators=eval_spec.configuration["n_estimators"],
        max_samples=eval_spec.configuration["max_samples"],
        max_features=eval_spec.configuration["max_features"],
        max_depth=eval_spec.configuration["max_depth"],
        min_samples_leaf=eval_spec.configuration["min_samples_leaf"],
        criterion=eval_spec.configuration["criterion"],
    )

    start = time.time()
    regr.fit(X_TRAIN, Y_TRAIN)
    fit_duration = time.time() - start

    y_pred = regr.predict(X_VALIDATE)
    objectives = {
        "R²": r2_score(Y_VALIDATE, y_pred),
        "Fit Duration": fit_duration,
        "Max Error": np.abs(Y_VALIDATE - y_pred).max(),
    }
    evaluation = eval_spec.create_evaluation(objectives=objectives)
    return evaluation


def main():
    logger = bbo.init_logger(logging.INFO)

    # Create an optimization run based on a parameterspace and optimizer choice
    optimizer = SpaceFilling(
        search_space=SPACE,
        objectives=[
            bbo.Objective("R²", greater_is_better=True),
            bbo.Objective("Max Error", greater_is_better=False),
            bbo.Objective("Fit Duration", greater_is_better=False),
        ],
    )

    # Fetch new configurations to evaluate until the optimization is done or
    # a given timeout is reached
    evaluations = run_optimization_loop(
        optimizer=optimizer,
        evaluation_function=evaluation_function,
        timeout_s=60.0,
    )

    logger.info(f"Evaluated {len(evaluations)} specifications")

    pareto_front = bbo.utils.filter_pareto_efficient(evaluations, optimizer.objectives)
    logger.info(f"{len(pareto_front)} evaluation(s) are pareto efficient")


if __name__ == "__main__":
    main()
