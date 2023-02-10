# Copyright (c) 2023 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Iterable, List, Optional, Tuple

import numpy as np
import parameterspace as ps
import torch
from sklearn.impute import SimpleImputer

from blackboxopt.base import ConstraintsError, Objective
from blackboxopt.evaluation import Evaluation


def filter_y_nans(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Filter rows jointly for `x` and `y`, where `y` is `NaN`.

    Args:
        x: Input tensor of shape `n x d` or `1 x n x d`.
        y: Input tensor of shape `n x m` or `1 x n x m`.

    Returns:
        - x_f: Filtered `x`.
        - y_f: Filtered `y`.

    Raises:
        ValueError: If input is 3D (batched representation) with first dimension not
            `1` (multiple batches).
    """
    if (len(x.shape) == 3 and x.shape[0] > 1) or (len(y.shape) == 3 and y.shape[0] > 1):
        raise ValueError("Multiple batches are not supported for now.")

    x_f = x.clone()
    y_f = y.clone()

    # filter rows jointly where y is NaN
    x_f = x_f[~torch.any(y_f.isnan(), dim=-1)]
    y_f = y_f[~torch.any(y_f.isnan(), dim=-1)]

    # cast n x d back to 1 x n x d if originally batch case
    if len(x.shape) == 3:
        x_f = x_f.reshape(torch.Size((1,)) + x_f.shape)
    if len(y.shape) == 3:
        y_f = y_f.reshape(torch.Size((1,)) + y_f.shape)

    return x_f, y_f


def impute_nans_with_constant(x: torch.Tensor, c: float = -1.0) -> torch.Tensor:
    """Impute `NaN` values with given constant value.

    Args:
        x: Input tensor of shape `n x d` or `b x n x d`.
        c: Constant used as fill value to replace `NaNs`.

    Returns:
        - x_i - `x` where all `NaN`s are replaced with given constant.
    """
    if x.numel() == 0:  # empty tensor, nothing to impute
        return x
    x_i = x.clone()

    # cast n x d to 1 x n x d (cover non-batch case)
    if len(x.shape) == 2:
        x_i = x_i.reshape(torch.Size((1,)) + x_i.shape)

    for b in range(x_i.shape[0]):
        x_1 = x_i[b, :, :]
        x_1 = torch.tensor(
            SimpleImputer(
                missing_values=np.nan, strategy="constant", fill_value=c
            ).fit_transform(x_1),
            dtype=x.dtype,
        )
        x_i[b, :, :] = x_1

    # cast 1 x n x d back to n x d if originally non-batch
    if len(x.shape) == 2:
        x_i = x_i.reshape(x.shape)
    return x_i


def to_numerical(
    evaluations: Iterable[Evaluation],
    search_space: ps.ParameterSpace,
    objective: Objective,
    constraint_names: Optional[List[str]] = None,
    batch_shape: torch.Size = torch.Size(),
    torch_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert evaluations to one `(#batch, #evaluations, #parameters)` tensor
    containing the numerical representations of the configurations and
    one `(#batch, #evaluations, 1)` tensor containing the loss representation of
    the evaluations' objective value (flips the sign for objective value
    if `objective.greater_is_better=True`) and optionally constraints value.

    Args:
        evaluations: List of evaluations that were collected during optimization.
        search_space: Search space used during optimization.
        objective: Objective that was used for optimization.
        constraint_names: Name of constraints that are used for optimization.
        batch_shape: Batch dimension(s) used for batched models.
        torch_dtype: Type of returned tensors.

    Returns:
        - X: Numerical representation of the configurations
        - Y: Numerical representation of the objective values and optionally constraints

    Raises:
        ValueError: If one of configurations is not valid w.r.t. search space.
        ValueError: If one of configurations includes parameters that are not part of
            the search space.
        ConstraintError: If one of the constraint names is not defined in evaluations.
    """
    # validate configuration values and dimensions
    parameter_names = search_space.get_parameter_names() + list(
        search_space.get_constant_names()
    )
    for e in evaluations:
        with warnings.catch_warnings():
            # we already raise error if search space not valid, thus can ignore warnings
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="Parameter"
            )
            if not search_space.check_validity(e.configuration):
                raise ValueError(
                    f"The provided configuration {e.configuration} is not valid."
                )
        if not set(parameter_names) >= set(e.configuration.keys()):
            raise ValueError(
                f"Mismatch in parameter names from search space {parameter_names} and "
                + f"configuration {e.configuration}"
            )

    X = torch.tensor(
        np.array([search_space.to_numerical(e.configuration) for e in evaluations]),
        dtype=torch_dtype,
    )
    X = X.reshape(*batch_shape + X.shape)
    Y = torch.tensor(
        np.array([[e.objectives[objective.name]] for e in evaluations], dtype=float),
        dtype=torch_dtype,
    )

    if objective.greater_is_better:
        Y *= -1

    if constraint_names is not None:
        try:
            Y_constraints = torch.tensor(
                np.array(
                    [[e.constraints[c] for c in constraint_names] for e in evaluations],
                    dtype=float,
                ),
                dtype=torch_dtype,
            )
            Y = torch.cat((Y, Y_constraints), dim=1)
        except KeyError as e:
            raise ConstraintsError(
                f"Constraint name {e} is not defined in input evaluations."
            )
        except TypeError:
            raise ConstraintsError(
                f"Constraint name(s) {constraint_names} are not defined in input "
                + "evaluations."
            )

    Y = Y.reshape(*batch_shape + Y.shape)

    return X, Y
