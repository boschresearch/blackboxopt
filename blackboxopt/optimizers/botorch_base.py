# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import warnings
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

from blackboxopt import (
    Evaluation,
    EvaluationSpecification,
    Objective,
    OptimizerNotReady,
    sort_evaluations,
)
from blackboxopt.base import (
    SingleObjectiveOptimizer,
    call_functions_with_evaluations_and_collect_errors,
    validate_objectives,
)

try:
    import numpy as np
    import parameterspace as ps
    import scipy.optimize as sci_opt
    import torch
    from botorch.acquisition import AcquisitionFunction
    from botorch.models.model import Model
    from botorch.optim import optimize_acqf
    from botorch.sampling.samplers import IIDNormalSampler
    from sklearn.impute import SimpleImputer

except ImportError as e:
    raise ImportError(
        "Unable to import BOTorch optimizer specific dependencies. "
        + "Make sure to install blackboxopt[botorch]"
    ) from e


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
    batch_shape: torch.Size = torch.Size(),
    torch_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert evaluations to one `(#batch, #evaluations, #parameters)` tensor
    containing the numerical representations of the configurations and
    one `(#batch, #evaluations, 1)` tensor containing the loss representation of
    the evaluations' objective value (flips the sign for objective value
    if `objective.greater_is_better=True`).

    Args:
        evaluations: List of evaluations that were collected during optimization.
        search_space: Search space used during optimization.
        objective: Objective that was used for optimization.
        batch_shape: Batch dimension(s) used for batched models.
        torch_dtype: Type of returned tensors.

    Returns:
        - X: Numerical representation of the configurations
        - Y: Numerical representation of the objective values

    Raises:
        ValueError: If one of configurations is not valid w.r.t. search space.
        ValueError: If one of configurations includes parameters that are not part of
            the search space.
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
    Y = Y.reshape(*batch_shape + Y.shape)

    if objective.greater_is_better:
        Y *= -1

    return X, Y


def init_af_opt_kwargs(af_opt_kwargs: Optional[dict]) -> dict:
    """Provide default initialization for the acquisition function optimizer
    configuration. Ensure that all mandatory fields are set to be used by BoTorch.

    Args:
        af_opt_kwargs: Acquisition function configuration.

    Returns:
        Acquisition function optimizer configuration, applicable for BoTorch's
            optimizer.
    """

    af_opt_config = {} if af_opt_kwargs is None else af_opt_kwargs

    # number of initial samples during AF optimization
    af_opt_config.setdefault("raw_samples", 1024)

    # number of restarts during AF optimization
    af_opt_config.setdefault("num_restarts", 4)

    return af_opt_config


class SingleObjectiveBOTorchOptimizer(SingleObjectiveOptimizer):
    def __init__(
        self,
        search_space: ps.ParameterSpace,
        objective: Objective,
        model: Model,
        acquisition_function_factory: Callable[[Model], AcquisitionFunction],
        af_optimizer_kwargs=None,
        num_initial_random_samples: int = 1,
        max_pending_evaluations: Optional[int] = 1,
        batch_shape: torch.Size = torch.Size(),
        logger: Optional[logging.Logger] = None,
        seed: Optional[int] = None,
        torch_dtype: torch.dtype = torch.float64,
    ):
        """Single objective BO optimizer that uses as a surrogate model the `model`
        object provided by user.

        The `model` is expected to be extended from BoTorch base model `Model` class,
        and does not require to be a GP model.

        Args:
            search_space: The space in which to optimize.
            objective: The objective to optimize.
            model: Surrogate model of `Model` type.
            acquisition_function_factory: Callable that produces an acquisition function
                instance, could also be a compatible acquisition function class.
                Only acquisition functions to be minimized are supported.
                Providing a partially initialized class is possible with, e.g.
                `functools.partial(UpperConfidenceBound, beta=6.0, maximize=False)`.
            af_optimizer_kwargs: Settings for acquisition function optimizer,
                see `botorch.optim.optimize_acqf`.
            num_initial_random_samples: Size of the initial space-filling design that
                is used before starting BO. The points are sampled randomly in the
                search space. If no random sampling is required, set it to 0.
            max_pending_evaluations: Maximum number of parallel evaluations. For
                sequential BO use the default value of 1. If no limit is required,
                set it to None.
            batch_shape: Batch dimension(s) used for batched models.
            logger: Custom logger.
            seed: A seed to make the optimization reproducible.
            torch_dtype: Torch data type used for storing the data. This needs to match
                the dtype of the model used
        """
        super().__init__(search_space=search_space, objective=objective, seed=seed)
        self.num_initial_random = num_initial_random_samples
        self.max_pending_evaluations = max_pending_evaluations
        self.batch_shape = batch_shape
        self.logger = logger or logging.getLogger("blackboxopt")

        self.torch_dtype = torch_dtype
        self.X = torch.empty(
            (*self.batch_shape, 0, len(search_space)), dtype=torch_dtype
        )
        self.losses = torch.empty((*self.batch_shape, 0, 1), dtype=torch_dtype)
        self.pending_specifications: Dict[int, EvaluationSpecification] = {}
        if seed is not None:
            torch.manual_seed(seed=seed)

        self.model = model
        self.acquisition_function_factory = acquisition_function_factory
        self.af_opt_kwargs = init_af_opt_kwargs(af_optimizer_kwargs)

    def _create_fantasy_model(self, model: Model) -> Model:
        """Create model with the pending specifications and model based
        outcomes added to the training data."""
        if not self.pending_specifications:
            # nothing to do when there are no pending specs
            return model

        pending_X = torch.tensor(
            np.array(
                [
                    self.search_space.to_numerical(e.configuration)
                    for e in self.pending_specifications.values()
                ]
            ),
            dtype=self.torch_dtype,
        )

        model = model.fantasize(pending_X, IIDNormalSampler(1), observation_noise=False)

        # model.fantasize extends model's X and Y with batch_size, even if originally
        # not given -> need to reshape these to their original representation
        n_samples = model.train_targets.size(-1)
        n_features = len(self.search_space)
        model.train_inputs[0] = model.train_inputs[0].reshape(
            torch.Size((*self.batch_shape, n_samples, n_features))
        )
        model.train_targets = model.train_targets.reshape(
            torch.Size((*self.batch_shape, n_samples, 1))
        )
        return model

    def _generate_evaluation_specification(self):
        """Optimize acquisition on fantasy model to pick next point."""
        fantasy_model = self._create_fantasy_model(self.model)
        fantasy_model.eval()

        # find next configuration by optimizing the acquisition function
        af = self.acquisition_function_factory(fantasy_model)
        if getattr(af, "maximize", False):
            raise ValueError(
                "Only acquisition functions that need to be minimized are supported. "
                f"The given {af.__class__.__name__} has maximize=True. "
                "One potential fix is using functools.partial("
                f"{af.__class__.__name__}, maximize=False) as the "
                "acquisition_function_factory init argument."
            )

        # numerical representation always lives on hypercube
        bounds = torch.tensor([[0, 1]] * len(self.search_space)).float().T

        configuration, _ = optimize_acqf(
            af,
            bounds=bounds,
            q=1,
            **self.af_opt_kwargs,
        )

        return EvaluationSpecification(
            configuration=self.search_space.from_numerical(configuration[0]),
        )

    def generate_evaluation_specification(self) -> EvaluationSpecification:
        """Call the optimizer specific function and append a unique integer id
        to the specification.

        Please refer to the docstring of
        `blackboxopt.base.SingleObjectiveOptimizer.generate_evaluation_specification`
        for a description of the method.
        """
        if (
            self.max_pending_evaluations
            and len(self.pending_specifications) == self.max_pending_evaluations
        ):
            raise OptimizerNotReady

        if self.num_initial_random > 0 and (
            self.X.size(-2) < self.num_initial_random
            or torch.nonzero(~torch.any(self.losses.isnan(), dim=1)).numel() == 0
        ):
            # We keep generating random samples until there are enough samples, and
            # at least one of them has a valid objective
            eval_spec = EvaluationSpecification(
                configuration=self.search_space.sample(),
            )
        else:
            eval_spec = self._generate_evaluation_specification()

        eval_id = self.X.size(-2) + len(self.pending_specifications)
        eval_spec.optimizer_info["evaluation_id"] = eval_id
        self.pending_specifications[eval_id] = eval_spec
        return eval_spec

    def _remove_pending_specifications(
        self, evaluations: Union[Evaluation, Iterable[Evaluation]]
    ):
        """Find and remove the corresponding entries in `self.pending_specifications`.

        Args:
            evaluations: List of completed evaluations.
        Raises:
            ValueError: If an evaluation is reported with an ID that was not issued
            by the optimizer, the method will fail.
        """
        _evals = [evaluations] if isinstance(evaluations, Evaluation) else evaluations

        for e in _evals:
            if "evaluation_id" not in e.optimizer_info:
                self.logger.debug("User provided EvaluationSpecification received.")
                continue

            if e.optimizer_info["evaluation_id"] not in self.pending_specifications:
                msg = (
                    "Unknown evaluation_id reported. This could indicate that the "
                    "evaluation has been reported before!"
                )
                self.logger.error(msg)
                raise ValueError(msg)

            del self.pending_specifications[e.optimizer_info["evaluation_id"]]

    def _append_evaluations_to_data(
        self, evaluations: Union[Evaluation, Iterable[Evaluation]]
    ):
        """Convert the reported evaluation into its numerical representation
        and append it to the training data.

        Args:
            evaluations: List of completed evaluations.
        """
        _evals = [evaluations] if isinstance(evaluations, Evaluation) else evaluations

        X, Y = to_numerical(
            _evals,
            self.search_space,
            self.objective,
            self.batch_shape,
            self.torch_dtype,
        )

        # fill in NaNs originating from inactive parameters (conditional spaces support)
        # botorch expect numerical representation of inputs to be within the unit
        # hypercube, thus we can't use the default c=-1.0
        X = impute_nans_with_constant(X, c=0.0)

        self.logger.debug(f"Next training configuration(s):{X}, {Y}")

        self.X = torch.cat([self.X, X], dim=-2)
        self.losses = torch.cat([self.losses, Y], dim=-2)

    def _update_internal_evaluation_data(
        self, evaluations: Iterable[Evaluation]
    ) -> None:
        """Check validity of the evaluations and do optimizer agnostic bookkeeping."""
        call_functions_with_evaluations_and_collect_errors(
            [
                functools.partial(validate_objectives, objectives=[self.objective]),
                self._remove_pending_specifications,
                self._append_evaluations_to_data,
            ],
            sort_evaluations(evaluations),
        )

    def report(self, evaluations: Union[Evaluation, Iterable[Evaluation]]) -> None:
        """A simple report method that conditions the model on data.
        This likely needs to be overridden for more specific BO implementations.
        """
        _evals = [evaluations] if isinstance(evaluations, Evaluation) else evaluations
        self._update_internal_evaluation_data(_evals)
        # Just for populating all relevant caches
        self.model.posterior(self.X)
        # The actual model update
        self.model.condition_on_observations(self.X, self.losses)

    def predict_model_based_best(self) -> Optional[Evaluation]:
        """Get the current configuration that is estimated to be the best (in terms of
        optimal objective value) without waiting for a reported evaluation of that
        configuration. Instead, the objective value estimation relies on BO's
        underlying model.

        This might return `None` in case there is no successfully evaluated
        configuration yet (thus, the optimizer has not been given training data yet).

        Returns:
            blackboxopt.evaluation.Evaluation
                The evaluated specification containing the estimated best configuration
                or `None` in case no evaluations have been reported yet.
        """
        if self.model.train_inputs[0].numel() == 0:
            return None

        def posterior_mean(x):
            # function to be optimized: posterior mean
            # scipy's minimize expects the following interface:
            #  - input: 1-D array with shape (n,)
            #  - output: float
            mean = self.model.posterior(torch.from_numpy(np.atleast_2d(x))).mean
            return mean.item()

        # prepare initial random samples and bounds for scipy's minimize
        n_init_samples = 10
        init_points = np.asarray(
            [
                self.search_space.to_numerical(self.search_space.sample())
                for _ in range(n_init_samples)
            ]
        )
        bounds = self.search_space.get_continuous_bounds()

        # use scipy's minimize to find optimum of the posterior mean
        optimized_points = [
            sci_opt.minimize(
                fun=posterior_mean,
                constraints=None,
                jac=False,
                x0=x,
                args=(),
                bounds=bounds,
                method="L-BFGS-B",
                options=None,
            )
            for x in init_points
        ]

        f_optimized = np.array(
            [np.atleast_1d(p.fun) for p in optimized_points]
        ).flatten()
        # get indexes of optimum value (with a tolerance)
        inds = np.argwhere(np.isclose(f_optimized, np.min(f_optimized)))
        # randomly select one index if there are multiple
        ind = np.random.choice(inds.flatten())

        # create Evaluation from the best estimated configuration
        best_x = optimized_points[ind].x
        best_y = posterior_mean(best_x)
        return Evaluation(
            configuration=self.search_space.from_numerical(best_x),
            objectives={
                self.objective.name: -1 * best_y
                if self.objective.greater_is_better
                else best_y
            },
        )
