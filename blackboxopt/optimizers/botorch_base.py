# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import warnings
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

from gpytorch.models import ExactGP
from parameterspace import ParameterSpace

from blackboxopt.base import (
    Objective,
    OptimizerNotReady,
    SingleObjectiveOptimizer,
    call_functions_with_evaluations_and_collect_errors,
    validate_objectives,
)
from blackboxopt.evaluation import Evaluation, EvaluationSpecification
from blackboxopt.utils import sort_evaluations

try:
    import numpy as np
    import parameterspace as ps
    import torch
    from botorch.acquisition import AcquisitionFunction
    from botorch.exceptions import BotorchTensorDimensionWarning
    from botorch.models.model import Model
    from botorch.optim import optimize_acqf, optimize_acqf_discrete
    from botorch.sampling import IIDNormalSampler

    from blackboxopt.optimizers.botorch_utils import (
        filter_y_nans,
        impute_nans_with_constant,
        predict_model_based_best,
        to_numerical,
    )

except ImportError as e:
    raise ImportError(
        "Unable to import BOTorch optimizer specific dependencies. "
        + "Make sure to install blackboxopt[botorch]"
    ) from e


def _get_numerical_points_from_discrete_space(space: ParameterSpace) -> np.ndarray:
    """Retrieve all points from a discrete space in the numerical representation"""
    points_along_dimensions = []
    for parameter_name in space.get_parameter_names():
        parameter = space.get_parameter_by_name(parameter_name)[
            "parameter"
        ]  # type:ignore
        if isinstance(parameter, ps.IntegerParameter):
            bounds = (parameter.bounds[0], parameter.bounds[1] + 1)
            points_along_dimensions.append(
                [parameter.val2num(v) for v in range(*bounds)]
            )
        elif isinstance(parameter, ps.OrdinalParameter) or isinstance(
            parameter, ps.CategoricalParameter
        ):
            points_along_dimensions.append(
                [parameter.val2num(v) for v in parameter.values]
            )
        else:
            raise ValueError(
                f"Only discrete parameters are allowed but got {parameter}"
            )
    points = np.meshgrid(*points_along_dimensions)
    return np.concatenate([p.reshape((p.size, 1)) for p in points], axis=-1)


def _acquisition_function_optimizer_factory(
    search_space: ps.ParameterSpace,
    af_opt_kwargs: Optional[dict],
    torch_dtype: torch.dtype,
) -> Callable[[AcquisitionFunction], Tuple[torch.Tensor, torch.Tensor]]:
    """Prepare either BoTorch's `optimize_acqf_discrete` or `optimize_acqf` depending
    on whether the search space is fully discrete or not and set required defaults if
    not overridden by `af_opt_kwargs`. If any of the af optimizer specific required
    kwargs are set, this overrides the automatic discrete space detection. In case an
    exclusively discrete space is detected and `num_random_choices` is not specified
    in `af_opt_kwargs`, the discrete acquisition function optimizer is using all
    possible combinations in the discrete space.

    Args:
        search_space: Search space used for optimization.
        af_opt_kwargs: Acquisition function optimizer configuration, e.g. containing
            values for `num_random_choices` for discrete optimization, and
            `num_restarts`, `raw_samples` for the continuous optimization case.
        torch_dtype: Torch tensor type.

    Returns:
        Acquisition function optimizer that takes an acquisition function and returns a
        candidate with its associate acquisition function value.
    """
    kwargs = {} if af_opt_kwargs is None else af_opt_kwargs.copy()

    space_has_continuous_parameters = any(
        search_space[n]["parameter"].is_continuous
        for n in search_space.get_parameter_names()
    )
    if "num_random_choices" not in kwargs and (
        "num_restarts" in kwargs
        or "raw_samples" in kwargs
        or space_has_continuous_parameters
    ):
        # continuous AF optimization
        return functools.partial(
            optimize_acqf,
            q=1,
            # The numerical representation always lives on the unit hypercube
            bounds=torch.tensor([[0, 1]] * len(search_space), dtype=torch_dtype).T,
            num_restarts=kwargs.pop("num_restarts", 4),
            raw_samples=kwargs.pop("raw_samples", 1024),
            **kwargs,
        )

    if "num_random_choices" not in kwargs and not space_has_continuous_parameters:
        # Optimize over the entire discrete search space, if the number of random
        # choices is not specified
        choices = torch.from_numpy(
            _get_numerical_points_from_discrete_space(search_space)
        ).to(torch_dtype)
    else:
        # Optimize over the desired number of samples from the discrete search space
        choices = torch.Tensor(
            # Converting a list of ndarrays to torch is slow => convert to ndarray first
            np.array(
                [
                    search_space.to_numerical(search_space.sample())
                    for _ in range(kwargs["num_random_choices"])
                ]
            )
        ).to(dtype=torch_dtype)
    return functools.partial(optimize_acqf_discrete, q=1, choices=choices, **kwargs)


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
                see `botorch.optim.optimize_acqf` and in case the whole search space
                is discrete: `botorch.optim.optimize_acqf_discrete`. The former can be
                enforced by providing `raw_samples` or `num_restarts`, the latter by
                providing `num_random_choices`.
            num_initial_random_samples: Size of the initial space-filling design that
                is used before starting BO. The points are sampled randomly in the
                search space. If no random sampling is required, set it to 0.
                When random sampling is enabled, but evaluations with missing objective
                values are reported, more specifications are sampled until
                `num_initial_random_samples` many valid evaluations were reported.
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
        self.af_optimizer_kwargs = af_optimizer_kwargs

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

        if isinstance(model, ExactGP):
            # ExactGP.fantasize extends model's X and Y with batch_size, even if
            # originally not given -> need to reshape these to their original
            # representation
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

        af = self.acquisition_function_factory(fantasy_model)
        if getattr(af, "maximize", False):
            raise ValueError(
                "Only acquisition functions that need to be minimized are supported. "
                f"The given {af.__class__.__name__} has maximize=True. "
                "One potential fix is using functools.partial("
                f"{af.__class__.__name__}, maximize=False) as the "
                "acquisition_function_factory init argument."
            )

        acquisition_function_optimizer = _acquisition_function_optimizer_factory(
            search_space=self.search_space,
            af_opt_kwargs=self.af_optimizer_kwargs,
            torch_dtype=self.torch_dtype,
        )
        configuration, _ = acquisition_function_optimizer(af)

        return EvaluationSpecification(
            configuration=self.search_space.from_numerical(configuration[0].numpy()),
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

        # Generate random samples until there are enough samples where at least one of
        # the objective values is available
        if self.num_initial_random > 0 and (
            sum(~torch.any(self.losses.isnan(), dim=1)) < self.num_initial_random
        ):
            eval_spec = EvaluationSpecification(
                configuration=self.search_space.sample(),
                optimizer_info={"model_based_pick": False},
            )
        else:
            eval_spec = self._generate_evaluation_specification()
            eval_spec.optimizer_info["model_based_pick"] = True

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
            objectives=[self.objective],
            batch_shape=self.batch_shape,
            torch_dtype=self.torch_dtype,
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

        x_filtered, y_filtered = filter_y_nans(self.X, self.losses)

        # The actual model update
        # Ignore BotorchTensorDimensionWarning which is always reported to make the user
        # aware that they are reponsible for the right input Tensors dimensionality.
        with warnings.catch_warnings():
            warnings.simplefilter(
                action="ignore", category=BotorchTensorDimensionWarning
            )
            self.model = self.model.condition_on_observations(x_filtered, y_filtered)

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
        return predict_model_based_best(
            model=self.model,
            objective=self.objective,
            search_space=self.search_space,
            torch_dtype=self.torch_dtype,
        )
