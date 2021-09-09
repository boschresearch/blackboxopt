# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import abc
import collections
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Type, Union

from parameterspace.base import SearchSpace

from blackboxopt.evaluation import Evaluation, EvaluationSpecification


class OptimizationComplete(Exception):
    """Exception that is raised when the optimization run is finished, e.g. when the
    budget has been exhausted.
    """


class OptimizerNotReady(Exception):
    """Exception that is raised when the optimizer is not ready to propose a new
    evaluation specification.
    """


class ObjectivesError(ValueError):
    """Raised on incomplete or missing objectives."""


class ConstraintsError(ValueError):
    """Raised on incomplete or missing constraints."""


class ContextError(ValueError):
    """Raised on incomplete or missing context information."""


class EvaluationsError(ValueError):
    """Raised on invalid evaluations.

    The problematic evaluations and their respective exceptions are passed in the
    `evaluations_with_errors` attribute.
    """

    def __init__(self, evaluations_with_errors: List[Tuple[Evaluation, Exception]]):
        self.message = (
            "An error with one or more evaluations occurred. Check the "
            "'evaluations_with_errors' attribute of this exception for details."
        )
        self.evaluations_with_errors = evaluations_with_errors


@dataclass
class Objective:
    name: str
    greater_is_better: bool


def _raise_on_duplicate_objective_names(objectives: List[Objective]) -> None:
    names = [o.name for o in objectives]
    duplications = [
        (name, count) for name, count in collections.Counter(names).items() if count > 1
    ]
    if duplications:
        raise ValueError(
            "All objectives need to have a unique name, but found duplications "
            + f"(objective name, occurrence count): {duplications}"
        )


def raise_on_unknown_or_incomplete(
    exception: Type[ValueError], known: Iterable[str], reported: Iterable[str]
) -> None:
    """Raise the given exception if not all known strings are contained in reported or
    the other way around.
    """
    known_set = set(known)
    reported_set = set(reported)

    unknown = reported_set - known_set
    if unknown:
        raise exception(
            f"Unknown reported: {list(unknown)}. Valid are only: {list(known_set)}"
        )
    missing = known_set - reported_set
    if missing:
        raise exception(f"Missing: {list(missing)}")


class Optimizer(abc.ABC):
    """Abstract base class for blackbox optimizer implementations."""

    def __init__(self, search_space: SearchSpace, seed: int = None) -> None:
        """Initialize the optimizer with an optional seed for reproducibility.

        Args:
            search_space: The search space to optimize.
            seed: A seed for the optimizer, which is also used to re-seed the provided
                search space.
        """
        super().__init__()
        self.search_space = search_space
        self.seed = seed

        if self.seed is not None:
            self.search_space.seed(self.seed)

    @abc.abstractmethod
    def get_evaluation_specification(self) -> EvaluationSpecification:
        """Get next configuration and settings to evaluate.

        Raises:
            OptimizationComplete: When the optimization run is finished, e.g. when the
                budget has been exhausted.
            OptimizerNotReady: When the optimizer is not ready to propose a new
                evaluation specification.
        """

    @abc.abstractmethod
    def report(self, evaluations: Union[Evaluation, Iterable[Evaluation]]) -> None:
        """Report one or more evaluated evaluation specifications.

        NOTE: Not all optimizers support reporting results for evaluation specifications
        that were not proposed by the optimizer.

        Args:
            evaluations: A single evaluated evaluation specifications, or an iterable
            of many.
        """


class SingleObjectiveOptimizer(Optimizer):
    def __init__(
        self,
        search_space: SearchSpace,
        objective: Objective,
        seed: int = None,
    ) -> None:
        """Initialize the optimizer with an optional seed for reproducibility.

        Args:
            search_space: The search space to optimize.
            objective: The objectives of the optimization.
            seed: A seed for the optimizer, which is also used to re-seed the provided
                search space.
        """
        super().__init__(search_space=search_space, seed=seed)
        self.objective = objective

    def report(self, evaluations: Union[Evaluation, Iterable[Evaluation]]) -> None:
        """Report one or multiple evaluations to the optimizer.

        All valid evaluations are processed. Faulty evaluations are not processed,
        instead an `EvaluationsError` is raised, which includes the problematic
        evaluations with their respective Exceptions in the `evaluations_with_errors`
        attribute.

        Args:
            evaluations: A single evaluated evaluation specifications, or an iterable
            of many.

        Raises:
            EvaluationsError: Raised when an evaluation could not be processed.
        """
        if isinstance(evaluations, Evaluation):
            evaluations = [evaluations]

        invalid_evaluations: List[Tuple[Evaluation, Exception]] = []
        for evaluation in evaluations:
            try:
                raise_on_unknown_or_incomplete(
                    exception=ObjectivesError,
                    known=[self.objective.name],
                    reported=evaluation.objectives.keys(),
                )
            except ObjectivesError as e:
                invalid_evaluations.append((evaluation, e))

        if invalid_evaluations:
            raise EvaluationsError(evaluations_with_errors=invalid_evaluations)


class MultiObjectiveOptimizer(Optimizer):
    def __init__(
        self,
        search_space: SearchSpace,
        objectives: List[Objective],
        seed: int = None,
    ) -> None:
        """Initialize the optimizer with an optional seed for reproducibility.

        Args:
            search_space: The search space to optimize.
            objectives: The objectives of the optimization.
            seed: A seed for the optimizer, which is also used to re-seed the provided
                search space.
        """
        super().__init__(search_space=search_space, seed=seed)

        _raise_on_duplicate_objective_names(objectives)
        self.objectives = objectives

    def report(self, evaluations: Union[Evaluation, Iterable[Evaluation]]) -> None:
        if isinstance(evaluations, Evaluation):
            evaluations = [evaluations]

        invalid_evaluations: List[Tuple[Evaluation, Exception]] = []
        for evaluation in evaluations:
            try:
                raise_on_unknown_or_incomplete(
                    exception=ObjectivesError,
                    known=[o.name for o in self.objectives],
                    reported=evaluation.objectives.keys(),
                )
            except ObjectivesError as e:
                invalid_evaluations.append((evaluation, e))

        if invalid_evaluations:
            raise EvaluationsError(evaluations_with_errors=invalid_evaluations)
