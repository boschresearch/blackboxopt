import functools
from typing import Callable, Iterable, Optional, Union

import optuna
import parameterspace as ps

from blackboxopt.base import (
    Objective,
    SingleObjectiveOptimizer,
    call_functions_with_evaluations_and_collect_errors,
    validate_objectives,
)
from blackboxopt.evaluation import Evaluation, EvaluationSpecification


class OptunaBasedSingleObjectiveOptimizer(SingleObjectiveOptimizer):
    def __init__(
        self,
        search_space: ps.ParameterSpace,
        objective: Objective,
        sampler_factory: Callable[[Optional[int]], optuna.samplers.BaseSampler],
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(search_space, objective, seed)
        self._optuna_study = optuna.create_study(
            sampler=sampler_factory(seed=seed),
            direction=optuna.study.StudyDirection.MAXIMIZE
            if objective.greater_is_better
            else optuna.study.StudyDirection.MINIMIZE,
        )

    def generate_evaluation_specification(self) -> EvaluationSpecification:
        trial = self._optuna_study.ask()

        # TODO: Revise after implementation of
        #       https://github.com/boschresearch/parameterspace/issues/47
        config = self.search_space._constants.copy()
        for p in [
            self.search_space.get_parameter_by_name(n)
            for n in self.search_space.get_parameter_names()
        ]:
            if not p["condition"].empty():
                raise ValueError(
                    "Optuna wrapper does not support conditional parameters"
                )
            p = p["parameter"]
            if isinstance(p, ps.IntegerParameter):
                config[p.name] = trial.suggest_int(
                    name=p.name, low=p.bounds[0], high=p.bounds[1], step=1
                )
            elif isinstance(p, ps.ContinuousParameter):
                config[p.name] = trial.suggest_uniform(
                    name=p.name, low=p.bounds[0], high=p.bounds[1]
                )
            elif isinstance(p, ps.CategoricalParameter) or isinstance(
                p, ps.OrdinalParameter
            ):
                config[p.name] = trial.suggest_categorical(
                    name=p.name, choices=p.values
                )
            else:
                raise NotImplementedError(
                    f"Parameter type {type(p)} not supported by optuna wrapper"
                )

        return EvaluationSpecification(
            config, optimizer_info={"trial": trial._trial_id}
        )

    def report(self, evaluations: Union[Evaluation, Iterable[Evaluation]]) -> None:
        if isinstance(evaluations, Evaluation):
            evaluations = [evaluations]

        call_functions_with_evaluations_and_collect_errors(
            [
                functools.partial(validate_objectives, objectives=[self.objective]),
                self._report,
            ],
            evaluations,
        )

    def _report(self, evaluation: Evaluation):
        self._optuna_study.tell(
            trial=evaluation.optimizer_info["trial"],
            values=evaluation.objectives[self.objective.name],
        )
