# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np


@dataclass
class EvaluationSpecification(Mapping[str, Any]):
    configuration: dict = field(
        metadata={"Description": "The configuration to be evaluated next."}
    )
    settings: dict = field(
        default_factory=dict,
        metadata={
            "Description": "Additional settings like the fidelity or target task."
        },
    )
    optimizer_info: dict = field(
        default_factory=dict,
        metadata={"Description": "Information about and for internal optimizer state."},
    )
    created_unixtime: float = field(
        default_factory=time.time,
        metadata={"Description": "Creation time of the evaluation specificiation."},
    )

    def keys(self):
        return self.__dataclass_fields__.keys()  # pylint: disable=no-member

    def get_evaluation(
        self,
        objectives: Dict[str, Optional[float]],
        user_info: Optional[dict] = None,
        stacktrace: Optional[str] = None,
        finished_unixtime: Optional[float] = None,
    ):
        """Create blackboxopt.Evaluation based on this evaluation specification."""
        evaluation = Evaluation(
            objectives=objectives, user_info=user_info, stacktrace=stacktrace, **self
        )

        if finished_unixtime is not None:
            evaluation.finished_unixtime = finished_unixtime

        return evaluation

    def __getitem__(self, key):
        if key not in self.__dataclass_fields__:  # pylint: disable=no-member
            raise KeyError(
                f"Only dataclass fields are accessible via __getitem__, '{key}' is not."
            )

        return deepcopy(getattr(self, key))

    def __iter__(self):
        return self.__dataclass_fields__.__iter__  # pylint: disable=no-member

    def __len__(self):
        return self.__dataclass_fields__.__len__  # pylint: disable=no-member

    def to_json(self, **json_dump_kwargs):
        return json.dumps(asdict(self), **json_dump_kwargs)


@dataclass
class _EvaluationBase:
    """Helper dataclass to allow the Evaluation class to have attributes with defaults
    while still having attributes without default values. To make this happen, the
    non-default attributes need to be defined / inherited before the ones with defaults.

    Attributes:
        objectives: [description]

    Raises:
        ValueError: [description]
    """

    objectives: Dict[str, Optional[float]]

    def __post_init__(self):
        available_objective_values = np.array(
            [o for o in self.objectives.values() if o is not None], dtype=float
        )
        if np.isnan(available_objective_values).any():
            raise ValueError(
                f"Objective values contain NaN: {self.objectives}\n"
                + "Please use None instead of NaN."
            )


@dataclass
class Evaluation(EvaluationSpecification, _EvaluationBase):
    """An evaluated specification with a timestamp indicating the time of the
    evaluation, and a result dictionary for all objective values.

    NOTE: `NaN` is not allowed as an objective value, use `None` instead.
    """

    finished_unixtime: float = field(
        default_factory=time.time,
        metadata={"Description": "Timestamp at completion of this evaluation."},
    )

    stacktrace: Optional[str] = field(
        default=None,
        metadata={
            "Description": "The stacktrace in case an unhandled exception occurred "
            + "inside the evaluation function."
        },
    )

    user_info: Optional[dict] = field(
        default=None,
        metadata={"Description": "Miscellaneous information provided by the user."},
    )

    def get_specification(
        self, reset_created_unixtime: bool = False
    ) -> EvaluationSpecification:
        """Get the evaluation specifiation for which this result was evaluated."""
        eval_spec_kwargs = deepcopy(
            dict(
                configuration=self.configuration,
                settings=self.settings,
                optimizer_info=self.optimizer_info,
            )
        )

        if reset_created_unixtime:
            return EvaluationSpecification(
                created_unixtime=time.time(), **eval_spec_kwargs
            )

        return EvaluationSpecification(
            created_unixtime=self.created_unixtime, **eval_spec_kwargs
        )

    @property
    def any_objective_none(self) -> bool:
        return any([v is None for v in self.objectives.values()])

    @property
    def all_objectives_none(self) -> bool:
        return all([v is None for v in self.objectives.values()])


@dataclass
class _EvaluationWithConstraintsBase:
    constraints: Dict[str, Optional[float]] = field(
        metadata={
            "Description": "For each constraint name the float value indicates "
            + "how much the constraint was satisfied, with negative values implying "
            + "a violated and positive values indicating a satisfied constraint."
        },
    )


@dataclass
class EvaluationWithConstraints(Evaluation, _EvaluationWithConstraintsBase):
    pass
