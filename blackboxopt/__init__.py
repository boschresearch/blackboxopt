__version__ = "4.6.2"

from parameterspace import ParameterSpace

from .base import (
    ConstraintsError,
    ContextError,
    EvaluationsError,
    Objective,
    ObjectivesError,
    OptimizationComplete,
    Optimizer,
    OptimizerNotReady,
)
from .evaluation import Evaluation, EvaluationSpecification
from .utils import sort_evaluations
