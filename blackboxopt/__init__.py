__version__ = "4.6.1"

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
