import optuna
import pytest

from blackboxopt.optimizers.optuna_wrapper import OptunaBasedSingleObjectiveOptimizer
from blackboxopt.optimizers.testing import ALL_REFERENCE_TESTS


@pytest.mark.parametrize("reference_test", ALL_REFERENCE_TESTS)
def test_optuna_based_single_objective_optimizer(reference_test):
    reference_test(
        OptunaBasedSingleObjectiveOptimizer,
        optimizer_kwargs={
            "sampler_factory": lambda seed: optuna.samplers.RandomSampler(seed=seed)
        },
    )
