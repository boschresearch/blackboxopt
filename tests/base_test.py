# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from blackboxopt.base import (
    Objective,
    ObjectivesError,
    _raise_on_duplicate_objective_names,
    raise_on_unknown_or_incomplete,
)


def test_raise_on_unknown_or_incomplete_objectives():
    raise_on_unknown_or_incomplete(
        exception=ObjectivesError,
        known=["mse"],
        reported=["mse"],
    )

    with pytest.raises(ObjectivesError) as exception:
        raise_on_unknown_or_incomplete(
            exception=ObjectivesError,
            known=["mse"],
            reported=["mse", "surprise"],
        )
    assert "mse" in str(exception.value).lower()
    assert "unknown" in str(exception.value).lower()
    assert "surprise" in str(exception.value)

    with pytest.raises(ObjectivesError) as exception:
        raise_on_unknown_or_incomplete(
            exception=ObjectivesError,
            known=["mse", "r²"],
            reported=["mse"],
        )
    assert "missing" in str(exception.value).lower()
    assert "r²" in str(exception.value)


def test_raise_on_duplicate_objective_names():
    objectives_without_duplicates = [
        Objective("mse", False),
        Objective("R²", True),
        Objective("UnicornSparkles", True),
    ]
    _raise_on_duplicate_objective_names(objectives_without_duplicates)

    objectives_with_duplicates = objectives_without_duplicates + [
        Objective("mse", False),
        Objective("mse", False),
        Objective("R²", True),
    ]
    with pytest.raises(ValueError) as exception:
        _raise_on_duplicate_objective_names(objectives_with_duplicates)

    assert "('R²', 2)" in str(exception.value)
    assert "('mse', 3)" in str(exception.value)
