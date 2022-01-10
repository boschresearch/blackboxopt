# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from blackboxopt.evaluation import Evaluation, EvaluationSpecification


def test_evaluation_with_inf_values():
    Evaluation({"mse": float("Inf"), "r²": 0.2}, {}, {})
    Evaluation({"mse": float("-Inf"), "r²": 0.2}, {}, {})


def test_evaluation_with_optional_objective_values():
    Evaluation({"mse": None, "r²": 0.2}, {}, {})
    Evaluation({"mse": None, "r²": None}, {}, {})


def test_evaluation_with_nan_objective_value():
    with pytest.raises(ValueError):
        Evaluation({"mse": float("NaN"), "r²": 0.2}, {}, {})


def test_unpack_specification_into_result():
    result = Evaluation(
        {"mse": 0.0, "r²": 1.0},
        **EvaluationSpecification(
            configuration={"p1": 1.2},
            settings={"fidelity": 1.0},
            optimizer_info={"id": 123},
        ),
    )
    assert "p1" in result.configuration
    assert "id" in result.optimizer_info
    assert "fidelity" in result.settings


def test_evaluate_specification_into_result():
    evaluation_spec = EvaluationSpecification(
        configuration={"p1": 1.2},
        settings={"fidelity": 1.0},
        optimizer_info={"id": 123},
    )
    result = evaluation_spec.create_evaluation(objectives={"mse": 0.0, "r²": 1.0})
    assert "p1" in result.configuration
    assert "id" in result.optimizer_info
    assert "fidelity" in result.settings


def test_evaluation_result_independent_from_specification():
    spec = EvaluationSpecification(
        configuration={"p1": 1.2},
        settings={"fidelity": 1.0},
        optimizer_info={"id": 123},
    )

    result = Evaluation(objectives={"mse": 0.0, "r²": 1.0}, **spec)
    result.configuration["p1"] = -1.0
    assert spec.configuration["p1"] == 1.2
    assert result.configuration["p1"] == -1.0


def test_get_specification_from_result_is_independent():
    spec = EvaluationSpecification(
        configuration={"p1": 1.2},
        settings={"fidelity": 1.0},
        optimizer_info={"id": 123},
    )

    result = Evaluation(objectives={"mse": 0.0, "r²": 1.0}, **spec)
    new_spec = result.get_specification()

    new_spec.settings["fidelity"] = 2.0
    assert new_spec.settings["fidelity"] == 2.0
    assert result.settings["fidelity"] == 1.0
    assert spec.settings["fidelity"] == 1.0


def test_to_json():
    spec = EvaluationSpecification(
        configuration={"p1": 1.2},
        settings={"fidelity": 1.0},
        optimizer_info={"id": 123},
        created_unixtime=1.0,
    )
    spec_json = spec.to_json()
    assert spec_json == (
        '{"configuration": {"p1": 1.2}, "settings": {"fidelity": 1.0}, '
        '"optimizer_info": {"id": 123}, "created_unixtime": 1.0, "context": null}'
    )

    result = Evaluation(
        objectives={"mse": 0.0, "r²": 1.0}, finished_unixtime=2.0, **spec
    )
    result_json = result.to_json()
    assert result_json == (
        '{"objectives": {"mse": 0.0, "r\\u00b2": 1.0}, "configuration": {"p1": 1.2}, '
        '"settings": {"fidelity": 1.0}, "optimizer_info": {"id": 123}, '
        '"created_unixtime": 1.0, "context": null, "constraints": null, '
        '"finished_unixtime": 2.0, "stacktrace": null, "user_info": null}'
    )


def test_to_dict():
    spec = EvaluationSpecification(
        configuration={"p1": 1.2},
        settings={"fidelity": 1.0},
        optimizer_info={"id": 123},
        created_unixtime=1.0,
    )
    spec_dict = spec.dict()
    assert spec_dict["configuration"] == spec.configuration
    assert spec_dict["settings"] == spec.settings
    assert spec_dict["optimizer_info"] == spec.optimizer_info
    assert spec_dict["created_unixtime"] == spec.created_unixtime
    assert spec_dict["context"] is None

    result = Evaluation(
        objectives={"mse": 0.0, "r²": 1.0}, finished_unixtime=2.0, **spec
    )
    result_dict = result.dict()
    assert result_dict["objectives"] == result.objectives
    assert result_dict["finished_unixtime"] == result.finished_unixtime
    assert result_dict["user_info"] is None
    assert result_dict["stacktrace"] is None


def test_get_specification_from_evaluation():
    eval_spec = EvaluationSpecification(
        configuration={"p1": 1.2},
        settings={"fidelity": 1.0},
        optimizer_info={"id": 123},
        context={"temperature": 25.3},
    )
    result = eval_spec.create_evaluation({"mse": 0.0, "r²": 1.0})

    assert result.get_specification() == eval_spec

    derived_eval_spec = result.get_specification(reset_created_unixtime=True)
    assert result.created_unixtime != derived_eval_spec.created_unixtime
    assert result.configuration == derived_eval_spec.configuration
    assert result.settings == derived_eval_spec.settings
    assert result.optimizer_info == derived_eval_spec.optimizer_info
