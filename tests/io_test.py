# Copyright (c) 2023 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0


import parameterspace as ps
import pytest

import blackboxopt as bbo


def test_save_and_load_study_pickle(tmp_path):
    tmp_file = tmp_path / "out.json"

    search_space = ps.ParameterSpace()
    search_space.add(ps.IntegerParameter("p", (-10, 10)))
    objectives = [bbo.Objective("loss", False), bbo.Objective("score", True)]
    evaluations = [
        bbo.Evaluation(configuration={"p": 1}, objectives={"loss": 1.0, "score": 3.0}),
        bbo.Evaluation(configuration={"p": 2}, objectives={"loss": 0.1, "score": 2.0}),
        bbo.Evaluation(configuration={"p": 3}, objectives={"loss": 0.0, "score": 1.0}),
    ]
    bbo.io.save_study_as_pickle(search_space, objectives, evaluations, tmp_file)

    # Check that default overwrite=False causes IOError on existing file
    with pytest.raises(IOError, match=str(tmp_file)):
        bbo.io.save_study_as_pickle(search_space, objectives, evaluations, tmp_file)

    loaded_study = bbo.io.load_study_from_pickle(tmp_file)

    assert loaded_study[1] == objectives
    assert loaded_study[2] == evaluations
    for _ in range(128):
        assert search_space.sample() == loaded_study[0].sample()


def test_save_and_load_study_pickle_fails_on_missing_output_directory():
    pickle_file_path = "/this/directory/does/not/exist/pickles/out.pkl"
    with pytest.raises(IOError, match=pickle_file_path):
        bbo.io.save_study_as_pickle(
            search_space=ps.ParameterSpace(),
            objectives=[],
            evaluations=[],
            pickle_file_path=pickle_file_path,
        )


def test_save_and_load_study_json(tmp_path):
    tmp_file = tmp_path / "out.json"

    search_space = ps.ParameterSpace()
    search_space.add(ps.IntegerParameter("p", (-10.0, 10.0)))
    objectives = [bbo.Objective("loss", False), bbo.Objective("score", True)]
    evaluations = [
        bbo.Evaluation(configuration={"p": 1}, objectives={"loss": 1.0, "score": 3.0}),
        bbo.Evaluation(configuration={"p": 2}, objectives={"loss": 0.1, "score": 2.0}),
        bbo.Evaluation(configuration={"p": 3}, objectives={"loss": 0.0, "score": 1.0}),
    ]
    bbo.io.save_study_as_json(search_space, objectives, evaluations, tmp_file)

    # Check that default overwrite=False causes ValueError on existing file
    with pytest.raises(IOError):
        bbo.io.save_study_as_json(search_space, objectives, evaluations, tmp_file)

    loaded_study = bbo.io.load_study_from_json(tmp_file)

    assert loaded_study[1] == objectives
    assert loaded_study[2] == evaluations
    for _ in range(128):
        assert search_space.sample() == loaded_study[0].sample()


def test_save_and_load_study_json_fails_on_missing_output_directory():
    json_file_path = "/this/directory/does/not/exist/jsons/out.json"
    with pytest.raises(IOError, match=json_file_path):
        bbo.io.save_study_as_json(
            search_space=ps.ParameterSpace(),
            objectives=[],
            evaluations=[],
            json_file_path=json_file_path,
        )


def test_save_and_load_study_json_fails_with_complex_type_in_evaluation(tmp_path):
    tmp_file = tmp_path / "out.json"

    search_space = ps.ParameterSpace()
    objectives = []
    evaluations = [
        bbo.Evaluation(
            configuration={},
            objectives={},
            user_info={"complex typed value": ps.ParameterSpace()},
        ),
    ]

    with pytest.raises(TypeError):
        bbo.io.save_study_as_json(search_space, objectives, evaluations, tmp_file)
