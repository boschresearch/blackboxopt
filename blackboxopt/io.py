# Copyright (c) 2023 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

import json
import lzma
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import parameterspace as ps

from blackboxopt.base import Objective
from blackboxopt.evaluation import Evaluation


def save_study_as_json(
    search_space: ps.ParameterSpace,
    objectives: List[Objective],
    evaluations: List[Evaluation],
    json_file_path: os.PathLike,
    overwrite: bool = False,
):
    """Save space, objectives and evaluations as json at `json_file_path`."""
    _file_path = Path(json_file_path)
    if not _file_path.parent.exists():
        raise IOError(
            f"The parent directory for {_file_path} does not exist, please create it."
        )
    if _file_path.exists() and not overwrite:
        raise IOError(f"{_file_path} exists and overwrite is False")

    with open(_file_path, "w", encoding="UTF-8") as fh:
        json.dump(
            {
                "search_space": search_space.to_dict(),
                "objectives": [o.__dict__ for o in objectives],
                "evaluations": [e.__dict__ for e in evaluations],
            },
            fh,
        )


def load_study_from_json(
    json_file_path: os.PathLike,
) -> Tuple[ps.ParameterSpace, List[Objective], List[Evaluation]]:
    """Load space, objectives and evaluations from a given `json_file_path`."""
    with open(json_file_path, "r", encoding="UTF-8") as fh:
        study = json.load(fh)

    search_space = ps.ParameterSpace.from_dict(study["search_space"])
    objectives = [Objective(**o) for o in study["objectives"]]
    evaluations = [Evaluation(**e) for e in study["evaluations"]]

    return search_space, objectives, evaluations


def save_study_as_pickle(
    search_space: ps.ParameterSpace,
    objectives: List[Objective],
    evaluations: List[Evaluation],
    pickle_file_path: os.PathLike,
    overwrite: bool = False,
):
    """Save space, objectives and evaluations as an lzma compressed pickle."""
    _file_path = Path(pickle_file_path)
    if not _file_path.parent.exists():
        raise IOError(
            f"The parent directory for {_file_path} does not exist, please create it."
        )
    if _file_path.exists() and not overwrite:
        raise IOError(f"{_file_path} exists and overwrite is False")

    with lzma.open(_file_path, "wb") as fh:
        pickle.dump(
            {
                "search_space": search_space,
                "objectives": objectives,
                "evaluations": evaluations,
            },
            fh,
        )


def load_study_from_pickle(
    pickle_file_path: os.PathLike,
) -> Tuple[ps.ParameterSpace, List[Objective], List[Evaluation]]:
    """Load space, objectives and evaluations from a given lzma compressed pickle."""
    with lzma.open(pickle_file_path, "rb") as fh:
        study = pickle.load(fh)

    return study["search_space"], study["objectives"], study["evaluations"]
