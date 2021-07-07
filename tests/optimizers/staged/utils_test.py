# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0

from blackboxopt import Evaluation, Objective
from blackboxopt.optimizers.staged.utils import best_evaluation_at_highest_fidelity


def test_best_evaluation_at_highest_fidelity():
    assert best_evaluation_at_highest_fidelity([], Objective("loss", False)) is None

    evals = [
        Evaluation({"loss": 0.1, "score": 1.0}, {}, {"fidelity": 0.4}),
        Evaluation({"loss": 1.0, "score": None}, {}, {"fidelity": 0.4}),
        Evaluation({"loss": None, "score": None}, {}, {"fidelity": 1.0}),
        Evaluation({"loss": 42.0, "score": None}, {}, {"fidelity": 0.9}),
    ]
    best = best_evaluation_at_highest_fidelity(evals, Objective("loss", False))
    assert best is not None
    assert best.settings["fidelity"] == 0.9
    assert best.objectives["loss"] == 42.0
