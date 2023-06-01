from pathlib import Path

import pytest

from blackboxopt.examples import multi_objective_multi_param


@pytest.mark.parametrize("example_module", [multi_objective_multi_param])
@pytest.mark.integration_test
def test_full_loop_examples(tmp_path, monkeypatch, example_module):
    if not example_module:
        return

    # Some examples run until a timeout of 60sec. As we are in a hurry,
    # we add/overwrite that argument for the loop:
    run_sequential = example_module.run_optimization_loop

    def run_sequential_mocked(*args, **kwargs):
        kwargs["timeout_s"] = 5
        return run_sequential(*args, **kwargs)

    monkeypatch.setattr(example_module, "run_optimization_loop", run_sequential_mocked)

    # Some examples write output files. Let's redirect those to temporary path:
    if getattr(example_module, "REPORT_PATH", None):
        new_path = tmp_path / Path(example_module.REPORT_PATH).name
        monkeypatch.setattr(example_module, "REPORT_PATH", str(new_path.absolute()))

    # Actually execute the example
    example_module.main()
