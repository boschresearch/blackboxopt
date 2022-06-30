from pathlib import Path

import toml

from blackboxopt import __version__


def test_pyproject_toml_version_matches_dunder_version():
    pyproject_toml = toml.load(Path(__file__).parent.parent / "pyproject.toml")
    assert pyproject_toml["tool"]["poetry"]["version"] == __version__
