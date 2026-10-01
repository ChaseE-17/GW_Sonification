"""Execute the teaching notebooks top to bottom (``pytest --notebooks``; needs network)."""

from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).parents[1] / "notebooks").glob("*.ipynb"))


@pytest.mark.notebook
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs(path, tmp_path, monkeypatch):
    import nbformat
    from nbclient import NotebookClient

    monkeypatch.chdir(tmp_path)
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={
        "metadata": {"path": str(tmp_path)}}).execute()
