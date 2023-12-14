from pathlib import Path
import pytest
import re
import subprocess


markdown_paths = [path for path in Path(".").glob("*/*.md") if not str(path).startswith(".")]


@pytest.mark.parametrize("markdown_path", markdown_paths, ids=map(str, markdown_paths))
def test_notebooks(markdown_path: Path, tmp_path: Path) -> None:
    # Check the notebook exists.
    notebook_path = markdown_path.with_suffix(".ipynb")
    if not notebook_path.is_file():
        raise FileNotFoundError(notebook_path)

    # Generate the notebook in a temp directory.
    tmp_notebook_path = tmp_path / notebook_path.name
    subprocess.check_call(["jupytext", str(markdown_path), "--output", str(tmp_notebook_path)])

    # Check that they're up to date after stripping random ids.
    pattern = re.compile(r'"id": "\w+"')
    actual_text = pattern.sub("", tmp_notebook_path.read_text())
    expected_text = pattern.sub("", notebook_path.read_text())
    assert actual_text == expected_text, f"{markdown_path} and {notebook_path} do not contain " \
        "equivalent content. Do you need to update the notebook?"
