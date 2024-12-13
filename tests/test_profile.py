from pathlib import Path
import pytest
import sys

sys.path.append(str(Path(__file__).parent.parent / "profile"))
from run_profile import __main__  # noqa: E402


@pytest.mark.parametrize("method", ["sample", "variational"])
@pytest.mark.parametrize("parameterization", [
    "graph_centered", "graph_non_centered", "fourier_centered", "fourier_non_centered",
    "standard_centered", "standard_non_centered"
])
def test_run_profile(method, parameterization) -> None:
    __main__([method, parameterization, "1", "--n=16", "--ignore_converged"])
