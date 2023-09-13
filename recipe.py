from cook import create_task, Task
from cook.contexts import create_group
import itertools as it
import numpy as np
import os
from pathlib import Path
from typing import Literal

FAST = "FAST" in os.environ


create_task("requirements", action="pip-compile -v", targets=["requirements.txt"],
            dependencies=["requirements.in"])
create_task("pip-sync", action="pip-sync", dependencies=["requirements.txt"])


# Run the profiling experiments with different sample sizes and noise scales.
SIZES = 16 * 2 ** np.arange(11)
FOURIER_ONLY_SIZE_THRESHOLD = 16 * 2 ** 9
LOG10_NOISE_SCALES = np.linspace(-1, 1, 7)
PARAMETERIZATIONS = [
    "graph_centered", "graph_non_centered", "fourier_centered", "fourier_non_centered",
    "standard_centered", "standard_non_centered"
]


def create_profile_task(
        method: Literal["sample", "variational"], parameterization: str, log10_sigma: float,
        size: int, max_chains: int = None,
        timeout: float = None, iter_sampling: int = None, train_frac: float = 1, suffix: str = ""
        ) -> Task:
    """
    Create a task for a profiling.

    Args:
        method: Stan inference method to use (see :code:`cmdstanpy.CmdStanModel.[method]`).
        parameterizations: One of the parameterizations in the :code:`PARAMETERIZATIONS` variable
            above.
        log10_sigma: Observation noise scale; the marginal variance of the GP is 1.
        size: Number of observations per example.
        max_chains: Maximum number of chains to run per example. This limits the amount of time
            spent on small :code:`size` experiments because we don't need to run the experiment for
            the whole :code:`timeout` seconds to get a good idea of sampling time.
        timeout: Maximum duration to run an experiment for before aborting.
        iter_sampling: Number of posterior draws after warmup.
        train_frac: Fraction of observations to use for fitting.
        suffix: Suffix to add to the name of the task and output filename.

    Returns:
        Task to run the profiling experiment.
    """
    timeout = timeout or (10 if FAST else 60)
    max_chains = max_chains or (2 if FAST else 20)
    iter_sampling = iter_sampling or (10 if FAST else 100)
    name = f"log10_noise_scale-{log10_sigma:.3f}_size-{size}{suffix}"
    target = f"profile/results/{method}/{parameterization}/{name}.pkl"
    args = [
        "python", "profile/run_profile.py", method, parameterization, 10 ** log10_sigma, target,
        f"--iter_sampling={iter_sampling}", f"--n={size}", f"--max_chains={max_chains}",
        f"--timeout={timeout}", f"--train_frac={train_frac}",
    ]
    dependencies = [
        "profile/run_profile.py",
        "profile/data.stan",
        f"profile/{parameterization}.stan",
    ]
    create_task(name=f"profile:{method}-{parameterization}-{name}", action=args, targets=[target],
                dependencies=dependencies)


profile_group: create_group
with create_group("profile") as profile_group:
    product = it.product(PARAMETERIZATIONS, LOG10_NOISE_SCALES, SIZES)
    for parameterization, log10_sigma, size in product:
        # Only run Fourier methods if the size threshold is exceeded.
        if size >= FOURIER_ONLY_SIZE_THRESHOLD and not parameterization.startswith("fourier"):
            continue
        create_profile_task("sample", parameterization, log10_sigma, size)

    # Add variational inference.
    for parameterization, log10_sigma in it.product(PARAMETERIZATIONS, LOG10_NOISE_SCALES):
        create_profile_task("variational", parameterization, log10_sigma, 1024, train_frac=0.8)
        # Here, we use a long timeout and many samples to ensure we get the distributions right.
        create_profile_task(
            "sample", parameterization, log10_sigma, 1024, train_frac=0.8, suffix="-train-test",
            iter_sampling=100 if FAST else 500, timeout=60 if FAST else 300
        )

    # Add a one-off task to calculate statistics for the abstract with 10k observations.
    create_profile_task("sample", "fourier_centered", 0, 10_000, timeout=300)
    create_profile_task("sample", "fourier_non_centered", 0, 10_000, timeout=300)


# Run the notebooks to generate figures.
figures = []
for example in ["kernels", "padding", "profile", "trees", "tube"]:
    ipynb = Path(example, f"{example}.ipynb")
    md = ipynb.with_suffix(".md")
    create_task(f"{example}:nb", dependencies=[md], targets=[ipynb],
                action=f"jupytext --to notebook {md}")
    targets = [ipynb.with_suffix(".png"), ipynb.with_suffix(".html")]
    task = create_task(
        f"{example}:fig", dependencies=[ipynb], targets=targets,
        action=f"jupyter nbconvert --to=html --execute --ExecutePreprocessor.timeout=-1 {ipynb}"
    )
    if example == "profile":
        task.task_dependencies.append(profile_group.task)
    figures.append(targets[0])


# Task that reproduces all outputs.
create_task("figures", dependencies=figures)


def delete_compiled_stan_files(_: Task) -> None:
    # Find all Stan files and remove compiled versions if they exist.
    for path in Path(".").glob("**/*.stan"):
        path = path.with_suffix("")
        if path.is_file():
            os.unlink(path)
            print(f"removed {path}")


create_task("rm-compiled", action=delete_compiled_stan_files)
