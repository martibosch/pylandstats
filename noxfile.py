"""Task runner for the developer.

# Usage

```
nox -l            # list of sessions.
nox -s <session>  # execute a session
nox -k <keyword>  # execute some session
```

"""

import os

import nox

os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})
nox.options.reuse_existing_virtualenvs = 1


@nox.session
def doc(session):
    """Build the documentation in a Nox environment."""
    command = "pdm sync --clean -G doc --no-self"
    session.run_install(*command.split(), external=True)

    # for documentation, we don't need the other backends
    command = "pip install . -C setup-args=-Dtransonic-backend=python"
    session.run_install(*command.split(), external=True)

    session.run("sphinx-build", "docs", "docs/_build")
    print(f"file://{os.getcwd()}/docs/_build/index.html")


@nox.session
def wheel(session):
    """Build the wheel."""
    session.install("build", "twine")
    session.run("python", "-m", "build")
    session.run("twine", "check", "dist/*")


@nox.session(venv_backend="mamba")
def test(session):
    """Run the test in a Nox environment."""
    command = "pdm sync --clean --prod -G test --no-self"
    session.run_install(*command.split(), external=True)

    session.conda_install("gdal>=3.3", channels=["conda-forge"])
    session.install(".", "--no-deps", external=True)

    session.run(
        "pytest",
        "-v",
        "-s",
        "--cov=pylandstats",
        "--cov-append",
        "--cov-report=xml",
        "--cov-report",
        "term-missing",
        "tests",
    )
