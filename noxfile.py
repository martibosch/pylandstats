"""Nox."""

import nox

PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12", "3.13"]


@nox.session(python=PYTHON_VERSIONS, venv_backend="mamba|micromamba|conda")
def tests(session):
    """Run tests with pytest, and collect coverage."""
    session.conda_install("gdal", channel=["conda-forge"])
    session.install(".[dev,test]")

    session.run(
        "pytest",
        "-s",
        "--cov=pylandstats",
        "--cov-append",
        "--cov-report=xml",
        "--cov-report=term-missing",
        "tests",
    )


@nox.session(name="build", venv_backend="mamba|micromamba|conda")
def build(session):
    """Build package and documentation."""
    session.conda_install("gdal", channel=["conda-forge"])
    session.install(".[doc,dev,test]")

    session.run("python", "-m", "build")
    session.run("sphinx-build", "docs", "docs/_build")
    session.run("twine", "check", "dist/*")
