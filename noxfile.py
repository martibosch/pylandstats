"""Task runner for the developer

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
    command = "pdm sync --clean -G doc"
    session.run_install(*command.split(), external=True)
    session.run("sphinx-build", "docs", "docs/_build")


@nox.session
def wheel(session):
    session.install("build", "twine")
    session.run("python", "-m", "build")
    session.run("twine", "check", "dist/*")


@nox.session
def test(session):
    command = "pdm sync --clean -G test"
    session.run_install(*command.split(), external=True)

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
