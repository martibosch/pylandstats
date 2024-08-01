# Setup a development environment

Once [PDM] and [Nox] are installed, it is very easy to setup a development environment for
PyLandStats. On most systems, one can install PDM and Nox with [Pipx], with something like:

```sh
python3 -m pip install pipx
python3 -m pipx ensurepath
```

and then in a new terminal:

```sh
pipx install pdm
pipx install nox
```

Once PDM is installed, clone the PyLandStats repo and run `make` from the root
directory. This should install a dedicated local virtual environment `.venv`.
You can then activate it and run the tests.

Note that there are few other targets in the `Makefile` useful for developers. In
particular, it is good to periodically recompute the dependencies written in
the `pdm.lock` file (with `make lock`) to check if new packages uploaded on PyPI
do not break PyLandStats. It is reasonable to do this in a dedicated PR.

[nox]: https://nox.thea.codes
[pdm]: https://pdm-project.org
[pipx]: https://github.com/pypa/pipx
