"""pylandstats setup script."""

from pathlib import Path

from setuptools import setup

# pythran imports must go AFTER setuptools imports
# See: https://github.com/pypa/setuptools/issues/309 and https://bit.ly/300HKtK
from transonic.dist import init_transonic_extensions, make_backend_files

here = Path(__file__).parent.absolute()

backend = "pythran"

paths = ["pylandstats/landscape.py"]
make_backend_files([here / path for path in paths], backend=backend)

compile_args = ("-O3", "-DUSE_XSIMD")

extensions = init_transonic_extensions(
    "pylandstats", compile_args=compile_args, backend=backend
)

setup(
    ext_modules=extensions,
)
