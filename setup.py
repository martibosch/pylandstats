# coding=utf-8

import platform
import sys
from io import open  # compatible encoding parameter
from pathlib import Path

from setuptools import find_packages, setup
# pythran imports must go AFTER setuptools imports
# See: https://github.com/pypa/setuptools/issues/309 and https://bit.ly/300HKtK
from transonic.dist import init_transonic_extensions, make_backend_files

if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version >= 3.6 required.")

__version__ = '2.1.3'

classifiers = [
    "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
]

here = Path(__file__).parent.absolute()

# Get the long description from the README file
with open(here / "README.md", encoding="utf-8") as f:
    long_description = f.read()

# get the dependencies and installs
with open(here / "requirements.txt", encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

# Extra dependencies for geometric operations
# we deliberately do not set any lower nor upper bounds on `geopandas`
# dependency so that people might install its cythonized version
geo = ["geopandas", "shapely >= 1.0.0"]

install_requires = [x.strip() for x in all_reqs if "git+" not in x]
dependency_links = [
    x.strip().replace("git+", "") for x in all_reqs if x.startswith("git+")
]

if platform.system() == "Windows":
    backend = "numba"
    install_requires.append("numba")
else:
    backend = "pythran"

paths = ["pylandstats/landscape.py"]
make_backend_files([here / path for path in paths], backend=backend)

if platform.system() == "Linux":
    compile_args = ("-O3", "-DUSE_XSIMD")
else:
    compile_args = ("-O3", )

extensions = init_transonic_extensions("pylandstats",
                                       compile_args=compile_args,
                                       backend=backend)

setup(
    name="pylandstats",
    version=__version__,
    description="Open-source Python library to compute landscape metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=classifiers,
    url="https://github.com/martibosch/pylandstats",
    author="Martí Bosch",
    author_email="marti.bosch@epfl.ch",
    license="GPL-3.0",
    packages=find_packages(exclude=["docs", "tests*"]),
    include_package_data=True,
    install_requires=install_requires,
    extras_require={"geo": geo},
    dependency_links=dependency_links,
    ext_modules=extensions,
)
