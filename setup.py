"""Build extensions."""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        name="pylandstats.adjacency",
        sources=["pylandstats/adjacency.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(extensions),
)
