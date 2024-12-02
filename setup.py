"""Build extensions."""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        name="adjacency",  # Name of the compiled module
        sources=["pylandstats/adjacency.pyx"],  # Path to your .pyx file
        include_dirs=[np.get_include()],  # Include NumPy headers
    )
]

setup(
    ext_modules=cythonize(extensions),
)
