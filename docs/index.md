# PyLandStats documentation!

Open-source library to compute landscape metrics in the Python ecosystem (NumPy, pandas, matplotlib...).

**Citation**: Bosch M. 2019. "PyLandStats: An open-source Pythonic library to compute landscape metrics". _PLOS ONE, 14(12), 1-19_. [doi.org/10.1371/journal.pone.0225734](https://doi.org/10.1371/journal.pone.0225734)

```{toctree}
---
hidden:
maxdepth: 2
---
user-guide
api
changelog
contributing
```

See the [user guide](user-guide.md) for a tutorial and thorough overview of PyLandStats, and the [API reference](api.md) for the API documentation. The data preprocessing pipeline that derives the example datasets used in the user guide is kept in the [pylandstats-notebooks](https://github.com/martibosch/pylandstats-notebooks) repository.

## Features

- Compute pandas DataFrames of landscape metrics at the patch, class and landscape level
- Analyze the spatiotemporal evolution of landscapes
- Analyze landscape changes across environmental gradients (zonal analysis)

## Using PyLandStats

The easiest way to install PyLandStats is with conda:

```bash
conda install -c conda-forge pylandstats
```

which will install PyLandStats and all of its dependencies. Alternatively, you can install PyLandStats using pip:

```bash
pip install pylandstats
```

Nevertheless, note that the `BufferAnalysis` and `SpatioTemporalBufferAnalysis` classes make use of [geopandas](https://github.com/geopandas/geopandas), which cannot be installed with pip. If you already have [the dependencies for geopandas](https://geopandas.readthedocs.io/en/latest/install.html#dependencies) installed in your system, you might then install PyLandStats with the `geo` extras as in:

```bash
pip install pylandstats[geo]
```

and you will be able to use the `BufferAnalysis` and `SpatioTemporalBufferAnalysis` classes (without having to use conda).

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
