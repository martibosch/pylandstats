[![PyPI version fury.io](https://badge.fury.io/py/pylandstats.svg)](https://pypi.python.org/pypi/pylandstats/)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/pylandstats.svg)](https://anaconda.org/conda-forge/pylandstats)
[![Documentation Status](https://readthedocs.org/projects/pylandstats/badge/?version=latest)](https://pylandstats.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://github.com/martibosch/pylandstats/workflows/tests/badge.svg?branch=main)](https://github.com/martibosch/pylandstats/actions?query=workflow%3Atests)
[![Coverage Status](https://coveralls.io/repos/github/martibosch/pylandstats/badge.svg?branch=master)](https://coveralls.io/github/martibosch/pylandstats?branch=master)
[![GitHub license](https://img.shields.io/github/license/martibosch/pylandstats.svg)](https://github.com/martibosch/pylandstats/blob/master/LICENSE)

# PyLandStats

Open-source Pythonic library to compute landscape metrics within the PyData stack (NumPy, pandas, matplotlib...)

**Citation**: Bosch M. 2019. "PyLandStats: An open-source Pythonic library to compute landscape metrics". *PLOS ONE, 14(12), 1-19*. [doi.org/10.1371/journal.pone.0225734](https://doi.org/10.1371/journal.pone.0225734)

## Features

* Read GeoTiff files of land use/cover:

    ```python
    import pylandstats as pls

    ls = pls.Landscape('data/vaud_g100_clc00_V18_5.tif')

    ls.plot_landscape(legend=True)
    ```

    ![landscape-vaud](figures/landscape.png)

* Compute pandas data frames of landscape metrics at the patch, class and landscape level:

    ```python
    class_metrics_df = ls.compute_class_metrics_df(metrics=['proportion_of_landscape', 'edge_density'])
    class_metrics_df
    ```

    | class_val | proportion_of_landscape | edge_density |
    | --------: | ----------------------: | -----------: |
    |         1 |                   7.702 |        4.459 |
    |         2 |                  92.298 |        4.459 |

* Analyze the spatio-temporal evolution of landscapes:

    ```python
    input_fnames = [
        'data/vaud_g100_clc00_V18_5.tif',
        'data/vaud_g100_clc06_V18_5a.tif',
        'data/vaud_g100_clc12_V18_5a.tif'
    ]

    sta = pls.SpatioTemporalAnalysis(
        input_fnames, metrics=[
            'proportion_of_landscape',
            'edge_density',
            'fractal_dimension_am',
            'landscape_shape_index',
            'shannon_diversity_index'
        ], classes=[1], dates=[2000, 2006, 2012],
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for metric, ax in zip(
        ['proportion_of_landscape', 'edge_density', 'fractal_dimension_am'], axes):
        sta.plot_metric(metric, class_val=1, ax=ax)
    fig.suptitle('Class-level metrics (urban)')
    ```

    ![spatiotemporal-analysis](figures/spatiotemporal.png)

* Zonal analysis of landscapes

See the [documentation](https://pylandstats.readthedocs.io/en/latest/?badge=latest) and the [pylandstats-notebooks](https://github.com/martibosch/pylandstats-notebooks) repository for a more complete overview.

## Installation

The easiest way to install PyLandStats is with conda:

    $ conda install -c conda-forge pylandstats

which will install PyLandStats and all of its dependencies. Alternatively, you can install PyLandStats using pip:

    $ pip install pylandstats


Nevertheless, note that in order to define zones by vector geometries in `ZonalAnalysis`, or in order to use the the `BufferAnalysis` and `SpatioTemporalBufferAnalysis` classes, PyLandStats requires [geopandas](https://github.com/geopandas/geopandas), which cannot be installed with pip. If you already have [the dependencies for geopandas](https://geopandas.readthedocs.io/en/latest/install.html#dependencies) installed in your system, you might then install PyLandStats with the `geo` extras as in:

    $ pip install pylandstats[geo]

and you will be able to use the aforementioned features (without having to use conda).

### Development install

To install a development version of PyLandStats, you can first use conda to create an environment with all the dependencies and activate it as in:

    $ conda create -n pylandstats -c conda-forge geopandas matplotlib-base rasterio scipy openblas
    $ conda activate pylandstats

and then clone the repository and use pip to install it in development mode

    $ git clone https://github.com/martibosch/pylandstats.git
    $ cd pylandstats/
    $ pip install -e .

## Acknowledgments

* The computation of the adjacency matrix in [transonic](https://github.com/fluiddyn/transonic) has been implemented by Pierre Augier ([paugier](https://github.com/paugier)).
* With the support of the École Polytechnique Fédérale de Lausanne (EPFL)
* The Corine Land Cover datasets used for the test datasets were produced with funding by the European Union
