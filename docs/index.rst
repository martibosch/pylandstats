PyLandStats documentation!
======================================

Open-source library to compute landscape metrics in the Python ecosystem (NumPy, pandas, matplotlib...).

**Citation**: Bosch M. 2019. "PyLandStats: An open-source Pythonic library to compute landscape metrics". *PLOS ONE, 14(12), 1-19*. `doi.org/10.1371/journal.pone.0225734 <https://doi.org/10.1371/journal.pone.0225734>`_

.. toctree::
   :maxdepth: 1
   :caption: User Guide:

   user-guide/overview
   user-guide/landscape-analysis
   user-guide/spatiotemporal-analysis
   user-guide/zonal-analysis
   user-guide/spatiotemporal-zonal-analysis
   user-guide/spatial-signature-analysis

.. toctree::
   :maxdepth: 1
   :caption: Reference Guide:

   landscape
   spatiotemporal
   zonal
   spatiotemporal-zonal
   spatial-signature-analysis

.. toctree::
   :maxdepth: 1
   :caption: Development:

   changelog
   contributing

See the user guide above for a tutorial and thorough overview of PyLandStats, and the reference guide for the API documentation. The data preprocessing pipeline that derives the example datasets used in the user guide is kept in the `pylandstats-notebooks <https://github.com/martibosch/pylandstats-notebooks>`_ repository.

Features
--------

* Compute pandas DataFrames of landscape metrics at the patch, class and landscape level
* Analyze the spatiotemporal evolution of landscapes
* Analyze landscape changes across environmental gradients (zonal analysis)

Using PyLandStats
-----------------

The easiest way to install PyLandStats is with conda:

    $ conda install -c conda-forge pylandstats

which will install PyLandStats and all of its dependencies. Alternatively, you can install PyLandStats using pip:

    $ pip install pylandstats

Nevertheless, note that the `BufferAnalysis` and `SpatioTemporalBufferAnalysis` classes make use of `geopandas <https://github.com/geopandas/geopandas>`_, which cannot be installed with pip. If you already have `the dependencies for geopandas <https://geopandas.readthedocs.io/en/latest/install.html#dependencies>`_ installed in your system, you might then install PyLandStats with the `geo` extras as in:

    $ pip install pylandstats[geo]

and you will be able to use the `BufferAnalysis` and `SpatioTemporalBufferAnalysis` classes (without having to use conda).




Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
