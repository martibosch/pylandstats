PyLandStats documentation!
======================================

Open-source Pythonic library to compute landscape metrics within the PyData stack (NumPy, pandas, matplotlib...)

.. toctree::
   :maxdepth: 1
   :caption: Reference Guide:
             
   landscape
   spatiotemporal
   gradient
   spatiotemporal-buffer

.. toctree::
   :maxdepth: 1
   :caption: Development:
             
   changelog
   contributing

This documentation is intended as an API reference. See also:

* `pylandstats-notebooks <https://github.com/martibosch/pylandstats-notebooks>`_ repository (tutorial/thorough overview of PyLandStats)
* `swiss-urbanization <https://github.com/martibosch/swiss-urbanization>`_ repository (example application of PyLandStats to evaluate the spatiotemporal patterns of urbanization in three Swiss urban agglomerations)
             
Features
--------

* Compute pandas DataFrames of landscape metrics at the patch, class and landscape level
* Analyze the spatiotemporal evolution of landscapes
* Analyze landscape changes accross environmental gradients

Using PyLandStats
-----------------

To install use pip:

.. code-block:: bash

    $ pip install pylandstats


If you want to use the ``BufferAnalysis`` class, you will need `geopandas <https://github.com/geopandas/geopandas>`_. The easiest way to install it is via `conda-forge <https://conda-forge.org/>`_ as in:

.. code-block:: bash
                
    $ conda install -c conda-forge geopandas
    
and then install PyLandStats with the ``geo`` extras as in:

.. code-block:: bash
                
    $ pip install pylandstats[geo]
   
Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
