==================
Landscape analysis
==================

---------------------------
List of implemented metrics
---------------------------

The metrics of PyLandStats `are computed according to its definitions in FRAGSTATS <https://github.com/martibosch/pylandstats-notebooks/blob/master/notebooks/A01-fragstats-comparison.ipynb>`_. 

The notation for the metrics below is as follows:

* the letters with suffixes :math:`a_{i,j}, p_{i,j}, h_{i,j}` respectively represent the area, perimeter, and distance to the nearest neighboring patch of the same class of the patch :math:`j` of class :math:`i`.
* the letters with suffixes :math:`e_{i,k}, g_{i,k}` respectively represent the total edge between and number of pixel adjacencies between classes :math:`i` and :math:`k`
* the capital letters :math:`A, N, E` respectively represent the total area, total number of patches and total edge of the landscape

Like FRAGSTATS, PyLandStats features six distribution-statistics metrics for each patch-level metric, which consist in a statistical aggregation of the values computed for each patch of a class or the whole landscape:

* the mean, which can be computed by adding a  `_mn` suffix to the method name, e.g., `area_mn`
* the area-weighted mean, which can be computed by adding a  `_am` suffix to the method name, e.g., `area_am`
* the median, which can be computed by adding a  `_md` suffix to the method name, e.g., `area_md`
* the range, which can be computed by adding a  `_ra` suffix to the method name, e.g., `area_ra`
* the standard deviation, which can be computed by adding a  `_sd` suffix to the method name, e.g., `area_sd`
* the coefficient of variation, which can be computed by adding a  `_cv` suffix to the method name, e.g., `area_cv`

note that the distribution-statistics metrics do not appear in the documentation below.  

See the `FRAGSTATS documentation <https://www.umass.edu/landeco/research/fragstats/documents/fragstats_documents.html>`_ for more information.

Patch-level metrics
===================

Area, density, edge
-------------------

.. automethod:: pylandstats.Landscape.area
.. automethod:: pylandstats.Landscape.perimeter

Shape
-----

.. automethod:: pylandstats.Landscape.perimeter_area_ratio
.. automethod:: pylandstats.Landscape.shape_index
.. automethod:: pylandstats.Landscape.fractal_dimension

Aggregation
-----------
                
.. automethod:: pylandstats.Landscape.euclidean_nearest_neighbor

Class-level and landscape-level metrics
=======================================

Area, density, edge
-------------------

.. automethod:: pylandstats.Landscape.total_area
.. automethod:: pylandstats.Landscape.proportion_of_landscape
.. automethod:: pylandstats.Landscape.number_of_patches
.. automethod:: pylandstats.Landscape.patch_density
.. automethod:: pylandstats.Landscape.largest_patch_index
.. automethod:: pylandstats.Landscape.total_edge
.. automethod:: pylandstats.Landscape.edge_density

Aggregation
-----------
                
.. automethod:: pylandstats.Landscape.landscape_shape_index
                
Landscape-level metrics
=======================

Contagion, interspersion
------------------------

.. automethod:: pylandstats.Landscape.contagion
.. automethod:: pylandstats.Landscape.shannon_diversity_index
                
                
-----------------------------
Computing metrics data frames
-----------------------------
                
.. automethod:: pylandstats.Landscape.compute_patch_metrics_df
.. automethod:: pylandstats.Landscape.compute_class_metrics_df
.. automethod:: pylandstats.Landscape.compute_landscape_metrics_df

-------------------------
Plotting landscape raster
-------------------------

.. automethod:: pylandstats.Landscape.plot_landscape
