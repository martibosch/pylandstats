==================
Landscape analysis
==================

---------------------------
List of implemented metrics
---------------------------

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
