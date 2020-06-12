# Change log

## 2.1.1 (12/06/2020)

* corrected `ZonalGridAnalysis` docstring
* corrected computation of `num_zone_cols` in `ZonalGridAnalysis`

## 2.1.0 (11/05/2020)

* configured flake8 in setup.cfg and added flake8 test in travis.yml
* added `compute_zonal_statistics` method to `ZonalAnalysis` and aded `ZonalGridAnalysis` class
* Implemented the `effective_mesh_size` metric

## 2.0.0a1 (24/09/2019)

* gradient -> zonal
* corrected cmap arg: default from rcParams, accept str
* updates in the docs (changed link to notebook + cleaned Makefile)

## 2.0.0a0 (20/09/2019)

* corrected shapely version in setup.py (bumpversion messed it)
* updated bumpversion for release candidates
* catch TypeError for existing metrics but at the wrong level
* always import modules, not methods/classes
* consistent API: all data frames obtained with `compute` methods

## 1.1.1 (18/09/2019)

* corrected rst typo in `total_area` docs
* fixed missing perimeter distribution statistic metric and dried class constants definitions
* fixed patch edge array computation when computing ENN (otherwise the speed-up of 1.1.0 is not effective)

## 1.1.0 (17/09/2019)

* speed-up (~x2) in `Landscape.compute_patch_euclidean_nearest_neighbor`: compute pixel-to-pixel distances for patch edges only

## 1.0.2 (25/07/2019)

* fix landscape array dtype in `SpatioTemporalBufferAnalysis`
* included `LICENSE` in `MANIFEST.in`

## 1.0.1 (24/07/2019)

* deleted Python 2 classifiers in `setup.py`
* fix ValueError message for `landscape_crs` and `landscape_transform` in `BufferAnalysis`
* fix landscape array dtype in `GradientAnalysis` and `BufferAnalysis`

## 1.0.0 (18/07/2019)

* dropped Python 2 support
* added `SpatioTemporalBufferAnalysis.plot_landscapes` method
* added `buffer_dist_legend` argument and docs in `SpatioTemporalBufferAnalysis. plot_metric`
* fix proper metric data frame properties in `SpatioTemporalBufferAnalysis`
* pass `transform` argument when initializating `MultiLandscape` instances (i.e., `SpatioTemporalAnalysis`, `BufferAnalysis`, `GradientAnalysis` and `SpatioTemporalBufferAnalysis`)
* `plot_landscape` and `plot_landscapes` with rasterio.plot.show
* changed `feature_{name,values}` for `attribute_{name,values}` in `MultiLandscape` (abstract) class
* dropped `plot_metrics` method

## 0.6.1 (02/07/2019)

* cell width and length comparisons with `numpy.isclose` to deal with imprecisions that come from float pixel resolutions (e.g., in GeoTIFF files)
* critical fix: moved pythran signatures from `compute.pythran` to `compute.py` so that the build is properly done when pip-installing

## 0.6.0 (01/07/2019)

* flat array approach to the computation of the adjacency matrix with pythran to improve performance (plus fixes a bug on the computation of `total_edge`)
* initialization of `scipy.spatial.cKDTree` with keyword arguments `balanced_tree` and `compact_nodes` set to `False`

## 0.5.0 (28/05/2019)

* methods plotting multiple axes return only the figure instead of a tuple with the figure and the axes
* settings module that allow configuring metrics' labels in plots, defaulting to FRAGSTATS abbreviations
* chaged CRS in tests to work with pyproj >= 2.0.0
* corrected `figlength` for `figwidth`
* warn when computing metrics that require an unmet minimum number of patches or classes (and their computation returns nan)
* all tests with `unittest.TestCase` assert methods

## 0.4.1 (03/04/2019)

* added docstrings for `MultiLandscape`, `GradientAnalysis` and `BufferAnalysis`
* raise `ValueError` when using buffer rings around a polygon geometry

## 0.4.0 (03/04/2019)

* implemented `SpatioTemporalBufferAnalysis` with a dedicated `plot_metric` method
* added buffer ring-wise analysis through a `buffer_rings` boolean argument in `BufferAnalysis`

## 0.3.1 (29/03/2019)

* float equality comparations with numpy `isclose` method

## 0.3.0 (28/03/2019)

* implemented `GradientAnalysis` and `BufferAnalysis`
* added optional `geopandas` dependences
* created abstract `MultiLandscape` class
* `Landscape` initialization from ndarray or geotiff (dropped `read_geotiff` method)
* implemented `contagion`
* convolution-based adjacency dataframe
* fixed bug with `class_cond` in `Landscape.compute_arr_edge`

## 0.2.0 (18/03/2019)

* implemented `euclidean_nearest_neighbor` and all its corresponding class/landscape distribution statistic metrics
* fixed dtype of `self.classes` in `landscape.Landscape`
* set default argument `nodata=None` (instead of `nodata=0`) for `landscape.read_geotiff`
* changed test input data files
* implemented `__len__` of `spatiotemporal.SpatioTemporalAnalysis`

## 0.1.1 (22/01/2019)

* corrected if-else statements involving `None` variables

## 0.1.0 (22/01/2019)

* initial release
