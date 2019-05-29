# Change log

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
