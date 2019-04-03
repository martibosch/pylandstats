# Change log

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
