# Change log

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
