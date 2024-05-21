# Change log

## v3.0.0rc2 (2024-05-21)

### :boom: BREAKING CHANGES

- due to [`bbef272`](https://github.com/martibosch/pylandstats/commit/bbef272ca515193410267dbec6d1b0e6c9a73b9d) - rename kws -> kwargs *(commit by @martibosch)*:

  rename kws -> kwargs

### :sparkles: New Features

- [`2dafaaf`](https://github.com/martibosch/pylandstats/commit/2dafaaff35e3caffae52b8ed4507818c54e1a13a) - use dask to compute metrics' dataframes in parallel *(commit by @martibosch)*
- [`a47f333`](https://github.com/martibosch/pylandstats/commit/a47f333294b311b02fafd54131ca72758559ce03) - multilandscape.landscape_ser, DRY compute metrics dfs, dask *(commit by @martibosch)*
- [`d480243`](https://github.com/martibosch/pylandstats/commit/d480243c43326fe831bb354a9d4d17fcba47d1e9) - single-class (or landscape-level) zonal gdf *(commit by @martibosch)*

### :bug: Bug Fixes

- [`ea7cde5`](https://github.com/martibosch/pylandstats/commit/ea7cde54e9dea2539abdae0356a6d02384f0e606) - missing fragstats abbrevs for core (also use f-strings) *(commit by @martibosch)*
- [`d074f32`](https://github.com/martibosch/pylandstats/commit/d074f32629f7837445d5a7dae7afe6e52940d507) - len in SpatioTemporalZonalAnalysis *(commit by @martibosch)*
- [`1ce4a4e`](https://github.com/martibosch/pylandstats/commit/1ce4a4e764a0ff10b25cb066dc8bf26a7676ae97) - min two classes warning for entropy (credits to @kareed1) *(commit by @martibosch)*
- [`72c1cc9`](https://github.com/martibosch/pylandstats/commit/72c1cc9e90dc093eb90edff4fc603167be81f551) - except zerodivisionerror in disjunct_core_area_am *(commit by @martibosch)*
- [`6fa816c`](https://github.com/martibosch/pylandstats/commit/6fa816ccd79c948b02df3e7d77ba672cf52cb952) - catch fiona DriverError for `zones` when fiona is installed *(commit by @martibosch)*

### :recycle: Refactors

- [`bbef272`](https://github.com/martibosch/pylandstats/commit/bbef272ca515193410267dbec6d1b0e6c9a73b9d) - rename kws -> kwargs *(commit by @martibosch)*

## v3.0.0rc1 (2023-11-20)

### Feat

- core area metrics (#45)

### Fix

- apply `to_numeric` in stza compute_class_metrics_df
- fillna with 0 for core area metrics

## v3.0.0rc0 (2023-09-11)

### Feat

- 3.0.0rc0 (#40)

### Fix

- manual cbuildwheel steps (until action is in marketplace) (#28)
- single quotes in release.yml (see actions/runner/issues/866) (#27)
- manage dtype for adj/reclassif arrays; fix reclassif arr init

## 2.4.2 (19/11/2021)

- fix sum(level) -> groupby(level).sum (pandas deprecation)
- dropped six for metaclasses (using direct ABC inheritance)

## 2.4.1 (19/11/2021)

- GitHub Actions tests in develop branch too

## 2.4.0 (13/11/2021)

- added methods to compute information theory-based metrics by Nowosad et Stepinski (2019).
- using pre-commit with black, isort, flake8 and pydocstyle
- CI (tests) and CD (release to PyPI) with github actions

## 2.3.0 (23/04/2021)

- improved exception caching in compute metric data frames methods
- init `BufferAnalysis` with `Landscape` with non-None `transform`
- test required init args in `Landscape` and `ZonalGridAnalysis`
- test `ValueError` in data frames with patch/class/landscape-only metrics
- several corrections in docstrings/comments/error msgs
- fix: removed `SpatioTemporalBufferAnalysis._landscape_metrics_df` (dropped this kind of caching in v2.0.0)
- using python3 `super` with no arguments
- added `neighborhood_rule` argument (to choose 4/8-cell adjacencies)
- fixed legend in `plot_landscape` and added `legend_kws` arg
- fixed multilandscape plot with one landscape, e.g., 1-zone analysis
- fixed values of zone_arr (to id zones) to avoid nodata=0 confusion
- default nodata (when providing numpy arrays) from settings module

## 2.2.1 (12/11/2020)

- `ZonalAnalysis.attribute_name` from `masks` when providing geoseries
- avoid reprojecting equivalent CRS (pyproj `__eq__` vs. `is_exact_same`) in `ZonalAnalysis`

## 2.2.0 (05/11/2020)

- fill `NaN` values according to the metric in `MultiLandscape.compute_class_metrics_df`
- consistent imports so that there are no direct class/function calls, and no module abbreviations in the documentation
- zonal analysis with more general `masks` argument, which also accepts vector geometries (and replaces the now deprecated `masks_arr`)

## 2.1.3 (02/09/2020)

- ensure numeric dtypes in metrics data frames
- `np.nan` nodata in img passed to `rasterio.plot.show` in `plot_landscape`

## 2.1.2 (03/08/2020)

- more robust `loc` (with index and columns) to compute metrics data frames in `MultiLandscape`

## 2.1.1 (12/06/2020)

- corrected `ZonalGridAnalysis` docstring
- corrected computation of `num_zone_cols` in `ZonalGridAnalysis`

## 2.1.0 (11/05/2020)

- configured flake8 in setup.cfg and added flake8 test in travis.yml
- added `compute_zonal_statistics` method to `ZonalAnalysis` and aded `ZonalGridAnalysis` class
- Implemented the `effective_mesh_size` metric

## 2.0.0a1 (24/09/2019)

- gradient -> zonal
- corrected cmap arg: default from rcParams, accept str
- updates in the docs (changed link to notebook + cleaned Makefile)

## 2.0.0a0 (20/09/2019)

- corrected shapely version in setup.py (bumpversion messed it)
- updated bumpversion for release candidates
- catch TypeError for existing metrics but at the wrong level
- always import modules, not methods/classes
- consistent API: all data frames obtained with `compute` methods

## 1.1.1 (18/09/2019)

- corrected rst typo in `total_area` docs
- fixed missing perimeter distribution statistic metric and dried class constants definitions
- fixed patch edge array computation when computing ENN (otherwise the speed-up of 1.1.0 is not effective)

## 1.1.0 (17/09/2019)

- speed-up (~x2) in `Landscape.compute_patch_euclidean_nearest_neighbor`: compute pixel-to-pixel distances for patch edges only

## 1.0.2 (25/07/2019)

- fix landscape array dtype in `SpatioTemporalBufferAnalysis`
- included `LICENSE` in `MANIFEST.in`

## 1.0.1 (24/07/2019)

- deleted Python 2 classifiers in `setup.py`
- fix ValueError message for `landscape_crs` and `landscape_transform` in `BufferAnalysis`
- fix landscape array dtype in `GradientAnalysis` and `BufferAnalysis`

## 1.0.0 (18/07/2019)

- dropped Python 2 support
- added `SpatioTemporalBufferAnalysis.plot_landscapes` method
- added `buffer_dist_legend` argument and docs in `SpatioTemporalBufferAnalysis. plot_metric`
- fix proper metric data frame properties in `SpatioTemporalBufferAnalysis`
- pass `transform` argument when initializating `MultiLandscape` instances (i.e., `SpatioTemporalAnalysis`, `BufferAnalysis`, `GradientAnalysis` and `SpatioTemporalBufferAnalysis`)
- `plot_landscape` and `plot_landscapes` with rasterio.plot.show
- changed `feature_{name,values}` for `attribute_{name,values}` in `MultiLandscape` (abstract) class
- dropped `plot_metrics` method

## 0.6.1 (02/07/2019)

- cell width and length comparisons with `numpy.isclose` to deal with imprecisions that come from float pixel resolutions (e.g., in GeoTIFF files)
- critical fix: moved pythran signatures from `compute.pythran` to `compute.py` so that the build is properly done when pip-installing

## 0.6.0 (01/07/2019)

- flat array approach to the computation of the adjacency matrix with pythran to improve performance (plus fixes a bug on the computation of `total_edge`)
- initialization of `scipy.spatial.cKDTree` with keyword arguments `balanced_tree` and `compact_nodes` set to `False`

## 0.5.0 (28/05/2019)

- methods plotting multiple axes return only the figure instead of a tuple with the figure and the axes
- settings module that allow configuring metrics' labels in plots, defaulting to FRAGSTATS abbreviations
- chaged CRS in tests to work with pyproj >= 2.0.0
- corrected `figlength` for `figwidth`
- warn when computing metrics that require an unmet minimum number of patches or classes (and their computation returns nan)
- all tests with `unittest.TestCase` assert methods

## 0.4.1 (03/04/2019)

- added docstrings for `MultiLandscape`, `GradientAnalysis` and `BufferAnalysis`
- raise `ValueError` when using buffer rings around a polygon geometry

## 0.4.0 (03/04/2019)

- implemented `SpatioTemporalBufferAnalysis` with a dedicated `plot_metric` method
- added buffer ring-wise analysis through a `buffer_rings` boolean argument in `BufferAnalysis`

## 0.3.1 (29/03/2019)

- float equality comparations with numpy `isclose` method

## 0.3.0 (28/03/2019)

- implemented `GradientAnalysis` and `BufferAnalysis`
- added optional `geopandas` dependences
- created abstract `MultiLandscape` class
- `Landscape` initialization from ndarray or geotiff (dropped `read_geotiff` method)
- implemented `contagion`
- convolution-based adjacency dataframe
- fixed bug with `class_cond` in `Landscape.compute_arr_edge`

## 0.2.0 (18/03/2019)

- implemented `euclidean_nearest_neighbor` and all its corresponding class/landscape distribution statistic metrics
- fixed dtype of `self.classes` in `landscape.Landscape`
- set default argument `nodata=None` (instead of `nodata=0`) for `landscape.read_geotiff`
- changed test input data files
- implemented `__len__` of `spatiotemporal.SpatioTemporalAnalysis`

## 0.1.1 (22/01/2019)

- corrected if-else statements involving `None` variables

## 0.1.0 (22/01/2019)

- initial release
