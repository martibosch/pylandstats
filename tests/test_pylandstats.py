import unittest

import affine
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import geometry

import pylandstats as pls

plt.switch_backend('agg')  # only for testing purposes


class TestImports(unittest.TestCase):
    def test_base_imports(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import rasterio
        from scipy import ndimage, spatial, stats

    def test_geo_imports(self):
        import geopandas as gpd
        from shapely.geometry.base import BaseGeometry


class TestLandscape(unittest.TestCase):
    def setUp(self):
        ls_arr = np.load('tests/input_data/ls250_06.npy')
        self.ls = pls.Landscape(ls_arr, res=(250, 250))

    def test_io(self):
        ls = pls.Landscape('tests/input_data/ls250_06.tif')
        # resolutions are not exactly 250, they are between [249, 251], so we
        # need to use a large delta
        self.assertAlmostEqual(ls.cell_width, 250, delta=1)
        self.assertAlmostEqual(ls.cell_height, 250, delta=1)
        self.assertAlmostEqual(ls.cell_area, 250 * 250, delta=250)

    def test_metrics_parameters(self):
        ls = self.ls

        for patch_metric in pls.Landscape.PATCH_METRICS:
            method = getattr(ls, patch_metric)
            self.assertIsInstance(method(), pd.DataFrame)
            self.assertIsInstance(method(class_val=ls.classes[0]), pd.Series)

        for class_metric in pls.Landscape.CLASS_METRICS:
            self.assertTrue(
                np.isreal(getattr(ls, class_metric)(class_val=ls.classes[0])))

        for landscape_metric in pls.Landscape.LANDSCAPE_METRICS:
            self.assertTrue(np.isreal(getattr(ls, landscape_metric)()))

    def test_metric_dataframes(self):
        ls = self.ls
        patch_df = ls.compute_patch_metrics_df()
        self.assertTrue(
            np.all(
                patch_df.columns.drop('class_val') ==
                pls.Landscape.PATCH_METRICS))
        self.assertEqual(patch_df.index.name, 'patch_id')
        self.assertRaises(ValueError, ls.compute_patch_metrics_df, ['foo'])

        class_df = ls.compute_class_metrics_df()
        self.assertEqual(
            len(class_df.columns.difference(pls.Landscape.CLASS_METRICS)), 0)
        self.assertEqual(class_df.index.name, 'class_val')
        self.assertRaises(ValueError, ls.compute_class_metrics_df, ['foo'])

        landscape_df = ls.compute_landscape_metrics_df()
        self.assertEqual(
            len(
                landscape_df.columns.difference(
                    pls.Landscape.LANDSCAPE_METRICS)), 0)
        self.assertEqual(len(landscape_df.index), 1)
        self.assertRaises(ValueError, ls.compute_landscape_metrics_df, ['foo'])

    def test_landscape_metrics_value_ranges(self):
        ls = self.ls

        # basic tests of the `Landscape` class' attributes
        self.assertNotIn(ls.nodata, ls.classes)
        self.assertGreater(ls.landscape_area, 0)

        class_val = ls.classes[0]
        # label_arr = ls._get_label_arr(class_val)

        # patch-level metrics
        assert (ls.area()['area'] > 0).all()
        assert (ls.perimeter()['perimeter'] > 0).all()
        assert (ls.perimeter_area_ratio()['perimeter_area_ratio'] > 0).all()
        assert (ls.shape_index()['shape_index'] >= 1).all()
        _fractal_dimension_ser = ls.fractal_dimension()['fractal_dimension']
        assert (_fractal_dimension_ser >= 1).all() and (_fractal_dimension_ser
                                                        <= 2).all()
        # TODO: assert 0 <= ls.contiguity_index(patch_arr) <= 1
        # ACHTUNG: euclidean nearest neighbor can be nan for classes with less
        # than two patches
        assert (ls.euclidean_nearest_neighbor()['euclidean_nearest_neighbor']
                .dropna() > 0).all()
        # TODO: assert 0 <= ls.proximity(patch_arr) <= 1

        # class-level metrics
        assert ls.total_area(class_val) > 0
        assert 0 < ls.proportion_of_landscape(class_val) < 100
        assert ls.patch_density(class_val) > 0
        assert 0 < ls.largest_patch_index(class_val) < 100
        assert ls.total_edge(class_val) >= 0
        assert ls.edge_density(class_val) >= 0

        # the value ranges of mean, area-weighted mean and median aggregations
        # are going to be the same as their respective original metrics
        mean_suffixes = ['_mn', '_am', '_md']
        # the value ranges of the range, standard deviation and coefficient of
        # variation  will always be nonnegative as long as the means are
        # nonnegative as well (which is the case of all of the metrics
        # implemented so far)
        var_suffixes = ['_ra', '_sd', '_cv']

        for mean_suffix in mean_suffixes:
            assert getattr(ls, 'area' + mean_suffix)(class_val) > 0
            assert getattr(ls,
                           'perimeter_area_ratio' + mean_suffix)(class_val) > 0
            assert getattr(ls, 'shape_index' + mean_suffix)(class_val) >= 1
            assert 1 <= getattr(
                ls, 'fractal_dimension' + mean_suffix)(class_val) <= 2
            # assert 0 <= getattr(
            #     ls, 'contiguity_index' + mean_suffix)(class_val) <= 1
            # assert getattr(ls, 'proximity' + mean_suffix)(class_val) >= 0
            # ACHTUNG: euclidean nearest neighbor can be nan for classes with
            # less than two patches
            enn = getattr(
                ls, 'euclidean_nearest_neighbor' + mean_suffix)(class_val)
            assert enn > 0 or np.isnan(enn)

        for var_suffix in var_suffixes:
            assert getattr(ls, 'area' + mean_suffix)(class_val) >= 0
            assert getattr(ls,
                           'perimeter_area_ratio' + var_suffix)(class_val) >= 0
            assert getattr(ls, 'shape_index' + var_suffix)(class_val) >= 0
            assert getattr(ls,
                           'fractal_dimension' + var_suffix)(class_val) >= 0
            # assert getattr(
            #    ls, 'contiguity_index' + var_suffix)(class_val) >= 0
            # assert getattr(ls, 'proximity' + var_suffix)(class_val) >= 0
            # ACHTUNG: euclidean nearest neighbor can be nan for classes with
            # less than two patches
            enn = getattr(ls,
                          'euclidean_nearest_neighbor' + var_suffix)(class_val)
            assert enn >= 0 or np.isnan(enn)

        # TODO: assert 0 < ls.interspersion_juxtaposition_index(
        #           class_val) <= 100
        assert ls.landscape_shape_index(class_val) >= 1

        # landscape-level metrics
        assert ls.total_area() > 0
        assert ls.patch_density() > 0
        assert 0 < ls.largest_patch_index() < 100
        assert ls.total_edge() >= 0
        assert ls.edge_density() >= 0
        assert 0 < ls.largest_patch_index() <= 100
        assert ls.total_edge() >= 0
        assert ls.edge_density() >= 0

        # for class_val in ls.classes:
        #     print('num_patches', class_val, ls._get_num_patches(class_val))
        #     print('patch_areas', len(ls._get_patch_areas(class_val)))

        # raise ValueError

        for mean_suffix in mean_suffixes:
            assert getattr(ls, 'area' + mean_suffix)() > 0
            assert getattr(ls, 'perimeter_area_ratio' + mean_suffix)() > 0
            assert getattr(ls, 'shape_index' + mean_suffix)() >= 1
            assert 1 <= getattr(ls, 'fractal_dimension' + mean_suffix)() <= 2
            # assert 0 <= getattr(ls, 'contiguity_index' + mean_suffix)() <= 1
            # assert getattr(ls, 'proximity' + mean_suffix)() >= 0
            # ACHTUNG: euclidean nearest neighbor can be nan for classes with
            # less than two patches
            enn = getattr(ls, 'euclidean_nearest_neighbor' + mean_suffix)()
            assert enn > 0 or np.isnan(enn)
        for var_suffix in var_suffixes:
            assert getattr(ls, 'area' + var_suffix)() > 0
            assert getattr(ls, 'perimeter_area_ratio' + var_suffix)() >= 0
            assert getattr(ls, 'shape_index' + var_suffix)() >= 0
            assert getattr(ls, 'fractal_dimension' + var_suffix)() >= 0
            # assert getattr(ls, 'contiguity_index' + var_suffix)() >= 0
            # assert getattr(ls, 'proximity' + var_suffix)() >= 0
            # ACHTUNG: euclidean nearest neighbor can be nan for classes with
            # less than two patches
            enn = getattr(ls, 'euclidean_nearest_neighbor' + var_suffix)()
            assert enn >= 0 or np.isnan(enn)

        assert 0 < ls.contagion() <= 100
        # TODO: assert 0 < ls.interspersion_juxtaposition_index() <= 100
        assert ls.shannon_diversity_index() >= 0

    def test_plot_landscape(self):
        # returned axis must be instances of matplotlib axes
        self.assertIsInstance(self.ls.plot_landscape(), plt.Axes)


class TestMultiLandscape(unittest.TestCase):
    def setUp(self):
        from pylandstats.multilandscape import MultiLandscape

        self.landscapes = [
            pls.Landscape(
                np.load('tests/input_data/ls100_06.npy'), res=(100, 100)),
            pls.Landscape(
                np.load('tests/input_data/ls250_06.npy'), res=(250, 250))
        ]
        self.landscape_fps = [
            'tests/input_data/ls100_06.tif', 'tests/input_data/ls250_06.tif'
        ]
        self.feature_name = 'resolution'
        self.feature_values = [100, 250]
        self.inexistent_class_val = 999

        # use this class just for testing purposes
        class InstantiableMultiLandscape(MultiLandscape):
            def __init__(self, *args, **kwargs):
                super(InstantiableMultiLandscape, self).__init__(
                    *args, **kwargs)

        self.InstantiableMultiLandscape = InstantiableMultiLandscape

    def test_multilandscape_init(self):
        from pylandstats.multilandscape import MultiLandscape

        # test that we cannot instantiate an abstract class
        self.assertRaises(TypeError, MultiLandscape)

        # test that if we init a MultiLandscape from filepaths, Landscape
        # instances are automaticaly built
        ml = self.InstantiableMultiLandscape(
            self.landscape_fps, self.feature_name, self.feature_values)
        for landscape in ml.landscapes:
            self.assertIsInstance(landscape, pls.Landscape)

        # from this point on, always instantiate from filepaths

        # test that constructing a MultiLandscape with inexistent metrics and
        # inexistent classes raises a ValueError
        self.assertRaises(ValueError, self.InstantiableMultiLandscape,
                          self.landscape_fps, self.feature_name,
                          self.feature_values, metrics=['foo'])
        self.assertRaises(ValueError, self.InstantiableMultiLandscape,
                          self.landscape_fps, self.feature_name,
                          self.feature_values,
                          classes=[self.inexistent_class_val])

        # test that constructing a MultiLandscape where the list of values of
        # the identifying features `feature_values` (in this example, the list
        # of resolutions `[100, 250]`) mismatches the length of the list of
        # landscapes raises a ValueError
        self.assertRaises(ValueError, self.InstantiableMultiLandscape,
                          self.landscape_fps, self.feature_name, [250])

    def test_multilandscape_dataframes(self):
        ml = self.InstantiableMultiLandscape(
            self.landscape_fps, self.feature_name, self.feature_values)
        # test that `class_metrics_df` and `landscape_metrics_df` are well
        # constructed
        class_metrics_df = ml.class_metrics_df
        feature_values = getattr(ml, ml.feature_name)
        self.assertTrue(
            np.all(class_metrics_df.columns == pls.Landscape.CLASS_METRICS))
        self.assertTrue(
            np.all(class_metrics_df.index == pd.MultiIndex.from_product(
                [ml.classes, feature_values])))
        landscape_metrics_df = ml.landscape_metrics_df
        self.assertTrue(
            np.all(landscape_metrics_df.columns ==
                   pls.Landscape.LANDSCAPE_METRICS))
        self.assertTrue(np.all(landscape_metrics_df.index == feature_values))

        # now test the same but with an analysis that only considers a subset
        # of metrics and a subset of classes
        ml_metrics = ['total_area', 'edge_density', 'proportion_of_landscape']
        ml_classes = self.landscapes[0].classes[:2]
        ml = self.InstantiableMultiLandscape(
            self.landscapes, self.feature_name, self.feature_values,
            metrics=ml_metrics, classes=ml_classes)

        class_metrics_df = ml.class_metrics_df
        feature_values = getattr(ml, ml.feature_name)
        self.assertTrue(
            np.all(class_metrics_df.columns == np.intersect1d(
                ml_metrics, pls.Landscape.CLASS_METRICS)))
        self.assertTrue(
            np.all(class_metrics_df.index == pd.MultiIndex.from_product(
                [ml_classes, feature_values])))
        landscape_metrics_df = ml.landscape_metrics_df
        self.assertTrue(
            np.all(landscape_metrics_df.columns == np.intersect1d(
                ml_metrics, pls.Landscape.LANDSCAPE_METRICS)))
        self.assertTrue(np.all(landscape_metrics_df.index == feature_values))

    def test_multilandscape_metric_kws(self):
        # Instantiate two multilandscape analyses, one with FRAGSTATS'
        # defaults and the other with keyword arguments specifying the total
        # area in meters and including the boundary in the computation of the
        # total edge.
        ml = self.InstantiableMultiLandscape(
            self.landscape_fps, self.feature_name, self.feature_values)
        ml_kws = self.InstantiableMultiLandscape(
            self.landscape_fps, self.feature_name, self.feature_values,
            metrics_kws={
                'total_area': {
                    'hectares': False
                },
                'total_edge': {
                    'count_boundary': True
                }
            })

        # For all feature values and all classes, metric values in hectares
        # should be less than in meters, and excluding boundaries should be
        # less or equal than including them
        for feature_value in getattr(ml, ml.feature_name):
            landscape_metrics = ml.landscape_metrics_df.loc[feature_value]
            landscape_metrics_kws = ml_kws.landscape_metrics_df.loc[
                feature_value]
            self.assertLess(landscape_metrics['total_area'],
                            landscape_metrics_kws['total_area'])
            self.assertLessEqual(landscape_metrics['total_edge'],
                                 landscape_metrics_kws['total_edge'])

            for class_val in ml.classes:
                class_metrics = ml.class_metrics_df.loc[class_val,
                                                        feature_value]
                class_metrics_kws = ml_kws.class_metrics_df.loc[class_val,
                                                                feature_value]

                # It could be that for some feature values, some classes are
                # not present within the respective Landscape. If so, all of
                # the metrics will be `nan`, both for the analysis with and
                # without keyword arguments. Otherwise, we just perform the
                # usual checks
                if class_metrics.isnull().all():
                    self.assertTrue(class_metrics_kws.isnull().all())
                else:
                    self.assertLess(class_metrics['total_area'],
                                    class_metrics_kws['total_area'])
                    self.assertLessEqual(
                        class_metrics['total_edge'],
                        class_metrics_kws['total_edge'] + .0004)

    def test_multilandscape_plot_metrics(self):
        ml = self.InstantiableMultiLandscape(
            self.landscape_fps, self.feature_name, self.feature_values)

        existent_class_val = ml.classes[0]

        # inexistent metrics should raise a ValueError
        self.assertRaises(ValueError, ml.plot_metric, 'foo')
        # inexistent classes should raise a ValueError
        self.assertRaises(ValueError, ml.plot_metric, 'patch_density',
                          {'class_val': self.inexistent_class_val})
        # `proportion_of_landscape` can only be computed at the class level,
        # so plotting it at the landscape level (with the default argument
        # `class_val=None`) must raise a ValueError
        self.assertRaises(ValueError, ml.plot_metric,
                          'proportion_of_landscape')
        # conversely, `shannon_diversity_index` can only be computed at the
        # landscape level, so plotting it at the class level must raise a
        # ValueError
        self.assertRaises(ValueError, ml.plot_metric,
                          'shannon_diversity_index',
                          {'class_val': existent_class_val})

        # TODO: test legend and figsize

        # test that there is only one line when plotting a single metric at
        # the landscape level
        ax = ml.plot_metric('patch_density', class_val=None)
        self.assertEqual(len(ax.lines), 1)
        # test that there are two lines if we add the plot of a single metric
        # (e.g., at the level of an existent class) to the previous axis
        ax = ml.plot_metric('patch_density', class_val=existent_class_val,
                            ax=ax)
        self.assertEqual(len(ax.lines), 2)
        # test that the x data of any line corresponds to the feature values
        for line in ax.lines:
            self.assertTrue(
                np.all(line.get_xdata() == getattr(ml, ml.feature_name)))

        # test that there are two axes if we plot two metrics
        fig, axes = ml.plot_metrics(class_val=existent_class_val,
                                    metrics=['edge_density', 'patch_density'])
        self.assertEqual(len(axes), 2)

    def test_plot_landscapes(self):
        ml = self.InstantiableMultiLandscape(
            self.landscape_fps, self.feature_name, self.feature_values)

        fig, axes = ml.plot_landscapes()

        # there must be one column for each landscape
        self.assertEqual(len(axes), len(ml))

        # returned axes must be instances of matplotlib axes
        for ax in axes:
            self.assertIsInstance(ax, plt.Axes)


class TestSpatioTemporalAnalysis(unittest.TestCase):
    def setUp(self):
        # we will only test reading from filepaths because the consistency
        # between passing `Landscape` objects or filepaths is already tested
        # in `TestMultiLandscape`
        self.landscape_fps = [
            'tests/input_data/ls250_06.tif', 'tests/input_data/ls250_12.tif'
        ]
        self.dates = [2006, 2012]
        self.inexistent_class_val = 999

    def test_spatiotemporalanalysis_init(self):
        # test that the `feature_name` is dates, and that if the `dates`
        # argument is not provided when instantiating a
        # `SpatioTemporalAnalysis`, the dates attribute is properly and
        # automatically generated
        sta = pls.SpatioTemporalAnalysis(self.landscape_fps)
        self.assertEqual(sta.feature_name, 'dates')
        self.assertEqual(len(sta), len(sta.dates))

    def test_spatiotemporalanalysis_dataframes(self):
        # test with the default constructor
        sta = pls.SpatioTemporalAnalysis(self.landscape_fps)

        # test that `class_metrics_df` and `landscape_metrics_df` are well
        # constructed
        class_metrics_df = sta.class_metrics_df
        self.assertTrue(
            np.all(class_metrics_df.index == pd.MultiIndex.from_product(
                [sta.classes, sta.dates])))
        landscape_metrics_df = sta.landscape_metrics_df
        self.assertTrue(np.all(landscape_metrics_df.index == sta.dates))

        # now test the same but with an analysis that only considers a
        # subset of metrics and a subset of classes
        sta_metrics = ['total_area', 'edge_density', 'proportion_of_landscape']
        sta_classes = sta.classes[:2]
        sta = pls.SpatioTemporalAnalysis(self.landscape_fps,
                                         metrics=sta_metrics,
                                         classes=sta_classes, dates=self.dates)

        class_metrics_df = sta.class_metrics_df
        self.assertTrue(
            np.all(class_metrics_df.index == pd.MultiIndex.from_product(
                [sta_classes, self.dates])))
        landscape_metrics_df = sta.landscape_metrics_df
        self.assertTrue(np.all(landscape_metrics_df.index == self.dates))

    def test_spatiotemporalanalysis_plot_metrics(self):
        sta = pls.SpatioTemporalAnalysis(self.landscape_fps, dates=self.dates)

        # test for `None` (landscape-level) and an existing class (class-level)
        for class_val in [None, sta.classes[0]]:
            # test that the x data of the line corresponds to the dates
            self.assertTrue(
                np.all(
                    sta.plot_metric('patch_density', class_val=class_val)
                    .lines[0].get_xdata() == self.dates))


class TestGradientAnalysis(unittest.TestCase):
    def setUp(self):
        self.masks_arr = np.load('tests/input_data/masks_arr.npy')
        self.landscape = pls.Landscape(
            np.load('tests/input_data/ls250_06.npy'), res=(250, 250))
        self.landscape_fp = 'tests/input_data/ls250_06.tif'
        self.landscape_transform = affine.Affine(
            249.96431809611167, 0.0, 4037084.1862939927, 0.0,
            -250.7188576750866, 2631436.6068059015)
        self.landscape_crs = {'init': 'epsg:3035'}
        # for buffer analysis
        self.geom = geometry.Point(6.6327025, 46.5218269)
        self.geom_crs = {'init': 'epsg:4326'}
        self.buffer_dists = [10000, 15000, 20000]

    def test_gradient_init(self):
        # test that the feature names and values are consistent with the
        # provided `masks_arr`
        ga = pls.GradientAnalysis(self.landscape, self.masks_arr)
        self.assertEqual(ga.feature_name, 'feature_values')
        self.assertEqual(len(ga), len(self.masks_arr))
        self.assertEqual(len(ga), len(ga.feature_values))

        # test that if we init a GradientAnalysis from filepaths, Landscape
        # instances are automaticaly built, and the feature names and values
        # are also consistent with the provided `masks_arr`
        ga = pls.GradientAnalysis(self.landscape_fp, self.masks_arr)
        for landscape in ga.landscapes:
            self.assertIsInstance(landscape, pls.Landscape)
        self.assertEqual(ga.feature_name, 'feature_values')
        self.assertEqual(len(ga), len(self.masks_arr))
        self.assertEqual(len(ga), len(ga.feature_values))

        # from this point on, always instantiate from filepaths

    def test_buffer_init(self):
        naive_gser = gpd.GeoSeries([self.geom])
        gser = gpd.GeoSeries([self.geom], crs=self.geom_crs)

        # test that we cannot init from a shapely geometry without providing
        # its crs
        self.assertRaises(ValueError, pls.BufferAnalysis, self.landscape_fp,
                          self.geom, self.buffer_dists)
        # test that we cannot init with a landscape that does not have crs and
        # transform information, even when providing the `base_mask` arguments
        # properly
        for base_mask in [self.geom, naive_gser, gser]:
            self.assertRaises(ValueError, pls.BufferAnalysis, self.landscape,
                              base_mask, self.buffer_dists,
                              {'base_mask_crs': self.geom_crs})
            self.assertRaises(
                ValueError, pls.BufferAnalysis, self.landscape, base_mask,
                self.buffer_dists, {
                    'base_mask_crs': self.geom_crs,
                    'landscape_crs': self.landscape_crs
                })
            self.assertRaises(
                ValueError, pls.BufferAnalysis, self.landscape, base_mask,
                self.buffer_dists, {
                    'base_mask_crs': self.geom_crs,
                    'landscape_transform': self.landscape_transform
                })

        # test that we can properly instantiate it from:
        # 1. a landscape filepath, shapely geometry, and its crs
        # 2. a landscape filepath, naive geopandas GeoSeries (with no crs set)
        #    and its crs
        # 3. a landscape filepath, geopandas GeoSeries with crs set
        # 4. a landscape filepath, geopandas GeoSeries with crs set and a crs (
        #    which will override the crs of the GeoSeries)
        # 5. any of the above but changing the landscape filepath for a
        #    Landscape instance with its crs and transform
        for ba in [
                pls.BufferAnalysis(self.landscape_fp, self.geom,
                                   self.buffer_dists,
                                   base_mask_crs=self.geom_crs),
                pls.BufferAnalysis(self.landscape_fp, naive_gser,
                                   self.buffer_dists,
                                   base_mask_crs=self.geom_crs),
                pls.BufferAnalysis(self.landscape_fp, gser, self.buffer_dists),
                pls.BufferAnalysis(self.landscape_fp, gser, self.buffer_dists,
                                   base_mask_crs=self.geom_crs),
                pls.BufferAnalysis(
                    self.landscape, gser, self.buffer_dists,
                    base_mask_crs=self.geom_crs,
                    landscape_crs=self.landscape_crs,
                    landscape_transform=self.landscape_transform)
        ]:
            self.assertEqual(ba.feature_name, 'buffer_dists')
            self.assertEqual(len(ba), len(ba.masks_arr))
            self.assertEqual(len(ba), len(ba.buffer_dists))

        # test that buffer rings are properly instantiated
        ba_rings = pls.BufferAnalysis(self.landscape_fp, gser,
                                      self.buffer_dists, buffer_rings=True)
        # the `buffer_dists` attribute must be a string of the form '{r}-{R}'
        # where r and R respectively represent the smaller and larger radius
        # that compose each ring
        for buffer_ring_str in ba_rings.buffer_dists:
            self.assertIn('-', buffer_ring_str)

        # compare it with the default instance (with the argument
        # `buffer_rings=False`) that does not consider rings but cumulatively
        # considers the inner areas in each mask. The first mask will in fact
        # be the same in both cases (the region that goes from 0 to the first
        # item of `self.buffer_dists`), but the successive masks will be
        # always larger in the default instance (since they will have the
        # surface of the corresponding ring plus the surface of the inner
        # region that is excluded when `buffer_rings=True`)
        ba = pls.BufferAnalysis(self.landscape_fp, gser, self.buffer_dists)
        for mask_arr, ring_mask_arr in zip(ba.masks_arr, ba_rings.masks_arr):
            self.assertGreaterEqual(np.sum(mask_arr), np.sum(ring_mask_arr))

    def test_buffer_plot_metrics(self):
        ba = pls.BufferAnalysis(self.landscape_fp, self.geom,
                                self.buffer_dists, base_mask_crs=self.geom_crs)

        # test for `None` (landscape-level) and an existing class (class-level)
        for class_val in [None, ba.classes[0]]:
            # test that the x data of the line corresponds to `buffer_dists`
            self.assertTrue(
                np.all(
                    ba.plot_metric('patch_density', class_val=class_val).lines[
                        0].get_xdata() == self.buffer_dists))


class TestSpatioTemporalBufferAnalysis(unittest.TestCase):
    def setUp(self):
        self.landscape_fps = [
            'tests/input_data/ls250_06.tif', 'tests/input_data/ls250_12.tif'
        ]
        self.dates = [2006, 2012]
        self.base_mask = gpd.GeoSeries([geometry.Point(6.6327025, 46.5218269)],
                                       crs={'init': 'epsg:4326'})
        self.buffer_dists = [10000, 15000, 20000]

    def test_spatiotemporalbufferanalysis_init(self):
        # we will just test the base init, the rest of functionalities have
        # already been tested above (in `TestSpatioTemporalAnalysis` and
        # `TestGradientAnalysis`)
        stba = pls.SpatioTemporalBufferAnalysis(
            self.landscape_fps, self.base_mask, self.buffer_dists,
            dates=self.dates)
        self.assertEqual(len(stba.buffer_dists), len(stba.stas))
        for sta in stba.stas:
            self.assertEqual(sta.dates, self.dates)

    def test_spatiotemporalbufferanalysis_plot_metric(self):
        stba = pls.SpatioTemporalBufferAnalysis(
            self.landscape_fps, self.base_mask, self.buffer_dists)

        # test for `None` (landscape-level) and an existing class (class-level)
        for class_val in [None, stba.stas[0].classes[0]]:
            ax = stba.plot_metric('patch_density', class_val=class_val)
            # test that there is a line for each buffer distance
            self.assertEqual(len(ax.lines), len(self.buffer_dists))
            # test that there is a legend label for each buffer distance
            handles, labels = ax.get_legend_handles_labels()
            self.assertEqual(len(labels), len(self.buffer_dists))
