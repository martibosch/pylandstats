import os
import shutil
import unittest
import warnings
from os import path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from shapely import geometry

import pylandstats as pls

plt.switch_backend('agg')  # only for testing purposes
geom_crs = 'epsg:4326'


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
        ls_arr = np.load('tests/input_data/ls250_06.npy', allow_pickle=True)
        self.ls = pls.Landscape(ls_arr, res=(250, 250))

    def test_io(self):
        ls = pls.Landscape('tests/input_data/ls250_06.tif')
        # resolutions are not exactly 250, they are between [249, 251], so we
        # need to use a large delta
        self.assertAlmostEqual(ls.cell_width, 250, delta=1)
        self.assertAlmostEqual(ls.cell_height, 250, delta=1)
        self.assertAlmostEqual(ls.cell_area, 250 * 250, delta=250)

        # test that the transform is None if we instantiate a landscape from
        # an ndarray (without providing the `transform` argument, but that it
        # is not none we get the landscape transform from a raster path
        self.assertIsNone(self.ls.transform)
        self.assertIsNotNone(ls.transform)

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

    def test_neighborhood_rules(self):
        # test that providing a value different than 8 or 4 raises a
        # `ValueError`
        with self.assertRaises(ValueError) as cm:
            pls.Landscape('tests/input_data/ls250_06.tif',
                          neighborhood_rule='2')
            self.assertIn('is not among', str(cm.exception))
        # test that we can provide the argument as int as long as it is 8 or 4
        for neighborhood_rule in (8, 4):
            self.assertEqual(
                pls.Landscape(
                    'tests/input_data/ls250_06.tif',
                    neighborhood_rule=neighborhood_rule).neighborhood_rule,
                str(neighborhood_rule))
        # test that there is at least the same number of patches with the Moore
        # neighborhood than with Von Neumann's
        ls_moore = pls.Landscape('tests/input_data/ls250_06.tif',
                                 neighborhood_rule='8')
        ls_von_neumann = pls.Landscape('tests/input_data/ls250_06.tif',
                                       neighborhood_rule='4')
        self.assertLessEqual(ls_moore.number_of_patches(),
                             ls_von_neumann.number_of_patches())

    def test_metrics_warnings(self):
        # test that warnings are raised

        # class-level metrics
        # euclidean nearest neighbor will return nan (and raise an informative
        # warning) if there is not at least two patches of each class. Let us
        # test this by creating a landscape with a background of class 1 and a
        # single patch of class 2
        arr = np.ones((4, 4))
        arr[1:-1, 1:-1] = 2
        ls = pls.Landscape(arr, res=(1, 1))

        # let us test that both the computation at the class-level (1 and 2)
        # and at the landscape level (`class_val` of `None`) raise at least one
        # warning (the exact number of warnings raised can be different in
        # Python 2 and 3)
        for class_val in [1, 2, None]:
            with warnings.catch_warnings(record=True) as w:
                ls.euclidean_nearest_neighbor(class_val)
                self.assertGreater(len(w), 0)

        # landscape-level metrics
        # some landscape-level metrics require at least two classes.
        ls = pls.Landscape(np.ones((4, 4)), res=(1, 1))
        for method in ['contagion', 'shannon_diversity_index']:
            with warnings.catch_warnings(record=True) as w:
                getattr(ls, method)()
                self.assertGreater(len(w), 0)

    def test_metric_dataframes(self):
        ls = self.ls
        patch_metrics = set(pls.Landscape.PATCH_METRICS)
        class_metrics = set(pls.Landscape.CLASS_METRICS)
        landscape_metrics = set(pls.Landscape.LANDSCAPE_METRICS)

        patch_df = ls.compute_patch_metrics_df()
        self.assertTrue(
            np.all(
                patch_df.columns.drop('class_val') ==
                pls.Landscape.PATCH_METRICS))
        self.assertEqual(patch_df.index.name, 'patch_id')
        # try that raised ValueErrors have different error messages depending
        # on the context
        with self.assertRaises(ValueError) as cm:
            ls.compute_patch_metrics_df(metrics=['foo'])
            self.assertIn('is not among', str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            for metric in class_metrics.union(landscape_metrics):
                ls.compute_patch_metrics_df(metrics=[metric])
            self.assertIn('cannot be computed', str(cm.exception))

        class_df = ls.compute_class_metrics_df()
        self.assertEqual(
            len(class_df.columns.difference(pls.Landscape.CLASS_METRICS)), 0)
        self.assertEqual(class_df.index.name, 'class_val')
        # try that raised ValueErrors have different error messages depending
        # on the context
        with self.assertRaises(ValueError) as cm:
            ls.compute_class_metrics_df(metrics=['foo'])
            self.assertIn('is not among', str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            for metric in patch_metrics.union(
                    landscape_metrics.difference(class_metrics)):
                ls.compute_class_metrics_df(metrics=[metric])
            self.assertIn('cannot be computed', str(cm.exception))

        landscape_df = ls.compute_landscape_metrics_df()
        self.assertEqual(
            len(
                landscape_df.columns.difference(
                    pls.Landscape.LANDSCAPE_METRICS)), 0)
        self.assertEqual(len(landscape_df.index), 1)
        # try that raised ValueErrors have different error messages depending
        # on the context
        with self.assertRaises(ValueError) as cm:
            ls.compute_landscape_metrics_df(metrics=['foo'])
            self.assertIn('is not among', str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            for metric in patch_metrics.union(
                    class_metrics.difference(landscape_metrics)):
                ls.compute_landscape_metrics_df(metrics=[metric])
            self.assertIn('cannot be computed', str(cm.exception))

    def test_landscape_metrics_value_ranges(self):
        ls = self.ls

        # basic tests of the `Landscape` class' attributes
        self.assertNotIn(ls.nodata, ls.classes)
        self.assertGreater(ls.landscape_area, 0)

        class_val = ls.classes[0]
        # label_arr = ls._get_label_arr(class_val)

        # patch-level metrics
        self.assertTrue((ls.area()['area'] > 0).all())
        self.assertTrue((ls.perimeter()['perimeter'] > 0).all())
        self.assertTrue(
            (ls.perimeter_area_ratio()['perimeter_area_ratio'] > 0).all())
        self.assertTrue((ls.shape_index()['shape_index'] >= 1).all())
        _fractal_dimension_ser = ls.fractal_dimension()['fractal_dimension']
        self.assertTrue((_fractal_dimension_ser >= 1).all()
                        and (_fractal_dimension_ser <= 2).all())
        # TODO: assert 0 <= ls.contiguity_index(patch_arr) <= 1
        # ACHTUNG: euclidean nearest neighbor can be nan for classes with less
        # than two patches
        self.assertTrue((ls.euclidean_nearest_neighbor()
                         ['euclidean_nearest_neighbor'].dropna() > 0).all())
        # TODO: assert 0 <= ls.proximity(patch_arr) <= 1

        # class-level metrics
        self.assertGreater(ls.total_area(class_val), 0)
        self.assertTrue(0 < ls.proportion_of_landscape(class_val) < 100)
        self.assertTrue(ls.patch_density(class_val) > 0)
        self.assertTrue(0 < ls.largest_patch_index(class_val) < 100)
        self.assertGreaterEqual(ls.total_edge(class_val), 0)
        self.assertGreaterEqual(ls.edge_density(class_val), 0)

        # the value ranges of mean, area-weighted mean and median aggregations
        # are going to be the same as their respective original metrics
        mean_suffixes = ['_mn', '_am', '_md']
        # the value ranges of the range, standard deviation and coefficient of
        # variation  will always be nonnegative as long as the means are
        # nonnegative as well (which is the case of all of the metrics
        # implemented so far)
        var_suffixes = ['_ra', '_sd', '_cv']

        for mean_suffix in mean_suffixes:
            self.assertGreater(getattr(ls, 'area' + mean_suffix)(class_val), 0)
            self.assertGreater(
                getattr(ls, 'perimeter_area_ratio' + mean_suffix)(class_val),
                0)
            self.assertGreaterEqual(
                getattr(ls, 'shape_index' + mean_suffix)(class_val), 1)
            self.assertTrue(1 <= getattr(ls, 'fractal_dimension' +
                                         mean_suffix)(class_val) <= 2)
            # assert 0 <= getattr(
            #     ls, 'contiguity_index' + mean_suffix)(class_val) <= 1
            # assert getattr(ls, 'proximity' + mean_suffix)(class_val) >= 0
            # ACHTUNG: euclidean nearest neighbor can be nan for classes with
            # less than two patches
            enn = getattr(ls, 'euclidean_nearest_neighbor' +
                          mean_suffix)(class_val)
            self.assertTrue(enn > 0 or np.isnan(enn))

        for var_suffix in var_suffixes:
            self.assertGreaterEqual(
                getattr(ls, 'area' + mean_suffix)(class_val), 0)
            self.assertGreaterEqual(
                getattr(ls, 'perimeter_area_ratio' + var_suffix)(class_val), 0)
            self.assertGreaterEqual(
                getattr(ls, 'shape_index' + var_suffix)(class_val), 0)
            self.assertGreaterEqual(
                getattr(ls, 'fractal_dimension' + var_suffix)(class_val), 0)
            # assert getattr(
            #    ls, 'contiguity_index' + var_suffix)(class_val) >= 0
            # assert getattr(ls, 'proximity' + var_suffix)(class_val) >= 0
            # ACHTUNG: euclidean nearest neighbor can be nan for classes with
            # less than two patches
            enn = getattr(ls,
                          'euclidean_nearest_neighbor' + var_suffix)(class_val)
            self.assertTrue(enn >= 0 or np.isnan(enn))

        # TODO: assert 0 < ls.interspersion_juxtaposition_index(
        #           class_val) <= 100
        self.assertGreaterEqual(ls.landscape_shape_index(class_val), 1)
        self.assertTrue(
            ls.cell_area / ls.landscape_area <=
            ls.effective_mesh_size(class_val) <= ls.landscape_area, 1)

        # landscape-level metrics
        self.assertGreater(ls.total_area(), 0)
        self.assertGreater(ls.patch_density(), 0)
        self.assertTrue(0 < ls.largest_patch_index() < 100)
        self.assertGreaterEqual(ls.total_edge(), 0)
        self.assertGreaterEqual(ls.edge_density(), 0)
        self.assertTrue(0 < ls.largest_patch_index() <= 100)
        self.assertGreaterEqual(ls.total_edge(), 0)
        self.assertGreaterEqual(ls.edge_density(), 0)

        # for class_val in ls.classes:
        #     print('num_patches', class_val, ls._get_num_patches(class_val))
        #     print('patch_areas', len(ls._get_patch_areas(class_val)))

        # raise ValueError

        for mean_suffix in mean_suffixes:
            self.assertGreater(getattr(ls, 'area' + mean_suffix)(), 0)
            self.assertGreater(
                getattr(ls, 'perimeter_area_ratio' + mean_suffix)(), 0)
            self.assertGreaterEqual(
                getattr(ls, 'shape_index' + mean_suffix)(), 1)
            self.assertTrue(
                1 <= getattr(ls, 'fractal_dimension' + mean_suffix)() <= 2)
            # assert 0 <= getattr(ls, 'contiguity_index' + mean_suffix)() <= 1
            # assert getattr(ls, 'proximity' + mean_suffix)() >= 0
            # ACHTUNG: euclidean nearest neighbor can be nan for classes with
            # less than two patches
            enn = getattr(ls, 'euclidean_nearest_neighbor' + mean_suffix)()
            self.assertTrue(enn > 0 or np.isnan(enn))
        for var_suffix in var_suffixes:
            self.assertGreater(getattr(ls, 'area' + var_suffix)(), 0)
            self.assertGreaterEqual(
                getattr(ls, 'perimeter_area_ratio' + var_suffix)(), 0)
            self.assertGreaterEqual(
                getattr(ls, 'shape_index' + var_suffix)(), 0)
            self.assertGreaterEqual(
                getattr(ls, 'fractal_dimension' + var_suffix)(), 0)
            # assert getattr(ls, 'contiguity_index' + var_suffix)() >= 0
            # assert getattr(ls, 'proximity' + var_suffix)() >= 0
            # ACHTUNG: euclidean nearest neighbor can be nan for classes with
            # less than two patches
            enn = getattr(ls, 'euclidean_nearest_neighbor' + var_suffix)()
            self.assertTrue(enn >= 0 or np.isnan(enn))

        self.assertGreaterEqual(ls.landscape_shape_index(), 1)
        self.assertTrue(
            ls.cell_area / ls.landscape_area <= ls.effective_mesh_size() <=
            ls.landscape_area, 1)
        self.assertTrue(0 < ls.contagion() <= 100)
        # TODO: assert 0 < ls.interspersion_juxtaposition_index() <= 100
        self.assertGreaterEqual(ls.shannon_diversity_index(), 0)

    def test_transonic(self):
        ls_arr = np.load('tests/input_data/ls250_06.npy', allow_pickle=True)
        ls = pls.Landscape(ls_arr, res=(250, 250))
        adjacency_df = ls._adjacency_df
        self.assertIsInstance(adjacency_df, pd.DataFrame)

    def test_plot_landscape(self):
        # first test for a landscape without affine transform (instantiated
        # from an ndarray and without providing a non-None `transform`
        # argument)
        ax = self.ls.plot_landscape()
        # returned axis must be instances of matplotlib axes
        self.assertIsInstance(ax, plt.Axes)

        # now do the same test for a landscape with affine transform (e.g.,
        # instantiated from a raster file)
        ls = pls.Landscape('tests/input_data/ls250_06.tif')
        ax = ls.plot_landscape()
        self.assertIsInstance(ax, plt.Axes)
        # and further test that the plot bounds correspond to the transform's
        # offsets
        self.assertAlmostEqual(ax.get_xlim()[0], ls.transform.xoff)
        self.assertAlmostEqual(ax.get_ylim()[1], ls.transform.yoff)

        # test legend arguments
        self.assertIsNone(ls.plot_landscape(legend=False).get_legend())
        self.assertIsNotNone(ls.plot_landscape(legend=True))
        self.assertIsNotNone(
            ls.plot_landscape(legend=True,
                              legend_kws=dict(loc='center right')))


class TestMultiLandscape(unittest.TestCase):
    def setUp(self):
        from pylandstats import multilandscape

        self.landscapes = [
            pls.Landscape(
                np.load('tests/input_data/ls100_06.npy', allow_pickle=True),
                res=(100, 100)),
            pls.Landscape(
                np.load('tests/input_data/ls250_06.npy', allow_pickle=True),
                res=(250, 250))
        ]
        self.landscape_fps = [
            'tests/input_data/ls100_06.tif', 'tests/input_data/ls250_06.tif'
        ]
        self.attribute_name = 'resolution'
        self.attribute_values = [100, 250]
        self.inexistent_class_val = 999

        # use this class just for testing purposes
        class InstantiableMultiLandscape(multilandscape.MultiLandscape):
            def __init__(self, *args, **kwargs):
                super(InstantiableMultiLandscape,
                      self).__init__(*args, **kwargs)

        self.InstantiableMultiLandscape = InstantiableMultiLandscape

    def test_multilandscape_init(self):
        from pylandstats import multilandscape

        # test that we cannot instantiate an abstract class
        self.assertRaises(TypeError, multilandscape.MultiLandscape)

        # test that if we init a MultiLandscape from filepaths, Landscape
        # instances are automaticaly built
        ml = self.InstantiableMultiLandscape(self.landscape_fps,
                                             self.attribute_name,
                                             self.attribute_values)
        for landscape in ml.landscapes:
            self.assertIsInstance(landscape, pls.Landscape)
        # test that we can pass keyword arguments to the `Landscape`
        # instantiation when providing filepaths
        landscape_kws = {'neighborhood_rule': '4'}
        ml = self.InstantiableMultiLandscape(self.landscape_fps,
                                             self.attribute_name,
                                             self.attribute_values,
                                             **landscape_kws)
        for landscape in ml.landscapes:
            self.assertEqual(landscape.neighborhood_rule,
                             landscape_kws['neighborhood_rule'])
        # test that if we instantiate providing `Landscape` instances, the
        # provided keyword arguments are ignored
        ml = self.InstantiableMultiLandscape(self.landscapes,
                                             self.attribute_name,
                                             self.attribute_values,
                                             **landscape_kws)
        for landscape in ml.landscapes:
            self.assertNotEqual(landscape.neighborhood_rule,
                                landscape_kws['neighborhood_rule'])

        # from this point on, always instantiate from filepaths

        # test that constructing a MultiLandscape where the list of values of
        # the identifying attributes `attribute_values` (in this example, the
        # list of resolutions `[100, 250]`) mismatches the length of the list
        # of landscapes raises a ValueError
        self.assertRaises(ValueError, self.InstantiableMultiLandscape,
                          self.landscape_fps, self.attribute_name, [250])

    def test_multilandscape_dataframes(self):
        ml = self.InstantiableMultiLandscape(self.landscape_fps,
                                             self.attribute_name,
                                             self.attribute_values)
        # test that the data frames that result from `compute_class_metrics_df`
        # and `compute_landscape_metrics_df` are well constructed
        class_metrics_df = ml.compute_class_metrics_df()
        attribute_values = getattr(ml, ml.attribute_name)
        self.assertTrue(
            np.all(class_metrics_df.columns == pls.Landscape.CLASS_METRICS))
        self.assertTrue(
            np.all(class_metrics_df.index == pd.MultiIndex.from_product(
                [ml.present_classes, attribute_values])))
        landscape_metrics_df = ml.compute_landscape_metrics_df()
        self.assertTrue(
            np.all(landscape_metrics_df.columns ==
                   pls.Landscape.LANDSCAPE_METRICS))
        self.assertTrue(np.all(landscape_metrics_df.index == attribute_values))

        # now test the same but with an analysis that only considers a subset
        # of metrics and a subset of classes
        metrics = ['total_area', 'edge_density', 'proportion_of_landscape']
        classes = self.landscapes[0].classes[:2]

        class_metrics_df = ml.compute_class_metrics_df(metrics=metrics,
                                                       classes=classes)
        attribute_values = getattr(ml, ml.attribute_name)
        self.assertTrue(np.all(class_metrics_df.columns == metrics))
        self.assertTrue(
            np.all(class_metrics_df.index == pd.MultiIndex.from_product(
                [classes, attribute_values])))
        # 'proportion_of_landscape' cannot be computed at the landscape level
        # (TODO: test for that elsewhere)
        landscape_metrics = metrics[:2]
        landscape_metrics_df = ml.compute_landscape_metrics_df(
            metrics=landscape_metrics)
        self.assertTrue(
            np.all(landscape_metrics_df.columns == landscape_metrics))
        self.assertTrue(np.all(landscape_metrics_df.index == attribute_values))

    def test_multilandscape_class_metrics_fillna(self):
        ml = self.InstantiableMultiLandscape(self.landscape_fps,
                                             self.attribute_name,
                                             self.attribute_values)
        fillna_df = ml.compute_class_metrics_df()
        nofillna_df = ml.compute_class_metrics_df(fillna=False)

        for metric in fillna_df.columns:
            if metric in self.InstantiableMultiLandscape.METRIC_FILLNA_DICT:
                # this metric should be zero in the first data frame and `NaN`
                # in the second one
                self.assertEqual(
                    np.count_nonzero(
                        fillna_df[metric][nofillna_df[metric].isna()]), 0)
            else:
                # the columns for this metric should be identical in both data
                # frames
                self.assertTrue(fillna_df[metric].equals(nofillna_df[metric]))

    def test_multilandscape_metric_kws(self):
        # Instantiate two multilandscape analyses, one with FRAGSTATS'
        # defaults and the other with keyword arguments specifying the total
        # area in meters and including the boundary in the computation of the
        # total edge.
        ml = self.InstantiableMultiLandscape(self.landscape_fps,
                                             self.attribute_name,
                                             self.attribute_values)
        metrics_kws = {
            'total_area': {
                'hectares': False
            },
            'perimeter_area_ratio_mn': {
                'hectares': True
            },
            'total_edge': {
                'count_boundary': True
            }
        }

        class_metrics_df = ml.compute_class_metrics_df()
        class_metrics_kws_df = ml.compute_class_metrics_df(
            metrics_kws=metrics_kws)
        landscape_metrics_df = ml.compute_landscape_metrics_df()
        landscape_metrics_kws_df = ml.compute_landscape_metrics_df(
            metrics_kws=metrics_kws)
        # For all attribute values and all classes, metric values in hectares
        # should be less than in meters, and excluding boundaries should be
        # less or equal than including them
        for attribute_value in getattr(ml, ml.attribute_name):
            landscape_metrics = landscape_metrics_df.loc[attribute_value]
            landscape_metrics_kws = landscape_metrics_kws_df.loc[
                attribute_value]
            self.assertLess(landscape_metrics['total_area'],
                            landscape_metrics_kws['total_area'])
            self.assertLessEqual(landscape_metrics['total_edge'],
                                 landscape_metrics_kws['total_edge'])

            for class_val in ml.present_classes:
                class_metrics = class_metrics_df.loc[class_val,
                                                     attribute_value]
                class_metrics_kws = class_metrics_kws_df.loc[class_val,
                                                             attribute_value]

                # It could be that for some attribute values, some classes are
                # not present within the respective Landscape. If so, some
                # metrics (e.g., 'perimeter_area_ratio_mn') will be `nan`,
                # both for the analysis with and without keyword arguments.
                # For area and edge metrics, PyLandStats inteprets such `nan`
                # values as 0, which is why we need to include the equality in
                # the comparison
                if class_metrics.isnull().all():
                    self.assertTrue(class_metrics_kws.isnull().all())
                else:
                    self.assertLessEqual(class_metrics['total_area'],
                                         class_metrics_kws['total_area'])
                    # we need quite some tolerance because pixel resolutions in
                    # raster files might be wierd float values, e.g., 99.13213
                    # instead of 100 (meters)
                    self.assertLessEqual(
                        class_metrics['total_edge'],
                        1.01 * class_metrics_kws['total_edge'])

    def test_multilandscape_plot_metrics(self):
        ml = self.InstantiableMultiLandscape(self.landscape_fps,
                                             self.attribute_name,
                                             self.attribute_values)

        existent_class_val = ml.present_classes[0]
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
        # test that the x data of any line corresponds to the attribute values
        for line in ax.lines:
            self.assertTrue(
                np.all(line.get_xdata() == getattr(ml, ml.attribute_name)))

        # test metric label arguments/settings
        # when passing default arguments, the axis ylabel must be the one set
        # within the settings module
        self.assertEqual(
            ml.plot_metric('edge_density').get_ylabel(),
            pls.settings.metric_label_dict['edge_density'])
        # when passing `metric_legend=False`, the axis ylabel must be an empty
        # string
        self.assertEqual(
            ml.plot_metric('edge_density', metric_legend=False).get_ylabel(),
            '')
        # when passing an non-empty string as `metric_label`, the axis ylabel
        # must be such string
        self.assertEqual(
            ml.plot_metric('edge_density', metric_label='foo').get_ylabel(),
            'foo')

        # test that asking to plot an inexistent metric raises a `ValueError`
        # try that raised ValueErrors have different error messages depending
        # on the context
        with self.assertRaises(ValueError) as cm:
            ml.plot_metric('foo')
            self.assertIn('is not among', str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            ml.plot_metric('proportion_of_landscape')
            self.assertIn('cannot be computed', str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            ml.plot_metric('contagion', class_val=1)
            self.assertIn('cannot be computed', str(cm.exception))

    def test_plot_landscapes(self):
        ml = self.InstantiableMultiLandscape(self.landscape_fps,
                                             self.attribute_name,
                                             self.attribute_values)

        fig = ml.plot_landscapes()

        # there must be one column for each landscape
        self.assertEqual(len(fig.axes), len(ml))

        # returned axes must be instances of matplotlib axes
        for ax in fig.axes:
            self.assertIsInstance(ax, plt.Axes)

        # test that by default, the dimensions of the resulting will come from
        # matplotlib's settings
        rc_figwidth, rc_figheight = plt.rcParams['figure.figsize']
        figwidth, figheight = fig.get_size_inches()
        # we have `len(ml)` axis, therefore the actual `figwidth` must be
        # `len(ml) * rc_figwidth`
        self.assertAlmostEqual(figwidth, len(ml) * rc_figwidth)
        self.assertAlmostEqual(figheight, rc_figheight)
        # if instead, we customize the figure size, the dimensions of the
        # resulting figure must be the customized ones
        custom_figsize = (10, 10)
        fig = ml.plot_landscapes(subplots_kws={'figsize': custom_figsize})
        figwidth, figheight = fig.get_size_inches()
        self.assertAlmostEqual(custom_figsize[0], figwidth)
        self.assertAlmostEqual(custom_figsize[1], figheight)


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
        # test that the `attribute_name` is dates, and that if the `dates`
        # argument is not provided when instantiating a
        # `SpatioTemporalAnalysis`, the dates attribute is properly and
        # automatically generated
        sta = pls.SpatioTemporalAnalysis(self.landscape_fps)
        self.assertEqual(sta.attribute_name, 'dates')
        self.assertEqual(len(sta), len(sta.dates))

        # test the `neighborhood_rule` argument
        neighborhood_rule = '4'
        other_neighborhood_rule = '8'
        # test that if provided and the elements of `landscapes` are filepaths,
        # the value is passed to each landscape
        for ls in pls.SpatioTemporalAnalysis(
                self.landscape_fps, dates=self.dates,
                neighborhood_rule=neighborhood_rule).landscapes:
            self.assertEqual(ls.neighborhood_rule, neighborhood_rule)
        # test that if provided and the elements of `landscapes` are
        # `Landscape` instances, the value is ignored
        for ls in pls.SpatioTemporalAnalysis([
                pls.Landscape(landscape_fp,
                              neighborhood_rule=other_neighborhood_rule)
                for landscape_fp in self.landscape_fps
        ], dates=self.dates, neighborhood_rule=neighborhood_rule).landscapes:
            self.assertEqual(ls.neighborhood_rule, other_neighborhood_rule)
        # test that if not provided and the elements of `landscapes` are
        # filepaths, the default value is taken
        for ls in pls.SpatioTemporalAnalysis(self.landscape_fps,
                                             dates=self.dates).landscapes:
            self.assertEqual(ls.neighborhood_rule,
                             pls.settings.DEFAULT_NEIGHBORHOOD_RULE)

    def test_spatiotemporalanalysis_dataframes(self):
        # test with the default constructor
        sta = pls.SpatioTemporalAnalysis(self.landscape_fps)

        # test that the data frames that result from `compute_class_metrics_df`
        # and `compute_landscape_metrics_df` are well constructed
        class_metrics_df = sta.compute_class_metrics_df()
        self.assertTrue(
            np.all(class_metrics_df.index == pd.MultiIndex.from_product(
                [sta.present_classes, sta.dates])))
        landscape_metrics_df = sta.compute_landscape_metrics_df()
        self.assertTrue(np.all(landscape_metrics_df.index == sta.dates))

        # now test the same but with an analysis that only considers a
        # subset of metrics and a subset of classes
        metrics = ['total_area', 'edge_density', 'proportion_of_landscape']
        classes = sta.present_classes[:2]

        class_metrics_df = sta.compute_class_metrics_df(
            metrics=metrics, classes=classes)
        self.assertTrue(
            np.all(class_metrics_df.index == pd.MultiIndex.from_product(
                [classes, sta.dates])))
        # 'proportion_of_landscape' cannot be computed at the landscape level
        # (TODO: test for that elsewhere)
        landscape_metrics = metrics[:2]
        landscape_metrics_df = sta.compute_landscape_metrics_df(
            metrics=landscape_metrics)
        self.assertTrue(np.all(landscape_metrics_df.index == sta.dates))

    def test_spatiotemporalanalysis_plot_metrics(self):
        sta = pls.SpatioTemporalAnalysis(self.landscape_fps, dates=self.dates)

        # test for `None` (landscape-level) and an existing class (class-level)
        for class_val in [None, sta.present_classes[0]]:
            # test that the x data of the line corresponds to the dates
            self.assertTrue(
                np.all(
                    sta.plot_metric('patch_density', class_val=class_val).
                    lines[0].get_xdata() == self.dates))


class TestZonaAlnalysis(unittest.TestCase):
    def setUp(self):
        self.masks_arr = np.load('tests/input_data/masks_arr.npy',
                                 allow_pickle=True)
        self.landscape = pls.Landscape(
            np.load('tests/input_data/ls250_06.npy', allow_pickle=True),
            res=(250, 250))
        self.landscape_fp = 'tests/input_data/ls250_06.tif'
        with rio.open(self.landscape_fp) as src:
            self.landscape_transform = src.transform
            self.landscape_crs = src.crs
        self.masks_fp = 'tests/input_data/gmb-lausanne'
        # for buffer analysis
        self.geom = geometry.Point(6.6327025, 46.5218269)
        self.buffer_dists = [10000, 15000, 20000]

        self.tmp_dir = path.join('tests/tmp')
        os.mkdir(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_zonal_init(self):
        # test that the attribute names and values are consistent with the
        # provided `masks_arr`
        za = pls.ZonalAnalysis(self.landscape, masks=self.masks_arr)
        self.assertEqual(za.attribute_name, 'attribute_values')
        self.assertEqual(len(za), len(self.masks_arr))
        self.assertEqual(len(za), len(za.attribute_values))

        # test that if we init a `ZonalAnalysis` from filepaths, Landscape
        # instances are automaticaly built, and the attribute names and values
        # are also consistent with the provided `masks_arr`
        za = pls.ZonalAnalysis(self.landscape_fp, masks=self.masks_arr)
        for landscape in za.landscapes:
            self.assertIsInstance(landscape, pls.Landscape)
        self.assertEqual(za.attribute_name, 'attribute_values')
        self.assertEqual(len(za), len(self.masks_arr))
        self.assertEqual(len(za), len(za.attribute_values))

        # test passing GeoSeries, GeoDataFrame and geopandas files as `masks`
        masks_gdf = gpd.read_file(self.masks_fp)
        masks_index_col = 'GMDNAME'

        # first test the GeoSeries, which works like the others except that we
        # cannot set a column as the zone index
        masks_gser = masks_gdf['geometry'].copy()
        za = pls.ZonalAnalysis(self.landscape_fp, masks=masks_gser)
        self.assertLessEqual(len(za), len(masks_gdf))
        za = pls.ZonalAnalysis(self.landscape_fp, masks=masks_gser,
                               masks_index_col=masks_index_col)
        self.assertLessEqual(len(za), len(masks_gdf))
        # also test that attribute name is properly set when using geoseries
        # as `masks` note that if a non-None `attriubte_name` is provided, it
        # always takes precedence
        attribute_name = 'foo'
        # first test for a geoseries with name and unnamed index
        masks_gser.name = 'bar'
        za = pls.ZonalAnalysis(self.landscape_fp, masks=masks_gser)
        self.assertEqual(za.attribute_name, masks_gser.name)
        za = pls.ZonalAnalysis(self.landscape_fp, masks=masks_gser,
                               attribute_name=attribute_name)
        self.assertEqual(za.attribute_name, attribute_name)
        # now test that for a geoseries with name and named index, the
        # geoseries name takes precedence
        masks_gser.index.name = 'name'
        za = pls.ZonalAnalysis(self.landscape_fp, masks=masks_gser)
        self.assertEqual(za.attribute_name, masks_gser.name)
        za = pls.ZonalAnalysis(self.landscape_fp, masks=masks_gser,
                               attribute_name=attribute_name)
        self.assertEqual(za.attribute_name, attribute_name)
        # finally test that for an unnamed geoseries with named index, the
        # geoseries index name is taken
        masks_gser.name = None
        za = pls.ZonalAnalysis(self.landscape_fp, masks=masks_gser)
        self.assertEqual(za.attribute_name, masks_gser.index.name)
        za = pls.ZonalAnalysis(self.landscape_fp, masks=masks_gser,
                               attribute_name=attribute_name)
        self.assertEqual(za.attribute_name, attribute_name)

        # now test the GeoDataFrame and geopandas file
        for masks in self.masks_fp, masks_gdf:
            # test init
            za = pls.ZonalAnalysis(self.landscape_fp, masks=masks)
            self.assertLessEqual(len(za), len(masks_gdf))
            self.assertTrue(
                np.all(np.isin(getattr(za, za.attribute_name),
                               masks_gdf.index)))
            # test that we can set a column as the zone index
            za = pls.ZonalAnalysis(self.landscape_fp, masks=masks,
                                   masks_index_col=masks_index_col)
            self.assertTrue(
                np.all(
                    np.isin(getattr(za, masks_index_col),
                            masks_gdf[masks_index_col])))

        # test the `neighborhood_rule` argument
        neighborhood_rule = '4'
        other_neighborhood_rule = '8'
        # test that if provided and `landscape` is a filepath, the
        # value is passed to each landscape
        for ls in pls.ZonalAnalysis(
                self.landscape_fp, masks=self.masks_arr,
                neighborhood_rule=neighborhood_rule).landscapes:
            self.assertEqual(ls.neighborhood_rule, neighborhood_rule)
        # test that if provided and `landscape` is a `Landscape` instance, the
        # value is ignored
        for ls in pls.ZonalAnalysis(
                pls.Landscape(self.landscape_fp,
                              neighborhood_rule=other_neighborhood_rule),
                masks=self.masks_arr,
                neighborhood_rule=neighborhood_rule).landscapes:
            self.assertEqual(ls.neighborhood_rule, other_neighborhood_rule)
        # test that if not provided and `landscape` is a filepath, the default
        # value is taken
        # from this point on, always instantiate from filepaths
        for ls in pls.ZonalAnalysis(self.landscape_fp,
                                    masks=self.masks_arr).landscapes:
            self.assertEqual(ls.neighborhood_rule,
                             pls.settings.DEFAULT_NEIGHBORHOOD_RULE)

    def test_zonal_plot_metrics(self):
        za = pls.ZonalAnalysis(self.landscape_fp, masks=self.masks_arr)

        # test for `None` (landscape-level) and an existing class (class-level)
        for class_val in [None, za.present_classes[0]]:
            # test that the x data of the line corresponds to the attribute
            # values
            self.assertTrue(
                np.all(
                    za.plot_metric('patch_density', class_val=class_val).
                    lines[0].get_xdata() == za.attribute_values))

    def test_compute_zonal_statistics_arr(self):
        za = pls.ZonalAnalysis(self.landscape_fp, masks=self.masks_arr)

        # test that the array has the proper shape
        zs_arr = za.compute_zonal_statistics_arr('patch_density')
        zs_arr.shape = za.landscape_meta['height'], za.landscape_meta['width']

        # test that a zonal statistics array computed at the class level has
        # at least as many nan values than its landscape-level counterpart
        self.assertGreaterEqual(
            np.isnan(zs_arr).sum(),
            np.isnan(
                za.compute_zonal_statistics_arr(
                    'patch_density', class_val=za.present_classes[0])).sum())

        # test that the zonal statistics when excluding boundaries should be
        # less or equal than including them
        self.assertLessEqual(
            np.nansum(za.compute_zonal_statistics_arr('total_edge')),
            np.nansum(
                za.compute_zonal_statistics_arr(
                    'total_edge', metric_kws={'count_boundary': True})))

        # test that passing `dst_filepath` dumps a raster file
        dst_filepath = path.join(self.tmp_dir, 'foo.tif')
        za.compute_zonal_statistics_arr('patch_density',
                                        dst_filepath=dst_filepath)
        self.assertTrue(path.exists(dst_filepath))

    def test_buffer_init(self):
        naive_gser = gpd.GeoSeries([self.geom])
        gser = gpd.GeoSeries([self.geom], crs=geom_crs)

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
                              {'base_mask_crs': geom_crs})
            self.assertRaises(ValueError, pls.BufferAnalysis, self.landscape,
                              base_mask, self.buffer_dists, {
                                  'base_mask_crs': geom_crs,
                                  'landscape_crs': self.landscape_crs
                              })
            self.assertRaises(
                ValueError, pls.BufferAnalysis, self.landscape, base_mask,
                self.buffer_dists, {
                    'base_mask_crs': geom_crs,
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
                                   self.buffer_dists, base_mask_crs=geom_crs),
                pls.BufferAnalysis(self.landscape_fp, naive_gser,
                                   self.buffer_dists, base_mask_crs=geom_crs),
                pls.BufferAnalysis(self.landscape_fp, gser, self.buffer_dists),
                pls.BufferAnalysis(self.landscape_fp, gser, self.buffer_dists,
                                   base_mask_crs=geom_crs),
                pls.BufferAnalysis(
                    self.landscape, gser, self.buffer_dists,
                    base_mask_crs=geom_crs, landscape_crs=self.landscape_crs,
                    landscape_transform=self.landscape_transform)
        ]:
            self.assertEqual(ba.attribute_name, 'buffer_dists')
            self.assertEqual(len(ba), len(ba.masks_arr))
            self.assertEqual(len(ba), len(ba.buffer_dists))

        # test that we cannot instantiate a `BufferAnalysis` with
        # `buffer_rings=True` if `base_mask` is a polygon or a GeoSeries
        # containing a polygon
        polygon = self.geom.buffer(1000)  # this will return a polygon instance
        self.assertRaises(ValueError, pls.BufferAnalysis, self.landscape_fp,
                          polygon, self.buffer_dists, {
                              'buffer_rings': True,
                              'base_mask_crs': geom_crs
                          })
        polygon_gser = gpd.GeoSeries([polygon], crs=geom_crs)
        self.assertRaises(ValueError, pls.BufferAnalysis, self.landscape_fp,
                          polygon_gser, self.buffer_dists, {
                              'buffer_rings': True,
                          })

        # test that otherwise, buffer rings are properly instantiated
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

    def test_grid_init(self):
        # test init by number of zone rows/cols
        num_zone_rows, num_zone_cols = 10, 10
        zga = pls.ZonalGridAnalysis(self.landscape_fp,
                                    num_zone_rows=num_zone_rows,
                                    num_zone_cols=num_zone_cols)
        self.assertEqual(len(zga.data_zones), num_zone_rows * num_zone_cols)
        self.assertGreaterEqual(len(zga.data_zones), len(zga.masks_arr))

        # test init by zone pixel width/height
        zone_pixel_width, zone_pixel_height = 100, 100
        zga = pls.ZonalGridAnalysis(self.landscape_fp,
                                    zone_pixel_width=zone_pixel_width,
                                    zone_pixel_height=zone_pixel_height)

        # test the `neighborhood_rule` argument
        neighborhood_rule = '4'
        other_neighborhood_rule = '8'
        # test that if provided and `landscape` is a filepath, the
        # value is passed to each landscape
        for ls in pls.ZonalGridAnalysis(
                self.landscape_fp, zone_pixel_width=zone_pixel_width,
                zone_pixel_height=zone_pixel_height,
                neighborhood_rule=neighborhood_rule).landscapes:
            self.assertEqual(ls.neighborhood_rule, neighborhood_rule)
        # test that if provided and `landscape` is a `Landscape` instance, the
        # value is ignored
        for ls in pls.ZonalGridAnalysis(
                pls.Landscape(self.landscape_fp,
                              neighborhood_rule=other_neighborhood_rule),
                zone_pixel_width=zone_pixel_width,
                zone_pixel_height=zone_pixel_height,
                neighborhood_rule=neighborhood_rule).landscapes:
            self.assertEqual(ls.neighborhood_rule, other_neighborhood_rule)
        # test that if not provided and `landscape` is a filepath, the default
        # value is taken
        # from this point on, always instantiate from filepaths
        for ls in pls.ZonalGridAnalysis(
                self.landscape_fp, zone_pixel_width=zone_pixel_width,
                zone_pixel_height=zone_pixel_height).landscapes:
            self.assertEqual(ls.neighborhood_rule,
                             pls.settings.DEFAULT_NEIGHBORHOOD_RULE)


class TestSpatioTemporalBufferAnalysis(unittest.TestCase):
    def setUp(self):
        self.landscape_fps = [
            'tests/input_data/ls250_06.tif', 'tests/input_data/ls250_12.tif'
        ]
        self.dates = [2006, 2012]
        self.base_mask = gpd.GeoSeries([geometry.Point(6.6327025, 46.5218269)],
                                       crs=geom_crs)
        self.buffer_dists = [10000, 15000, 20000]

    def test_spatiotemporalbufferanalysis_init(self):
        # we will just test the base init, the rest of functionalities have
        # already been tested above (in `TestSpatioTemporalAnalysis` and
        # `TestZonalAnalysis`)
        stba = pls.SpatioTemporalBufferAnalysis(self.landscape_fps,
                                                self.base_mask,
                                                self.buffer_dists,
                                                dates=self.dates)
        self.assertEqual(len(stba.buffer_dists), len(stba.stas))
        for sta in stba.stas:
            self.assertEqual(sta.dates, self.dates)

        # test the `neighborhood_rule` argument
        neighborhood_rule = '4'
        other_neighborhood_rule = '8'
        # test that if provided and the elements of `landscapes` are filepaths,
        # the value is passed to each landscape
        for sta in pls.SpatioTemporalBufferAnalysis(
                self.landscape_fps, self.base_mask, self.buffer_dists,
                dates=self.dates, neighborhood_rule=neighborhood_rule).stas:
            for ls in sta.landscapes:
                self.assertEqual(ls.neighborhood_rule, neighborhood_rule)
        # test that if provided and the elements of `landscapes` are
        # `Landscape` instances, the value is ignored
        with rio.open(self.landscape_fps[0]) as src:
            landscape_crs = src.crs
            landscape_transform = src.transform
        for sta in pls.SpatioTemporalBufferAnalysis(
            [
                pls.Landscape(landscape_fp,
                              neighborhood_rule=other_neighborhood_rule)
                for landscape_fp in self.landscape_fps
            ], self.base_mask, self.buffer_dists, landscape_crs=landscape_crs,
                landscape_transform=landscape_transform, dates=self.dates,
                neighborhood_rule=neighborhood_rule).stas:
            for ls in sta.landscapes:
                self.assertEqual(ls.neighborhood_rule, other_neighborhood_rule)
        # test that if not provided and the elements of `landscapes` are
        # filepaths, the default value is taken
        for sta in pls.SpatioTemporalBufferAnalysis(self.landscape_fps,
                                                    self.base_mask,
                                                    self.buffer_dists,
                                                    dates=self.dates).stas:
            for ls in sta.landscapes:
                self.assertEqual(ls.neighborhood_rule,
                                 pls.settings.DEFAULT_NEIGHBORHOOD_RULE)

    def test_spatiotemporalbufferanalysis_dataframes(self):
        stba = pls.SpatioTemporalBufferAnalysis(self.landscape_fps,
                                                self.base_mask,
                                                self.buffer_dists,
                                                dates=self.dates)

        # test that the data frames that result from `compute_class_metrics_df`
        # and `compute_landscape_metrics_df` are well constructed
        class_metrics_df = stba.compute_class_metrics_df()
        self.assertTrue(
            np.all(class_metrics_df.index == pd.MultiIndex.from_product(
                [stba.buffer_dists, stba.present_classes, stba.dates])))
        landscape_metrics_df = stba.compute_landscape_metrics_df()
        self.assertTrue(
            np.all(landscape_metrics_df.index == pd.MultiIndex.from_product(
                [stba.buffer_dists, stba.dates])))

        # now test the same but with an analysis that only considers a
        # subset of metrics and a subset of classes
        metrics = ['total_area', 'edge_density', 'proportion_of_landscape']
        classes = stba.present_classes[:2]

        class_metrics_df = stba.compute_class_metrics_df(
            metrics=metrics, classes=classes)
        self.assertTrue(
            np.all(class_metrics_df.index == pd.MultiIndex.from_product(
                [stba.buffer_dists, classes, stba.dates])))
        # 'proportion_of_landscape' cannot be computed at the landscape level
        # (TODO: test for that elsewhere)
        landscape_metrics = metrics[:2]
        landscape_metrics_df = stba.compute_landscape_metrics_df(
            metrics=landscape_metrics)
        self.assertTrue(
            np.all(landscape_metrics_df.index == pd.MultiIndex.from_product(
                [stba.buffer_dists, stba.dates])))

    def test_spatiotemporalbufferanalysis_plot_metric(self):
        stba = pls.SpatioTemporalBufferAnalysis(self.landscape_fps,
                                                self.base_mask,
                                                self.buffer_dists)

        # test for `None` (landscape-level) and an existing class (class-level)
        for class_val in [None, stba.stas[0].present_classes[0]]:
            ax = stba.plot_metric('patch_density', class_val=class_val)
            # test that there is a line for each buffer distance
            self.assertEqual(len(ax.lines), len(self.buffer_dists))
            # test that there is a legend label for each buffer distance
            handles, labels = ax.get_legend_handles_labels()
            self.assertEqual(len(labels), len(self.buffer_dists))

    def test_plot_landscapes(self):
        stba = pls.SpatioTemporalBufferAnalysis(self.landscape_fps,
                                                self.base_mask,
                                                self.buffer_dists)

        fig = stba.plot_landscapes()

        # there must be one column for each buffer distance and one row for
        # each date
        self.assertEqual(len(fig.axes),
                         len(stba.buffer_dists) * len(stba.dates))

        # returned axes must be instances of matplotlib axes
        for ax in fig.axes:
            self.assertIsInstance(ax, plt.Axes)

        # test that by default, the dimensions of the resulting will come from
        # matplotlib's settings
        rc_figwidth, rc_figheight = plt.rcParams['figure.figsize']
        figwidth, figheight = fig.get_size_inches()
        # the actual `figwidth` must be `len(stba.buffer_dists) * rc_figwidth`
        # and `figheight` must be `len(stba.dates) * rc_figheight`
        self.assertAlmostEqual(figwidth, len(stba.buffer_dists) * rc_figwidth)
        self.assertAlmostEqual(figheight, len(stba.dates) * rc_figheight)
        # if instead, we customize the figure size, the dimensions of the
        # resulting figure must be the customized ones
        custom_figsize = (10, 10)
        fig = stba.plot_landscapes(subplots_kws={'figsize': custom_figsize})
        figwidth, figheight = fig.get_size_inches()
        self.assertAlmostEqual(custom_figsize[0], figwidth)
        self.assertAlmostEqual(custom_figsize[1], figheight)

        # first row has the date as title
        for date, ax in zip(stba.dates, fig.axes):
            self.assertEqual(str(date), ax.get_title())
        # first column has the buffer distance as `ylabel`
        for buffer_dist, i in zip(stba.buffer_dists,
                                  range(0, len(fig.axes), len(stba.dates))):
            self.assertEqual(str(buffer_dist), fig.axes[i].get_ylabel())
