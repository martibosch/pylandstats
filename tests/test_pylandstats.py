"""pylandstats tests."""

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

plt.switch_backend("agg")  # only for testing purposes
geom_crs = "epsg:4326"

tests_dir = "tests"
tests_data_dir = path.join(tests_dir, "data")


class TestImports(unittest.TestCase):
    def test_base_imports(self):
        pass

    def test_geo_imports(self):
        pass


class TestLandscape(unittest.TestCase):
    def setUp(self):
        self.ls_arr = np.load(
            path.join(tests_data_dir, "ls250_06.npy"), allow_pickle=True
        )
        self.ls = pls.Landscape(self.ls_arr, res=(250, 250))
        self.landscape_fp = path.join(tests_data_dir, "ls250_06.tif")

    def test_io(self):
        # test that if we provide a ndarray, we also need to provide the resolution
        with self.assertRaises(ValueError) as cm:
            ls = pls.Landscape(self.ls_arr)
        self.assertIn("must be provided", str(cm.exception))

        ls = pls.Landscape(self.landscape_fp)
        # resolutions are not exactly 250, they are between [249, 251], so we need to
        # use a large delta
        self.assertAlmostEqual(ls.cell_width, 250, delta=1)
        self.assertAlmostEqual(ls.cell_height, 250, delta=1)
        self.assertAlmostEqual(ls.cell_area, 250 * 250, delta=250)

        # test that the transform is None if we instantiate a landscape from an ndarray
        # (without providing the `transform` argument, but that it is not none we get
        # the landscape transform from a raster path
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
                np.isreal(getattr(ls, class_metric)(class_val=ls.classes[0]))
            )

        for landscape_metric in pls.Landscape.LANDSCAPE_METRICS:
            self.assertTrue(np.isreal(getattr(ls, landscape_metric)()))

    def test_neighborhood_rules(self):
        # test that providing a value different than 8 or 4 raises a `ValueError`
        with self.assertRaises(ValueError) as cm:
            pls.Landscape(self.landscape_fp, neighborhood_rule="2")
        self.assertIn("is not among", str(cm.exception))
        # test that we can provide the argument as int as long as it is 8 or 4
        for neighborhood_rule in (8, 4):
            self.assertEqual(
                pls.Landscape(
                    self.landscape_fp, neighborhood_rule=neighborhood_rule
                ).neighborhood_rule,
                str(neighborhood_rule),
            )
        # test that there is at least the same number of patches with the Moore
        # neighborhood than with Von Neumann's
        ls_moore = pls.Landscape(self.landscape_fp, neighborhood_rule="8")
        ls_von_neumann = pls.Landscape(self.landscape_fp, neighborhood_rule="4")
        self.assertLessEqual(
            ls_moore.number_of_patches(), ls_von_neumann.number_of_patches()
        )

    def test_metrics_warnings(self):
        # test that warnings are raised

        # class-level metrics
        # euclidean nearest neighbor will return nan (and raise an informative warning)
        # if there is not at least two patches of each class. Let us test this by
        # creating a landscape with a background of class 1 and a single patch of class
        # 2
        arr = np.ones((4, 4))
        arr[1:-1, 1:-1] = 2
        ls = pls.Landscape(arr, res=(1, 1))

        # let us test that both the computation at the class-level (1 and 2) and at the
        # landscape level (`class_val` of `None`) raise at least one warning (the exact
        # number of warnings raised can be different in Python 2 and 3)
        for class_val in [1, 2, None]:
            with warnings.catch_warnings(record=True) as w:
                ls.euclidean_nearest_neighbor(class_val=class_val)
                self.assertGreater(len(w), 0)

        # landscape-level metrics
        # some landscape-level metrics require at least two classes.
        ls = pls.Landscape(np.ones((4, 4)), res=(1, 1))
        for method in pls.Landscape.ENTROPY_METRICS:
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
            np.all(patch_df.columns.drop("class_val") == pls.Landscape.PATCH_METRICS)
        )
        self.assertEqual(patch_df.index.name, "patch_id")
        # try that raised ValueErrors have different error messages depending on the
        # context
        with self.assertRaises(ValueError) as cm:
            ls.compute_patch_metrics_df(metrics=["foo"])
        self.assertIn("is not among", str(cm.exception))
        for metric in class_metrics.union(landscape_metrics):
            with self.assertRaises(ValueError) as cm:
                ls.compute_patch_metrics_df(metrics=[metric])
            self.assertIn("cannot be computed", str(cm.exception))

        class_df = ls.compute_class_metrics_df()
        self.assertEqual(
            len(class_df.columns.difference(pls.Landscape.CLASS_METRICS)), 0
        )
        self.assertEqual(class_df.index.name, "class_val")
        # try that raised ValueErrors have different error messages depending on the
        # context
        with self.assertRaises(ValueError) as cm:
            ls.compute_class_metrics_df(metrics=["foo"])
        self.assertIn("is not among", str(cm.exception))
        for metric in patch_metrics.union(landscape_metrics.difference(class_metrics)):
            with self.assertRaises(ValueError) as cm:
                ls.compute_class_metrics_df(metrics=[metric])
            self.assertIn("cannot be computed", str(cm.exception))

        landscape_df = ls.compute_landscape_metrics_df()
        self.assertEqual(
            len(landscape_df.columns.difference(pls.Landscape.LANDSCAPE_METRICS)),
            0,
        )
        self.assertEqual(len(landscape_df.index), 1)
        # try that raised ValueErrors have different error messages depending on the
        # context
        with self.assertRaises(ValueError) as cm:
            ls.compute_landscape_metrics_df(metrics=["foo"])
        self.assertIn("is not among", str(cm.exception))
        for metric in patch_metrics.union(class_metrics.difference(landscape_metrics)):
            with self.assertRaises(ValueError) as cm:
                ls.compute_landscape_metrics_df(metrics=[metric])
            self.assertIn("cannot be computed", str(cm.exception))

    def test_metrics_value_ranges(self):
        ls = self.ls

        # basic tests of the `Landscape` class' attributes
        self.assertNotIn(ls.nodata, ls.classes)
        self.assertGreater(ls.landscape_area, 0)

        class_val = ls.classes[0]
        # label_arr = ls._get_label_arr(class_val)

        # patch-level metrics
        self.assertTrue((ls.area()["area"] > 0).all())
        self.assertTrue((ls.perimeter()["perimeter"] > 0).all())
        self.assertTrue((ls.perimeter_area_ratio()["perimeter_area_ratio"] > 0).all())
        self.assertTrue((ls.shape_index()["shape_index"] >= 1).all())
        _fractal_dimension_ser = ls.fractal_dimension()["fractal_dimension"]
        self.assertTrue((_fractal_dimension_ser <= 2).all())
        # ACHTUNG: ugly hardcoded tolerance to correct for mysterious errors in GitHub
        # Actions with some Python versions
        self.assertTrue((_fractal_dimension_ser >= 1 - 1e-3).all())
        self.assertTrue((ls.core_area()["core_area"] >= 0).all())
        self.assertTrue((ls.number_of_core_areas()["number_of_core_areas"] >= 0).all())
        _core_area_index_ser = ls.core_area_index()["core_area_index"]
        self.assertTrue((_core_area_index_ser >= 0).all())
        self.assertTrue((_core_area_index_ser < 100).all())
        # TODO: assert 0 <= ls.contiguity_index(patch_arr) <= 1
        # ACHTUNG: euclidean nearest neighbor can be nan for classes with less than two
        # patches
        self.assertTrue(
            (
                ls.euclidean_nearest_neighbor()["euclidean_nearest_neighbor"].dropna()
                > 0
            ).all()
        )
        # TODO: assert 0 <= ls.proximity(patch_arr) <= 1

        # class-level metrics
        self.assertGreater(ls.total_area(class_val=class_val), 0)
        self.assertTrue(0 < ls.proportion_of_landscape(class_val) < 100)
        self.assertTrue(ls.patch_density(class_val=class_val) > 0)
        self.assertTrue(0 < ls.largest_patch_index(class_val=class_val) < 100)
        self.assertGreaterEqual(ls.total_edge(class_val=class_val), 0)
        self.assertGreaterEqual(ls.edge_density(class_val=class_val), 0)
        self.assertGreaterEqual(ls.total_core_area(class_val=class_val), 0)
        self.assertGreaterEqual(
            ls.core_area_proportion_of_landscape(class_val=class_val), 0
        )
        self.assertGreaterEqual(
            ls.number_of_disjunct_core_areas(class_val=class_val), 0
        )

        # the value ranges of mean, area-weighted mean and median aggregations are going
        # to be the same as their respective original metrics
        mean_suffixes = ["_mn", "_am", "_md"]
        # the value ranges of the range, standard deviation and coefficient of variation
        # will always be nonnegative as long as the means are nonnegative as well (which
        # is the case of all of the metrics implemented so far)
        var_suffixes = ["_ra", "_sd", "_cv"]

        for mean_suffix in mean_suffixes:
            self.assertGreater(
                getattr(ls, "area" + mean_suffix)(class_val=class_val), 0
            )
            self.assertGreater(
                getattr(ls, "perimeter_area_ratio" + mean_suffix)(class_val=class_val),
                0,
            )
            self.assertGreaterEqual(
                getattr(ls, "shape_index" + mean_suffix)(class_val=class_val), 1
            )
            self.assertTrue(
                1
                <= getattr(ls, "fractal_dimension" + mean_suffix)(class_val=class_val)
                <= 2
            )
            self.assertGreaterEqual(
                getattr(ls, "core_area" + mean_suffix)(class_val=class_val), 0
            )
            self.assertGreaterEqual(
                getattr(ls, "disjunct_core_area" + mean_suffix)(class_val=class_val), 0
            )
            # self.assertGreaterEqual(
            #     getattr(ls, "number_of_core_areas" + mean_suffix)(
            #         class_val=class_val),
            #     0,
            # )
            cai = getattr(ls, "core_area_index" + mean_suffix)(class_val=class_val)
            self.assertGreaterEqual(cai, 0)
            self.assertLess(cai, 100)
            # assert 0 <= getattr(
            #     ls, 'contiguity_index' + mean_suffix)(class_val) <= 1
            # assert getattr(ls, 'proximity' + mean_suffix)(class_val) >= 0
            # ACHTUNG: euclidean nearest neighbor can be nan for classes with less than
            # two patches
            enn = getattr(ls, "euclidean_nearest_neighbor" + mean_suffix)(
                class_val=class_val
            )
            self.assertTrue(enn > 0 or np.isnan(enn))

        for var_suffix in var_suffixes:
            self.assertGreaterEqual(
                getattr(ls, "area" + mean_suffix)(class_val=class_val), 0
            )
            self.assertGreaterEqual(
                getattr(ls, "perimeter_area_ratio" + var_suffix)(class_val=class_val), 0
            )
            self.assertGreaterEqual(
                getattr(ls, "shape_index" + var_suffix)(class_val=class_val), 0
            )
            self.assertGreaterEqual(
                getattr(ls, "fractal_dimension" + var_suffix)(class_val=class_val), 0
            )
            # assert getattr(
            #    ls, 'contiguity_index' + var_suffix)(class_val) >= 0
            # assert getattr(ls, 'proximity' + var_suffix)(class_val) >= 0
            # ACHTUNG: euclidean nearest neighbor can be nan for classes with less than
            # two patches
            enn = getattr(ls, "euclidean_nearest_neighbor" + var_suffix)(
                class_val=class_val
            )
            self.assertTrue(enn >= 0 or np.isnan(enn))

        # TODO: assert 0 < ls.interspersion_juxtaposition_index(
        #           class_val) <= 100
        self.assertGreaterEqual(ls.landscape_shape_index(class_val=class_val), 1)
        self.assertTrue(
            ls.cell_area / ls.landscape_area
            <= ls.effective_mesh_size(class_val=class_val)
            <= ls.landscape_area,
            1,
        )

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
            self.assertGreater(getattr(ls, "area" + mean_suffix)(), 0)
            self.assertGreater(getattr(ls, "perimeter_area_ratio" + mean_suffix)(), 0)
            self.assertGreaterEqual(getattr(ls, "shape_index" + mean_suffix)(), 1)
            self.assertTrue(1 <= getattr(ls, "fractal_dimension" + mean_suffix)() <= 2)
            # assert 0 <= getattr(ls, 'contiguity_index' + mean_suffix)() <= 1
            # assert getattr(ls, 'proximity' + mean_suffix)() >= 0
            # ACHTUNG: euclidean nearest neighbor can be nan for classes with less than
            # two patches
            enn = getattr(ls, "euclidean_nearest_neighbor" + mean_suffix)()
            self.assertTrue(enn > 0 or np.isnan(enn))
        for var_suffix in var_suffixes:
            self.assertGreater(getattr(ls, "area" + var_suffix)(), 0)
            self.assertGreaterEqual(
                getattr(ls, "perimeter_area_ratio" + var_suffix)(), 0
            )
            self.assertGreaterEqual(getattr(ls, "shape_index" + var_suffix)(), 0)
            self.assertGreaterEqual(getattr(ls, "fractal_dimension" + var_suffix)(), 0)
            # assert getattr(ls, 'contiguity_index' + var_suffix)() >= 0
            # assert getattr(ls, 'proximity' + var_suffix)() >= 0
            # ACHTUNG: euclidean nearest neighbor can be nan for classes with less than
            # two patches
            enn = getattr(ls, "euclidean_nearest_neighbor" + var_suffix)()
            self.assertTrue(enn >= 0 or np.isnan(enn))

        self.assertGreaterEqual(ls.landscape_shape_index(), 1)
        self.assertTrue(
            ls.cell_area / ls.landscape_area
            <= ls.effective_mesh_size()
            <= ls.landscape_area,
            1,
        )
        for method in [
            "entropy",
            "shannon_diversity_index",
            "joint_entropy",
            "conditional_entropy",
            "mutual_information",
        ]:
            self.assertGreaterEqual(getattr(ls, method)(), 0)
        self.assertTrue(0 < ls.relative_mutual_information(), 1)
        self.assertTrue(0 < ls.contagion() <= 100)

        # TODO: assert 0 < ls.interspersion_juxtaposition_index() <= 100

    def test_transonic(self):
        ls_arr = np.load(path.join(tests_data_dir, "ls250_06.npy"), allow_pickle=True)
        ls = pls.Landscape(ls_arr, res=(250, 250))
        adjacency_df = ls._adjacency_df
        self.assertIsInstance(adjacency_df, pd.DataFrame)

    def test_plot_landscape(self):
        # first test for a landscape without affine transform (instantiated from an
        # ndarray and without providing a non-None `transform` argument)
        ax = self.ls.plot_landscape()
        # returned axis must be instances of matplotlib axes
        self.assertIsInstance(ax, plt.Axes)

        # now do the same test for a landscape with affine transform (e.g., instantiated
        # from a raster file)
        ls = pls.Landscape(self.landscape_fp)
        ax = ls.plot_landscape()
        self.assertIsInstance(ax, plt.Axes)
        # and further test that the plot bounds correspond to the transform's offsets
        self.assertAlmostEqual(ax.get_xlim()[0], ls.transform.xoff)
        self.assertAlmostEqual(ax.get_ylim()[1], ls.transform.yoff)

        # test legend arguments
        self.assertIsNone(ls.plot_landscape(legend=False).get_legend())
        self.assertIsNotNone(ls.plot_landscape(legend=True))
        self.assertIsNotNone(
            ls.plot_landscape(legend=True, legend_kwargs=dict(loc="center right"))
        )


class TestMultiLandscape(unittest.TestCase):
    def setUp(self):
        from pylandstats import multilandscape

        self.landscapes = [
            pls.Landscape(
                np.load(path.join(tests_data_dir, "ls100_06.npy"), allow_pickle=True),
                res=(100, 100),
            ),
            pls.Landscape(
                np.load(path.join(tests_data_dir, "ls250_06.npy"), allow_pickle=True),
                res=(250, 250),
            ),
        ]
        self.landscape_fps = [
            path.join(tests_data_dir, "ls100_06.tif"),
            path.join(tests_data_dir, "ls250_06.tif"),
        ]
        self.attribute_name = "resolution"
        self.attribute_values = [100, 250]
        self.inexistent_class_val = 999

        # use this class just for testing purposes
        class InstantiableMultiLandscape(multilandscape.MultiLandscape):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        self.InstantiableMultiLandscape = InstantiableMultiLandscape

    def test_multilandscape_init(self):
        from pylandstats import multilandscape

        # test that we cannot instantiate an abstract class
        self.assertRaises(TypeError, multilandscape.MultiLandscape)

        # test that if we init a MultiLandscape from filepaths, Landscape instances are
        # automatically built
        ml = self.InstantiableMultiLandscape(
            self.landscape_fps, self.attribute_name, self.attribute_values
        )
        for landscape in ml.landscape_ser:
            self.assertIsInstance(landscape, pls.Landscape)
        # test that we can pass keyword arguments to the `Landscape` instantiation when
        # providing filepaths
        landscape_kwargs = {"neighborhood_rule": "4"}
        ml = self.InstantiableMultiLandscape(
            self.landscape_fps,
            self.attribute_name,
            self.attribute_values,
            **landscape_kwargs,
        )
        for landscape in ml.landscape_ser:
            self.assertEqual(
                landscape.neighborhood_rule, landscape_kwargs["neighborhood_rule"]
            )
        # test that if we instantiate providing `Landscape` instances, the provided
        # keyword arguments are ignored
        ml = self.InstantiableMultiLandscape(
            self.landscapes,
            self.attribute_name,
            self.attribute_values,
            **landscape_kwargs,
        )
        for landscape in ml.landscape_ser:
            self.assertNotEqual(
                landscape.neighborhood_rule, landscape_kwargs["neighborhood_rule"]
            )

        # from this point on, always instantiate from filepaths

        # test that constructing a MultiLandscape where the list of values of the
        # identifying attributes `attribute_values` (in this example, the list of
        # resolutions `[100, 250]`) mismatches the length of the list of landscapes
        # raises a ValueError
        self.assertRaises(
            ValueError,
            self.InstantiableMultiLandscape,
            self.landscape_fps,
            self.attribute_name,
            [250],
        )

    def test_multilandscape_dataframes(self):
        ml = self.InstantiableMultiLandscape(
            self.landscape_fps, self.attribute_name, self.attribute_values
        )
        # test that the data frames that result from `compute_class_metrics_df` and
        # `compute_landscape_metrics_df` are well constructed
        class_metrics_df = ml.compute_class_metrics_df()
        attribute_values = ml.landscape_ser.index
        self.assertTrue(np.all(class_metrics_df.columns == pls.Landscape.CLASS_METRICS))
        self.assertTrue(
            np.all(
                class_metrics_df.index
                == pd.MultiIndex.from_product([ml.present_classes, attribute_values])
            )
        )
        landscape_metrics_df = ml.compute_landscape_metrics_df()
        self.assertTrue(
            np.all(landscape_metrics_df.columns == pls.Landscape.LANDSCAPE_METRICS)
        )
        self.assertTrue(np.all(landscape_metrics_df.index == attribute_values))

        # now test the same but with an analysis that only considers a subset of metrics
        # and a subset of classes
        metrics = ["total_area", "edge_density", "proportion_of_landscape"]
        classes = ml.landscape_ser.iloc[0].classes[:2]

        class_metrics_df = ml.compute_class_metrics_df(metrics=metrics, classes=classes)
        attribute_values = ml.landscape_ser.index
        self.assertTrue(np.all(class_metrics_df.columns == metrics))
        self.assertTrue(
            np.all(
                class_metrics_df.index
                == pd.MultiIndex.from_product([classes, attribute_values])
            )
        )
        # 'proportion_of_landscape' cannot be computed at the landscape level
        # (TODO: test for that elsewhere)
        landscape_metrics = metrics[:2]
        landscape_metrics_df = ml.compute_landscape_metrics_df(
            metrics=landscape_metrics
        )
        self.assertTrue(np.all(landscape_metrics_df.columns == landscape_metrics))
        self.assertTrue(np.all(landscape_metrics_df.index == attribute_values))

    def test_multilandscape_class_metrics_fillna(self):
        ml = self.InstantiableMultiLandscape(
            self.landscape_fps, self.attribute_name, self.attribute_values
        )
        fillna_df = ml.compute_class_metrics_df()
        nofillna_df = ml.compute_class_metrics_df(fillna=False)

        for metric in fillna_df.columns:
            if metric in self.InstantiableMultiLandscape.METRIC_FILLNA_DICT:
                # this metric should be zero in the first data frame and `NaN` in the
                # second one
                self.assertEqual(
                    np.count_nonzero(fillna_df[metric][nofillna_df[metric].isna()]),
                    0,
                )
            else:
                # the columns for this metric should be identical in both data frames
                self.assertTrue(fillna_df[metric].equals(nofillna_df[metric]))

    def test_multilandscape_metrics_kwargs(self):
        # Instantiate two multilandscape analyses, one with FRAGSTATS' defaults and the
        # other with keyword arguments specifying the total area in meters and including
        # the boundary in the computation of the total edge.
        ml = self.InstantiableMultiLandscape(
            self.landscape_fps, self.attribute_name, self.attribute_values
        )
        # entropy metrics that accept a `base` keyword argument (all entropy metrics
        # except for Shannon's diversity index and contagion)
        entropy_metrics = [
            "entropy",
            "joint_entropy",
            "conditional_entropy",
            "mutual_information",
        ]
        metrics_kwargs = {
            "total_area": {"hectares": False},
            "perimeter_area_ratio_mn": {"hectares": True},
            "total_edge": {"count_boundary": True},
            **{entropy_metric: {"base": 10} for entropy_metric in entropy_metrics},
        }

        class_metrics_df = ml.compute_class_metrics_df()
        class_metrics_kwargs_df = ml.compute_class_metrics_df(
            metrics_kwargs=metrics_kwargs
        )
        landscape_metrics_df = ml.compute_landscape_metrics_df()
        landscape_metrics_kwargs_df = ml.compute_landscape_metrics_df(
            metrics_kwargs=metrics_kwargs
        )
        for attribute_value in ml.landscape_ser.index:
            # For all attribute values and all classes, metric values in hectares should
            # be less than in meters, and excluding boundaries should be less or equal
            # than including them
            landscape_metrics = landscape_metrics_df.loc[attribute_value]
            landscape_metrics_kwargs = landscape_metrics_kwargs_df.loc[attribute_value]
            self.assertLess(
                landscape_metrics["total_area"],
                landscape_metrics_kwargs["total_area"],
            )
            self.assertLessEqual(
                landscape_metrics["total_edge"],
                landscape_metrics_kwargs["total_edge"],
            )

            for class_val in ml.present_classes:
                class_metrics = class_metrics_df.loc[class_val, attribute_value]
                class_metrics_kwargs = class_metrics_kwargs_df.loc[
                    class_val, attribute_value
                ]

                # It could be that for some attribute values, some classes are not
                # present within the respective Landscape. If so, some metrics (e.g.,
                # 'perimeter_area_ratio_mn') will be `nan`, both for the analysis with
                # and without keyword arguments.
                # For area and edge metrics, PyLandStats interprets such `nan` values as
                # 0, which is why we need to include the equality in the comparison
                if class_metrics.isnull().all():
                    self.assertTrue(class_metrics_kwargs.isnull().all())
                else:
                    self.assertLessEqual(
                        class_metrics["total_area"],
                        class_metrics_kwargs["total_area"],
                    )
                    # we need quite some tolerance because pixel resolutions in raster
                    # files might be weird float values, e.g., 99.13213 instead of 100
                    # (meters)
                    self.assertLessEqual(
                        class_metrics["total_edge"],
                        1.01 * class_metrics_kwargs["total_edge"],
                    )

            # For all attribute values, entropy metric values computed with the default
            # base=2 should be greater than those computed with a custom base=10
            for entropy_metric in entropy_metrics:
                self.assertGreater(
                    landscape_metrics[entropy_metric],
                    landscape_metrics_kwargs[entropy_metric],
                )

    def test_multilandscape_plot_metrics(self):
        ml = self.InstantiableMultiLandscape(
            self.landscape_fps, self.attribute_name, self.attribute_values
        )

        existent_class_val = ml.present_classes[0]
        # TODO: test legend and figsize

        # test that there is only one line when plotting a single metric at the
        # landscape level
        ax = ml.plot_metric("patch_density", class_val=None)
        self.assertEqual(len(ax.lines), 1)
        # test that there are two lines if we add the plot of a single metric (e.g., at
        # the level of an existent class) to the previous axis
        ax = ml.plot_metric("patch_density", class_val=existent_class_val, ax=ax)
        self.assertEqual(len(ax.lines), 2)
        # test that the x data of any line corresponds to the attribute values
        for line in ax.lines:
            self.assertTrue(np.all(line.get_xdata() == ml.landscape_ser.index))

        # test metric label arguments/settings
        # when passing default arguments, the axis ylabel must be the one set within the
        # settings module
        self.assertEqual(
            ml.plot_metric("edge_density").get_ylabel(),
            pls.settings.metric_label_dict["edge_density"],
        )
        # when passing `metric_legend=False`, the axis ylabel must be an empty string
        self.assertEqual(
            ml.plot_metric("edge_density", metric_legend=False).get_ylabel(),
            "",
        )
        # when passing an non-empty string as `metric_label`, the axis ylabel must be
        # such string
        self.assertEqual(
            ml.plot_metric("edge_density", metric_label="foo").get_ylabel(),
            "foo",
        )

        # test that asking to plot an inexistent metric raises a `ValueError` try that
        # raised ValueErrors have different error messages depending on the context
        with self.assertRaises(ValueError) as cm:
            ml.plot_metric("foo")
        self.assertIn("is not among", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            ml.plot_metric("proportion_of_landscape")
        self.assertIn("cannot be computed", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            ml.plot_metric("contagion", class_val=1)
        self.assertIn("cannot be computed", str(cm.exception))

    def test_plot_landscapes(self):
        ml = self.InstantiableMultiLandscape(
            self.landscape_fps, self.attribute_name, self.attribute_values
        )

        fig = ml.plot_landscapes()

        # there must be one column for each landscape
        self.assertEqual(len(fig.axes), len(ml))

        # returned axes must be instances of matplotlib axes
        for ax in fig.axes:
            self.assertIsInstance(ax, plt.Axes)

        # test that by default, the dimensions of the resulting will come from
        # matplotlib's settings
        rc_figwidth, rc_figheight = plt.rcParams["figure.figsize"]
        figwidth, figheight = fig.get_size_inches()
        # we have `len(ml)` axis, therefore the actual `figwidth` must be
        # `len(ml) * rc_figwidth`
        self.assertAlmostEqual(figwidth, len(ml) * rc_figwidth)
        self.assertAlmostEqual(figheight, rc_figheight)
        # if instead, we customize the figure size, the dimensions of the resulting
        # figure must be the customized ones
        custom_figsize = (10, 10)
        fig = ml.plot_landscapes(subplots_kwargs={"figsize": custom_figsize})
        figwidth, figheight = fig.get_size_inches()
        self.assertAlmostEqual(custom_figsize[0], figwidth)
        self.assertAlmostEqual(custom_figsize[1], figheight)


class TestSpatioTemporalAnalysis(unittest.TestCase):
    def setUp(self):
        # we will only test reading from filepaths because the consistency between
        # passing `Landscape` objects or filepaths is already tested in
        # `TestMultiLandscape`
        self.landscape_fps = [
            path.join(tests_data_dir, "ls250_06.tif"),
            path.join(tests_data_dir, "ls250_12.tif"),
        ]
        self.dates = [2006, 2012]
        self.inexistent_class_val = 999

    def test_spatiotemporalanalysis_init(self):
        # test that the landscape index name is dates, and that if the `dates` argument
        # is not provided when instantiating a `SpatioTemporalAnalysis`, the dates
        # attribute is properly and automatically generated
        sta = pls.SpatioTemporalAnalysis(self.landscape_fps)
        self.assertEqual(sta.landscape_ser.index.name, "dates")
        self.assertEqual(len(sta), len(sta.landscape_ser))

        # test the `neighborhood_rule` argument
        neighborhood_rule = "4"
        other_neighborhood_rule = "8"
        # test that if provided and the elements of `landscapes` are filepaths, the
        # value is passed to each landscape
        for ls in pls.SpatioTemporalAnalysis(
            self.landscape_fps,
            dates=self.dates,
            neighborhood_rule=neighborhood_rule,
        ).landscape_ser:
            self.assertEqual(ls.neighborhood_rule, neighborhood_rule)
        # test that if provided and the elements of `landscapes` are `Landscape`
        # instances, the value is ignored
        for ls in pls.SpatioTemporalAnalysis(
            [
                pls.Landscape(landscape_fp, neighborhood_rule=other_neighborhood_rule)
                for landscape_fp in self.landscape_fps
            ],
            dates=self.dates,
            neighborhood_rule=neighborhood_rule,
        ).landscape_ser:
            self.assertEqual(ls.neighborhood_rule, other_neighborhood_rule)
        # test that if not provided and the elements of `landscapes` are filepaths, the
        # default value is taken
        for ls in pls.SpatioTemporalAnalysis(
            self.landscape_fps, dates=self.dates
        ).landscape_ser:
            self.assertEqual(
                ls.neighborhood_rule, pls.settings.DEFAULT_NEIGHBORHOOD_RULE
            )

    def test_spatiotemporalanalysis_dataframes(self):
        # test with the default constructor
        sta = pls.SpatioTemporalAnalysis(self.landscape_fps)

        # test that the data frames that result from `compute_class_metrics_df` and
        # `compute_landscape_metrics_df` are well constructed
        class_metrics_df = sta.compute_class_metrics_df()
        self.assertTrue(
            np.all(
                class_metrics_df.index
                == pd.MultiIndex.from_product(
                    [sta.present_classes, sta.landscape_ser.index]
                )
            )
        )
        landscape_metrics_df = sta.compute_landscape_metrics_df()
        self.assertTrue(np.all(landscape_metrics_df.index == sta.landscape_ser.index))

        # now test the same but with an analysis that only considers a subset of metrics
        # and a subset of classes
        metrics = ["total_area", "edge_density", "proportion_of_landscape"]
        classes = sta.present_classes[:2]

        class_metrics_df = sta.compute_class_metrics_df(
            metrics=metrics, classes=classes
        )
        self.assertTrue(
            np.all(
                class_metrics_df.index
                == pd.MultiIndex.from_product([classes, sta.landscape_ser.index])
            )
        )
        # 'proportion_of_landscape' cannot be computed at the landscape level (TODO:
        # test for that elsewhere)
        landscape_metrics = metrics[:2]
        landscape_metrics_df = sta.compute_landscape_metrics_df(
            metrics=landscape_metrics
        )
        self.assertTrue(np.all(landscape_metrics_df.index == sta.landscape_ser.index))

    def test_spatiotemporalanalysis_plot_metrics(self):
        sta = pls.SpatioTemporalAnalysis(self.landscape_fps, dates=self.dates)

        # test for `None` (landscape-level) and an existing class (class-level)
        for class_val in [None, sta.present_classes[0]]:
            # test that the x data of the line corresponds to the dates
            self.assertTrue(
                np.all(
                    sta.plot_metric("patch_density", class_val=class_val)
                    .lines[0]
                    .get_xdata()
                    == self.dates
                )
            )


class TestZonaAlnalysis(unittest.TestCase):
    def setUp(self):
        self.landscape_fp = path.join(tests_data_dir, "ls250_06.tif")
        with rio.open(self.landscape_fp) as src:
            self.landscape_transform = src.transform
            self.landscape_crs = src.crs
        self.zones_fp = path.join(tests_data_dir, "gmb-lausanne.gpkg")
        self.zone_gdf = gpd.read_file(self.zones_fp)
        self.label_arr = np.load(
            path.join(tests_data_dir, "label_arr.npy"), allow_pickle=True
        )
        # for buffer analysis
        self.geom = geometry.Point(6.6327025, 46.5218269)
        self.buffer_dists = [10000, 15000, 20000]

        self.tmp_dir = path.join(tests_dir, "tmp")
        os.mkdir(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_zonal_init(self):
        # test passing GeoSeries, list-like of geometries, GeoDataFrame and geopandas
        # files as `zones`
        # first test the GeoSeries and list-like of shapely geometries, which work like
        # the others except that we cannot set a column as the zone index
        zone_gser = self.zone_gdf["geometry"].copy()
        za = pls.ZonalAnalysis(self.landscape_fp, zone_gser)
        self.assertLessEqual(len(za), len(self.zone_gdf))
        self.assertTrue(np.all(za.zone_gser.index == zone_gser.index))
        # also test that landscape index name is properly set when using geoseries as
        # `zones` first test for a geoseries with name and unnamed index
        zone_gser.name = "bar"
        za = pls.ZonalAnalysis(self.landscape_fp, zone_gser)
        self.assertEqual(za.landscape_ser.index.name, zone_gser.name)
        # now test that for a named geoseries with a named index, the geoseries index
        # name takes precedence
        zone_gser.index.name = "name"
        za = pls.ZonalAnalysis(self.landscape_fp, zone_gser)
        self.assertEqual(za.landscape_ser.index.name, zone_gser.index.name)
        # test that for an named geoseries with an unnamed index, the geoseries name is
        # taken
        zone_gser.index.name = None
        za = pls.ZonalAnalysis(self.landscape_fp, zone_gser)
        self.assertEqual(za.landscape_ser.index.name, zone_gser.name)
        # test overriding the zone index
        zone_index = zone_gser.index + 1
        za = pls.ZonalAnalysis(self.landscape_fp, zone_gser, zone_index=zone_index)
        self.assertTrue(np.all(za.zone_gser.index == zone_index))
        # test overriding the zone index with a named index
        zone_index = zone_index.rename("foo")
        za = pls.ZonalAnalysis(self.landscape_fp, zone_gser, zone_index=zone_index)
        self.assertEqual(za.landscape_ser.index.name, zone_index.name)
        # test that for a list-like of shapely geometries, the CRS of the landscape is
        # taken
        zones = list(zone_gser)
        za = pls.ZonalAnalysis(self.landscape_fp, zones)
        self.assertEqual(za.zone_gser.crs, self.landscape_crs)

        # now test the GeoDataFrame and geopandas file
        zone_index_col = "GMDNAME"
        for zones in self.zones_fp, self.zone_gdf:
            # test init
            za = pls.ZonalAnalysis(self.landscape_fp, zones)
            self.assertLessEqual(len(za), len(self.zone_gdf))
            self.assertTrue(
                np.all(np.isin(za.landscape_ser.index, self.zone_gdf.index))
            )
            # test that we can set a column as the zone index
            za = pls.ZonalAnalysis(self.landscape_fp, zones, zone_index=zone_index_col)
            self.assertTrue(
                np.all(
                    np.isin(
                        za.zone_gser.index,
                        self.zone_gdf[zone_index_col],
                    )
                )
            )

        # test that we can still pass a raster labelled array
        # test that the landscape index name and values are consistent with the provided
        # `label_arr`
        za = pls.ZonalAnalysis(self.landscape_fp, self.label_arr)
        self.assertEqual(za.landscape_ser.index.name, "zone")
        # the number of zones must be equal to the number of unique labels (excluding
        # the nodata value)
        zone_nodata = 0
        zones = list(set(np.unique(self.label_arr)).difference({zone_nodata}))
        self.assertEqual(len(za), len(zones))
        # test that Landscape instances are automatically built
        for landscape in za.landscape_ser:
            self.assertIsInstance(landscape, pls.Landscape)
        # test that the zone index corresponds to the zone labels
        self.assertTrue(np.all(np.isin(zones, za.zone_gser.index)))
        # test that we can override the zone index
        zone_index = pd.Series(range(1, len(zones) + 1), name="foo")
        za = pls.ZonalAnalysis(self.landscape_fp, self.label_arr, zone_index=zone_index)
        self.assertEqual(za.landscape_ser.index.name, "foo")
        self.assertTrue(np.all(za.zone_gser.index == zone_index))

        # test that we can override the nodata value
        zone_nodata = 255
        label_arr = np.where(self.label_arr != 0, self.label_arr, zone_nodata)
        # in this case, if we do not specify the nodata value, we will have another zone
        # (since by default, 0 are considered nodata)
        self.assertEqual(
            len(pls.ZonalAnalysis(self.landscape_fp, label_arr)), len(zones) + 1
        )
        self.assertEqual(
            len(
                pls.ZonalAnalysis(self.landscape_fp, label_arr, zone_nodata=zone_nodata)
            ),
            len(zones),
        )

        # test the `neighborhood_rule` argument
        neighborhood_rule = "4"
        # test that if provided, the value is passed to each landscape
        for ls in pls.ZonalAnalysis(
            self.landscape_fp,
            zone_gser,
            neighborhood_rule=neighborhood_rule,
        ).landscape_ser:
            self.assertEqual(ls.neighborhood_rule, neighborhood_rule)
        # test that if not provided, the default value is taken
        for ls in pls.ZonalAnalysis(self.landscape_fp, self.label_arr).landscape_ser:
            self.assertEqual(
                ls.neighborhood_rule, pls.settings.DEFAULT_NEIGHBORHOOD_RULE
            )

    def test_zonal_plot_metrics(self):
        za = pls.ZonalAnalysis(self.landscape_fp, self.zone_gdf)

        # test for `None` (landscape-level) and an existing class (class-level)
        for class_val in [None, za.present_classes[0]]:
            # test that the x data of the line corresponds to the attribute values,
            # i.e., the zone ids
            self.assertTrue(
                np.all(
                    za.plot_metric("patch_density", class_val=class_val)
                    .lines[0]
                    .get_xdata()
                    == za.zone_gser.index
                )
            )

    def test_compute_zonal_statistics_gdf(self):
        za = pls.ZonalAnalysis(self.landscape_fp, self.zone_gdf)

        # test that the gdf has the proper shape (number of zones, number of metrics +
        # geometry column)
        for class_val in [None, za.present_classes[0]]:
            metrics = ["patch_density"]
            zs_gdf = za.compute_zonal_statistics_gdf(
                metrics=metrics, class_val=class_val
            )
            self.assertEqual(zs_gdf.shape, (len(self.zone_gdf), len(metrics) + 1))
            # test that the crs is set correctly
            self.assertEqual(zs_gdf.crs, self.zone_gdf.crs)
            # test that the geometry column is not None
            self.assertFalse(zs_gdf.geometry.isna().any())

            # test that the zonal statistics when excluding boundaries should be less or
            # equal than including them
            metric = "total_edge"
            metric_kwargs = {"count_boundary": True}
            self.assertLessEqual(
                za.compute_zonal_statistics_gdf(metrics=[metric], class_val=class_val)[
                    metric
                ].sum(),
                za.compute_zonal_statistics_gdf(
                    metrics=[metric],
                    class_val=class_val,
                    metrics_kwargs={metric: metric_kwargs},
                )[metric].sum(),
            )

    def test_buffer_init(self):
        naive_gser = gpd.GeoSeries([self.geom])
        gser = gpd.GeoSeries([self.geom], crs=geom_crs)
        naive_gdf = gpd.GeoDataFrame(geometry=naive_gser)
        gdf = gpd.GeoDataFrame(geometry=gser)

        # test that we cannot init from a shapely geometry without providing its crs
        self.assertRaises(
            ValueError,
            pls.BufferAnalysis,
            self.landscape_fp,
            self.geom,
            self.buffer_dists,
        )

        # test that we can properly instantiate it from:
        # 1. a landscape filepath, shapely geometry, and its crs
        # 2. a landscape filepath, naive geopandas GeoSeries/GeoDataFrame (with no crs
        #    set) and its crs
        # 3. a landscape filepath, geopandas GeoSeries/GeoDataFrame with crs set
        # 4. a landscape filepath, geopandas GeoSeries/GeoDataFrame with crs set and a
        #    crs (which will override the crs of the GeoSeries)
        for ba in [
            pls.BufferAnalysis(
                self.landscape_fp,
                self.geom,
                self.buffer_dists,
                base_geom_crs=geom_crs,
            ),
            pls.BufferAnalysis(
                self.landscape_fp,
                naive_gser,
                self.buffer_dists,
                base_geom_crs=geom_crs,
            ),
            pls.BufferAnalysis(self.landscape_fp, gser, self.buffer_dists),
            pls.BufferAnalysis(
                self.landscape_fp,
                gser,
                self.buffer_dists,
                base_geom_crs=geom_crs,
            ),
            pls.BufferAnalysis(
                self.landscape_fp,
                naive_gdf,
                self.buffer_dists,
                base_geom_crs=geom_crs,
            ),
            pls.BufferAnalysis(self.landscape_fp, gdf, self.buffer_dists),
            pls.BufferAnalysis(
                self.landscape_fp,
                gdf,
                self.buffer_dists,
                base_geom_crs=geom_crs,
            ),
        ]:
            self.assertEqual(ba.landscape_ser.index.name, "buffer_dist")
            self.assertEqual(len(ba), len(ba.zone_gser))

        # test that we cannot instantiate a `BufferAnalysis` with `buffer_rings=True` if
        # `base_mask` is a polygon or a GeoSeries containing a polygon
        polygon = self.geom.buffer(1000)  # this will return a polygon instance
        self.assertRaises(
            ValueError,
            pls.BufferAnalysis,
            self.landscape_fp,
            polygon,
            self.buffer_dists,
            buffer_rings=True,
            base_geom_crs=geom_crs,
        )
        polygon_gser = gpd.GeoSeries([polygon], crs=geom_crs)
        self.assertRaises(
            ValueError,
            pls.BufferAnalysis,
            self.landscape_fp,
            polygon_gser,
            self.buffer_dists,
            buffer_rings=True,
        )

        # test that otherwise, buffer rings are properly instantiated
        ba_rings = pls.BufferAnalysis(
            self.landscape_fp, gser, self.buffer_dists, buffer_rings=True
        )
        # the `buffer_dists` attribute must be a string of the form '{r}-{R}' where r
        # and R respectively represent the smaller and larger radius that compose each
        # ring
        for buffer_ring_str in ba_rings.zone_gser.index:
            self.assertIn("-", buffer_ring_str)

        # compare it with the default instance (with the argument `buffer_rings=False`)
        # that does not consider rings but cumulatively considers the inner areas in
        # each zone. The first zone will in fact be the same in both cases (the region
        # that goes from 0 to the first item of `self.buffer_dists`), but the successive
        # zones will be always larger in the default instance (since they will have the
        # surface of the corresponding ring plus the surface of the inner region that is
        # excluded when `buffer_rings=True`)
        ba = pls.BufferAnalysis(self.landscape_fp, gser, self.buffer_dists)
        for zone_gser, ring_zone_gser in zip(ba.zone_gser, ba_rings.zone_gser):
            self.assertGreaterEqual(zone_gser.area, ring_zone_gser.area)

    def test_grid_init(self):
        # test init by number of zone rows/cols
        num_zone_rows, num_zone_cols = 10, 10
        zga = pls.ZonalGridAnalysis(
            self.landscape_fp,
            num_zone_rows=num_zone_rows,
            num_zone_cols=num_zone_cols,
        )
        # there are at most as many zones as num_zone_rows * num_zone_cols, because
        # cells with no valid data are excluded from `zone_gser`
        self.assertLessEqual(len(zga.zone_gser), num_zone_rows * num_zone_cols)

        # test init by zone width/height
        zone_width, zone_height = 1000, 1000
        zga = pls.ZonalGridAnalysis(
            self.landscape_fp,
            zone_width=zone_width,
            zone_height=zone_height,
        )

        # test that init must provide one arg for each dimension (i.e,
        # `num_zone_cols`/`zone_width` and `num_zone_rows`/`zone_height`)
        for kws in [
            {},
            {
                "zone_height": zone_height,
                "num_zone_rows": num_zone_rows,
            },
            {
                "zone_width": zone_width,
                "num_zone_cols": num_zone_cols,
            },
        ]:
            with self.assertRaises(ValueError) as cm:
                zga = pls.ZonalGridAnalysis(self.landscape_fp, **kws)
            self.assertIn("must be provided", str(cm.exception))

        # test the `neighborhood_rule` argument
        neighborhood_rule = "4"
        # test that the value is passed to each landscape
        for ls in pls.ZonalGridAnalysis(
            self.landscape_fp,
            zone_width=zone_width,
            zone_height=zone_height,
            neighborhood_rule=neighborhood_rule,
        ).landscape_ser:
            self.assertEqual(ls.neighborhood_rule, neighborhood_rule)
        # test that if not provided, the default value is taken
        for ls in pls.ZonalGridAnalysis(
            self.landscape_fp,
            zone_width=zone_width,
            zone_height=zone_height,
        ).landscape_ser:
            self.assertEqual(
                ls.neighborhood_rule, pls.settings.DEFAULT_NEIGHBORHOOD_RULE
            )


class TestSpatioTemporalZonalAnalysis(unittest.TestCase):
    def setUp(self):
        self.landscape_fps = [
            path.join(tests_data_dir, "ls250_06.tif"),
            path.join(tests_data_dir, "ls250_12.tif"),
        ]
        self.dates = [2006, 2012]
        # zonal
        self.zone_gser = pls.ZonalAnalysis(
            self.landscape_fps[0],
            np.load(path.join(tests_data_dir, "label_arr.npy"), allow_pickle=True),
        ).zone_gser
        # buffer
        self.base_geom = gpd.GeoSeries(
            [geometry.Point(6.6327025, 46.5218269)], crs=geom_crs
        )
        self.buffer_dists = [10000, 15000, 20000]
        # grid
        self.num_zone_rows, self.num_zone_cols = 10, 10
        # self.zone_width, self.zone_height = 1000, 1000
        self.init_combinations = zip(
            [
                pls.SpatioTemporalZonalAnalysis,
                pls.SpatioTemporalBufferAnalysis,
                pls.SpatioTemporalZonalGridAnalysis,
            ],
            [[self.zone_gser], [self.base_geom, self.buffer_dists], []],
            [
                {},
                {},
                {
                    "num_zone_cols": self.num_zone_cols,
                    "num_zone_rows": self.num_zone_rows,
                },
            ],
            ["zone", "buffer_dist", "grid_cell"],
        )

    def test_init(self):
        # we will just test the base init, the rest of functionalities have already been
        # tested above (in `TestSpatioTemporalAnalysis` and `TestZonalAnalysis`)
        for _class, init_args, init_kwargs, attr_name in self.init_combinations:
            # test zones and dates
            stza = _class(
                self.landscape_fps, *init_args, dates=self.dates, **init_kwargs
            )
            # test that we have the same zone labels in `zone_gser` and `landscape_ser`
            self.assertTrue(
                (
                    stza.zone_gser.index
                    == stza.landscape_ser.index.get_level_values(
                        stza.zone_gser.index.name
                    ).unique()
                ).all()
            )
            # test the `zone_gser` index name
            self.assertEqual(stza.zone_gser.index.name, attr_name)
            # test that we have the same date labels in `dates` and `landscape_ser`
            self.assertTrue(
                (
                    self.dates
                    == stza.landscape_ser.index.get_level_values("date").unique()
                ).all()
            )

            # test the `neighborhood_rule` argument
            # test that if not provided, the default value is taken
            for ls in stza.landscape_ser:
                self.assertEqual(
                    ls.neighborhood_rule,
                    pls.settings.DEFAULT_NEIGHBORHOOD_RULE,
                )
            neighborhood_rule = "4"
            # test that if provided, the value is passed to each landscape
            stza = _class(
                self.landscape_fps,
                *init_args,
                neighborhood_rule=neighborhood_rule,
                **init_kwargs,
            )
            for ls in stza.landscape_ser:
                self.assertEqual(ls.neighborhood_rule, neighborhood_rule)

    def test_dataframes(self):
        for _class, init_args, init_kwargs, _ in self.init_combinations:
            stza = _class(
                self.landscape_fps, *init_args, dates=self.dates, **init_kwargs
            )
            zone_index = stza.zone_gser.index
            dates = stza.landscape_ser.index.get_level_values("date").unique()

            # test that the data frames that result from `compute_class_metrics_df` and
            # `compute_landscape_metrics_df` are well constructed, i.e., they are a
            # subset of all class/zone/date combinations
            class_metrics_df = stza.compute_class_metrics_df()
            self.assertTrue(
                class_metrics_df.index.isin(
                    pd.MultiIndex.from_product(
                        [stza.present_classes, zone_index, dates]
                    )
                ).all()
            )
            landscape_metrics_df = stza.compute_landscape_metrics_df()
            self.assertTrue(
                landscape_metrics_df.index.isin(
                    pd.MultiIndex.from_product([zone_index, dates])
                ).all()
            )

            # now test the same but with an analysis that only considers a subset of
            # metrics and a subset of classes
            metrics = ["total_area", "edge_density", "proportion_of_landscape"]
            classes = stza.present_classes[:2]

            class_metrics_df = stza.compute_class_metrics_df(
                metrics=metrics, classes=classes
            )
            self.assertTrue(
                class_metrics_df.index.isin(
                    pd.MultiIndex.from_product([classes, zone_index, dates])
                ).all()
            )
            # 'proportion_of_landscape' cannot be computed at the landscape level (TODO:
            # test for that elsewhere)
            landscape_metrics = metrics[:2]
            landscape_metrics_df = stza.compute_landscape_metrics_df(
                metrics=landscape_metrics
            )
            self.assertTrue(
                landscape_metrics_df.index.isin(
                    pd.MultiIndex.from_product([zone_index, dates])
                ).all()
            )

    def test_compute_zonal_statistics_gdf(self):
        for _class, init_args, init_kwargs, _ in self.init_combinations:
            stza = _class(
                self.landscape_fps, *init_args, dates=self.dates, **init_kwargs
            )
            # test that the gdf has the proper shape (number of zones, number of metrics
            # + geometry column)
            for class_val in [None, stza.present_classes[0]]:
                metrics = ["patch_density"]
                zs_gdf = stza.compute_zonal_statistics_gdf(
                    metrics=metrics, class_val=class_val
                )
                self.assertLessEqual(
                    zs_gdf.shape[0],
                    len(stza.zone_gser),
                )
                self.assertEqual(zs_gdf.shape[1], len(metrics) * len(self.dates) + 1)
                # test that the crs is set correctly
                self.assertEqual(zs_gdf.crs, self.zone_gser.crs)
                # test that the geometry column is not None
                self.assertFalse(zs_gdf.geometry.isna().any())

                # test that the zonal statistics when excluding boundaries should be
                # less or equal than including them
                metric = "total_edge"
                metric_kwargs = {"count_boundary": True}
                self.assertLessEqual(
                    stza.compute_zonal_statistics_gdf(
                        metrics=[metric], class_val=class_val
                    )[metric]
                    .sum()
                    .sum(),
                    stza.compute_zonal_statistics_gdf(
                        metrics=[metric],
                        class_val=class_val,
                        metrics_kwargs={metric: metric_kwargs},
                    )[metric]
                    .sum()
                    .sum(),
                )

    def test_plot_metric(self):
        for _class, init_args, init_kwargs, _ in self.init_combinations:
            stza = _class(
                self.landscape_fps, *init_args, dates=self.dates, **init_kwargs
            )
            # test for `None` (landscape-level) and an existing class (class-level)
            metric = "patch_density"
            for class_val in [None, stza.present_classes[0]]:
                ax = stza.plot_metric(metric, class_val=class_val)
                # test that there is a line for each zone with that metric
                if class_val is None:
                    metric_ser = stza.compute_landscape_metrics_df(metrics=[metric])[
                        metric
                    ]
                else:
                    metric_ser = stza.compute_class_metrics_df(
                        metrics=[metric], classes=[class_val]
                    )[metric]
                num_lines = len(
                    metric_ser.index.get_level_values(
                        stza.zone_gser.index.name
                    ).unique()
                )
                self.assertEqual(
                    len(ax.lines),
                    num_lines,
                )
                # test that there is a legend label for each zone
                _, labels = ax.get_legend_handles_labels()
                self.assertEqual(len(labels), num_lines)

    def test_plot_landscapes(self):
        for _class, init_args, init_kwargs, _ in self.init_combinations:
            stza = _class(
                self.landscape_fps, *init_args, dates=self.dates, **init_kwargs
            )
            dates = stza.landscape_ser.index.get_level_values("date").unique()

            fig = stza.plot_landscapes()

            # there must be one column for each buffer distance and one row for each
            # date
            self.assertEqual(len(fig.axes), len(stza.zone_gser) * len(dates))

            # returned axes must be instances of matplotlib axes
            for ax in fig.axes:
                self.assertIsInstance(ax, plt.Axes)

            # test that by default, the dimensions of the resulting will come from
            # matplotlib's settings
            rc_figwidth, rc_figheight = plt.rcParams["figure.figsize"]
            figwidth, figheight = fig.get_size_inches()
            # the actual `figwidth` must be `len(stba.buffer_dists) * rc_figwidth` and
            # `figheight` must be `len(stba.dates) * rc_figheight`
            self.assertAlmostEqual(figwidth, len(stza.zone_gser) * rc_figwidth)
            self.assertAlmostEqual(figheight, len(dates) * rc_figheight)
            # if instead, we customize the figure size, the dimensions of the resulting
            # figure must be the customized ones
            custom_figsize = (10, 10)
            fig = stza.plot_landscapes(subplots_kwargs={"figsize": custom_figsize})
            figwidth, figheight = fig.get_size_inches()
            self.assertAlmostEqual(custom_figsize[0], figwidth)
            self.assertAlmostEqual(custom_figsize[1], figheight)

            # first row has the date as title
            for date, ax in zip(dates, fig.axes):
                self.assertEqual(str(date), ax.get_title())
            # first column has the buffer distance as `ylabel`
            for zone, i in zip(
                stza.zone_gser.index, range(0, len(fig.axes), len(dates))
            ):
                self.assertEqual(str(zone), fig.axes[i].get_ylabel())
