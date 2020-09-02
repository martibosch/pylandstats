from __future__ import division

import platform
import warnings
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio import plot
from scipy import ndimage, spatial, stats
from transonic import boost, set_backend_for_this_module

if platform.system() == "Windows":
    backend = "numba"
else:
    backend = "pythran"

set_backend_for_this_module(backend)

__all__ = ["Landscape"]

# sometimes pixel resolutions in GeoTIFF files are floats therefore
# compparisons (e.g., `cell_width == cell_height`) should allow for some
# tolerance (i.e., using `np.isclose`)
CELLLENGTH_RTOL = 0.001
KERNEL_MOORE = ndimage.generate_binary_structure(2, 2)


@boost
def compute_adjacency_arr(padded_arr: "uint32[:,:]", num_classes: "int"):
    # flat-array approach to pixel adjacency from link below:
    # https://ilovesymposia.com/2016/12/20/numba-in-the-real-world/
    # the first axis of `adjacency_arr` is of fixed size of 2 and serves to
    # distinguish between vertical and horizontal adjacencies (we could also
    # use a tuple of two 2-D arrays)
    # adjacency_arr = np.zeros((2, num_classes + 1, num_classes + 1),
    #                          dtype=np.uint32)
    num_cols_adjacency = num_classes + 1
    horizontal_adjacency_arr = np.zeros(
        num_cols_adjacency * num_cols_adjacency, dtype=np.uint32)
    vertical_adjacency_arr = np.zeros(num_cols_adjacency * num_cols_adjacency,
                                      dtype=np.uint32)
    num_cols_pixel = padded_arr.shape[1]
    flat_arr = padded_arr.ravel()
    # steps_to_neighbours as argument to distinguish between vertical/
    # horizontal adjacencies
    # steps_to_neighbours = [1, num_cols, -1, -num_cols]
    horizontal_neighbours = [1, -1]
    vertical_neighbours = [num_cols_pixel, -num_cols_pixel]
    start = num_cols_pixel + 1
    end = len(flat_arr) - start
    for i in range(start, end):
        class_i = flat_arr[i]
        # class_left = flat_arr[i - 1]
        # class_right = flat_arr[i + 1]
        # class_above = flat_arr[i - num_cols]
        # class_below = flat_arr[i + num_cols]
        # adjacency_arr[0, class_i, class_left] += 1
        # adjacency_arr[0, class_i, class_right] += 1
        # adjacency_arr[1, class_i, class_above] += 1
        # adjacency_arr[1, class_i, class_below] += 1
        for neighbour in horizontal_neighbours:
            # adjacency_arr[0, class_i, flat_arr[i + neighbour]] += 1
            horizontal_adjacency_arr[class_i + num_cols_adjacency *
                                     flat_arr[i + neighbour]] += 1
        for neighbour in vertical_neighbours:
            # adjacency_arr[1, class_i, flat_arr[i + neighbour]] += 1
            vertical_adjacency_arr[class_i + num_cols_adjacency *
                                   flat_arr[i + neighbour]] += 1

    return np.stack(
        (horizontal_adjacency_arr, vertical_adjacency_arr)).reshape(
            (2, num_cols_adjacency, num_cols_adjacency))


class Landscape:
    """Class representing a raster landscape upon which the landscape metrics
    will be computed
    """

    def __init__(self, landscape, res=None, nodata=None, transform=None,
                 **kwargs):
        """
        Parameters
        ----------
        landscape : np.ndarray or str, file object or pathlib.Path object
            A landscape array with pixel values corresponding to a set of land
            use/land cover classes, or a filename or URL, a file object opened
            in binary ('rb') mode, or a Path object. If not a `np.ndarray`,
            `landscape` will be passed to `rasterio.open`
        res : tuple, optional
            The (x, y) resolution of the dataset. Required if `landscape` is a
            `np.ndarray`
        nodata : int, optional
            Value to be assigned to pixels with no data. It will be set to 0
            if `landscape` is a `np.ndarray`
        transform : affine.Affine, optional
            Transformation from pixel coordinates to coordinate reference
            system. If `landscape` is a path to a GeoTiff, this argument will
            be ignored and extracted from the raster's metadata instead
        **kwargs : optional
            Keyword arguments to be passed to `rasterio.open`. Ignored if
            `landscape` is an `np.ndarray`
        """
        if isinstance(landscape, np.ndarray):
            landscape_arr = np.copy(landscape)
            if res is None:
                raise ValueError(
                    "If `landscape` is a `np.ndarray`, `res` must be provided")
            if nodata is None:
                nodata = 0
        else:
            with rasterio.open(landscape, nodata=nodata, **kwargs) as src:
                landscape_arr = src.read(1)
                if res is None:
                    res = src.res
                if nodata is None:
                    nodata = src.nodata
                transform = src.transform

        self.landscape_arr = landscape_arr
        self.cell_width, self.cell_height = res
        self.cell_area = res[0] * res[1]
        self.nodata = nodata
        self.transform = transform
        # by default, numpy creates arrays of floats. Instead, land use/land
        # cover rasters are often of integer dtypes. Therefore, we will
        # explicitly set the dtype of the landscape classes to ensure
        # consistency
        classes = np.array(sorted(np.unique(landscape_arr)),
                           dtype=self.landscape_arr.dtype)
        classes = classes[classes != nodata]
        classes = classes[~np.isnan(classes)]
        self.classes = classes

    ###########################################################################
    # common utilities

    # constants

    PATCH_METRICS = [
        "area",
        "perimeter",
        "perimeter_area_ratio",
        "shape_index",
        "fractal_dimension",
        "euclidean_nearest_neighbor",
    ]  # 'contiguity_index', 'proximity'

    DISTR_METRICS = [
        patch_metric + "_" + suffix for patch_metric in PATCH_METRICS
        for suffix in ["mn", "am", "md", "ra", "sd", "cv"]
    ]

    CLASS_METRICS = [
        "total_area",
        "proportion_of_landscape",
        "number_of_patches",
        "patch_density",
        "largest_patch_index",
        "total_edge",
        "edge_density",
        "landscape_shape_index",
        "effective_mesh_size",
    ] + DISTR_METRICS

    LANDSCAPE_METRICS = ([
        "total_area",
        "number_of_patches",
        "patch_density",
        "largest_patch_index",
        "total_edge",
        "edge_density",
        "landscape_shape_index",
        "effective_mesh_size",
    ] + DISTR_METRICS + ["contagion", "shannon_diversity_index"])

    # compute methods

    def class_label(self, class_val):
        return ndimage.label(self.landscape_arr == class_val, KERNEL_MOORE)

    # compute methods to obtain a scalar from an array

    def compute_arr_perimeter(self, arr):
        return (np.sum(arr[1:, :] != arr[:-1, :]) * self.cell_width +
                np.sum(arr[:, 1:] != arr[:, :-1]) * self.cell_height)

    # compute methods to obtain patchwise scalars

    def compute_patch_areas(self, label_arr):
        # we could use `ndimage.find_objects`, but since we do not need to
        # preserve the feature shapes, `np.bincount` is much faster
        return np.bincount(label_arr.ravel())[1:] * self.cell_area

    def compute_patch_perimeters(self, label_arr):
        # NOTE: performance comparison of `patch_perimeters` as np.array of
        # fixed size with `patch_perimeters[i] = ...` within the loop is
        # slower and less Pythonic but can lead to better performances if
        # optimized via Cython/numba
        patch_perimeters = []
        # `ndimage.find_objects` only finds the (rectangular) bounds; there
        # might be parts of other patches within such bounds, so we need to
        # check which pixels correspond to the patch of interest. Since
        # `ndimage.label` labels patches with an enumeration starting by 1, we
        # can use Python's built-in `enumerate`
        # NOTE: feature-wise iteration could this be done with
        # `ndimage.labeled_comprehension(
        #     label_arr, label_arr, np.arange(1, num_patches + 1),
        #     _compute_arr_perimeter, np.float, default=None)`
        # ?
        # I suspect no, because each feature array is flattened, which does
        # not allow for the computation of the perimeter or other shape metrics
        for i, patch_slice in enumerate(ndimage.find_objects(label_arr),
                                        start=1):
            patch_arr = np.pad(
                label_arr[patch_slice] == i,
                pad_width=1,
                mode="constant",
                constant_values=False,
            )  # self.nodata

            patch_perimeters.append(self.compute_arr_perimeter(patch_arr))

        return patch_perimeters

    def compute_patch_euclidean_nearest_neighbor(self, label_arr):
        # label_arr, num_patches = self.class_label(class_val)

        if np.max(label_arr) < 2:  # num_patches < 2
            return np.array([np.nan])
        else:
            # we will first get only the edges of the patches, since the
            # shortest edge-to-edge distance between patches is certainly
            # going to be between pixels at their corresponding patch edge
            label_mask = label_arr != 0
            edges_mask = label_mask & ~ndimage.binary_erosion(
                label_mask, KERNEL_MOORE)
            edges_arr = label_arr * edges_mask

            # get coordinates with non-zero values
            # Note that `label_arr` will use zero values to indicate nodata
            # (even if our landscape raster uses a different nodata value,
            # i.e., `self.nodata`)
            I, J = np.nonzero(edges_arr)
            labels = label_arr[I, J]  # this gives all the non-zero labels
            coords = np.column_stack((I, J))

            # sort labels/coordinates by the feature value
            sorter = np.argsort(labels)
            labels = labels[sorter]
            coords = coords[sorter]

            # # begin CDIST
            # # get feature-vs-feature distance matrix
            # sq_dists = spatial.distance.cdist(coords, coords,
            #                                   'sqeuclidean')
            # start_idx = np.flatnonzero(np.r_[1, np.diff(labels)])
            # nonzero_vs_feat = np.minimum.reduceat(sq_dists, start_idx,
            #                                       axis=1)
            # feat_vs_feat = np.minimum.reduceat(
            #     nonzero_vs_feat, start_idx, axis=0)

            # # get min edge-to-edge distance to closest patch of the same
            # # class
            # feat_vs_feat[feat_vs_feat == 0] = np.nan
            # enn = np.sqrt(np.nanmin(feat_vs_feat, axis=1))
            # # end CDIST

            # begin KDTree
            unique_labels = np.unique(labels)

            enn = np.empty(len(unique_labels))
            for unique_label in unique_labels:
                # we build a KDTree with all the coords that are not part of
                # the current feature
                tree = spatial.cKDTree(
                    coords[labels != unique_label],
                    balanced_tree=False,
                    compact_nodes=False,
                )
                # now, for each coord of the current feature, we query the
                # closest coord of the tree (which does not include points of
                # the current feature)
                mindist, minid = tree.query(coords[labels == unique_label])
                # note that `mindist` and `minid` will be 1D arrays, whose
                # lengths correspond to the number of pixels within the
                # current feature.
                # Each position of `mindist` and `mindid` matches the
                # corresponding pixel of the current feature to its closest
                # neighbor from the non-feature tree. Since we are only
                # interested in the closest distance, we will just get
                # `min(mindist)`. Note that because of the symmetry, we could
                # use `minid` to assign this same distance to the counterpart
                # of `unique_label`.
                # Nevertheless, the overheads of maintaining the required data
                # structure would most likely exceed any potential gains.
                # We use `unique_label - 1` to obtain the corresponding 0-based
                # index
                enn[unique_label - 1] = min(mindist)
            # end KDTree

            if np.isclose(self.cell_width, self.cell_height,
                          rtol=CELLLENGTH_RTOL):
                enn *= self.cell_width
            else:
                enn *= np.sqrt(self.cell_area)

            return enn

    # compute metrics from area and perimeter series

    def compute_shape_index(self, area_ser, perimeter_ser):
        # scalar version of this method
        # if self.cell_width != self.cell_height:
        #     # this is rare and not even supported in FRAGSTATS. We could
        #     # calculate the perimeter in terms of cell counts in a
        #     # dedicated function and then adjust for a square standard,
        #     # but I believe it is not worth the effort. So we will just
        #     # return the base formula without adjusting for the square
        #     # standard
        #     return .25 * perimeter / np.sqrt(area)
        # else:
        #     area_cells = area / self.cell_area
        #     # we could also divide by `self.cell_height`
        #     perimeter_cells = perimeter / self.cell_width
        #     n = np.floor(np.sqrt(area_cells))
        #     m = area_cells - n**2
        #     if m == 0:
        #         min_p = 4 * n
        #     elif n**2 < area_cells and area_cells <= n * (n + 1):
        #         min_p = 4 * n + 2
        #     else:  # elif area_cells > n * (n + 1):
        #         min_p = 4 * n + 4

        #     return perimeter_cells / min_p

        if np.isclose(self.cell_width, self.cell_height, rtol=CELLLENGTH_RTOL):
            area_cells_ser = area_ser / self.cell_area
            # we could also divide by `self.cell_height`
            perimeter_cells_ser = perimeter_ser / self.cell_width
            n = np.floor(np.sqrt(area_cells_ser))
            m = area_cells_ser - n**2
            min_p = np.ones(len(area_cells_ser))
            min_p = np.where(np.isclose(m, 0), 4 * n, min_p)
            min_p = np.where(
                (n**2 < area_cells_ser) & (area_cells_ser <= n * (n + 1)),
                4 * n + 2,
                min_p,
            )
            min_p = np.where(area_cells_ser > n * (n + 1), 4 * n + 4, min_p)

            return perimeter_cells_ser / min_p
        else:
            # this is rare and not even supported in FRAGSTATS. We could
            # calculate the perimeter in terms of cell counts in a
            # dedicated function and then adjust for a square standard,
            # but I believe it is not worth the effort. So we will just
            # return the base formula without adjusting for the square
            # standard
            return 0.25 * perimeter_ser / np.sqrt(area_ser)

    # properties

    @property
    def _num_patches_dict(self):
        try:
            return self._cached_num_patches_dict
        except AttributeError:
            self._cached_num_patches_dict = {
                class_val: self.class_label(class_val)[1]
                for class_val in self.classes
            }

            return self._cached_num_patches_dict

    @property
    def landscape_area(self):
        try:
            return self._landscape_area
        except AttributeError:
            if self.nodata == 0:
                # ~ x8 times faster
                landscape_num_cells = np.count_nonzero(self.landscape_arr)
            else:
                landscape_num_cells = np.sum(self.landscape_arr != self.nodata)

            self._landscape_area = landscape_num_cells * self.cell_area

            return self._landscape_area

    @property
    def _patch_class_ser(self):
        try:
            return self._cached_patch_class_ser
        except AttributeError:
            self._cached_patch_class_ser = pd.Series(
                np.concatenate([
                    np.full(self._num_patches_dict[class_val], class_val)
                    for class_val in self.classes
                ]),
                name="class_val",
            )

            return self._cached_patch_class_ser

    @property
    def _patch_area_ser(self):
        try:
            return self._cached_patch_area_ser
        except AttributeError:
            self._cached_patch_area_ser = pd.Series(
                np.concatenate([
                    self.compute_patch_areas(self.class_label(class_val)[0])
                    for class_val in self.classes
                ]),
                name="area",
            )

            return self._cached_patch_area_ser

    @property
    def _patch_perimeter_ser(self):
        try:
            return self._cached_patch_perimeter_ser
        except AttributeError:
            self._cached_patch_perimeter_ser = pd.Series(
                np.concatenate([
                    self.compute_patch_perimeters(
                        self.class_label(class_val)[0])
                    for class_val in self.classes
                ]),
                name="perimeter",
            )

            return self._cached_patch_perimeter_ser

    @property
    def _patch_euclidean_nearest_neighbor_ser(self):
        try:
            return self._cached_patch_euclidean_nearest_neighbor_ser
        except AttributeError:
            self._cached_patch_euclidean_nearest_neighbor_ser = pd.Series(
                np.concatenate([
                    self.compute_patch_euclidean_nearest_neighbor(
                        self.class_label(class_val)[0])
                    for class_val in self.classes
                ]),
                name="euclidean_nearest_neighbor",
            )

            return self._cached_patch_euclidean_nearest_neighbor_ser

    @property
    def _adjacency_df(self):
        try:
            return self._cached_adjacency_df
        except AttributeError:
            num_classes = len(self.classes)
            # first create a reclassified array with the landscape's shape
            # where each class value will be an int from 0 to `num_classes - 1`
            # and the nodata value will be an int of value `num_classes`
            reclassified_arr = np.copy(self.landscape_arr)
            for i, class_val in enumerate(self.classes):
                reclassified_arr[self.landscape_arr == class_val] = i
                reclassified_arr[self.landscape_arr ==
                                 self.nodata] = num_classes

            # pad the reclassified array with the nodata value (i.e.,
            # `num_classes` see comment above). Set dtype to `np.uint32` to
            # match the numba method signature of
            # `pylandstats_compute.compute_adjacency_arr`
            padded_arr = np.pad(
                reclassified_arr,
                pad_width=1,
                mode="constant",
                constant_values=num_classes,
            ).astype(np.uint32)

            # compute the adjacency array
            adjacency_arr = compute_adjacency_arr(padded_arr,
                                                  num_classes)  # .sum(axis=0)

            # put the adjacency array in the form of a pandas DataFrame
            adjacency_cols = np.concatenate([self.classes, [self.nodata]])
            adjacency_df = pd.DataFrame(
                index=pd.MultiIndex.from_product(
                    [["horizontal", "vertical"], adjacency_cols],
                    names=["direction", "class_val"],
                ),
                columns=adjacency_cols,
            )
            adjacency_df.loc["horizontal"] = adjacency_arr[0]
            adjacency_df.loc["vertical"] = adjacency_arr[1]

            # cache it
            self._cached_adjacency_df = adjacency_df

            return self._cached_adjacency_df

    # small utilities to get patch areas/perimeters for a particular class only

    def _get_patch_area_ser(self, class_val=None):
        if class_val is None:
            patch_area_ser = self._patch_area_ser
        else:
            patch_area_ser = self._patch_area_ser[self._patch_class_ser ==
                                                  class_val]

        # TODO: return a copy? even when `class_val` is set and thus
        # `patch_area_ser` is a slice: although we would not have alias
        # problems, we would get a `SettingWithCopyWarning` form `pandas`
        return patch_area_ser

    def _get_patch_perimeter_ser(self, class_val=None, copy=False):
        if class_val is None:
            patch_perimeter_ser = self._patch_perimeter_ser
        else:
            patch_perimeter_ser = self._patch_perimeter_ser[
                self._patch_class_ser == class_val]

        # TODO: return a copy? even when `class_val` is set and thus
        # `patch_perimeter_ser` is a slice: although we would not have alias
        # problems, we would get a `SettingWithCopyWarning` form `pandas`
        return patch_perimeter_ser

    def _get_patch_euclidean_nearest_neighbor_ser(self, class_val=None,
                                                  copy=False):
        if class_val is None:
            patch_euclidean_nearest_neighbor_ser = (
                self._patch_euclidean_nearest_neighbor_ser)
        else:
            patch_euclidean_nearest_neighbor_ser = \
                self._patch_euclidean_nearest_neighbor_ser[
                    self._patch_class_ser == class_val]

        # TODO: return a copy? even when `class_val` is set and thus
        # `patch_perimeter_ser` is a slice: although we would not have alias
        # problems, we would get a `SettingWithCopyWarning` form `pandas`
        return patch_euclidean_nearest_neighbor_ser

    # metric distribution statistics

    def _metric_reduce(
            self,
            class_val,
            patch_metric_method,
            patch_metric_method_kws,
            reduce_method,
    ):
        if patch_metric_method_kws is None:
            patch_metrics = patch_metric_method(class_val)
        else:
            patch_metrics = patch_metric_method(class_val,
                                                **patch_metric_method_kws)
        if class_val is None:
            # ACHTUNG: dropping columns from a `pd.DataFrame` until leaving it
            # with only one column will still return a `pd.DataFrame`, so we
            # must convert to `pd.Series` manually (e.g., with `iloc`)
            patch_metrics = patch_metrics.drop("class_val", axis=1).iloc[:, 0]

        return reduce_method(patch_metrics)

    def _metric_mn(self, class_val, patch_metric_method,
                   patch_metric_method_kws=None):
        return self._metric_reduce(class_val, patch_metric_method,
                                   patch_metric_method_kws, np.mean)

    def _metric_am(self, class_val, patch_metric_method,
                   patch_metric_method_kws=None):
        # `area` can be `pd.Series` or `pd.DataFrame`
        area = self.area(class_val)

        if class_val is None:
            area = area["area"]

        return self._metric_reduce(
            class_val,
            patch_metric_method,
            patch_metric_method_kws,
            partial(np.average, weights=area),
        )

    def _metric_md(self, class_val, patch_metric_method,
                   patch_metric_method_kws=None):
        return self._metric_reduce(class_val, patch_metric_method,
                                   patch_metric_method_kws, np.median)

    def _metric_ra(self, class_val, patch_metric_method,
                   patch_metric_method_kws=None):
        return self._metric_reduce(
            class_val,
            patch_metric_method,
            patch_metric_method_kws,
            lambda ser: ser.max() - ser.min(),
        )

    def _metric_sd(self, class_val, patch_metric_method,
                   patch_metric_method_kws=None):
        return self._metric_reduce(class_val, patch_metric_method,
                                   patch_metric_method_kws, np.std)

    def _metric_cv(
            self,
            class_val,
            patch_metric_method,
            patch_metric_method_kws=None,
            percent=True,
    ):
        metric_cv = self._metric_reduce(
            class_val,
            patch_metric_method,
            patch_metric_method_kws,
            stats.variation,
        )
        if percent:
            metric_cv *= 100

        return metric_cv

    ###########################################################################
    # patch-level metrics

    # area and edge metrics

    def area(self, class_val=None, hectares=True):
        """
        The area of each patch of the landscape

        .. math::
           AREA = a_{i,j} \\quad [hec] \\; or \\; [m]

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed for the corresponding
            class only, otherwise it will be computed for all the classes of
            the landscape
        hectares : bool, default True
            Whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        AREA : pd.Series if `class_val` is provided, pd.DataFrame otherwise
            AREA > 0, without limit
        """

        # class_ser = self._patch_class_ser
        # area_ser = self._patch_area_ser.copy()
        area_ser = self._get_patch_area_ser(class_val)

        if hectares:
            # ACHTUNG: very important to copy to ensure that we do not modify
            # the 'area' values if converting to hectares nor we return a
            # variable with the reference to the property
            # `self._patch_areas_ser`
            area_ser = area_ser.copy()
            area_ser /= 10000

        if class_val is None:
            return pd.DataFrame({
                "class_val": self._patch_class_ser,
                "area": area_ser
            })
        else:
            return area_ser

    def perimeter(self, class_val=None):
        """
        The perimeter of each patch of the landscape

        .. math::
           PERIM = p_{i,j} \\quad [m]

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed for the corresponding
            class only, otherwise it will be computed for all the classes of
            the landscape

        Returns
        -------
        PERIM : pd.Series if `class_val` is provided, pd.DataFrame otherwise
            PERIM > 0, without limit
        """

        # class_ser = self._patch_class_ser
        # perimeter_ser = self._patch_perimeter_ser
        perimeter_ser = self._get_patch_perimeter_ser(class_val)

        if class_val is None:
            return pd.DataFrame({
                "class_val": self._patch_class_ser,
                "perimeter": perimeter_ser
            })
        else:
            return perimeter_ser

    # shape

    def perimeter_area_ratio(self, class_val=None, hectares=True):
        """
        The ratio between the perimeter and area of each patch of the
        landscape. Measures shape complexity, however it varies with the size
        of the patch, e.g, for the same shape, larger patches will have a
        smaller perimeter-area ratio.

        .. math::
           PARA = \\frac{p_{i,j}}{a_{i,j}} \\quad [m/hec] \\; or \\; [m/m^2]

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed for the corresponding
            class only, otherwise it will be computed for all the classes of
            the landscape
        hectares : bool, default True
            Whether the area should be converted to hectares (tends to yield
            more legible values for the metric)

        Returns
        -------
        PARA : pd.Series if `class_val` is provided, pd.DataFrame otherwise
            PARA > 0, without limit
        """

        # class_ser = self._patch_class_ser
        # area_ser = self._patch_area_ser.copy()
        area_ser = self._get_patch_area_ser(class_val)
        perimeter_ser = self._get_patch_perimeter_ser(class_val)

        if hectares:
            # ACHTUNG: very important to copy to ensure that we do not modify
            # the 'area' values if converting to hectares nor we return a
            # variable with the reference to the property
            # `self._patch_areas_ser`
            area_ser = area_ser.copy()
            area_ser /= 10000

        perimeter_area_ratio_ser = perimeter_ser / area_ser

        if class_val is None:
            return pd.DataFrame({
                "class_val":
                self._patch_class_ser,
                "perimeter_area_ratio":
                perimeter_area_ratio_ser,
            })
        else:
            # ensure that the returned `pd.Series` has a name (so `seaborn`
            # plots can automatically label the axes)
            perimeter_area_ratio_ser.name = "perimeter_area_ratio"
            return perimeter_area_ratio_ser

    def shape_index(self, class_val=None):
        """
        A measure of shape complexity, similar to the perimeter-area ratio,
        but correcting for its size problem by adjusting for a standard square
        shape.

        .. math::
           SHAPE = \\frac{.25 \\; p_{i,j}}{\\sqrt{a_{i,j}}}

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed for the corresponding
            class only, otherwise it will be computed for all the classes of
            the landscape

        Returns
        -------
        SHAPE : pd.Series if `class_val` is provided, pd.DataFrame otherwise
            SHAPE >= 1, without limit ; SHAPE equals 1 when the patch
            is maximally compact, and increases without limit as patch shape
            becomes more regular
        """

        area_ser = self._get_patch_area_ser(class_val)
        perimeter_ser = self._get_patch_perimeter_ser(class_val)

        shape_index_ser = self.compute_shape_index(area_ser, perimeter_ser)

        if class_val is None:
            return pd.DataFrame({
                "class_val": self._patch_class_ser,
                "shape_index": shape_index_ser,
            })
        else:
            # ensure that the returned `pd.Series` has a name (so `seaborn`
            # plots can automatically label the axes)
            shape_index_ser.name = "shape_index"
            return shape_index_ser

    def fractal_dimension(self, class_val=None):
        """
        A measure of shape complexity appropriate across a wide range of patch
        sizes

        .. math::
           FRAC = \\frac{2 \\; ln (.25 \\; p_{i,j})}{ln (a_{i,j})}

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed for the corresponding
            class only, otherwise it will be computed for all the classes of
            the landscape

        Returns
        -------
        FRAC : pd.Series if `class_val` is provided, pd.DataFrame otherwise
            1 <= FRAC <=2 ; for a two-dimensional patch, FRAC approaches 1 for
            very simple shapes such as squares, and approaches 2 for complex
            plane-filling shapes
        """

        area_ser = self._get_patch_area_ser(class_val)
        perimeter_ser = self._get_patch_perimeter_ser(class_val)

        # TODO: separate staticmethod?
        fractal_dimension_ser = (2 * np.log(0.25 * perimeter_ser) /
                                 np.log(area_ser))

        if class_val is None:
            return pd.DataFrame({
                "class_val": self._patch_area_ser,
                "fractal_dimension": fractal_dimension_ser,
            })
        else:
            # ensure that the returned `pd.Series` has a name (so `seaborn`
            # plots can automatically label the axes)
            fractal_dimension_ser.name = "fractal_dimension"
            return fractal_dimension_ser

    def continguity_index(self, class_val=None):
        """
        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed for the corresponding
            class only, otherwise it will be computed for all the classes of
            the landscape

        Returns
        -------
        contig : float
            0 <= contig <= 1 ; contig equals 0 for a one-pixel
            patch and increases to a limit of 1 as patch contiguity increases
        """

        # TODO
        raise NotImplementedError

    # aggregation metrics (formerly isolation, proximity)

    def euclidean_nearest_neighbor(self, class_val=None):
        """
        Distance to the nearest neighboring patch of the same class based on
        the shortest edge-to-edge distance

        .. math::
           ENN = h_{i,j} \\quad [m]

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed for the corresponding
            class only, otherwise it will be computed for all the classes of
            the landscape

        Returns
        -------
        ENN : numeric
            ENN > 0, without limit ; ENN approaches 0 as the distance to the
            nearest neighbors decreases
        """

        euclidean_nearest_neighbor_ser = \
            self._get_patch_euclidean_nearest_neighbor_ser(class_val)

        if class_val is None:
            for class_val in self.classes:
                num_patches = self._num_patches_dict[class_val]
                if num_patches < 2:
                    warnings.warn(
                        "Class {} has less than 2 patches. ".format(class_val)
                        +
                        "Euclidean-nearest-neighbor might contain nan values",
                        RuntimeWarning,
                    )

            return pd.DataFrame({
                "class_val":
                self._patch_class_ser,
                "euclidean_nearest_neighbor":
                euclidean_nearest_neighbor_ser,
            })
        else:
            num_patches = self._num_patches_dict[class_val]
            if num_patches < 2:
                warnings.warn(
                    "Class {} has less than 2 patches. ".format(class_val) +
                    "Euclidean-nearest-neighbor might contain nan values",
                    RuntimeWarning,
                )

            return euclidean_nearest_neighbor_ser

    def proximity(self, search_radius, class_val=None):
        """
        Parameters
        ----------
        search_radius : numeric
            Search radius defining the neighborhood at which the metric will
            be computed for each patch
        class_val : int, optional
            If provided, the metric will be computed for the corresponding
            class only, otherwise it will be computed for all the classes of
            the landscape

        Returns
        -------
        prox : float
            prox >= 0 ; prox equals 0 if a patch has no neighbors, and
            increases as the neighborhood is occupied by patches of the same
            type and those patches become more contiguous (or less fragmented)
        """

        # TODO
        raise NotImplementedError

    ###########################################################################
    # class-level and landscape-level metrics

    # area, density, edge

    def total_area(self, class_val=None, hectares=True):
        """
        Total area. If `class_val` is provided, the metric is computed at the
        class level as in:

        .. math::
           TA_i = \\sum_{j=1}^{n_i} a_{i,j} \\quad [hec] \\; or \\; [m] \\quad
           (class \\; i)

        otherwise, the metric is computed at the landscape level as in:

        .. math::
           TA = A \\quad [hec] \\; or \\; [m] \\quad (landscape)

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            Whether the area should be converted to hectares (tends to yield
            more legible values for the metric)

        Returns
        -------
        TA : numeric
        """

        if class_val is None:
            total_area = self.landscape_area
        else:
            area_ser = self._get_patch_area_ser(class_val)
            total_area = np.sum(area_ser)

        if hectares:
            total_area /= 10000

        return total_area

    def proportion_of_landscape(self, class_val, percent=True):
        """
        Measures the proportional abundance of a particular class within the
        landscape. It is computed at the class level as in:

        .. math::
           PLAND = \\frac{1}{A} \\sum_j^{n_i} a_{i,j}

        Parameters
        ----------
        class_val : int
            Class for which the metric should be computed
        percent : bool, default True
            Whether the index should be expressed as proportion or converted
            to percentage. If True, this method returns FRAGSTATS' percentage
            of landscape (PLAND)

        Returns
        -------
        PLAND : numeric
            0 < PLAND <= 100 ; PLAND approaches 0 when the occurrence of the
            corresponding class becomes increasingly rare, and approaches 100
            when the entire landscape consists of a single patch of such class.
        """

        numerator = np.sum(self._get_patch_area_ser(class_val))

        if percent:
            numerator *= 100

        return numerator / self.landscape_area

    def number_of_patches(self, class_val=None):
        """
        Number of patches. If `class_val` is provided, the metric is computed
        at the class level as in:

        .. math::
           NP_i = n_i \\quad (class \\; i)

        otherwise, the metric is computed at the landscape level as in:

        .. math::
           NP = N \\quad (landscape)

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        NP : int
            NP >= 1, without limit
        """
        if class_val is None:
            num_patches = np.sum(list(self._num_patches_dict.values()))
        else:
            num_patches = self._num_patches_dict[class_val]

        return num_patches

    def patch_density(self, class_val=None, percent=True, hectares=True):
        """
        Density of class patches, which is arguably more useful than the
        number of patches since it facilitates comparison among landscapes of
        different sizes. If `class_val` is provided, the metric is computed at
        the class level as in:

        .. math::
           PD_i = \\frac{n_i}{A} \\quad [1/hec] \\; or \\; [1/m^2] \\quad
           (class \\; i)

        otherwise, the metric is computed at the landscape level as in:

        .. math::
           PD = \\frac{N}{A} \\quad [1/hec] \\; or \\; [1/m^2] \\quad
           (landscape)

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        percent : bool, default True
            Whether the index should be expressed as proportion or converted
            to percentage
        hectares : bool, default True
            Whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        PD : numeric
            PD > 0, constrained by cell size ; maximum PD is attained when
            every cell is a separate patch
        """

        # TODO: DRY and use `self.number_of_patches` as in:
        # `numerator = self.number_of_patches(class_val)`
        # or avoid reusing metric's methods?
        if class_val is None:
            numerator = np.sum(list(self._num_patches_dict.values()))
        else:
            numerator = self._num_patches_dict[class_val]

        if percent:
            numerator *= 100
        if hectares:
            numerator *= 10000

        return numerator / self.landscape_area

    def largest_patch_index(self, class_val=None, percent=True):
        """
        The proportion of total landscape comprised by the largest patch. If
        `class_val` is provided, the metric is computed at the class level as
        in:

        .. math::
           LPI_i = \\frac{1}{A} \\max_{j=1}^{n_i} a_{i,j} \\quad (class \\; i)

        otherwise, the metric is computed at the landscape level as in:

        .. math::
           LPI = \\frac{1}{A} \\max a_{i,j} \\quad (landscape)

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        percent : bool, default True
            Whether the index should be expressed as proportion or converted
            to percentage

        Returns
        -------
        LPI : numeric
            0 < LPI <= 100 (or 0 < LPI <= 1 if percent argument is False) ;
            LPI approaches 0 when the largest patch of the corresponding class
            is increasingly small, and approaches its maximum value when such
            largest patch comprises the totality of the landscape
        """

        area_ser = self._get_patch_area_ser(class_val)

        numerator = np.max(area_ser)

        if percent:
            numerator *= 100

        return numerator / self.landscape_area

    def total_edge(self, class_val=None, count_boundary=False):
        """
        Measure of the total edge length. If `class_val` is provided, the
        metric is computed at the class level as in:

        .. math::
           TE_i = \\sum_{k=1}^{m} e_{i,k} \\quad [m] \\quad (class \\; i)

        otherwise, the metric is computed at the landscape level as in:

        .. math::
           TE = E \\quad [m] \\quad (landscape)

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        count_boundary : bool, default False
            Whether the boundary of the landscape should be included in the
            total edge length

        Returns
        -------
        TE : numeric
            TE >= 0 ; TE equals 0 when the entire landscape and its border
            consist of the corresponding class
        """

        if class_val is None:
            if count_boundary:
                total_edge = self.compute_arr_perimeter(
                    np.pad(
                        self.landscape_arr,
                        pad_width=1,
                        mode="constant",
                        constant_values=self.nodata,
                    ))
            else:
                if np.isclose(self.cell_width, self.cell_height,
                              rtol=CELLLENGTH_RTOL):
                    adjacency_arr = np.triu(
                        self._adjacency_df.groupby(
                            level=1, sort=False).sum().drop(self.nodata).drop(
                                self.nodata, axis=1))
                    np.fill_diagonal(adjacency_arr, 0)
                    total_edge = np.sum(adjacency_arr) * self.cell_width
                else:
                    total_edge = 0
                    for direction, length in [
                        ("horizontal", self.cell_width),
                        ("vertical", self.cell_height),
                    ]:
                        adjacency_arr = np.triu(
                            self._adjacency_df.loc[direction].drop(
                                self.nodata).drop(self.nodata, axis=1))
                        # `np.fill_diagonal` acts inplace, however `np.triu`
                        # returns a copy so we do not need to worry about
                        # inadvently modfying `self._adjacency_df`
                        np.fill_diagonal(adjacency_arr, 0)
                        total_edge += np.sum(adjacency_arr) * length
        else:
            if count_boundary:
                # then the total edge is just the sum of the perimeters of all
                # the patches of the corresponding class
                perimeter_ser = self._get_patch_perimeter_ser(class_val)
                total_edge = np.sum(perimeter_ser)
            else:
                if np.isclose(self.cell_width, self.cell_height,
                              rtol=CELLLENGTH_RTOL):
                    total_edge = (np.sum(
                        self._adjacency_df.groupby(
                            level=1, sort=False).sum().drop(
                                [class_val, self.nodata])[class_val]) *
                                  self.cell_width)
                else:
                    total_edge = 0
                    for direction, length in [
                        ("horizontal", self.cell_width),
                        ("vertical", self.cell_height),
                    ]:
                        total_edge += (
                            np.sum(self._adjacency_df.loc[direction].drop(
                                [class_val, self.nodata])[class_val]) * length)

        return total_edge

    def edge_density(self, class_val=None, count_boundary=False,
                     hectares=True):
        """
        Measure of edge length per area unit, which facilitates comparison
        among landscapes of different sizes. If `class_val` is provided, the
        metric is computed at the class level as in:

        .. math::
           ED_i = \\frac{1}{A} \\sum_{k=1}^{m} e_{i,k} \\quad [m/hec] \\; or
           \\; [m/m^2] \\quad (class \\; i)

        otherwise, the metric is computed at the landscape level as in:

        .. math::
           ED = \\frac{E}{A} \\quad [m/hec] \\; or \\; [m/m^2] \\quad
           (landscape)

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        count_boundary : bool, default False
            Whether the boundary of the landscape should be considered
        hectares : bool, default True
            Whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        ED : numeric
            ED >= 0, without limit ; ED equals 0 when the entire landscape and
            its border consist of the corresponding patch class.
        """

        # TODO: we make an exception here of the "not reusing other metric's
        # methods within metric's methods" policy, since `total_edge` is a bit
        # puzzling to compute
        numerator = self.total_edge(class_val=class_val,
                                    count_boundary=count_boundary)

        if hectares:
            numerator *= 10000

        return numerator / self.landscape_area

    def area_mn(self, class_val=None, hectares=True):
        """
        Mean of the patch area distribution. See also the documentation of
        `area`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            Whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        area_mn : float
        """

        return self._metric_mn(class_val, self.area, {"hectares": hectares})

    def area_am(self, class_val=None, hectares=True):
        """
        Area-weighted mean of the patch area distribution. See also the
        documentation of `area`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            Whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        area_am : float
        """

        return self._metric_am(class_val, self.area, {"hectares": hectares})

    def area_md(self, class_val=None, hectares=True):
        """
        Median of the patch area distribution. See also the documentation of
        `area`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            Whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        area_md : float
        """

        return self._metric_md(class_val, self.area, {"hectares": hectares})

    def area_ra(self, class_val=None, hectares=True):
        """
        Range of the patch area distribution. See also the documentation of
        `area`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            Whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        area_ra : float
        """

        return self._metric_ra(class_val, self.area, {"hectares": hectares})

    def area_sd(self, class_val=None, hectares=True):
        """
        Standard deviation of the patch area distribution. See also the
        documentation of `area`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            Whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        area_sd : float
        """

        return self._metric_sd(class_val, self.area, {"hectares": hectares})

    def area_cv(self, class_val=None, percent=True):
        """
        Coefficient of variation of the patch area distribution. See also the
        documentation of `area`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        percent : bool, default True
            whether the index should be expressed as proportion or converted
            to percentage

        Returns
        -------
        area_cv : float
        """

        return self._metric_cv(class_val, self.area, percent=percent)

    def perimeter_mn(self, class_val=None):
        """
        Mean of the patch perimeter distribution. See also the documentation of
        `perimeter`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        perimeter_mn : float
        """

        return self._metric_mn(class_val, self.perimeter)

    def perimeter_am(self, class_val=None):
        """
        Area-weighted mean of the patch perimeter distribution. See also the
        documentation of `perimeter`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        perimeter_am : float
        """

        return self._metric_am(class_val, self.perimeter)

    def perimeter_md(self, class_val=None):
        """
        Median of the patch perimeter distribution. See also the documentation
        of `perimeter`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        perimeter_md : float
        """

        return self._metric_md(class_val, self.perimeter)

    def perimeter_ra(self, class_val=None):
        """
        Range of the patch perimeter distribution. See also the documentation
        of `perimeter`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        perimeter_ra : float
        """

        return self._metric_ra(class_val, self.perimeter)

    def perimeter_sd(self, class_val=None):
        """
        Standard deviation of the patch perimeter distribution. See also the
        documentation of `perimeter`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        perimeter_sd : float
        """

        return self._metric_sd(class_val, self.perimeter)

    def perimeter_cv(self, class_val=None, percent=True):
        """
        Coefficient of variation of the patch perimeter distribution. See also
        the documentation of `perimeter`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        percent : bool, default True
            whether the index should be expressed as proportion or converted
            to percentage

        Returns
        -------
        perimeter_cv : float
        """

        return self._metric_cv(class_val, self.perimeter, percent=percent)

    # shape

    def perimeter_area_ratio_mn(self, class_val=None, hectares=True):
        """
        Mean of the patch perimeter-area ratio distribution. See also the
        documentation of `perimeter_area_ratio`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            Whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        para_mn : float
        """

        return self._metric_mn(class_val, self.perimeter_area_ratio,
                               {"hectares": hectares})

    def perimeter_area_ratio_am(self, class_val=None, hectares=True):
        """
        Area-weighted mean of the patch perimeter-area ratio distribution. See
        also the documentation of `perimeter_area_ratio`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            Whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        para_am : float
        """

        return self._metric_am(class_val, self.perimeter_area_ratio,
                               {"hectares": hectares})

    def perimeter_area_ratio_md(self, class_val=None, hectares=True):
        """
        Median of the patch perimeter-area ratio distribution. See also the
        documentation of `perimeter_area_ratio`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            Whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        para_md : float
        """

        return self._metric_md(class_val, self.perimeter_area_ratio,
                               {"hectares": hectares})

    def perimeter_area_ratio_ra(self, class_val=None, hectares=True):
        """
        Range of the patch perimeter-area ratio distribution. See also the
        documentation of `perimeter_area_ratio`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            Whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        para_ra : float
        """

        return self._metric_ra(class_val, self.perimeter_area_ratio,
                               {"hectares": hectares})

    def perimeter_area_ratio_sd(self, class_val=None, hectares=True):
        """
        Standard deviation of the patch perimeter-area ratio distribution. See
        also the documentation of `perimeter_area_ratio`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            Whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        para_sd : float
        """

        return self._metric_sd(class_val, self.perimeter_area_ratio,
                               {"hectares": hectares})

    def perimeter_area_ratio_cv(self, class_val=None, percent=True):
        """
        Coefficient of variation of the patch perimeter-area ratio
        distribution. See also the documentation of `perimeter_area_ratio`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        percent : bool, default True
            Whether the index should be expressed as proportion or converted
            to percentage

        Returns
        -------
        para_cv : float
        """

        return self._metric_cv(class_val, self.perimeter_area_ratio,
                               percent=percent)

    def shape_index_mn(self, class_val=None):
        """
        Mean of the shape index distribution. See also the documentation of
        `shape_index`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        shape_mn : float
        """

        return self._metric_mn(class_val, self.shape_index)

    def shape_index_am(self, class_val=None):
        """
        Area-weighted mean of the shape index distribution. See also the
        documentation of `shape_index`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        shape_am : float
        """

        return self._metric_am(class_val, self.shape_index)

    def shape_index_md(self, class_val=None):
        """
        Median of the shape index distribution. See also the documentation of
        `shape_index`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        shape_md : float
        """

        return self._metric_md(class_val, self.shape_index)

    def shape_index_ra(self, class_val=None):
        """
        Range of the shape index distribution. See also the documentation of
        `shape_index`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        shape_ra : float
        """

        return self._metric_ra(class_val, self.shape_index)

    def shape_index_sd(self, class_val=None):
        """
        Standard deviation of the shape index distribution. See also the
        documentation of `shape_index`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        shape_sd : float
        """

        return self._metric_sd(class_val, self.shape_index)

    def shape_index_cv(self, class_val=None, percent=True):
        """
        Coefficient of variation of the shape index distribution. See also the
        documentation of `shape_index`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        percent : bool, default True
           Whether the index should be expressed as proportion or converted
            to percentage

        Returns
        -------
        shape_cv : float
        """

        return self._metric_cv(class_val, self.shape_index, percent=percent)

    def fractal_dimension_mn(self, class_val=None):
        """
        Mean of the fractal dimension distribution. See also the documentation
        of `fractal_dimension`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        frac_mn : float
        """

        return self._metric_mn(class_val, self.fractal_dimension)

    def fractal_dimension_am(self, class_val=None):
        """
        Area-weighted mean of the fractal dimension distribution. See also the
        documentation of `fractal_dimension`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        frac_am : float
        """

        return self._metric_am(class_val, self.fractal_dimension)

    def fractal_dimension_md(self, class_val=None):
        """
        Median of the fractal dimension distribution. See also the
        documentation of `fractal_dimension`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        frac_md : float
        """

        return self._metric_md(class_val, self.fractal_dimension)

    def fractal_dimension_ra(self, class_val=None):
        """
        Range of the fractal dimension distribution. See also the
        documentation of `fractal_dimension`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        frac_ra : float
        """

        return self._metric_ra(class_val, self.fractal_dimension)

    def fractal_dimension_sd(self, class_val=None):
        """
        Standard deviation of the fractal dimension distribution. See also the
        documentation of `fractal_dimension`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        frac_sd : float
        """

        return self._metric_sd(class_val, self.fractal_dimension)

    def fractal_dimension_cv(self, class_val=None, percent=True):
        """
        Coefficient of variation of the fractal dimension distribution. See
        also the documentation of `fractal_dimension`.

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        percent : bool, default True
            Whether the index should be expressed as proportion or converted
            to percentage

        Returns
        -------
        frac_cv : float
        """

        return self._metric_cv(class_val, self.fractal_dimension,
                               percent=percent)

    def continguity_index_mn(self, class_val=None):
        """
        See also the documentation of `Landscape.contiguity_index`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        contig_mn : float
        """

        # TODO
        raise NotImplementedError

    def continguity_index_am(self, class_val=None):
        """
        See also the documentation of `Landscape.contiguity_index`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        contig_am : float
        """

        # TODO
        raise NotImplementedError

    def continguity_index_md(self, class_val=None):
        """
        See also the documentation of `Landscape.contiguity_index`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        contig_md : float
        """

        # TODO
        raise NotImplementedError

    def continguity_index_ra(self, class_val=None):
        """
        See also the documentation of `Landscape.contiguity_index`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        contig_ra : float
        """

        # TODO
        raise NotImplementedError

    def continguity_index_sd(self, class_val=None):
        """
        See also the documentation of `Landscape.contiguity_index`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        contig_sd : float
        """

        # TODO
        raise NotImplementedError

    def continguity_index_cv(self, class_val=None):
        """
        See also the documentation of `Landscape.contiguity_index`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        contig_cv : float
        """

        # TODO
        raise NotImplementedError

    # isolation, proximity

    def proximity_mn(self, class_val=None):
        """
        See also the documentation of `Landscape.proximity`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        prox_mn : float
        """

        # TODO
        raise NotImplementedError

    def proximity_am(self, class_val=None):
        """
        See also the documentation of `Landscape.proximity`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        prox_am : float
        """

        # TODO
        raise NotImplementedError

    def proximity_md(self, class_val=None):
        """
        See also the documentation of `Landscape.proximity`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        prox_md : float
        """

        # TODO
        raise NotImplementedError

    def proximity_ra(self, class_val=None):
        """
        See also the documentation of `Landscape.proximity`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        prox_ra : float
        """

        # TODO
        raise NotImplementedError

    def proximity_sd(self, class_val=None):
        """
        See also the documentation of `Landscape.proximity`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        prox_sd : float
        """

        # TODO
        raise NotImplementedError

    def proximity_cv(self, class_val=None):
        """
        See also the documentation of `Landscape.proximity`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        prox_cv :
        """

        # TODO
        raise NotImplementedError

    def euclidean_nearest_neighbor_mn(self, class_val=None):
        """
        See also the documentation of `Landscape.euclidean_nearest_neighbor`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        enn_mn : float
        """
        return self._metric_mn(class_val, self.euclidean_nearest_neighbor)

    def euclidean_nearest_neighbor_am(self, class_val=None):
        """
        See also the documentation of `Landscape.euclidean_nearest_neighbor`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        enn_am : float
        """

        return self._metric_am(class_val, self.euclidean_nearest_neighbor)

    def euclidean_nearest_neighbor_md(self, class_val=None):
        """
        See also the documentation of `Landscape.euclidean_nearest_neighbor`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        enn_md : float
        """

        return self._metric_md(class_val, self.euclidean_nearest_neighbor)

    def euclidean_nearest_neighbor_ra(self, class_val=None):
        """
        See also the documentation of `Landscape.euclidean_nearest_neighbor`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        enn_ra : float
        """

        return self._metric_ra(class_val, self.euclidean_nearest_neighbor)

    def euclidean_nearest_neighbor_sd(self, class_val=None):
        """
        See also the documentation of `Landscape.euclidean_nearest_neighbor`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        enn_sd : float
        """

        return self._metric_sd(class_val, self.euclidean_nearest_neighbor)

    def euclidean_nearest_neighbor_cv(self, class_val=None, percent=True):
        """
        See also the documentation of `Landscape.euclidean_nearest_neighbor`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        percent : bool, default True
            Whether the index should be expressed as proportion or converted
            to percentage

        Returns
        -------
        enn_cv : float
        """

        return self._metric_cv(class_val, self.euclidean_nearest_neighbor,
                               percent=percent)

    # aggregation

    def landscape_shape_index(self, class_val=None):
        """
        Measure of class aggregation that provides a standardized measure of
        edginess that adjusts for the size of the landscape. If `class_val` is
        provided, the metric is computed at the class level as in:

        .. math::
           LSI_i = \\frac{.25 \\sum \\limits_{k=1}^{m} e_{i,k}}{\\sqrt{A}}
           \\quad (class \\; i)

        otherwise, the metric is computed at the landscape level as in:

        .. math::

           LSI = \\frac{.25 E}{\\sqrt{A}} \\quad (landscape)

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        LSI : float
            LSI >=1 ; LSI equals 1 when the entire landscape consists of a
            single patch of the corresponding class, and increases without
            limit as the patches of such class become more disaggregated.
        """

        # compute the total area
        if class_val is None:
            area = self.landscape_area
        else:
            area = np.sum(self._get_patch_area_ser(class_val))

        # TODO: we make an exception here of the "not reusing other metric's
        # methods within metric's methods" policy, since `total_edge` is a bit
        # puzzling to compute
        perimeter = self.total_edge(class_val, count_boundary=True)

        # `compute shape index` works on vectors, so we need to pass arrays as
        # arguments and then extract its first (and only element) in order to
        # return a scalar
        # TODO: use np.vectorize
        return self.compute_shape_index(np.array([area]),
                                        np.array([perimeter]))[0]

    # contagion, interspersion

    def interspersion_juxtaposition_index(self, class_val=None, percent=True):
        """

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        percent : bool, default True
            Whether the index should be expressed as proportion or converted
            to percentage


        Returns
        -------
        iji : float
            0 < iji <= 100 ; iji approaches 0 when the corresponding class is
            adjacent to only 1 other class and the number of classes increases,
            iji approaches its maximum when the corersponding class is equally
            adjacent to all other classes. Analogously, at the landscape level,
            iji approaches 0 when the distribution of adjacencies among classes
            becomes increasingly uneven, and approaches its maximum when all
            classes are equally adjacent to all other classes.
        """

        # TODO
        raise NotImplementedError

    def effective_mesh_size(self, class_val=None, hectares=True):
        """
        Measure of aggregation based on the cumulative patch size distribution.
        If `class_val` is provided, the metric is computed at the class level
        as in:

        .. math::
           MESH_i = \\frac{1}{A} \\sum_{j=1}^{n_i} a_{i,j}^2 \\quad [m] \\quad
           (class \\; i)

        otherwise, the metric is computed at the landscape level as in:

        .. math::
           MESH = \\frac{1}{A} \\sum_{i=1}^{m} \\sum_{j=1}^{n_i} a_{i,j}^2
           \\quad [m] \\quad (landscape)

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            Whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        mesh : float
            cell_area / A <= MESH <= A
        """
        mesh = np.sum(self._get_patch_area_ser(class_val)**2) / \
            self.landscape_area

        if hectares:
            mesh /= 10000

        return mesh

    ###########################################################################
    # landscape-level metrics

    # contagion, interspersion

    def contagion(self, percent=True):
        """
        Measure of aggregation that measures the probability that two random
        adjacent cells belong to the same class. It is computed at the
        landscape level as in:

        .. math::
           CONTAG = 1 + \\frac{
             \\sum \\limits_{i=1}^{m} \\sum \\limits_{k=1}^{m} \\Bigg[
               P_i \\frac{g_{i,k}}{\\sum \\limits_{k=1}^{m} g_{i,k}}
             \\Bigg] \\Bigg[ ln \\Bigg(
               P_i \\frac{g_{i,k}}{\\sum \\limits_{k=1}^{m} g_{i,k}}
             \\Bigg) \\Bigg]}{2 ln(m)}

        Parameters
        ----------
        percent : bool, default True
            Whether the index should be expressed as proportion or converted
            to percentage

        Returns
        -------
        CONTAG : float
            0 < CONTAG <= 100 ; CONTAG approaches 0 when the classes are
            maximally disaggregated (i.e., every cell is a patch of a
            different class) and interspersed (i.e., equal proportions of all
            pairwise adjacencies), and approaches its maximum when the
            landscape consists of a single patch.
        """

        if len(self.classes) < 2:
            warnings.warn(
                "Contagion can only be computed in landscapes with more than "
                "two classes of patches. Returning nan",
                RuntimeWarning,
            )
            return np.nan

        _contag = 0

        for i in self.classes:
            p_i = np.sum(self._get_patch_area_ser(i)) / self.landscape_area
            g_i = self._adjacency_df.groupby(level=1, sort=False).sum()[i]
            # print(g_i)
            g_i_sum = np.sum(g_i)
            # print(g_i_sum)
            for k in self.classes:
                q = p_i * g_i[k] / g_i_sum
                if q > 0:  # avoid zero-logarithm
                    _contag += q * np.log(q)
                # else:
                #     warnings.warn(
                #         "No adjacencies between classes {i} and {k}.".format(
                #             i=i, k=k) + "Ignoring nan", RuntimeWarning)

        contag = 1 + _contag / (2 * np.log(len(self.classes)))

        if percent:
            contag *= 100

        return contag

    # diversity

    def shannon_diversity_index(self):
        """
        Measure of diversity that reflects the number of classes present in
        the landscape as well as the relative abundance of each class. It is
        computed at the landscape level as in:

        .. math::
           SHDI = - \\sum \\limits_{i=1}^{m} \\Big( P_i \\; ln P_i \\Big)

        Returns
        -------
        SHDI : float
            SHDI >= 0 ; SHDI approaches 0 when the entire landscape consists
            of a single patch, and increases as the number of classes
            increases and/or the proportional distribution of area among
            classes becomes more equitable.
        """

        if len(self.classes) < 2:
            warnings.warn(
                "Shannon's Diversity Index can only be computed in landscapes "
                "with more than two classes of patches. Returning nan",
                RuntimeWarning,
            )
            return np.nan

        shdi = 0
        for class_val in self.classes:
            p_class = (np.sum(self._get_patch_area_ser(class_val)) /
                       self.landscape_area)
            shdi += p_class * np.log(p_class)

        return -shdi

    ###########################################################################
    # compute metrics data frames

    def compute_patch_metrics_df(self, metrics=None, metrics_kws=None):
        """
        Computes the patch-level metrics

        Parameters
        ----------
        metrics : list-like, optional
            A list-like of strings with the names of the metrics that should
            be computed. If None, all the implemented patch-level metrics will
            be computed.
        metrics_kws : dict, default None
            Dictionary mapping the keyword arguments (values) that should be
            passed to each metric method (key), e.g., to compute `area` in
            meters instead of hectares, metric_kws should map the string 'area'
            (method name) to {'hectares': False}. If `None`, each metric will
            be computed according to FRAGSTATS defaults.

        Returns
        -------
        df : pd.DataFrame
            Dataframe with the values computed for each patch (index) and
            metric (columns)
        """

        if metrics is None:
            metrics = Landscape.PATCH_METRICS

        if metrics_kws is None:
            metrics_kws = {}

        try:
            # # in order to avoid adding a duplicate 'class_val' column for
            # # each metric, we drop the 'class_val' column of each metric
            # # DataFrame except for the first
            # metric = metrics[0]
            # metrics_dfs = [getattr(self, metric)()]
            # for metric in metrics[1:]:

            metrics_dfs = [self._patch_class_ser]
            for metric in metrics:
                if metric in metrics_kws:
                    metric_kws = metrics_kws[metric]
                else:
                    metric_kws = {}

                metrics_dfs.append(
                    getattr(self,
                            metric)(**metric_kws).drop("class_val", axis=1))

        except AttributeError:
            raise ValueError("{metric} is not among {Landscape.PATCH_METRICS}")
        except TypeError:
            raise ValueError(
                "{metric} cannot be computed at the patch level".format(
                    metric=metric))

        df = pd.concat(metrics_dfs, axis=1)  # [['class_val'] + patch_metrics]
        df.index.name = "patch_id"

        return df

    def compute_class_metrics_df(self, metrics=None, classes=None,
                                 metrics_kws=None):
        """
        Computes the class-level metrics

        Parameters
        ----------
        metrics : list-like, optional
            A list-like of strings with the names of the metrics that should
            be computed. If None, all the implemented class-level metrics will
            be computed.
        classes : list-like, optional
            A list-like of ints or strings with the class values that should be
            considered in the context of this analysis case
        metrics_kws : dict, optional
            Dictionary mapping the keyword arguments (values) that should be
            passed to each metric method (key), e.g., to exclude the boundary
            from the computation of `total_edge`, metric_kws should map the
            string 'total_edge' (method name) to {'count_boundary': False}.
            If `None`, each metric will be computed according to FRAGSTATS
            defaults.

        Returns
        -------
        df : pd.DataFrame
            Dataframe with the values computed for each class (index) and
            metric (columns)
        """

        if metrics is None:
            metrics = Landscape.CLASS_METRICS
        else:
            # here and only here we need to check manually that none of the
            # provided metrics is a patch-level metric. Why? because the
            # methods to compute patch-level metrics and class-level metrics
            # have the same signature, so calling them would not raise any
            # `TypeError` - instead, since the methods to compute patch-level
            # metrics return series/data frames instead of scalar values, we
            # would obtain a malformed dataframe.
            for metric in metrics:
                if metric in Landscape.PATCH_METRICS:
                    raise ValueError(
                        "{metric} cannot be computed at the class level".
                        format(metric=metric))

        if classes is None:
            classes = self.classes

        if metrics_kws is None:
            metrics_kws = {}

        try:
            metrics_sers = []
            for metric in metrics:
                if metric in metrics_kws:
                    metric_kws = metrics_kws[metric]
                else:
                    metric_kws = {}

                metrics_sers.append(
                    pd.Series(
                        {
                            class_val: getattr(self, metric)(
                                class_val=class_val, **metric_kws)
                            for class_val in classes
                        },
                        name=metric,
                    ))

        except AttributeError:
            raise ValueError("{metric} is not among {metrics}".format(
                metric=metric, metrics=Landscape.CLASS_METRICS))
        except TypeError:
            raise ValueError(
                "{metric} cannot be computed at the class level".format(
                    metric=metric))

        df = pd.concat(metrics_sers, axis=1)
        df.index.name = "class_val"

        return df

    def compute_landscape_metrics_df(self, metrics=None, metrics_kws=None):
        """
        Computes the landscape-level metrics

        Parameters
        ----------
        metrics : list-like, optional
            A list-like of strings with the names of the metrics that should
            be computed. If None, all the implemented landscape-level metrics
            will be computed.
        metrics_kws : dict, optional
            Dictionary mapping the keyword arguments (values) that should be
            passed to each metric method (key), e.g., to exclude the boundary
            from the computation of `total_edge`, metric_kws should map the
            string 'total_edge' (method name) to {'count_boundary': False}.
            If `None`, each metric will be computed according to FRAGSTATS
            defaults.

        Returns
        -------
        df : pd.DataFrame
            Dataframe with the values computed at the landscape level (one row
            only) for each metric (columns)
        """

        if metrics is None:
            metrics = Landscape.LANDSCAPE_METRICS

        if metrics_kws is None:
            metrics_kws = {}

        try:
            metrics_dict = {}
            for metric in metrics:
                if metric in metrics_kws:
                    metric_kws = metrics_kws[metric]
                else:
                    metric_kws = {}

                metrics_dict[metric] = getattr(self, metric)(**metric_kws)

        except AttributeError:
            raise ValueError("{metric} is not among {metrics}".format(
                metric=metric, metrics=Landscape.LANDSCAPE_METRICS))
        except TypeError:
            raise ValueError(
                "{metric} cannot be computed at the landscape level".format(
                    metric=metric))

        return pd.DataFrame(metrics_dict, index=[0])

    def plot_landscape(self, cmap=None, ax=None, legend=False, figsize=None,
                       **show_kws):
        """
        Plots the landscape with a categorical legend by means of
        `rasterio.plot.show`

        Parameters
        -------
        cmap : str or `~matplotlib.colors.Colormap`, optional
            A Colormap instance
        ax : axis object, optional
            Plot in given axis; if None creates a new figure
        legend : bool, optional
            If ``True``, display the legend
        figsize: tuple of two numeric types, optional
            Size of the figure to create. Ignored if axis `ax` is provided
        **show_kws : optional
            Keyword arguments to be passed to `rasterio.plot.show`

        Returns
        -------
        ax : matplotlib axis
            axis with plot data
        """

        if cmap is None:
            cmap = plt.rcParams['image.cmap']

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_aspect("equal")

        ax = plot.show(
            np.where(self.landscape_arr != self.nodata, self.landscape_arr,
                     self.nodata), ax=ax, transform=self.transform, cmap=cmap,
            **show_kws)

        if legend:
            im = ax.get_images()[0]
            for class_val in self.classes:
                ax.plot(
                    ax.get_xlim()[0],
                    ax.get_ylim()[0],
                    "o",
                    c=cmap(im.norm(class_val)),
                    label=class_val,
                )

            ax.legend()

        return ax
