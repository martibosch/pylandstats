from __future__ import division

from functools import partial

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage, stats

__all__ = ['Landscape', 'read_geotiff']

KERNEL_HORIZONTAL = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.int8)
KERNEL_VERTICAL = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.int8)
KERNEL_MOORE = ndimage.generate_binary_structure(2, 2)


class Landscape:
    """Documentation for Landscape

    """

    def __init__(self, landscape_arr, res, nodata=0):
        self.landscape_arr = landscape_arr
        self.cell_width, self.cell_height = res
        self.cell_area = res[0] * res[1]
        self.nodata = nodata
        classes = np.array(sorted(np.unique(landscape_arr)))
        classes = np.delete(classes, nodata)
        classes = classes[~np.isnan(classes)]
        self.classes = classes

    ###########################################################################
    # common utilities

    # constants

    PATCH_METRICS = [
        'area', 'perimeter', 'perimeter_area_ratio', 'shape_index',
        'fractal_dimension'
    ]  # 'contiguity_index', 'euclidean_nearest_neighbor', 'proximity'

    _suffixes = ['mn', 'am', 'md', 'ra', 'sd', 'cv']

    CLASS_METRICS = [
        'total_area',
        'proportion_of_landscape',
        'number_of_patches',
        'patch_density',
        'largest_patch_index',
        'total_edge',
        'edge_density',
        'landscape_shape_index',
    ] + ['area_{}'.format(suffix) for suffix in _suffixes] + [
        'perimeter_area_ratio_{}'.format(suffix) for suffix in _suffixes
    ] + ['shape_index_{}'.format(suffix) for suffix in _suffixes
         ] + ['fractal_dimension_{}'.format(suffix) for suffix in _suffixes]

    LANDSCAPE_METRICS = [
        'total_area',
        'number_of_patches',
        'patch_density',
        'largest_patch_index',
        'total_edge',
        'edge_density',
        'landscape_shape_index',
    ] + ['area_{}'.format(suffix) for suffix in _suffixes] + [
        'perimeter_area_ratio_{}'.format(suffix) for suffix in _suffixes
    ] + ['shape_index_{}'.format(suffix) for suffix in _suffixes
         ] + ['fractal_dimension_{}'.format(suffix) for suffix in _suffixes]

    # compute methods

    def class_label(self, class_val):
        return ndimage.label(self.landscape_arr == class_val, KERNEL_MOORE)

    # compute methods to obtain a scalar from an array

    def compute_arr_perimeter(self, arr):
        return np.sum(arr[1:, :] != arr[:-1, :]) * self.cell_width + np.sum(
            arr[:, 1:] != arr[:, :-1]) * self.cell_height

    def compute_arr_edge(self, arr):
        """
        Computes the edge of a feature considering the landscape background in
        order to exclude the edges between the feature and nodata values
        """
        # check self.nodata in class_arr?
        class_cond = arr != self.nodata
        # class_with_bg_arr = np.copy(self.landscape_arr)
        # class_with_bg_arr[~class_cond] = self.landscape_arr[~class_cond]
        # get a 'boolean-like' integer array where one indicates that the cell
        # corresponds to some class value whereas zero indicates that the cell
        # corresponds to a nodata value
        data_arr = (self.landscape_arr != self.nodata).astype(np.int8)

        # use a convolution to determine which edges should be exluded from the
        # perimeter's width and height
        perimeter_width = np.sum(arr[1:, :] != arr[:-1, :]) + np.sum(
            ndimage.convolve(data_arr, KERNEL_VERTICAL)[class_cond] - 3)
        perimeter_height = np.sum(arr[:, 1:] != arr[:, :-1]) + np.sum(
            ndimage.convolve(data_arr, KERNEL_HORIZONTAL)[class_cond] - 3)

        return perimeter_width * self.cell_width + \
            perimeter_height * self.cell_height

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
        for i, patch_slice in enumerate(
                ndimage.find_objects(label_arr), start=1):
            patch_arr = np.pad(label_arr[patch_slice] == i, pad_width=1,
                               mode='constant',
                               constant_values=False)  # self.nodata

            patch_perimeters.append(self.compute_arr_perimeter(patch_arr))

        return patch_perimeters

    # compute metrics from area and perimeter series

    @staticmethod
    def compute_perimeter_area_ratio(area_ser, perimeter_ser):
        return perimeter_ser / area_ser

    @staticmethod
    def compute_shape_index(area_cells_ser, perimeter_cells_ser):
        n = np.floor(np.sqrt(area_cells_ser))
        m = area_cells_ser - n**2
        min_p = np.ones(len(area_cells_ser))
        min_p = np.where(m == 0, 4 * n, min_p)
        min_p = np.where(
            (n**2 < area_cells_ser) & (area_cells_ser <= n * (n + 1)),
            4 * n + 2, min_p)
        min_p = np.where(area_cells_ser > n * (n + 1), 4 * n + 4, min_p)

        return perimeter_cells_ser / min_p

    @staticmethod
    def compute_fractal_dimension(area_ser, perimeter_ser):
        return 2 * np.log(.25 * perimeter_ser) / np.log(area_ser)

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
                ]), name='class_val')

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
                ]), name='area')

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
                ]), name='perimeter')

            return self._cached_patch_perimeter_ser

    # metric distribution statistics

    def _metric_reduce(self, class_val, patch_metric_method,
                       patch_metric_method_kwargs, reduce_method):
        patch_metrics = patch_metric_method(class_val,
                                            **patch_metric_method_kwargs)
        if class_val is None:
            # ACHTUNG: dropping columns from a `pd.DataFrame` until leaving it
            # with only one column will still return a `pd.DataFrame`, so we
            # must convert to `pd.Series` manually (e.g., with `iloc`)
            patch_metrics = patch_metrics.drop('class_val', axis=1).iloc[:, 0]

        return reduce_method(patch_metrics)

    def _metric_mn(self, class_val, patch_metric_method,
                   patch_metric_method_kwargs={}):
        return self._metric_reduce(class_val, patch_metric_method,
                                   patch_metric_method_kwargs, np.mean)

    def _metric_am(self, class_val, patch_metric_method,
                   patch_metric_method_kwargs={}):
        # `area` can be `pd.Series` or `pd.DataFrame`
        area = self.area(class_val)

        if class_val is None:
            area = area['area']

        return self._metric_reduce(class_val, patch_metric_method,
                                   patch_metric_method_kwargs,
                                   partial(np.average, weights=area))

    def _metric_md(self, class_val, patch_metric_method,
                   patch_metric_method_kwargs={}):
        return self._metric_reduce(class_val, patch_metric_method,
                                   patch_metric_method_kwargs, np.median)

    def _metric_ra(self, class_val, patch_metric_method,
                   patch_metric_method_kwargs={}):
        return self._metric_reduce(class_val, patch_metric_method,
                                   patch_metric_method_kwargs,
                                   lambda ser: ser.max() - ser.min())

    def _metric_sd(self, class_val, patch_metric_method,
                   patch_metric_method_kwargs={}):
        return self._metric_reduce(class_val, patch_metric_method,
                                   patch_metric_method_kwargs, np.std)

    def _metric_cv(self, class_val, patch_metric_method,
                   patch_metric_method_kwargs={}, percent=True):
        metric_cv = self._metric_reduce(class_val, patch_metric_method,
                                        patch_metric_method_kwargs,
                                        stats.variation)
        if percent:
            metric_cv *= 100

        return metric_cv

    ###########################################################################
    # patch-level metrics

    # area and edge metrics

    def area(self, class_val=None, hectares=True):
        """

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed for the corresponding
            class only, otherwise it will be computed for all the classes of
            the landscape
        hectares : bool, default True
            whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        area : pd.Series if `class_val` is provided, pd.DataFrame otherwise
            area > 0, without limit
        """

        class_ser = self._patch_class_ser
        # ACHTUNG: very important to copy to ensure that we do not modify the
        # 'area' values if converting to hectares nor we return a variable
        # with the reference to the property `self._patch_areas_df`
        area_ser = self._patch_area_ser.copy()

        if hectares:
            area_ser /= 10000

        if class_val:
            return area_ser[class_ser == class_val]
        else:
            return pd.DataFrame({'class_val': class_ser, 'area': area_ser})

    def perimeter(self, class_val=None):
        """

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed for the corresponding
            class only, otherwise it will be computed for all the classes of
            the landscape

        Returns
        -------
        perim : pd.Series if `class_val` is provided, pd.DataFrame otherwise
            perim > 0, without limit
        """

        class_ser = self._patch_class_ser
        perimeter_ser = self._patch_perimeter_ser

        if class_val:
            return perimeter_ser[class_ser == class_val]
        else:
            return pd.DataFrame({
                'class_val': class_ser,
                'perimeter': perimeter_ser
            })

    # shape

    def perimeter_area_ratio(self, class_val=None, hectares=True):
        """

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed for the corresponding
            class only, otherwise it will be computed for all the classes of
            the landscape
        hectares : bool, default True
            whether the area should be converted to hectares (tends to yield
            more legible values for the metric)

        Returns
        -------
        para : pd.Series if `class_val` is provided, pd.DataFrame otherwise
            para > 0, without limit
        """

        # it is a bit silly to define a method in the `metric_utils` module to
        # compute the perimeter area ratio since it is a simple division
        area = self.area(class_val, hectares)
        perimeter = self.perimeter(class_val)

        if class_val:
            # both `perimeter` and `area` are `pd.Series`
            return Landscape.compute_perimeter_area_ratio(area, perimeter)
        else:
            # both `perimeter` and `area` are `pd.DataFrame`
            return pd.DataFrame({
                'class_val':
                area['class_val'],
                'perimeter_area_ratio':
                Landscape.compute_perimeter_area_ratio(area['area'],
                                                       perimeter['perimeter'])
            })

    def shape_index(self, class_val=None):
        """

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed for the corresponding
            class only, otherwise it will be computed for all the classes of
            the landscape

        Returns
        -------
        shape : pd.Series if `class_val` is provided, pd.DataFrame otherwise
            shape >= 1, without limit ; shape equals 1 when the patch
            is maximally compact, and increases without limit as patch shape
            becomes more regular
        """

        area = self.area(class_val, False)
        perimeter = self.perimeter(class_val)

        if class_val:
            # both `perimeter` and `area` are `pd.Series`
            if self.cell_width != self.cell_height:
                # this is rare and not even supported in FRAGSTATS. We could
                # calculate the perimeter in terms of cell counts in a
                # dedicated function and then adjust for a square standard,
                # but I believe it is not worth the effort. So we will just
                # return the base formula without adjusting for the square
                # standard
                return .25 * perimeter / np.sqrt(area)
            else:
                # we could also divide by `self.cell_height`
                return Landscape.compute_shape_index(
                    area / self.cell_area, perimeter / self.cell_width)

        else:
            # both `perimeter` and `area` are `pd.DataFrame`
            if self.cell_width != self.cell_height:
                # see comment above
                shape_index_ser = .25 * perimeter['perimeter'] / np.sqrt(
                    area['area'])
            else:
                shape_index_ser = pd.Series(
                    Landscape.compute_shape_index(
                        area['area'] / self.cell_area,
                        perimeter['perimeter'] / self.cell_width),
                    index=area.index)

            return pd.DataFrame({
                'class_val': area['class_val'],
                'shape_index': shape_index_ser
            })

    def fractal_dimension(self, class_val=None):
        """

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed for the corresponding
            class only, otherwise it will be computed for all the classes of
            the landscape


        Returns
        -------
        frac : pd.Series if `class_val` is provided, pd.DataFrame otherwise
            1 <= frac <=2 ; for a two-dimensional patch, frac approaches 1 for
            very simple shapes such as squares, and approaches 2 for complex
            plane-filling shapes
        """

        area = self.area(class_val, hectares=False)
        perimeter = self.perimeter(class_val)

        if class_val:
            # both `perimeter` and `area` are `pd.Series`
            return Landscape.compute_fractal_dimension(area, perimeter)
        else:
            # both `perimeter` and `area` are `pd.DataFrame`
            return pd.DataFrame({
                'class_val':
                area['class_val'],
                'fractal_dimension':
                Landscape.compute_fractal_dimension(area['area'],
                                                    perimeter['perimeter'])
            })

    def continguity_index(self, patch_arr):
        """

        Parameters
        ----------
        patch_arr :

        Returns
        -------
        contig : float
            0 <= contig <= 1 ; contig equals 0 for a one-pixel
            patch and increases to a limit of 1 as patch contiguity increases
        """

        # TODO
        raise NotImplementedError

    # aggregation metrics (formerly isolation, proximity)

    def euclidean_nearest_neighbor(self, patch_arr):
        """

        Parameters
        ----------
        patch_arr :

        Returns
        -------
        enn : float
            enn > 0, without limit ; enn approaches 0 as the distance to the
            nearest neighbors decreases
        """

        # TODO
        raise NotImplementedError

    def proximity(self, patch_arr, neighborhood):
        """

        Parameters
        ----------
        patch_arr :
        neighborhood :

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
        See also the documentation of `Landscape.area`

        Parameters
        ----------
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            whether the area should be converted to hectares (tends to yield
            more legible values for the metric)

        Returns
        -------
        ta : float
        """

        if class_val:
            return np.sum(self.area(class_val, hectares))
        else:
            total_area = self.landscape_area

            if hectares:
                total_area /= 10000

            return total_area

    def proportion_of_landscape(self, class_val, percent=True):
        """

        Parameters
        ----------
        class_val :
        percent : bool, default True
            whether the index should be expressed as proportion or converted
            to percentage. If True, this method returns FRAGSTATS' percentage
            of landscape (PLAND)

        Returns
        -------
        pland : float
            0 < pland <= 100 ; pland approaches 0 when the occurrence of the
            corresponding class becomes increasingly rare, and approaches 100
            when the entire landscape consists of a single patch of such class.
        """

        # whether this computes a class or landscape level metric will be
        # dealt within the `total_area` method according to the value of the
        # `class_val` argument
        numerator = self.total_area(class_val, hectares=False)
        if percent:
            numerator *= 100

        return numerator / self.landscape_area

    def number_of_patches(self, class_val=None):
        """

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        np : int
            np >= 1
        """
        if class_val:
            num_patches = self._num_patches_dict[class_val]
        else:
            num_patches = np.sum(list(self._num_patches_dict.values()))

        return num_patches

    def patch_density(self, class_val=None, percent=True, hectares=True):
        """

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        percent : bool, default True
            whether the index should be expressed as proportion or converted
            to percentage
        hectares : bool, default True
            whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        pd : float
            pd > 0, constrained by cell size ; maximum pd is attained when
            every cell is a separate patch
        """

        numerator = self.number_of_patches(class_val)
        if percent:
            numerator *= 100
        if hectares:
            numerator *= 10000

        return numerator / self.landscape_area

    def largest_patch_index(self, class_val=None, percent=True):
        """

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
        lpi : float
            0 < lpi <= 100 (or 0 < lpi <= 1 if percent argument is False) ;
            lpi approaches 0 when the largest patch of the corresponding class
            is increasingly small, and approaches its maximum value when such
            largest patch comprises the totality of the landscape
        """

        area = self.area(class_val, hectares=False)

        if class_val:
            numerator = np.max(area)
        else:
            numerator = np.max(area['area'])

        if percent:
            numerator *= 100

        return numerator / self.landscape_area

    def total_edge(self, class_val=None, count_boundary=False):
        """

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        count_boundary : bool, default False
            whether the boundary of the landscape should be included in the
            total edge length

        Returns
        -------
        te : float
            te >= 0 ; te equals 0 when the entire landscape and its border
            consist of the corresponding class
        """

        if class_val:
            if count_boundary:
                # then the total edge is just the sum of the perimeters of all
                # the patches of the corresponding class
                total_edge = np.sum(self.perimeter(class_val))
            else:
                total_edge = self.compute_arr_edge(
                    self.landscape_arr == class_val)
        else:
            if count_boundary:
                total_edge = self.compute_arr_perimeter(
                    np.pad(self.landscape_arr, pad_width=1, mode='constant',
                           constant_values=self.nodata))
            else:
                total_edge = self.compute_arr_edge(self.landscape_arr)

        return total_edge

    def edge_density(self, class_val=None, count_boundary=False,
                     hectares=True):
        """

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        count_boundary : bool, default False
            whether the boundary of the landscape should be included in the
            total edge length
        hectares : bool, default True
            whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        ed : float
            ed >= 0, without limit ; ed equals 0 when the entire landscape and
            its border consist of the corresponding patch class.
            Units: meters of edge per hectare/square meter.
        """

        numerator = self.total_edge(class_val=class_val,
                                    count_boundary=count_boundary)

        if hectares:
            numerator *= 10000

        return numerator / self.landscape_area

    def area_mn(self, class_val=None, hectares=True):
        """
        See also the documentation of `Landscape.area`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        area_mn : float
        """

        return self._metric_mn(class_val, self.area, {'hectares': hectares})

    def area_am(self, class_val=None, hectares=True):
        """
        See also the documentation of `Landscape.area`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        area_am : float
        """

        return self._metric_am(class_val, self.area, {'hectares': hectares})

    def area_md(self, class_val=None, hectares=True):
        """
        See also the documentation of `Landscape.area`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        area_md : float
        """

        return self._metric_md(class_val, self.area, {'hectares': hectares})

    def area_ra(self, class_val=None, hectares=True):
        """
        See also the documentation of `Landscape.area`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        area_ra : float
        """

        return self._metric_ra(class_val, self.area, {'hectares': hectares})

    def area_sd(self, class_val=None, hectares=True):
        """
        See also the documentation of `Landscape.area`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        area_sd : float
        """

        return self._metric_sd(class_val, self.area, {'hectares': hectares})

    def area_cv(self, class_val=None, percent=True):
        """
        See also the documentation of `Landscape.area`

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

    def landscape_shape_index(self, class_val=None):
        """

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        lsi : float
            lsi >=1 ; lsi equals 1 when the entire landscape consists of a
            single patch of the corresponding class, and increases without
            limit as the patches of such class become more disaggregated.
        """

        area = self.total_area(class_val, False)
        perimeter = self.total_edge(class_val, count_boundary=True)

        if self.cell_width != self.cell_height:
            # this is rare and not even supported in FRAGSTATS. We could
            # calculate the perimeter in terms of cell counts in a dedicated
            # function and then adjust for a square standard, but I believe it
            # is not worth the effort. So we will just return the base formula
            # without adjusting for the square standard
            return .25 * perimeter / np.sqrt(area)
        else:
            # we could also divide by `self.cell_height`
            return Landscape.compute_shape_index(
                [area / self.cell_area], [perimeter / self.cell_width])[0]

    # shape

    def perimeter_area_ratio_mn(self, class_val=None, hectares=True):
        """
        See also the documentation of `Landscape.perimeter_area_ratio`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        para_mn : float
        """

        return self._metric_mn(class_val, self.perimeter_area_ratio,
                               {'hectares': hectares})

    def perimeter_area_ratio_am(self, class_val=None, hectares=True):
        """
        See also the documentation of `Landscape.perimeter_area_ratio`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        para_am : float
        """

        return self._metric_am(class_val, self.perimeter_area_ratio,
                               {'hectares': hectares})

    def perimeter_area_ratio_md(self, class_val=None, hectares=True):
        """
        See also the documentation of `Landscape.perimeter_area_ratio`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        para_md : float
        """

        return self._metric_md(class_val, self.perimeter_area_ratio,
                               {'hectares': hectares})

    def perimeter_area_ratio_ra(self, class_val=None, hectares=True):
        """
        See also the documentation of `Landscape.perimeter_area_ratio`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        para_ra : float
        """

        return self._metric_ra(class_val, self.perimeter_area_ratio,
                               {'hectares': hectares})

    def perimeter_area_ratio_sd(self, class_val=None, hectares=True):
        """
        See also the documentation of `Landscape.perimeter_area_ratio`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        para_sd : float
        """

        return self._metric_sd(class_val, self.perimeter_area_ratio,
                               {'hectares': hectares})

    def perimeter_area_ratio_cv(self, class_val=None, percent=True):
        """
        See also the documentation of `Landscape.perimeter_area_ratio`

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
        para_cv : float
        """

        return self._metric_cv(class_val, self.perimeter_area_ratio,
                               percent=percent)

    def shape_index_mn(self, class_val=None):
        """
        See also the documentation of `Landscape.shape_index`

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
        See also the documentation of `Landscape.shape_index`

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
        See also the documentation of `Landscape.shape_index`

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
        See also the documentation of `Landscape.shape_index`

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
        See also the documentation of `Landscape.shape_index`

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
        See also the documentation of `Landscape.shape_index`

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
        shape_cv : float
        """

        return self._metric_cv(class_val, self.shape_index, percent=percent)

    def fractal_dimension_mn(self, class_val=None):
        """
        See also the documentation of `Landscape.fractal_dimension`

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
        See also the documentation of `Landscape.fractal_dimension`

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
        See also the documentation of `Landscape.fractal_dimension`

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
        See also the documentation of `Landscape.fractal_dimension`

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
        See also the documentation of `Landscape.fractal_dimension`

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
        See also the documentation of `Landscape.fractal_dimension`

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

        # TODO
        # label_arr, num_patches = self.class_label(class_val)

        # if num_patches == 0:
        #     return np.nan
        # elif num_patches < 2:
        #     return 0
        # else:
        #     I, J = np.nonzero(label_arr)
        #     labels = label_arr[I, J]
        #     coords = np.column_stack((I, J))

        #     sorter = np.argsort(labels)
        #     labels = labels[sorter]
        #     coords = coords[sorter]

        #     sq_dists = cdist(coords, coords, 'sqeuclidean')

        #     start_idx = np.flatnonzero(np.r_[1, np.diff(labels)])
        #     nonzero_vs_feat = np.minimum.reduceat(
        #         sq_dists, start_idx, axis=1)
        #     feat_vs_feat = np.minimum.reduceat(nonzero_vs_feat, start_idx,
        #                                        axis=0)

        #     # Get lower triangle and zero distances to nan
        #     b = np.tril(np.sqrt(feat_vs_feat))
        #     b[b == 0] = np.nan

        #     # Calculate mean and multiply with cellsize
        #     return np.nanmean(b) * self.cell_area
        raise NotImplementedError

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

        # TODO
        raise NotImplementedError

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

        # TODO
        raise NotImplementedError

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

        # TODO
        raise NotImplementedError

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
        enn_sd :
        """

        # TODO
        raise NotImplementedError

    def euclidean_nearest_neighbor_cv(self, class_val=None):
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
        enn_cv : float
        """

        # TODO
        raise NotImplementedError

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
            whether the index should be expressed as proportion or converted
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

    ###########################################################################
    # landscape-level metrics

    # contagion, interspersion

    def contagion(self, percent=True):
        """
        Parameters
        ----------
        percent : bool, default True
            whether the index should be expressed as proportion or converted
            to percentage

        Returns
        -------
        cont : float
            0 < contag <= 100 ; contag approaches 0 when the classes are
            maximally disaggregated (i.e., every cell is a patch of a
            different class) and interspersed (i.e., equal proportions of all
            pairwise adjacencies), and approaches its maximum when the
            landscape consists of a single patch.
        """

        # TODO
        raise NotImplementedError

    # diversity

    def shannon_diversity_index(self):
        """

        Returns
        -------
        shdi : float
            shdi >= 0 ; shdi approaches 0 when the entire landscape consists
            of a single patch, and increases as the number of classes
            increases and/or the proportional distribution of area among
            classes becomes more equitable.
        """

        # TODO
        raise NotImplementedError

    def patch_metrics_df(self, metrics=None):

        if metrics is None:
            metrics = Landscape.PATCH_METRICS
        # so far we do not support metric-wise kwargs in this method, so we
        # only work with FRAGSTATS defaults. More customized metrics might be
        # computed individually with their dedicated method

        # df = getattr(self, patch_metrics[0])()

        # for patch_metric in patch_metrics[1:]:
        #     df = pd.concat(
        #         [df, getattr(self, patch_metric)().drop('class_val', 1)],
        #         axis=1)
        # return df

        try:
            # in order to avoid adding a duplicate 'class_val' column for each
            # metric, we drop the 'class_val' column of each metric DataFrame
            # except for the first
            metric = metrics[0]
            metrics_dfs = [getattr(self, metric)()]
            for metric in metrics[1:]:
                metrics_dfs.append(
                    getattr(self, metric)().drop('class_val', axis=1))

        except AttributeError:
            raise ValueError("{metric} is not among {Landscape.PATCH_METRICS}")

        df = pd.concat(metrics_dfs, axis=1)  # [['class_val'] + patch_metrics]
        df.index.name = 'patch_id'

        return df

    def class_metrics_df(self, metrics=None):

        if metrics is None:
            metrics = Landscape.CLASS_METRICS

        try:
            metrics_sers = []
            for metric in metrics:
                metrics_sers.append(
                    pd.Series({
                        class_val: getattr(self, metric)(class_val)
                        for class_val in self.classes
                    }, name=metric))
        except AttributeError:
            raise ValueError("{metric} is not among {metrics}".format(
                metric=metric, metrics=Landscape.CLASS_METRICS))

        df = pd.concat(metrics_sers, axis=1)
        df.index.name = 'class_val'

        return df

    def landscape_metrics_df(self, metrics=None):

        if metrics is None:
            metrics = Landscape.LANDSCAPE_METRICS

        try:
            metrics_dict = {}
            for metric in metrics:
                metrics_dict[metric] = getattr(self, metric)()

        except AttributeError:
            raise ValueError("{metric} is not among {metrics}".format(
                metric=metric, metrics=Landscape.LANDSCAPE_METRICS))

        return pd.DataFrame(metrics_dict, index=[0])


def read_geotiff(fp, nodata=0, **kwargs):
    """
    See also the documentation of `rasterio.open`

    Parameters
    ----------
    fp : str, file object or pathlib.Path object
        A filename or URL, a file object opened in binary ('rb') mode,
        or a Path object. It will be passed to `rasterio.open`
    nodata : int, float, or nan; default 0
        Defines the pixel value to be interpreted as not valid data.
    **kwargs : optional
        Keyword arguments to be passed to `rasterio.open`

    Returns
    -------
    result : Landscape
    """
    with rasterio.open(fp, nodata=nodata, **kwargs) as src:
        ls_arr = src.read(1)
        res = src.res

    return Landscape(ls_arr, res, nodata=nodata)
