from __future__ import division

from functools import partial

import numpy as np
from scipy import ndimage, stats

from . import settings

# from scipy.spatial.distance import cdist

__all__ = ['Landscape']

KERNEL_HORIZONTAL = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.int8)
KERNEL_VERTICAL = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.int8)
KERNEL_MOORE = ndimage.generate_binary_structure(2, 2)


class Landscape:
    """Documentation for Landscape

    """

    def __init__(self, landscape_arr, res, nodata=0,
                 use_cache=settings.USE_LANDSCAPE_CACHE):
        self.landscape_arr = landscape_arr
        self.cell_width, self.cell_height = res
        self.cell_area = res[0] * res[1]
        self.nodata = nodata
        classes = np.array(sorted(np.unique(landscape_arr)))
        classes = np.delete(classes, nodata)
        classes = classes[~np.isnan(classes)]
        self.classes = classes

        self.use_cache = use_cache
        if use_cache:
            # cache
            self._class_arr_dict = dict()
            self._label_dict = dict()
            self._patch_areas_dict = dict()
            self._patch_perimeters_dict = dict()
            # TODO: worth caching perimeter area ratio, fractal dimension...?
            # self._patch_perimter_area_ratios_dict = dict()

    ###########################################################################
    # common utilities

    # compute methods to obtain a scalar from an array

    def _compute_patch_area(self, patch_arr, cell_counts=False):
        if self.nodata == 0:
            # ~ x8 times faster
            area = np.count_nonzero(patch_arr)
        else:
            area = np.sum(patch_arr != self.nodata)

        if not cell_counts:
            area *= self.cell_area

        return area

    def _compute_patch_perimeter(self, patch_arr, cell_counts=False):
        arr = np.pad(patch_arr, pad_width=1, mode='constant',
                     constant_values=False)  # self.nodata

        perimeter_width = np.sum(arr[1:, :] != arr[:-1, :])
        perimeter_height = np.sum(arr[:, 1:] != arr[:, :-1])

        if not cell_counts:
            perimeter_width *= self.cell_width
            perimeter_height *= self.cell_height

        return perimeter_width + perimeter_height

    def _compute_patch_perimeter_area_ratio(self, patch_arr):
        return self._compute_patch_perimeter(
            patch_arr) / self._compute_patch_area(patch_arr)

    def _compute_patch_shape_index(self, patch_arr):
        area = self._compute_patch_area(patch_arr, cell_counts=False)
        # the method below ensures that every adjacency, even within a class
        # value and nodata (the landscape boundary) is counted as patch
        # perimeter. This is also how it is done in FRAGSTATS.
        perimeter = self._compute_patch_perimeter(patch_arr, cell_counts=False)

        # n := size of the smallest containing integer square
        n = np.floor(np.sqrt(area))
        m = area - n**2
        if m == 0:
            min_perimeter = 4 * n
        elif n**2 < area < n * (n + 1):
            min_perimeter = 4 * n + 2
        else:  # assert `area > n * (n + 1)`
            min_perimeter = 4 * n + 4

        return perimeter / min_perimeter

    def _compute_patch_fractal_dimension(self, patch_arr):
        return 2 * np.log(
            .25 * self._compute_patch_perimeter(patch_arr)) / np.log(
                self._compute_patch_area(patch_arr))

    def _compute_class_area(self, class_arr, cell_counts=False):
        return self._compute_patch_area(class_arr, cell_counts=cell_counts)

    def _compute_class_perimeter(self, class_arr, cell_counts=False,
                                 count_boundary=False):
        perimeter_width = np.sum(class_arr[1:, :] != class_arr[:-1, :])
        perimeter_height = np.sum(class_arr[:, 1:] != class_arr[:, :-1])

        if not count_boundary:
            # check self.nodata in class_arr?
            class_cond = class_arr != self.nodata
            # class_with_bg_arr = np.copy(self.landscape_arr)
            # class_with_bg_arr[~class_cond] = self.landscape_arr[~class_cond]
            # get a 'boolean-like' integer array where one indicates that the
            # cell corresponds to some class value whereas zero indicates that
            # the cell corresponds to a nodata value
            data_arr = (self.landscape_arr != self.nodata).astype(np.int8)

            perimeter_width += np.sum(
                ndimage.convolve(data_arr, KERNEL_VERTICAL)[class_cond] - 3)
            perimeter_height += np.sum(
                ndimage.convolve(data_arr, KERNEL_HORIZONTAL)[class_cond] - 3)

        if not cell_counts:
            perimeter_width *= self.cell_width
            perimeter_height *= self.cell_height

        return perimeter_width + perimeter_height

    def _compute_landscape_area(self, cell_counts=False):
        return self._compute_patch_area(self.landscape_arr,
                                        cell_counts=cell_counts)

    # compute methods to obtain class and patch-label arrays

    def _compute_class_arr(self, class_val):
        return self.landscape_arr == class_val

    def _compute_class_label(self, class_arr):
        # This returns a tuple with `label_arr` and `num_patches`
        # TODO: parameter for Von Neumann adjacency?
        # Moore neighborhood
        return ndimage.label(class_arr, KERNEL_MOORE)

    # compute methods to obtain patchwise scalars

    def _compute_patch_scalars(self, label_arr, method):
        # TODO: static method, or put in utils file
        # abstract method to map a value to each patch of `label_arr`
        # `patch_values` as np.array of fixed size with
        # `patch_values[i] = ...` within the loop is slower and less Pythonic
        # but can lead to better performances if optimized via Cython/numba
        patch_values = []
        # for patch_slice in ndimage.find_objects(label_arr):
        #     patch_values.append(method(label_arr[patch_slice]))
        # `ndimage.find_objects` only finds the (rectangular) bounds; there
        # might be parts of other patches within such bounds, so we need to
        # check which pixels correspond to the patch of interest. Since
        # `ndimage.label` labels patches with an enumeration starting by 1, we
        # can use Python's built-in `enumerate`
        for i, patch_slice in enumerate(
                ndimage.find_objects(label_arr), start=1):
            patch_values.append(method(label_arr[patch_slice] == i))
        return np.array(patch_values, dtype=np.float)

    def _compute_patch_areas(self, label_arr):
        # could use `_compute_patch_scalars`, but `np.bincount` is much faster
        return np.bincount(label_arr.ravel())[1:] * self.cell_area

    def _compute_patch_perimeters(self, label_arr):
        return self._compute_patch_scalars(label_arr,
                                           self._compute_patch_perimeter)

    def _compute_patch_perimeter_area_ratios(self, label_arr):
        return self._compute_patch_scalars(
            label_arr, self._compute_patch_perimeter_area_ratio)

    def _compute_patch_shape_indices(self, label_arr):
        return self._compute_patch_scalars(label_arr,
                                           self._compute_patch_shape_index)

    def _compute_patch_fractal_dimensions(self, label_arr):
        return self._compute_patch_scalars(
            label_arr, self._compute_patch_fractal_dimension)

    # cache of class-level arrays and lists of patchwise scalars

    def _get_from_cache_or_compute(self, class_val, cache_dict_name,
                                   compute_method, compute_method_args):
        # Assume that we do not pass kwargs, because we only cache FRAGSTATS'
        # defaults, which correspond to the methods' default kwarg values
        if self.use_cache:
            cache_dict = getattr(self, cache_dict_name)
            try:
                return cache_dict[class_val]
            except KeyError:
                element = compute_method(*compute_method_args)
                cache_dict[class_val] = element
                return element
        else:
            return compute_method(*compute_method_args)

    def _get_class_arr(self, class_val):
        return self._get_from_cache_or_compute(
            class_val, '_class_arr_dict', self._compute_class_arr, [class_val])

    def _get_label_arr(self, class_val):
        class_arr = self._get_class_arr(class_val)
        return self._get_from_cache_or_compute(class_val, '_label_dict',
                                               self._compute_class_label,
                                               [class_arr])[0]

    def _get_num_patches(self, class_val):
        class_arr = self._get_class_arr(class_val)
        return self._get_from_cache_or_compute(class_val, '_label_dict',
                                               self._compute_class_label,
                                               [class_arr])[1]

    def _get_patch_areas(self, class_val):
        label_arr = self._get_label_arr(class_val)
        return self._get_from_cache_or_compute(class_val, '_patch_areas_dict',
                                               self._compute_patch_areas,
                                               [label_arr])

    def _get_patch_perimeters(self, class_val):
        label_arr = self._get_label_arr(class_val)
        return self._get_from_cache_or_compute(
            class_val, '_patch_perimeters_dict',
            self._compute_patch_perimeters, [label_arr])

    def _get_patch_perimeter_area_ratios(self, class_val):
        label_arr = self._get_label_arr(class_val)
        return self._compute_patch_perimeter_area_ratios(label_arr)

    def _get_patch_shape_indices(self, class_val):
        label_arr = self._get_label_arr(class_val)
        return self._compute_patch_shape_indices(label_arr)

    def _get_patch_fractal_dimensions(self, class_val):
        # label_arr = self._get_label_arr(class_val)
        # return self._get_from_cache_or_compute(
        #     class_val, '_patch_fractal_dimensions_dict',
        #     self._compute_patch_fractal_dimensions, [label_arr])
        label_arr = self._get_label_arr(class_val)
        return self._compute_patch_fractal_dimensions(label_arr)

    @property
    def landscape_area(self):
        try:
            return self._landscape_area
        except AttributeError:
            self._landscape_area = self._compute_landscape_area()
            return self._landscape_area

    # metric distribution statistics

    def _metric_reduce(
            self,
            class_val,
            get_patch_scalars_method,
            patch_reduce_method,
    ):
        if class_val:
            patch_scalars = get_patch_scalars_method(class_val)
        else:
            patch_scalars = np.concatenate([
                get_patch_scalars_method(_class_val)
                for _class_val in self.classes
            ])

        return patch_reduce_method(patch_scalars)

    def _metric_mn(self, class_val, get_patch_scalars_method, hectares=False):
        metric_mn = self._metric_reduce(class_val, get_patch_scalars_method,
                                        np.mean)

        if hectares:
            metric_mn /= 10000

        return metric_mn

    def _metric_am(self, class_val, get_patch_scalars_method, hectares=False):
        if class_val:
            patch_areas = self._get_patch_areas(class_val)
        else:
            patch_areas = np.concatenate([
                self._get_patch_areas(_class_val)
                for _class_val in self.classes
            ])

        metric_am = self._metric_reduce(
            class_val, get_patch_scalars_method,
            partial(np.average, weights=patch_areas))

        if hectares:
            metric_am /= 10000

        return metric_am

    def _metric_sd(self, class_val, get_patch_scalars_method, hectares=False):
        metric_sd = self._metric_reduce(class_val, get_patch_scalars_method,
                                        np.std)

        if hectares:
            metric_sd /= 10000

        return metric_sd

    def _metric_cv(self, class_val, get_patch_scalars_method, percent=True):
        metric_cv = self._metric_reduce(class_val, get_patch_scalars_method,
                                        stats.variation)

        if percent:
            metric_cv *= 100

        return metric_cv

    ###########################################################################
    # patch-level metrics

    # area and edge metrics

    def area(self, patch_arr, hectares=True):
        """

        Parameters
        ----------
        patch_arr :
        hectares : bool, default True
            whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        area : float
            area > 0, without limit
        """

        area = self._compute_patch_area(patch_arr)

        if hectares:
            area /= 10000

        return area

    def perimeter(self, patch_arr):
        """

        Parameters
        ----------
        patch_arr :

        Returns
        -------
        perim : float
            perim > 0, without limit
        """

        # the default arguments are already `pad=True` and
        # `count_boundary=True`, which ensures that every adjacency, even
        # within a class value and nodata (the landscape boundary) is counted
        # as patch perimeter. This is also how it is done in FRAGSTATS
        return self._compute_patch_perimeter(patch_arr)

    # shape

    def perimeter_area_ratio(self, patch_arr):
        """

        Parameters
        ----------
        patch_arr :

        Returns
        -------
        para : float
            para > 0, without limit
        """

        # self.area(patch_arr, hectares=False) / self.perimeter(patch_arr)
        return self._compute_patch_perimeter_area_ratio(patch_arr)

    def shape_index(self, patch_arr):
        """

        Parameters
        ----------
        patch_arr :

        Returns
        -------
        shape : float
            shape >= 1, without limit ; shape equals 1 when the patch
            is maximally compact, and increases without limit as patch shape
            becomes more regular
        """

        return self._compute_patch_shape_index(patch_arr)

    def fractal_dimension(self, patch_arr):
        """

        Parameters
        ----------
        patch_arr :

        Returns
        -------
        frac : float
            1 <= frac <=2 ; for a two-dimensional patch, frac approaches 1 for
            very simple shapes such as squares, and approaches 2 for complex
            plane-filling shapes
        """

        return self._compute_patch_fractal_dimension(patch_arr)

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
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level
        hectares : bool, default True
            whether the landscape area should be converted to hectares (tends
            to yield more legible values for the metric)

        Returns
        -------
        ta : float
        """

        if class_val:
            total_area = np.sum(self._get_patch_areas(class_val))
        else:
            # this is safe to do for the `hectares` division below as far as
            # Python aliases are concerned
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

        if class_val:
            num_patches = self._get_num_patches(class_val)
        else:
            num_patches = np.sum([
                self._get_num_patches(_class_val)
                for _class_val in self.classes
            ])

        numerator = num_patches
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

        if class_val:
            patch_areas = self._get_patch_areas(class_val)
        else:
            patch_areas = np.concatenate([
                self._get_patch_areas(_class_val)
                for _class_val in self.classes
            ])

        numerator = np.max(patch_areas)
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
            # Alternative: check performance, check if same result. In any
            # case, it makes sense to use the cache methods, since patchwise
            # computations (even if less performant) might have already been
            # performed, or might be useful later
            # class_arr = self._get_class_arr(class_val)
            # total_edge = self._compute_arr_perimeter(class_arr)
            if count_boundary:
                # then the total edge is just the sum of the perimeters of all
                # the patches of the corresponding class
                total_edge = np.sum(self._get_patch_perimeters(class_val))
            else:
                total_edge = self._compute_class_perimeter(
                    self._get_class_arr(class_val))
        else:
            landscape_arr = np.copy(self.landscape_arr)
            if count_boundary:
                landscape_arr = np.pad(landscape_arr, pad_width=1,
                                       mode='constant',
                                       constant_values=self.nodata)
            total_edge = self._compute_class_perimeter(
                landscape_arr, count_boundary=count_boundary)

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

        return self._metric_mn(class_val, self._get_patch_areas, hectares)

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

        return self._metric_am(class_val, self._get_patch_areas, hectares)

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

        return self._metric_sd(class_val, self._get_patch_areas, hectares)

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

        return self._metric_cv(class_val, self._get_patch_areas,
                               percent=percent)

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

        if class_val:
            # total_edge = self.total_edge(class_val=class_val,
            #                              count_boundary=True)
            # return .25 * total_edge / np.sqrt(self.landscape_area)
            class_arr = self._get_class_arr(class_val)
            area = self._compute_class_area(class_arr, cell_counts=True)
            perimeter = self._compute_class_perimeter(
                class_arr, cell_counts=True, count_boundary=True)

            n = np.floor(np.sqrt(area))
            m = area - n**2
            if m == 0:
                min_perimeter = 4 * n
            elif n**2 < area < n * (n + 1):
                min_perimeter = 4 * n + 2
            else:  # assert `area > n * (n + 1)`
                min_perimeter = 4 * n + 4

            return perimeter / min_perimeter
        else:
            landscape_arr = np.pad(self.landscape_arr, pad_width=1,
                                   mode='constant',
                                   constant_values=self.nodata)
            perimeter = self._compute_class_perimeter(
                landscape_arr, cell_counts=True, count_boundary=True)
            area = self._compute_landscape_area(cell_counts=True)

            n = np.floor(np.sqrt(area))
            m = area - n**2
            if m == 0:
                min_perimeter = 4 * n
            elif n**2 < area < n * (n + 1):
                min_perimeter = 4 * n + 2
            else:  # assert `area > n * (n + 1)`
                min_perimeter = 4 * n + 4

            return perimeter / min_perimeter

    # shape

    def perimeter_area_ratio_mn(self, class_val=None):
        """
        See also the documentation of `Landscape.perimeter_area_ratio`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        para_mn : float
        """

        return self._metric_mn(class_val,
                               self._get_patch_perimeter_area_ratios)

    def perimeter_area_ratio_am(self, class_val=None):
        """
        See also the documentation of `Landscape.perimeter_area_ratio`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        para_an : float
        """

        return self._metric_am(class_val,
                               self._get_patch_perimeter_area_ratios)

    def perimeter_area_ratio_sd(self, class_val=None):
        """
        See also the documentation of `Landscape.perimeter_area_ratio`

        Parameters
        ----------
        class_val : int, optional
            If provided, the metric will be computed at the level of the
            corresponding class, otherwise it will be computed at the
            landscape level

        Returns
        -------
        para_sd : float
        """

        return self._metric_sd(class_val,
                               self._get_patch_perimeter_area_ratios)

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

        return self._metric_cv(
            class_val, self._get_patch_perimeter_area_ratios, percent=percent)

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

        return self._metric_mn(class_val, self._get_patch_shape_indices)

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

        return self._metric_am(class_val, self._get_patch_shape_indices)

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

        return self._metric_sd(class_val, self._get_patch_shape_indices)

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

        return self._metric_cv(class_val, self._get_patch_shape_indices,
                               percent=percent)

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

        return self._metric_mn(class_val, self._get_patch_fractal_dimensions)

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

        return self._metric_am(class_val, self._get_patch_fractal_dimensions)

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

        return self._metric_sd(class_val, self._get_patch_fractal_dimensions)

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

        return self._metric_cv(class_val, self._get_patch_fractal_dimensions,
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

    def continguity_index_aw(self, class_val=None):
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
        contig_aw : float
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

    def proximity_aw(self, class_val=None):
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
        prox_aw : float
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
        # label_arr = self._get_label_arr(class_val)
        # num_patches = self._get_num_patches(class_val)

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

    def euclidean_nearest_neighbor_aw(self, class_val=None):
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
        enn_aw : float
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
