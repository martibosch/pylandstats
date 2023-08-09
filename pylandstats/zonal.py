"""Zonal analysis."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from numpy.lib import stride_tricks
from rasterio import features

from . import landscape as pls_landscape
from . import multilandscape

try:
    import geopandas as gpd
    from shapely import geometry
    from shapely.geometry import base as geometry_base

    geo_imports = True
except ImportError:
    geo_imports = False

__all__ = ["ZonalAnalysis", "BufferAnalysis", "ZonalGridAnalysis"]


class ZonalAnalysis(multilandscape.MultiLandscape):
    """Zonal analysis."""

    def __init__(
        self,
        landscape,
        masks_arr=None,
        landscape_crs=None,
        landscape_transform=None,
        attribute_name=None,
        attribute_values=None,
        masks=None,
        masks_index_col=None,
        neighborhood_rule=None,
    ):
        """
        Initialize the zonal analysis.

        Parameters
        ----------
        landscape : `Landscape` or str, file-like object or pathlib.Path object
            A `Landscape` object or string/file-like object/pathlib.Path object that
            will be passed as the `landscape` argument of `Landscape.__init__`.
        masks_arr : list-like or numpy.ndarray, optional
            A list-like of numpy arrays of shape (width, height), i.e., of the same
            shape as the landscape raster image. Each array will serve to mask the
            base landscape and define a region of study for which the metrics will be
            computed separately. The same information can also be provided as a single
            array of shape (num_masks, width, height). Ignored if `masks` is provided.
        landscape_crs : str, dict or pyproj.CRS, optional
            The coordinate reference system of the landscapes. Used to dump rasters in
            the `compute_zonal_statistics_arr` method. Ignored if the passed-in
            `landscape` is a path to a raster dataset that already contains such
            information.
        landscape_transform : affine.Affine
            Transformation from pixel coordinates to coordinate reference system. Used
            to dump rasters in the `compute_zonal_statistics_arr` method. Ignored if
            the passed-in `landscape` is a path to a raster dataset that already
            contains such information.
        attribute_name : str, optional
            Name of the attribute that will distinguish each landscape.
        attribute_values : str, optional
            Values of the attribute that correspond to each of the landscapes.
        masks : list-like, numpy.ndarray, geopandas.GeoSeries, geopandas.GeoDataFrame, \
                str, file-like object or pathlib.Path object, optional
            This can either be:

            * A list-like of numpy arrays of shape (width, height), i.e., of the same
              shape as the landscape raster image. Each array will serve to mask the
              base landscape and define a region of study for which the metrics will be
              computed separately. The same information can also be provided as a single
              array of shape (num_masks, width, height).
            * A geopandas geo-series or geo-data frame.
            * A filename or URL, a file-like object opened in binary ('rb') mode, or a
              Path object that will be passed to `geopandas.read_file`.
        masks_index_col : str, optional
            Column of the `masks` geo-data frame that will be used as attribute values,
            i.e., index of the metrics data frames. Ignored if `masks` is not a geo-data
            frame or a geo-data frame file, e.g., a shapefile.
        neighborhood_rule : {'8', '4'}, optional
            Neighborhood rule to determine patch adjacencies, i.e: '8' (queen's
            case/Moore neighborhood) or '4' (rook's case/Von Neumann neighborhood).
            Ignored if `landscape` is a `Landscape` instance. If no value is provided
            and `landscape` is a file-like object or a path, the default value set in
            `settings.DEFAULT_NEIGHBORHOOD_RULE` will be taken.
        """
        # read input data/metadata
        if not isinstance(landscape, pls_landscape.Landscape):
            with rio.open(landscape) as src:
                landscape_crs = src.crs
            landscape = pls_landscape.Landscape(landscape)
        else:
            neighborhood_rule = landscape.neighborhood_rule
        landscape_arr = landscape.landscape_arr
        height, width = landscape_arr.shape
        if landscape.transform is not None:
            landscape_transform = landscape.transform

        # masks
        if masks_arr is not None:
            msg = (
                "The `masks_arr` parameter is deprecated and will be removed "
                "in a future version. Use the `masks` parameter instead"
            )
            warnings.warn(msg, FutureWarning)
        if masks is not None:
            if not geo_imports:
                # if geopandas is not installed, `masks` must be either a
                # list-like object or an ndarray - in both cases, an iterable
                try:
                    _ = iter(masks)
                except TypeError:
                    raise ImportError(
                        "If `masks` is not a list-like of numpy arrays or a "
                        "numpy array, it must be a `geopandas.GeoDataFrame` or"
                        " a vector-based spatial data file, which requires the"
                        " geopandas package."
                    )

                # rename the variable to `masks_arr` so that it is properly
                # used below
                masks_arr = masks
            else:
                if isinstance(masks, gpd.GeoSeries):
                    # if we have a GeoSeries, convert it to a GeoDataFrame so
                    # that we can use the same code
                    if attribute_name is None:
                        # if no attribute_name is provided, we will first see
                        # if there is a name in the geoseries or geoseries'
                        # index that we might use as attribute name
                        if masks.name:
                            attribute_name = masks.name
                        elif masks.index.name:
                            attribute_name = masks.index.name
                    masks = gpd.GeoDataFrame(geometry=masks, index=masks.index)
                    # since `masks_index_col` is meant to be a column of the
                    # geodataframe (or geodataframe file, e.g., shapefile)
                    # provided as the `masks` argument, if `masks` is a
                    # GeoSeries, such a column will not exist - we therefore
                    # set it to `None` so that we do not enter the respective
                    # "if" below and get errors
                    masks_index_col = None
                elif not isinstance(masks, gpd.GeoDataFrame):
                    try:
                        masks = gpd.read_file(masks)
                    except AttributeError:
                        # AttributeError: 'list'/'numpy.ndarray' object has no
                        # attribute 'startswith'
                        # we assume that `masks` is a list-like of numpy
                        # arrays or a numpy array, in which case we rename the
                        # variable to `masks_arr` so that it is properly used
                        # below
                        masks_arr = masks

                # at this point, `masks` can either be a GeoDataFrame (in
                # which case, we process it inside the "if" below) or a
                # list-like of numpy arrays/numpy array (in which case no
                # further pre-processing needs to be done)
                if isinstance(masks, gpd.GeoDataFrame):
                    # first of all, let us transform our geometries into the
                    # CRS of the landscape
                    try:
                        masks_gser = masks["geometry"].to_crs(landscape_crs)
                    except AttributeError as e:
                        # geopandas uses pyproj's `is_exact_same` method,
                        # which might return `False` for equivalent CRSs and
                        # raise "AttributeError: 'NoneType' object has no
                        # attribute 'is_empty'". To avoid that, we can try
                        # using the basic equality test for the CRSs and avoid
                        # reprojecting:
                        if masks.crs == landscape_crs:
                            masks_gser = masks["geometry"]
                        else:
                            raise e

                    # we first rasterize the geometries using the values of
                    # each geometry's index key in the raster
                    # to avoid confusing values used to map each zones (in
                    # `zone_arr`) with the landscape nodata value, we start
                    # with an array of zeros and use a sequence of integers
                    # starting at 1 to map each zone
                    zone_arr = features.rasterize(
                        shapes=(
                            (geom, val) for val, geom in enumerate(masks_gser, start=1)
                        ),
                        out_shape=landscape_arr.shape,
                        fill=0,
                        transform=landscape.transform,
                    )
                    # we now filter so that only the zone geometries that
                    # intersect the data region of our landscape are considered
                    # and also replace the zeros of `zone_arr` with the
                    # landscape nodata value
                    zone_arr = np.where(
                        (landscape_arr != landscape.nodata) & (zone_arr != 0),
                        zone_arr,
                        landscape.nodata,
                    )
                    # we now get all the non-nodata values (i.e., index keys of
                    # the GeoDataFrame) that intersect the data region of our
                    # landscape
                    zone_values = np.setdiff1d(np.unique(zone_arr), [landscape.nodata])
                    # we now transform `zone_arr` into a list of boolean masks
                    # that delineate the extent of each zone
                    masks_arr = [zone_arr == mask_id for mask_id in zone_values]

                    if masks_index_col is not None:
                        # to index the data frames of landscape metrics with
                        # the values of the `masks_index_col` of the
                        # GeoDataFrame (instead of the index keys), we set the
                        # `attribute_name` and `attribute_values` variables,
                        # which will be set as instance attributes below
                        attribute_name = masks_index_col
                        attribute_values = masks[masks_index_col].iloc[zone_values - 1]

        # we generate the landscapes of each zone here
        landscapes = [
            pls_landscape.Landscape(
                np.where(mask_arr, landscape_arr, landscape.nodata).astype(
                    landscape.landscape_arr.dtype
                ),
                res=(landscape.cell_width, landscape.cell_height),
                nodata=landscape.nodata,
                transform=landscape.transform,
                neighborhood_rule=neighborhood_rule,
            )
            for mask_arr in masks_arr
        ]

        # store `landscape_meta`/`masks_arr` as instance attributes so that we
        # can compute zonal statistics
        self.landscape_meta = dict(
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            transform=landscape_transform,
            crs=landscape_crs,
        )
        self.masks_arr = masks_arr
        # useful in `compute_zonal_statistics_arr` below
        self.filter_landscape_nodata = True

        # The attribute name will be `buffer_dists` for `BufferAnalysis` or
        # `transect_dist` for `TransectAnalysis`, but for any other custom use
        # of `ZonalAnalysis`, the user might provide (or not) a custom name
        if attribute_name is None:
            attribute_name = "attribute_values"

        # If the values for the distinguishing attribute are not provided, a
        # basic enumeration will be automatically generated
        if attribute_values is None:
            attribute_values = [i for i in range(len(masks_arr))]

        # now call the parent's init
        super().__init__(landscapes, attribute_name, attribute_values)

    def compute_zonal_statistics_arr(
        self,
        metric,
        class_val=None,
        metric_kws=None,
        dst_filepath=None,
        custom_meta=None,
    ):
        """
        Compute the zonal statistics of a metric over a landscape raster.

        Parameters
        ----------
        metric : str
            A string indicating the name of the metric for which the zonal statistics
            will be computed.
        class_val : int, optional
            If provided, the zonal statistics will be computed for the metric computed
            at the level of the corresponding class, otherwise they will be computed at
            the landscape level.
        metric_kws : dict, optional
            Keyword arguments to be passed to the method that computes the metric
            (specified in the `metric` argument) for each landscape.
        dst_filepath : str, file-like object or pathlib.Path object, optional
            Path to dump the zonal statistics raster. If not provided, no raster will be
            dumped.
        custom_meta : dict, optional
            Custom meta data for the output raster, consistent with the rasterio
            library.

        Returns
        -------
        zonal_statistics_arr : numpy.ndarray
            Two-dimensional array with the computed zonal statistics.
        """
        # ACHTUNG: do not confuse `metric_kws` and `metrics_kws`. The former
        # are the keyword arguments for the method to compute the metric. The
        # latter is a dict mapping the metric to such keyword argument (such
        # dict will be passed to the `compute_class_metrics_df`/
        # `compute_landscape_metrics_df` method)
        if metric_kws is None:
            metrics_kws = None
        else:
            metrics_kws = {metric: metric_kws}
        if class_val is None:
            zonal_metrics_df = self.compute_landscape_metrics_df(
                metrics=[metric], metrics_kws=metrics_kws
            )
            metric_ser = zonal_metrics_df[metric]
        else:
            zonal_metrics_df = self.compute_class_metrics_df(
                metrics=[metric], classes=[class_val], metrics_kws=metrics_kws
            )
            metric_ser = zonal_metrics_df.loc[class_val, metric]
        # ensure that we have numeric types (not strings)
        metric_ser = pd.to_numeric(metric_ser)

        # reconstruct the zonal statistics array
        zonal_statistics_arr = np.full(
            (self.landscape_meta["height"], self.landscape_meta["width"]),
            np.nan,
            dtype=metric_ser.dtype,
        )
        if self.filter_landscape_nodata:
            for metric_val, landscape, mask_arr in zip(
                metric_ser, self.landscapes, self.masks_arr
            ):
                zonal_statistics_arr[
                    (landscape.landscape_arr != landscape.nodata) & mask_arr
                ] = metric_val
        else:
            for metric_val, mask_arr in zip(metric_ser, self.masks_arr):
                zonal_statistics_arr[mask_arr] = metric_val

        # dump a raster
        if dst_filepath:
            dst_meta = self.landscape_meta.copy()
            dst_meta.update(dtype=zonal_statistics_arr.dtype)
            if custom_meta is None:
                dst_meta.update(nodata=np.nan)
            else:
                if "nodata" in custom_meta:
                    zonal_statistics_arr[np.isnan(zonal_statistics_arr)] = custom_meta[
                        "nodata"
                    ]
                dst_meta.update(**custom_meta)
            with rio.open(dst_filepath, "w", **dst_meta) as dst:
                dst.write(zonal_statistics_arr, 1)
        return zonal_statistics_arr


class BufferAnalysis(ZonalAnalysis):
    """Buffer analysis around a feature of interest."""

    def __init__(
        self,
        landscape,
        base_mask,
        buffer_dists,
        buffer_rings=False,
        base_mask_crs=None,
        landscape_crs=None,
        landscape_transform=None,
        neighborhood_rule=None,
    ):
        """
        Initialize the buffer analysis.

        Parameters
        ----------
        landscape : `Landscape` or str, file-like object or pathlib.Path object
            A `Landscape` object or of string/file-like object/pathlib.Path object that
            will be passed as the `landscape` argument of `Landscape.__init__`
        base_mask : shapely geometry or geopandas.GeoSeries
            Geometry that will serve as a base mask to buffer around.
        buffer_dists : list-like
            Buffer distances.
        buffer_rings : bool, default False
            If `False`, each buffer zone will consist of the whole region that lies
            within the respective buffer distance around the base mask. If `True`,
            buffer zones will take the form of rings around the base mask.
        base_mask_crs : str, dict or pyproj.CRS, optional
            The coordinate reference system of the base mask. Required if the base mask
            is a shapely geometry or a geopandas GeoSeries without the `crs` attribute
            set.
        landscape_crs : str, dict or pyproj.CRS, optional
            The coordinate reference system of the landscapes. Required if the passed-in
            landscapes are `Landscape` instances, ignored if they are paths to raster
            datasets that already contain such information.
        landscape_transform : affine.Affine
            Transformation from pixel coordinates to coordinate reference system.
            Required if the passed-in landscapes are `Landscape` instances, ignored if
            they are paths to raster datasets that already contain such information.
        neighborhood_rule : {'8', '4'}, optional
            Neighborhood rule to determine patch adjacencies, i.e: '8' (queen's
            case/Moore neighborhood) or '4' (rook's case/Von Neumann neighborhood).
            Ignored if `landscape` is a `Landscape` instance. If no value is provided
            and `landscape` is a file-like object or a path, the default value set in
            `settings.DEFAULT_NEIGHBORHOOD_RULE` will be taken.
        """
        # first check that we meet the package dependencies
        if not geo_imports:
            raise ImportError(
                "The `BufferAnalysis` class requires the 'geopandas' package."
            )

        # get `buffer_masks_arr` from a base geometry and a list of buffer
        # distances
        # 1. get a GeoSeries with the base mask geometry
        if isinstance(base_mask, geometry_base.BaseGeometry):
            if base_mask_crs is None:
                raise ValueError(
                    "If `base_mask` is a shapely geometry, `base_mask_crs` "
                    "must be provided"
                )
            # BufferSpatioTemporalAnalysis.get_buffer_masks_gser(
            base_mask_gser = gpd.GeoSeries(base_mask, crs=base_mask_crs)
        else:
            # we assume that `base_mask` is a geopandas GeoSeries
            if base_mask.crs is None:
                if base_mask_crs is None:
                    raise ValueError(
                        "If `base_mask` is a naive geopandas GeoSeries (with "
                        "no crs set), `base_mask_crs` must be provided"
                    )

                base_mask_gser = base_mask.copy()  # avoid alias/ref problems
                base_mask_gser.crs = base_mask_crs
            else:
                base_mask_gser = base_mask

        # 2. get the crs, transform and shape of the landscapes
        if isinstance(landscape, pls_landscape.Landscape):
            if landscape_crs is None:
                raise ValueError(
                    "If passing `Landscape` instances (instead of paths to "
                    "raster datasets), `landscape_crs` must be provided"
                )
            if landscape_transform is None:
                if landscape.transform is None:
                    raise ValueError(
                        "If passing `Landscape` instances (instead of paths to"
                        " raster datasets), either they have a non-None "
                        "`transform` attribute, either `landscape_transform` "
                        "must be provided"
                    )
                landscape_transform = landscape.transform
            landscape_shape = landscape.landscape_arr.shape
            # note that we DO NOT have to get `neighborhood_rule` from
            # `landscape` since this will be done when calling
            # `ZonalAnalysis.__init__` at the end of this method
        else:
            with rio.open(landscape) as src:
                landscape_crs = src.crs
                landscape_transform = src.transform
                landscape_shape = src.height, src.width

        # 3. buffer around base mask
        avg_longitude = base_mask_gser.to_crs(
            "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        ).unary_union.centroid.x
        # trick from OSMnx to be able to buffer in meters
        utm_zone = int(np.floor((avg_longitude + 180) / 6.0) + 1)
        # utm_crs = {
        #     'datum': 'WGS84',
        #     'ellps': 'WGS84',
        #     'proj': 'utm',
        #     'zone': utm_zone,
        #     'units': 'm'
        # }
        utm_crs = (
            f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 " "+units=m +no_defs"
        )
        base_mask_geom = base_mask_gser.to_crs(utm_crs).iloc[0]
        if buffer_rings:
            if not isinstance(base_mask_geom, geometry.Point):
                raise ValueError(
                    "Buffer rings can only work when `base_mask_geom` is a " "`Point`"
                )
            _buffer_dists = np.concatenate([[0], buffer_dists])
            buffer_dists = list(
                map(
                    lambda d: "{}-{}".format(d[0], d[1]),
                    zip(_buffer_dists[:-1], _buffer_dists[1:]),
                )
            )
            masks_gser = gpd.GeoSeries(
                [
                    base_mask_geom.buffer(_buffer_dists[i + 1])
                    - base_mask_geom.buffer(_buffer_dists[i])
                    for i in range(len(_buffer_dists) - 1)
                ],
                index=buffer_dists,
                crs=utm_crs,
            ).to_crs(landscape_crs)
        else:
            masks_gser = gpd.GeoSeries(
                [base_mask_geom.buffer(buffer_dist) for buffer_dist in buffer_dists],
                index=buffer_dists,
                crs=utm_crs,
            ).to_crs(landscape_crs)

        # 4. rasterize each mask
        num_rows, num_cols = landscape_shape
        buffer_masks_arr = np.zeros(
            (len(buffer_dists), num_rows, num_cols), dtype=np.uint8
        )
        for i in range(len(masks_gser)):
            buffer_masks_arr[i] = features.rasterize(
                [masks_gser.iloc[i]],
                out_shape=landscape_shape,
                transform=landscape_transform,
                dtype=np.uint8,
            )

        buffer_masks_arr = buffer_masks_arr.astype(bool)

        # now we can call the parent's init with the landscape and the
        # constructed buffer_masks_arr
        super().__init__(
            landscape,
            masks=buffer_masks_arr,
            landscape_crs=landscape_crs,
            landscape_transform=landscape_transform,
            attribute_name="buffer_dists",
            attribute_values=buffer_dists,
            neighborhood_rule=neighborhood_rule,
        )

    # override docs
    def compute_class_metrics_df(  # noqa: D102
        self, metrics=None, classes=None, metrics_kws=None, fillna=None
    ):
        return super().compute_class_metrics_df(
            metrics=metrics,
            classes=classes,
            metrics_kws=metrics_kws,
            fillna=fillna,
        )

    compute_class_metrics_df.__doc__ = (
        multilandscape._compute_class_metrics_df_doc.format(
            index_descr="multi-indexed by the class and buffer distance",
            index_return="class, buffer distance (multi-index)",
        )
    )

    def compute_landscape_metrics_df(  # noqa: D102
        self, metrics=None, metrics_kws=None
    ):
        return super().compute_landscape_metrics_df(
            metrics=metrics, metrics_kws=metrics_kws
        )

    compute_landscape_metrics_df.__doc__ = (
        multilandscape._compute_landscape_metrics_df_doc.format(
            index_descr="indexed by the buffer distance",
            index_return="buffer distance (index)",
        )
    )


class ZonalGridAnalysis(ZonalAnalysis):
    """Zonal analysis over a grid."""

    def __init__(
        self,
        landscape,
        num_zone_rows=None,
        num_zone_cols=None,
        zone_pixel_width=None,
        zone_pixel_height=None,
        landscape_crs=None,
        landscape_transform=None,
        neighborhood_rule=None,
    ):
        """
        Initialize the zonal grid analysis.

        Parameters
        ----------
        landscape : `Landscape` or str, file-like object or pathlib.Path object
            A `Landscape` object or of string/file-like object/pathlib.Path object that
            will be passed as the `landscape` argument of `Landscape.__init__`.
        num_zone_rows, num_zone_cols : int, optional
            The number of zone rows/columns into which the landscape will be separated.
            If the landscape dimensions and the desired zones do not divide evenly, the
            zones will be defined for the maximum subset (starting from the top, left
            corner) for which there is an even division. If not provided, then
            `num_pixel_width`/`num_pixel_height` must be provided.
        zone_pixel_width, zone_pixel_height : int, optional
            The width/height of each zone (in pixels). If the landscape dimensions and
            the desired zones do not divide evenly, the zones will be defined for the
            maximum subset (starting from the top, left corner) for which there is an
            even division. If not provided, then `num_zone_rows`/`num_zone_cols` must be
            provided.
        landscape_crs : str, dict or pyproj.CRS, optional
            The coordinate reference system of the landscapes. Required to reconstruct
            the zonal statistics rasters if the passed-in landscapes are `Landscape`
            instances, ignored if they are paths to raster datasets that already contain
            such information.
        landscape_transform : affine.Affine
            Transformation from pixel coordinates to coordinate reference system.
            Required if the passed-in landscapes are `Landscape` instances, ignored if
            they are paths to raster datasets that already contain such information.
        neighborhood_rule : {'8', '4'}, optional
            Neighborhood rule to determine patch adjacencies, i.e: '8' (queen's
            case/Moore neighborhood) or '4' (rook's case/Von Neumann neighborhood). If
            no value is provided, the value will be taken from `landscape` if it is an
            instance of `Landscape`, otherwise the default value set in
            `settings.DEFAULT_NEIGHBORHOOD_RULE` will be taken.
        """
        if not isinstance(landscape, pls_landscape.Landscape):
            with rio.open(landscape) as src:
                landscape_crs = src.crs
            landscape = pls_landscape.Landscape(landscape)
        else:
            # note that we DO HAVE to get the neighborhood from `landscape`
            # since we are bypassing the parent's (i.e., `ZonalAnalysis`)
            # initialization method at the end of this method
            neighborhood_rule = landscape.neighborhood_rule
        landscape_arr = landscape.landscape_arr
        height, width = landscape_arr.shape

        if zone_pixel_height is None:
            if num_zone_rows is None:
                raise ValueError(
                    "Either `num_zone_rows` or `zone_pixel_height` must be " "provided"
                )
            zone_pixel_height = height // num_zone_rows
        if zone_pixel_width is None:
            if num_zone_cols is None:
                raise ValueError(
                    "Either `num_zone_cols` or `zone_pixel_width` must be " "provided"
                )
            zone_pixel_width = width // num_zone_cols

        if num_zone_rows is None:
            num_zone_rows = height // zone_pixel_height
        if num_zone_cols is None:
            num_zone_cols = width // zone_pixel_width

        # raster meta
        # transform.from_origin(landscape_transform.c, landscape_transform.f)
        if landscape.transform is not None:
            landscape_transform = landscape.transform
        self.landscape_meta = dict(
            driver="GTiff",
            width=num_zone_cols,
            height=num_zone_rows,
            count=1,
            transform=landscape_transform
            * landscape_transform.scale(zone_pixel_width, zone_pixel_height),
            crs=landscape_crs,
        )

        # Based on `skimage.util.shape.view_as_blocks`
        # arr_shape = np.array([height, width])
        zone_shape = np.array([zone_pixel_height, zone_pixel_width])
        # num_even_rows, num_even_cols = arr_shape - arr_shape % zone_shape
        # landscape_arr[:num_even_rows, :num_even_cols]
        landscape_arrs = stride_tricks.as_strided(
            landscape_arr,
            # shape=tuple(arr_shape // zone_shape) + tuple(zone_shape),
            shape=(num_zone_rows, num_zone_cols) + tuple(zone_shape),
            strides=tuple(landscape_arr.strides * zone_shape) + landscape_arr.strides,
        )
        # the reshape could probably be done directly in the `as_strided` call
        # tuple(landscape_arrs.shape[0] * landscape_arrs.shape[1])
        landscape_arrs = landscape_arrs.reshape(
            (num_zone_cols * num_zone_rows,) + tuple(zone_shape)
        )
        # identify zones as their (row, col) position
        zone_ids = np.array(
            [(row, col) for row in range(num_zone_rows) for col in range(num_zone_cols)]
        )

        # check which zones actually contain only nans
        # nan_zones = np.full(len(masks), False)
        # for i, mask_arr in enumerate(masks):
        #     if np.any(landscape.landscape_arr[mask_arr] != landscape.nodata):
        #         nan_zones[i] = True
        # save this as instance attribute since we will need it to reconstruct
        # the zonal statistics raster
        self.data_zones = np.array(
            [
                np.any(landscape_arr != landscape.nodata)
                for landscape_arr in landscape_arrs
            ]
        )

        # We only need to consider zones that actually contain non-nan pixels
        landscapes = [
            pls_landscape.Landscape(
                landscape_arr,
                res=(landscape.cell_width, landscape.cell_height),
                nodata=landscape.nodata,
                neighborhood_rule=neighborhood_rule,
            )
            for landscape_arr in landscape_arrs[self.data_zones]
        ]
        zone_ids = list(map(tuple, zone_ids[self.data_zones]))

        # TODO: find a better way to DRY this (see comment just below)
        # build a list of numpy masks, each representing a grid cell of our
        # zonal analysis. Doing this here is rather silly, but it allows us to
        # re-use the `compute_zonal_statistics_arr` method of the
        # `ZonalAnalysis` class (at the expense of some performance loss,
        # though most-likely not too critical)
        # masks = []
        # # base_mask_arr = np.full((height, width), False)
        # for zone_row_start in range(0, height, zone_pixel_height):
        #     for zone_col_start in range(0, width, zone_pixel_width):
        #         # mask_arr = np.copy(base_mask_arr)
        #         mask_arr = np.full((height, width), False)
        #         mask_arr[zone_row_start:zone_row_start +
        #                  zone_pixel_height, zone_col_start:zone_col_start +
        #                  zone_pixel_width] = True
        #         masks.append(mask_arr)
        # # make it a numpy array, filter out the nan zones and store it as a
        # # class attribute
        # self.masks_arr = np.array(masks)[self.data_zones]
        masks = []
        for zone_rowcol in zone_ids:
            mask_arr = np.full(
                (self.landscape_meta["height"], self.landscape_meta["width"]),
                False,
            )
            mask_arr[zone_rowcol] = True
            masks.append(mask_arr)
        self.masks_arr = np.array(masks)

        # to reuse the `compute_zonal_statistics_arr` from `ZonalAnalysis`
        self.filter_landscape_nodata = False

        # Note that
        # # now we can call the parent's init with the landscape and the
        # # constructed masks. We only need to consider zones that actually
        # # contain non-nan pixels
        # zones = list(map(tuple, np.compress(nan_zones, zones, axis=0)))
        # super(ZonalGridAnalysis, self).__init__(
        #     landscape, np.compress(nan_zones, masks, axis=0),'zones', zones,
        #     crop_landscapes=False)

        # ACHTUNG: since we have built the landscapes here, we bypass the
        # parent's init (i.e., `ZonalAnalysis`), and call the grandparent's
        # init instead
        super(ZonalAnalysis, self).__init__(landscapes, "zones", zone_ids)

    def plot_landscapes(self, cmap=None, ax=None, figsize=None, **show_kws):
        """
        Plot the spatial distribution of the landscape zones.

        Parameters
        ----------
        cmap : str or `~matplotlib.colors.Colormap`, optional
            A Colormap instance.
        ax : axis object, optional
            Plot in given axis; if None creates a new figure.
        figsize : tuple of two numeric types, optional
            Size of the figure to create. Ignored if axis `ax` is provided.
        **show_kws : optional
            Keyword arguments to be passed to `rasterio.plot.show`.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Returns the `Axes` object with the plot drawn onto it.
        """
        if cmap is None:
            cmap = plt.rcParams["image.cmap"]

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_aspect("equal")

        if show_kws is None:
            show_kws = {}

        zone_arr = np.full_like(self.data_zones, np.nan, dtype=np.float32)
        zone_arr[self.data_zones] = np.random.random(np.sum(self.data_zones))

        ax.imshow(
            zone_arr.reshape(
                self.landscape_meta["height"], self.landscape_meta["width"]
            ),
            cmap=cmap,
            **show_kws,
        )

        return ax
