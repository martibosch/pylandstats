"""Zonal analysis."""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio import features, mask
from shapely import geometry
from shapely.geometry import base as geometry_base

try:
    from fiona import errors as fiona_errors
except ImportError:
    fiona_errors = None

from . import multilandscape
from .landscape import Landscape

__all__ = ["ZonalAnalysis", "BufferAnalysis", "ZonalGridAnalysis"]

ZONES_READ_ERRORS = [AttributeError, TypeError]
if fiona_errors is not None:
    ZONES_READ_ERRORS.append(fiona_errors.DriverError)
ZONES_READ_ERRORS = tuple(ZONES_READ_ERRORS)

_compute_zonal_statistics_gdf_doc = """
Compute the zonal statistics geo-data frame over the landscape raster.

Parameters
----------
metrics : list-like, optional
    A list-like of strings with the names of the metrics that should be computed. If
    `None`, all the implemented metrics at the specified level will be computed.
level : {{'class', 'landscape'}}, optional
    Whether the metrics should be computed at the class or landscape level. If `None`,
    the metrics will be computed (a) at the class level when a non-None `classes` is
    provided, otherwise (b) at the landscape level.
class_val : int, optional
    If provided, the metric will be computed at the level of the corresponding class,
    otherwise it will be computed at the landscape level.
metrics_kwargs : dict, optional
    Dictionary mapping the keyword arguments (values) that should be passed to
    each metric method (key), e.g., to exclude the boundary from the computation
    of `total_edge`, metric_kwargs should map the string 'total_edge' (method name)
    to {{'count_boundary': False}}. If `None`, each metric will be computed
    according to FRAGSTATS defaults.

Returns
-------
zonal_statistics_gdf : geopandas.GeoDataFrame
    Geo-data frame with the computed zonal statistics, with the zones as rows and
    {col_return} as columns.
"""


class ZonalAnalysis(multilandscape.MultiLandscape):
    """Zonal analysis."""

    def __init__(
        self,
        landscape_filepath,
        zones,
        *,
        zone_index=None,
        zone_nodata=0,
        neighborhood_rule=None,
    ):
        """Initialize the zonal analysis.

        Parameters
        ----------
        landscape_filepath : str, file-like object or pathlib.Path object
            A string/file-like object/pathlib.Path object with the landscape data.
        zones : geopandas.GeoSeries, geopandas.GeoDataFrame, list-like, str, \
            file-like object, pathlib.Path object, numpy.ndarray, optional
            This can either be:

            * A geopandas geo-series or geo-data frame.
            * A list-like of shapely geometries, in the CRS of the landscape.
            * A filename or URL, a file-like object opened in binary ('rb') mode, or a
              Path object that will be passed to `geopandas.read_file`.
            * A numpy array of the same shape as the landscape raster image, where each
              zone is labelled by a unique integer value. The values will be used to
              identify the zones (i.e., as index) - unless a different `zone_index` is
              provided.
        zone_index : list-like or str, optional
            Index used to identify zones (i.e., index of the metrics data frames). This
            can either be:

            * A list-like of index labels that will be positionally mapped to each zone.
            * A str with the name of a column only when `zones` is a geo-data frame or a
              geo-data frame file, e.g., a shapefile.
        zone_nodata : numeric, optional, default 0
            Value of the `zones` array that corresponds to no data. Only considered if
            `zones` is a numpy array of integer types.
        neighborhood_rule : {'8', '4'}, optional
            Neighborhood rule to determine patch adjacencies, i.e: '8' (queen's
            case/Moore neighborhood) or '4' (rook's case/Von Neumann neighborhood).
            Ignored if `landscape` is a `Landscape` instance. If no value is provided,
            the default value set in `settings.DEFAULT_NEIGHBORHOOD_RULE` will be taken.
        """
        # the overall approach to process the `zones` argument is: unless `zones` is
        # provided as a geo-series or a list-like of shapely geometries, we first
        # convert it to a geo-data frame, process it (mainly try to get the proper
        # index) and then convert it to a geo-series to store it as instance attribute

        # first, try to read the `zones` argument as a geo-data frame-like file
        try:
            zones = gpd.read_file(zones)
        except ZONES_READ_ERRORS:
            # Depending on the system and installed libraries, geopandas may raise an
            # `AttributeError`, `TypeError` or `fiona.errors.DriverError`. we let this
            # continue and try to read the `zones` argument differently below
            pass

        with rio.open(landscape_filepath) as src:
            if isinstance(zones, np.ndarray):
                # zones is an ndarray labelling each zone by a unique integer value
                # we first instantiate a geo-data frame because we will use the labels
                # as zone ids
                zones = gpd.GeoDataFrame(
                    [
                        (val, geometry.shape(geom))
                        for geom, val in features.shapes(zones, transform=src.transform)
                        if val != zone_nodata
                    ],
                    columns=["zone", "geometry"],
                    crs=src.crs,
                )
                if zone_index is None:
                    # if no zone indexing is provided, we will use the zone labels as
                    # index
                    # zones = zones.set_index("zone")
                    # instead of using `set_index`, we will just set `zone_index` to the
                    # column name "zone", so that the `set_index` is called when
                    # processing the geo-data frame below
                    zone_index = "zone"

            # at this point, unless `zones` was provided as geo series or list-like of
            # shapely geometries, `zones` must be a geo-data frame
            if isinstance(zones, gpd.GeoDataFrame):
                # if there is a non-None `zone_index`, use it
                if zone_index is not None:
                    # we get the index after calling `set_index` because this will give
                    # us the right index both when `zone_index` is a column name or a
                    # list-like.
                    # note that if zone_index is a list, pandas will try to interpret
                    # its values as column names, so we need to convert it to a numpy
                    # array/pandas series first so that the values are set as index.
                    # we will convert it to a pandas series so that we can set a name.
                    if isinstance(zone_index, list):
                        zone_index = pd.Series(zone_index, name="zone")
                    zone_index = zones.set_index(zone_index).index
                    # we now take just the "geometry" column and treat `zones` as
                    # GeoSeries.
                    zones = zones.geometry
                else:
                    # take just the "geometry" column, treat `zones` as GeoSeries but
                    # rename the index to "zone"
                    zones = zones.geometry.rename_axis("zone")

            # at this point, `zones` must be a geo-series or a list-like of shapely
            # geometries
            if not isinstance(zones, gpd.GeoSeries):
                # convert to a geo-series with the CRS of the landscape raster
                zones = gpd.GeoSeries(zones, crs=src.crs)
            else:
                # `zones` must be a geo-series, reproject into the CRS of the landscape
                # if needed
                if zones.crs != src.crs:
                    zones = zones.to_crs(src.crs)
            if zone_index is not None:
                # set the geo-series index
                zones = zones.set_axis(zone_index)

            # now that we have a processed geo series, store it as an instance attribute
            self.zone_gser = zones

            # now perform a masked read of the raster for the zone geometries
            landscapes = [
                Landscape(
                    zone_arr[0],
                    res=src.res,
                    nodata=src.nodata,
                    transform=zone_transform,
                    neighborhood_rule=neighborhood_rule,
                )
                for zone_arr, zone_transform in [
                    mask.mask(src, [geom], crop=True) for geom in zones
                ]
            ]

            # start: raster-based alternative ------------------------------------------
            # rasterize vector features into a labeled array and use it to generate the
            # zone landscapes
            # if types.is_list_like(zones):
            #     # if isinstance(zones[0], (geometry.Polygon, geometry.MultiPolygon)):
            #     zones = features.rasterize(
            #     [
            #         (geom, attribute_val)
            #         for attribute_val, geom in zip(
            #             attribute_values, zones.to_crs(src.crs)
            #         )
            #     ],
            #     out_shape=src.shape,
            #     transform=landscape_transform,
            #     fill=src.nodata,
            # )

            # # finally, zones is a labeled array
            # landscape_arr = src.read(1)
            # landscapes = [
            #     Landscape(landscape_arr[loc]) for loc in ndimage.find_objects(zones)
            # ]
            # end: raster-based alternative --------------------------------------------

        # now call the parent's init
        # for the parent class (MultiLandscape), set:
        # * `attribute_name` (by order of preference): (i) the series' index name if
        #    non-None, (ii) the series' name if non-None, or (iii) provide a default
        # * `attribute_values`: the index of the zones geo-series
        super().__init__(
            landscapes,
            self.zone_gser.index.name or self.zone_gser.name or "zone",
            self.zone_gser.index.values,
        )

    def compute_zonal_statistics_gdf(  # noqa: D102
        self,
        *,
        metrics=None,
        class_val=None,
        metrics_kwargs=None,
    ):
        if class_val is not None:
            zonal_metrics_df = self.compute_class_metrics_df(
                metrics=metrics, classes=[class_val], metrics_kwargs=metrics_kwargs
            ).loc[class_val]
        else:
            zonal_metrics_df = self.compute_landscape_metrics_df(
                metrics=metrics, metrics_kwargs=metrics_kwargs
            )

        # ensure that we have numeric types (not strings)
        # metric_ser = pd.to_numeric(metric_ser)

        # return a geo-data frame
        zone_name = self.zone_gser.index.name
        return gpd.GeoDataFrame(
            zonal_metrics_df.pivot_table(
                index=zone_name,
                columns=zonal_metrics_df.index.names.difference([zone_name]),
            ),
            geometry=self.zone_gser,
        )

    compute_zonal_statistics_gdf.__doc__ = _compute_zonal_statistics_gdf_doc.format(
        col_return="metrics"
    )


class BufferAnalysis(ZonalAnalysis):
    """Buffer analysis around a feature of interest."""

    def __init__(
        self,
        landscape_filepath,
        base_geom,
        buffer_dists,
        *,
        buffer_rings=False,
        base_geom_crs=None,
        neighborhood_rule=None,
    ):
        """Initialize the buffer analysis.

        Parameters
        ----------
        landscape_filepath : str, file-like object or pathlib.Path object
            A string/file-like object/pathlib.Path object with the landscape data.
        base_geom : shapely geometry, geopandas.GeoSeries or geopandas.GeoDataFrame
            Geometry that will serve as a base to buffer around.
        buffer_dists : list-like
            Buffer distances, in units of the landscape CRS.
        buffer_rings : bool, default Falsenn
            If `False`, each buffer zone will consist of the whole region that lies
            within the respective buffer distance around the base geometry. If `True`,
            buffer zones will take the form of rings around the base geometry.
        base_geom_crs : str, dict or pyproj.CRS, optional
            The coordinate reference system of the base geometry. Required if the base
            geometry is a shapely geometry or a geopandas GeoSeries without the `crs`
            attribute set.
        neighborhood_rule : {'8', '4'}, optional
            Neighborhood rule to determine patch adjacencies, i.e: '8' (queen's
            case/Moore neighborhood) or '4' (rook's case/Von Neumann neighborhood).
            Ignored if `landscape` is a `Landscape` instance. If no value is provided,
            the default value set in `settings.DEFAULT_NEIGHBORHOOD_RULE` will be taken.
        """
        # 1. get a GeoSeries with the base geometry
        if isinstance(base_geom, geometry_base.BaseGeometry):
            if base_geom_crs is None:
                raise ValueError(
                    "If `base_geom` is a shapely geometry, `base_geom_crs` must be"
                    " provided."
                )
            base_gser = gpd.GeoSeries(base_geom, crs=base_geom_crs)
        else:
            # we assume that `base_geom` is a geopandas GeoSeries/GeoDataFrame
            if isinstance(base_geom, gpd.GeoDataFrame):
                # in this case, we just select the geometry column and then treat it
                # like a geo series
                base_geom = base_geom.geometry
            if base_geom.crs is None:
                if base_geom_crs is None:
                    raise ValueError(
                        "If `base_geom` is a naive geopandas GeoSeries/GeoDataFrame"
                        " (with no crs set), `base_geom_crs` must be provided."
                    )

                base_gser = base_geom.copy()  # avoid alias/ref problems
                base_gser.crs = base_geom_crs
            else:
                base_gser = base_geom

        # 3. buffer around base mask
        avg_longitude = base_gser.to_crs(
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
        base_proj_geom = base_gser.to_crs(utm_crs).iloc[0]
        if buffer_rings:
            if not isinstance(base_proj_geom, geometry.Point):
                raise ValueError(
                    "Buffer rings can only work when `base_geom` is a `Point`."
                )
            _buffer_dists = np.concatenate([[0], buffer_dists])
            buffer_dists = list(
                map(
                    lambda d: "{}-{}".format(d[0], d[1]),
                    zip(_buffer_dists[:-1], _buffer_dists[1:]),
                )
            )
            zone_gser = gpd.GeoSeries(
                [
                    base_proj_geom.buffer(_buffer_dists[i + 1])
                    - base_proj_geom.buffer(_buffer_dists[i])
                    for i in range(len(_buffer_dists) - 1)
                ],
                index=buffer_dists,
                crs=utm_crs,
            )
        else:
            zone_gser = gpd.GeoSeries(
                [base_proj_geom.buffer(buffer_dist) for buffer_dist in buffer_dists],
                index=buffer_dists,
                crs=utm_crs,
            )

        # now we can call the parent's init with the landscape and the constructed
        # buffer geoseries
        super().__init__(
            landscape_filepath,
            zones=zone_gser.rename_axis("buffer_dist"),  # set the index name
            neighborhood_rule=neighborhood_rule,
        )

    # override docs
    def compute_class_metrics_df(  # noqa: D102
        self, *, metrics=None, classes=None, metrics_kwargs=None, fillna=None
    ):
        return super().compute_class_metrics_df(
            metrics=metrics,
            classes=classes,
            metrics_kwargs=metrics_kwargs,
            fillna=fillna,
        )

    compute_class_metrics_df.__doc__ = (
        multilandscape._compute_class_metrics_df_doc.format(
            index_descr="multi-indexed by the class and buffer distance",
            index_return="class, buffer distance (multi-index)",
        )
    )

    def compute_landscape_metrics_df(  # noqa: D102
        self, *, metrics=None, metrics_kwargs=None
    ):
        return super().compute_landscape_metrics_df(
            metrics=metrics, metrics_kwargs=metrics_kwargs
        )

    compute_landscape_metrics_df.__doc__ = (
        multilandscape._compute_landscape_metrics_df_doc.format(
            index_descr="indexed by the buffer distance",
            index_return="buffer distance (index)",
        )
    )


class ZonalGridAnalysis(ZonalAnalysis):
    """Zonal analysis over a grid."""

    @staticmethod
    def _get_grid_gser(
        bounds, num_zone_rows, num_zone_cols, zone_width, zone_height, offset
    ):
        # get the zone dimensions to generate the grid
        left, bottom, right, top = bounds

        total_width = right - left
        total_height = top - bottom

        # make sure that we have both the number of zone rows/columns and the zone
        # width/height
        if zone_width is None:
            try:
                zone_width = np.ceil(total_width / num_zone_cols)
            except TypeError:
                # num_zone_cols is also None
                raise ValueError(
                    "Either `num_zone_cols` or `zone_width` must be provided"
                )
        if zone_height is None:
            try:
                zone_height = np.ceil(total_height / num_zone_rows)
            except TypeError:
                # num_zone_rows is also None
                raise ValueError(
                    "Either `num_zone_rows` or `zone_height` must be provided"
                )
        if num_zone_cols is None:
            num_zone_cols = int(np.ceil(total_width / zone_width))

        if num_zone_rows is None:
            num_zone_rows = int(np.ceil(total_height / zone_height))

        # once we have the number of zone rows/columns and the zone width/height, we can
        # compute the grid
        if offset == "center":
            # center the grid on the raster bounds
            left = left - (num_zone_cols * zone_width - total_width) / 2
            top = top + (num_zone_rows * zone_height - total_height) / 2

        # generate a grid of size using numpy meshgrid
        grid_x, grid_y = np.meshgrid(
            np.arange(num_zone_cols) * zone_width + left,
            top - np.arange(num_zone_rows) * zone_height,
            indexing="xy",
        )

        # vectorize the grid as a geo series
        flat_grid_x = grid_x.flatten()
        flat_grid_y = grid_y.flatten()
        zones = pd.DataFrame(
            {
                "xmin": flat_grid_x,
                "ymin": flat_grid_y - zone_height,
                "xmax": flat_grid_x + zone_width,
                "ymax": flat_grid_y,
            }
        ).apply(lambda row: geometry.box(*row), axis=1)

        # # identify zones as their (row, col) position
        # zone_ids = np.array(
        #     [(row, col)
        #      for row in range(num_zone_rows) for col in range(num_zone_cols)]
        # )
        # return grid_gser.set_index(zone_ids)

        return zones

    def __init__(
        self,
        landscape_filepath,
        *,
        num_zone_cols=None,
        num_zone_rows=None,
        zone_width=None,
        zone_height=None,
        offset=None,
        neighborhood_rule=None,
    ):
        """Initialize the zonal grid analysis.

        Parameters
        ----------
        landscape_filepath : str, file-like object or pathlib.Path object
            A string/file-like object/pathlib.Path object with the landscape data.
        num_zone_cols, num_zone_rows : int, optional
            The number of zone columns/rows into which the landscape will be separated.
            If the landscape dimensions and the desired zones do not divide evenly, the
            zones will be defined for the minimum superset that covers the landscape
            bounds. If not provided, then `num_width`/`num_height` must be provided.
        zone_width, zone_height : numeric, optional
            The width/height of each zone (in units of the landscape CRS). If the
            landscape dimensions and the desired zones do not divide evenly, the zones
            will be defined for the minimum superset that covers the landscape bounds.
            If not provided, then `num_width`/`num_height` must be provided.
        offset : str, optional
            If set to "center", the and the landscape dimensions and the desired zones
            do not divide evenly, the grid is offsetted so that the landscape bounds are
            in the center of the grid. Otherwise, the grid starts at the top-left corner
            of the landscape. Ignored if the landscape dimensions and the desired zones
            divide evenly.
        neighborhood_rule : {'8', '4'}, optional
            Neighborhood rule to determine patch adjacencies, i.e: '8' (queen's
            case/Moore neighborhood) or '4' (rook's case/Von Neumann neighborhood). If
            no value is provided, the default value set in
            `settings.DEFAULT_NEIGHBORHOOD_RULE` will be taken.
        """
        with rio.open(landscape_filepath) as src:
            zone_gser = gpd.GeoSeries(
                ZonalGridAnalysis._get_grid_gser(
                    src.bounds,
                    num_zone_rows,
                    num_zone_cols,
                    zone_width,
                    zone_height,
                    offset,
                ),
                crs=src.crs,
            )

            # filter out zones that do not meet any valid data pixel
            def has_valid_data(geom):
                zone_arr, _ = mask.mask(src, [geom], crop=True)
                return np.any(zone_arr != src.nodata)

            zone_gser = zone_gser[zone_gser.apply(has_valid_data)]

        # now we can call the parent's init with the landscape and the constructed grid
        # geoseries
        super().__init__(
            landscape_filepath,
            zones=zone_gser.rename_axis("grid_cell"),  # set the index name
            neighborhood_rule=neighborhood_rule,
        )

    def plot_landscapes(self, *, cmap=None, ax=None, figsize=None, **plot_kwargs):
        """Plot the spatial distribution of the landscape zones.

        Parameters
        ----------
        cmap : str or `~matplotlib.colors.Colormap`, optional
            A Colormap instance.
        ax : axis object, optional
            Plot in given axis; if None creates a new figure.
        figsize : tuple of two numeric types, optional
            Size of the figure to create. Ignored if axis `ax` is provided.
        **plot_kwargs : optional
            Keyword arguments to be passed to `geopandas.GeoSeries.plot`.

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

        if plot_kwargs is None:
            plot_kwargs = {}

        gpd.GeoDataFrame(
            {"color": np.arange(len(self.zone_gser))}, geometry=self.zone_gser
        ).plot("color", ax=ax, cmap=cmap, **plot_kwargs)

        return ax
