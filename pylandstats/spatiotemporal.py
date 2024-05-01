"""Spatio-temporal analysis."""

import functools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio import mask

from . import multilandscape, settings, zonal
from .landscape import Landscape

__all__ = [
    "SpatioTemporalAnalysis",
    "SpatioTemporalZonalAnalysis",
    "SpatioTemporalBufferAnalysis",
    "SpatioTemporalZonalGridAnalysis",
]

_plot_metric_doc = """Plot the time series of the metric across {zone_descr}s.

Parameters
----------
metric : str
    A string indicating the name of the metric to plot.
class_val : int, optional
    If provided, the metric will be plotted at the level of the corresponding class,
    otherwise it will be plotted at the landscape level.
ax : axis object, optional
    Plot in given axis; if None creates a new figure.
metric_legend : bool, default True
    Whether the metric label should be displayed within the plot (as label of the
    y-axis).
metric_label : str, optional
    Label of the y-axis to be displayed if `metric_legend` is `True`. If the provided
    value is `None`, the label will be taken from the `settings` module.
{zone_var_name}_legend : bool, default True
    Whether a legend linking each plotted line to a {zone_descr} should be displayed
    within the plot.
fmt : str, default '--o'
    A format string for `matplotlib.pyplot.plot`.
plot_kwargs : dict, default None
    Keyword arguments to be passed to `matplotlib.pyplot.plot`.
subplots_kwargs : dict, default None
    Keyword arguments to be passed to `matplotlib.pyplot.subplots` only if no axis is
    given (through the `ax` argument).
metric_kwargs : dict, default None
    Keyword arguments to be passed to the method that computes the metric (specified in
    the `metric` argument) for each landscape.

Returns
-------
ax : matplotlib.axes.Axes
    Returns the `Axes` object with the plot drawn onto it.
"""


class SpatioTemporalAnalysis(multilandscape.MultiLandscape):
    """Spatio-temporal analysis."""

    def __init__(
        self, landscapes, *, dates=None, neighborhood_rule=None, **landscape_kwargs
    ):
        """Initialize the spatio-temporal analysis.

        Parameters
        ----------
        landscapes : list-like
            A list-like of `Landscape` instances or of strings/file-like/pathlib.Path
            objects so that each is passed as the `landscape` argument of
            `Landscape.__init__`.
        dates : list-like, optional
            A list-like of ints or strings that label the date of each snapshot of
            `landscapes` (for DataFrame indices and plot labels).
        neighborhood_rule : {'8', '4'}, optional
            Neighborhood rule to determine patch adjacencies, i.e: '8' (queen's
            case/Moore neighborhood) or '4' (rook's case/Von Neumann neighborhood).
            Ignored if the passed-in landscapes are `Landscape` instances. If no value
            is provided and the passed-in landscapes are file-like objects or paths, the
            default value set in `settings.DEFAULT_NEIGHBORHOOD_RULE` will be taken.
        landscape_kwargs : dict, optional
            Other keyword arguments to be passed to the instantiation of
           `pylandstats.Landscape` for each element of `landscapes`. Ignored if the
            elements of `landscapes` are already instances of `pylandstats.Landcape`.
        """
        if dates is None:
            dates = ["t{}".format(i) for i in range(len(landscapes))]

        # pop the `neighborhood_rule` from `landscape_kwargs` (this is merely done so
        # that the `neighborhood_rule` argument is explicitly documented in this method
        _ = landscape_kwargs.pop("neighborhood_rule", None)
        # call the parent's init
        super().__init__(
            landscapes,
            "dates",
            dates,
            neighborhood_rule=neighborhood_rule,
            **landscape_kwargs,
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
            index_descr="multi-indexed by the class and date",
            index_return="class, date (multi-index)",
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
            index_descr="indexed by the date", index_return="date (index)"
        )
    )

    # def plot_patch_metric(metric):
    #     # TODO: sns distplot?
    #     fig, ax = plt.subplots()
    #     ax.hist()


class SpatioTemporalZonalAnalysis(SpatioTemporalAnalysis, zonal.ZonalAnalysis):
    """Spatio-temporal zonal analysis."""

    def __init__(
        self,
        landscape_filepaths,
        zones,
        *,
        zone_index=None,
        zone_nodata=None,
        dates=None,
        neighborhood_rule=None,
    ):
        """Initialize the spatio-temporal zonal analysis.

        Parameters
        ----------
        landscape_filepaths : list-like
            A list-like of strings/file-like/pathlib.Path objects so that each is passed
            as the `landscape` argument of `Landscape.__init__`.
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
        dates : list-like, optional
            A list-like of ints or strings that label the date of each snapshot of
            `landscapes` (for DataFrame indices and plot labels).
        neighborhood_rule : {'8', '4'}, optional
            Neighborhood rule to determine patch adjacencies, i.e: '8' (queen's
            case/Moore neighborhood) or '4' (rook's case/Von Neumann neighborhood).
            Ignored if `landscape` is a `Landscape` instance. If no value is provided,
            the default value set in `settings.DEFAULT_NEIGHBORHOOD_RULE` will be taken.
        """
        # we first instantiate only a zonal analysis to use its (after initialization)
        # attributes
        za = zonal.ZonalAnalysis(
            landscape_filepaths[0],
            zones,
            zone_index=zone_index,
            zone_nodata=zone_nodata,
            neighborhood_rule=neighborhood_rule,
        )

        # while `ZonalAnalysis.__init__` will set the `zone_gser` attribute to the
        # instantiated object (stored in the variable `ba`), it will not set it to the
        # current `SpatioTemporalZonalAnalysis`, so we need to do it here
        self.zone_gser = za.zone_gser

        # init the landscape instances
        landscapes = []
        for _, zone_geom in za.zone_gser.items():
            for landscape in landscape_filepaths:
                with rio.open(landscape) as src:
                    zone_arr, zone_transform = mask.mask(src, [zone_geom], crop=True)
                    landscapes.append(
                        Landscape(
                            zone_arr[0],
                            res=src.res,
                            transform=zone_transform,
                            neighborhood_rule=neighborhood_rule,
                        )
                    )

        # TODO: DRY this from `SpatioTemporalAnalysis.__init__`
        if dates is None:
            dates = ["t{}".format(i) for i in range(len(landscape_filepaths))]

        # instantiate the series of landscapes but using a multi-index
        self.landscape_ser = pd.Series(
            landscapes,
            index=pd.MultiIndex.from_product(
                [self.zone_gser.index, dates],
                names=[self.zone_gser.index.name, "date"],
            ),
        )

        # get the all classes present in the provided landscapes
        # TODO: DRY this from `MultiLandscape.__init__`
        self.present_classes = functools.reduce(
            np.union1d,
            tuple(landscape.classes for landscape in self.landscape_ser),
        )

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
            index_descr="multi-indexed by the zone, class and date",
            index_return="zone, class, distance (multi-index)",
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
            index_descr="multi-indexed by the buffer distance and date",
            index_return="buffer distance, date (multi-index)",
        )
    )

    def compute_zonal_statistics_gdf(  # noqa: D102
        self, *, metrics=None, class_val=None, metrics_kwargs=None
    ):
        return super().compute_zonal_statistics_gdf(
            metrics=metrics, class_val=class_val, metrics_kwargs=metrics_kwargs
        )

    compute_zonal_statistics_gdf.__doc__ = (
        zonal._compute_zonal_statistics_gdf_doc.format(
            col_return="metrics and dates (multi-index)"
        )
    )

    def plot_metric(  # noqa: D102
        self,
        metric,
        *,
        class_val=None,
        ax=None,
        metric_legend=True,
        metric_label=None,
        zone_legend=True,
        fmt="--o",
        plot_kwargs=None,
        subplots_kwargs=None,
        metric_kwargs=None,
    ):
        # TODO: refactor this method so that it uses `class_metrics_df` and
        # `landscape_metrics_df` properties?
        if ax is None:
            if subplots_kwargs is None:
                subplots_kwargs = {}
            fig, ax = plt.subplots(**subplots_kwargs)

        if plot_kwargs is None:
            plot_kwargs = {}

        # start: compute metrics - TODO: DRY from multilandscape.plot_metric?
        if metric_kwargs is None:
            metric_kwargs = {}
        metrics_kwargs = {metric: metric_kwargs}
        metrics = [metric]
        if class_val is None:
            metric_df = self.compute_landscape_metrics_df(
                metrics=metrics, metrics_kwargs=metrics_kwargs
            )
        else:
            metric_df = self.compute_class_metrics_df(
                metrics=metrics, classes=[class_val], metrics_kwargs=metrics_kwargs
            ).loc[class_val]
        # end: compute metrics

        if "label" not in plot_kwargs:
            # avoid alias/reference issues
            _plot_kwargs = plot_kwargs.copy()
            for zone, zone_df in metric_df.groupby(level=0):
                _plot_kwargs["label"] = zone
                ax.plot(zone_df.loc[zone].index, zone_df.values, fmt, **_plot_kwargs)
        else:
            for zone, zone_df in metric_df.groupby(level=0):
                ax.plot(zone_df.loc[zone].index, zone_df.values, fmt, **plot_kwargs)

        # start: metric legend - TODO: DRY from multilandscape
        if metric_legend:
            if metric_label is None:
                # get the metric label from the settings, otherwise use the metric
                # method name, i.e., metric name in camel-case
                metric_label = settings.metric_label_dict.get(metric, metric)

            ax.set_ylabel(metric_label)
        # end metric legend

        if zone_legend:
            ax.legend()

        return ax

    plot_metric.__doc__ = _plot_metric_doc.format(
        zone_descr="zone",
        zone_var_name="zone",
    )

    def plot_landscapes(
        self,
        *,
        cmap=None,
        legend=True,
        subplots_kwargs=None,
        show_kwargs=None,
        subplots_adjust_kwargs=None,
    ):
        """Plot each landscape snapshot in a dedicated matplotlib axis.

        Parameters
        ----------
        cmap : str or `~matplotlib.colors.Colormap`, optional
            A Colormap instance.
        legend : bool, optional
            If ``True``, display the legend of the land use/cover color codes.
        subplots_kwargs : dict, default None
            Keyword arguments to be passed to `matplotlib.pyplot.subplots`.
        show_kwargs : dict, default None
            Keyword arguments to be passed to `rasterio.plot.show`.
        subplots_adjust_kwargs : dict, default None
            Keyword arguments to be passed to `matplotlib.pyplot.subplots_adjust`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure with its corresponding plots drawn into its axes.
        """
        # avoid alias/reference issues
        if subplots_kwargs is None:
            _subplots_kwargs = {}
        else:
            _subplots_kwargs = subplots_kwargs.copy()
        figsize = _subplots_kwargs.pop("figsize", None)
        dates = self.landscape_ser.index.get_level_values("date").unique()
        if figsize is None:
            figwidth, figheight = plt.rcParams["figure.figsize"]
            figsize = (
                figwidth * len(self.zone_gser),
                figheight * len(dates),
            )

        fig, axes = plt.subplots(
            len(self.zone_gser), len(dates), figsize=figsize, **_subplots_kwargs
        )

        if show_kwargs is None:
            show_kwargs = {}
        flat_axes = axes.flat
        for landscape in self.landscape_ser:
            ax = landscape.plot_landscape(
                cmap=cmap, ax=next(flat_axes), legend=legend, **show_kwargs
            )

        # labels in first row and column only
        for date, ax in zip(dates, axes[0]):
            ax.set_title(date)

        for zone, ax in zip(self.zone_gser.index, axes[:, 0]):
            ax.set_ylabel(zone)

        # adjust spacing between axes
        if subplots_adjust_kwargs is not None:
            fig.subplots_adjust(**subplots_adjust_kwargs)

        return fig


class SpatioTemporalBufferAnalysis(SpatioTemporalZonalAnalysis):
    """Spatio-temporal buffer analysis around a feature of interest."""

    def __init__(
        self,
        landscape_filepaths,
        base_geom,
        buffer_dists,
        *,
        buffer_rings=False,
        base_geom_crs=None,
        dates=None,
        neighborhood_rule=None,
    ):
        """Initialize the spatio-temporal buffer analysis.

        Parameters
        ----------
        landscape_filepaths : list-like
            A list-like of strings/file-like/pathlib.Path objects so that each is passed
            as the `landscape` argument of `Landscape.__init__`.
        base_geom : shapely geometry, geopandas.GeoSeries or geopandas.GeoDataFrame
            Geometry that will serve as a base to buffer around.
        buffer_dists : list-like
            Buffer distances, in units of the landscape CRS.
        buffer_rings : bool, default False
            If `False`, each buffer zone will consist of the whole region that lies
            within the respective buffer distance around the base geometry. If `True`,
            buffer zones will take the form of rings around the base geometry.
        base_geom_crs : str, dict or pyproj.CRS, optional
            The coordinate reference system of the base geometry. Required if the base
            geometry is a shapely geometry or a geopandas GeoSeries without the `crs`
            attribute set.
        dates : list-like, optional
            A list-like of ints or strings that label the date of each snapshot of
            `landscapes` (for DataFrame indices and plot labels).
        neighborhood_rule : {'8', '4'}, optional
            Neighborhood rule to determine patch adjacencies, i.e: '8' (queen's
            case/Moore neighborhood) or '4' (rook's case/Von Neumann neighborhood).
            Ignored if the passed-in landscapes are `Landscape` instances. If no value
            is provided and the passed-in landscapes are file-like objects or paths, the
            default value set in `settings.DEFAULT_NEIGHBORHOOD_RULE` will be taken.
        """
        ba = zonal.BufferAnalysis(
            landscape_filepaths[0],
            base_geom,
            buffer_dists,
            buffer_rings=buffer_rings,
            base_geom_crs=base_geom_crs,
        )
        # now we can call the parent's init with the landscapes, the constructed buffer
        # geoseries and forward the `dates` and `neighborhood_rule` args.
        super().__init__(
            landscape_filepaths,
            zones=ba.zone_gser,
            dates=dates,
            neighborhood_rule=neighborhood_rule,
        )

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
            index_descr="multi-indexed by the buffer distance, class and date",
            index_return="buffer distance, class, distance (multi-index)",
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
            index_descr="multi-indexed by the buffer distance and date",
            index_return="buffer distance, date (multi-index)",
        )
    )

    def plot_metric(  # noqa: D102
        self,
        metric,
        *,
        class_val=None,
        ax=None,
        metric_legend=True,
        metric_label=None,
        buffer_dist_legend=True,
        fmt="--o",
        plot_kwargs=None,
        subplots_kwargs=None,
        metric_kwargs=None,
    ):
        return super().plot_metric(
            metric,
            class_val=class_val,
            ax=ax,
            metric_legend=metric_legend,
            metric_label=metric_label,
            zone_legend=buffer_dist_legend,
            fmt=fmt,
            plot_kwargs=plot_kwargs,
            subplots_kwargs=subplots_kwargs,
            metric_kwargs=metric_kwargs,
        )

    plot_metric.__doc__ = _plot_metric_doc.format(
        zone_descr="buffer zone", zone_var_name="buffer_dist"
    )


class SpatioTemporalZonalGridAnalysis(SpatioTemporalZonalAnalysis):
    """Spatio-temporal zonal analysis around a grid."""

    def __init__(
        self,
        landscape_filepaths,
        *,
        num_zone_cols=None,
        num_zone_rows=None,
        zone_width=None,
        zone_height=None,
        offset=None,
        dates=None,
        neighborhood_rule=None,
    ):
        """Initialize the spatio-temporal zonal grid analysis.

        Parameters
        ----------
        landscape_filepaths : str, file-like object or pathlib.Path object
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
        dates : list-like, optional
            A list-like of ints or strings that label the date of each snapshot of
            `landscapes` (for DataFrame indices and plot labels).
        neighborhood_rule : {'8', '4'}, optional
            Neighborhood rule to determine patch adjacencies, i.e: '8' (queen's
            case/Moore neighborhood) or '4' (rook's case/Von Neumann neighborhood). If
            no value is provided, the default value set in
            `settings.DEFAULT_NEIGHBORHOOD_RULE` will be taken.
        """
        zga = zonal.ZonalGridAnalysis(
            landscape_filepaths[0],
            num_zone_cols=num_zone_cols,
            num_zone_rows=num_zone_rows,
            zone_width=zone_width,
            zone_height=zone_height,
            offset=offset,
            neighborhood_rule=neighborhood_rule,
        )
        # now we can call the parent's init with the landscapes, the constructed grid
        # geoseries and forward the `dates` and `neighborhood_rule` args.
        super().__init__(
            landscape_filepaths,
            zones=zga.zone_gser,
            dates=dates,
            neighborhood_rule=neighborhood_rule,
        )

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
            index_descr="multi-indexed by the grid cell, class and date",
            index_return="grid cell, class, distance (multi-index)",
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
            index_descr="multi-indexed by the grid cell and date",
            index_return="grid cell, date (multi-index)",
        )
    )

    def plot_metric(  # noqa: D102
        self,
        metric,
        *,
        class_val=None,
        ax=None,
        metric_legend=True,
        metric_label=None,
        grid_cell_legend=True,
        fmt="--o",
        plot_kwargs=None,
        subplots_kwargs=None,
        metric_kwargs=None,
    ):
        return super().plot_metric(
            metric,
            class_val=class_val,
            ax=ax,
            metric_legend=metric_legend,
            metric_label=metric_label,
            zone_legend=grid_cell_legend,
            fmt=fmt,
            plot_kwargs=plot_kwargs,
            subplots_kwargs=subplots_kwargs,
            metric_kwargs=metric_kwargs,
        )

    plot_metric.__doc__ = _plot_metric_doc.format(
        zone_descr="zone grid", zone_var_name="grid_cell"
    )
