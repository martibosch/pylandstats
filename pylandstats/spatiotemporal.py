from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import landscape as pls_landscape
from . import multilandscape, zonal

__all__ = ['SpatioTemporalAnalysis', 'SpatioTemporalBufferAnalysis']


class SpatioTemporalAnalysis(multilandscape.MultiLandscape):
    def __init__(self, landscapes, dates=None):
        """
        Parameters
        ----------
        landscapes : list-like
            A list-like of `Landscape` objects or of strings/file objects/
            pathlib.Path objects so that each is passed as the `landscape`
            argument of `Landscape.__init__`
        dates : list-like, optional
            A list-like of ints or strings that label the date of each
            snapshot of `landscapes` (for DataFrame indices and plot labels)
        """

        if dates is None:
            dates = ['t{}'.format(i) for i in range(len(landscapes))]

        # Call the parent's init
        super(SpatioTemporalAnalysis, self).__init__(landscapes, 'dates',
                                                     dates)

    # override docs
    def compute_class_metrics_df(self, metrics=None, classes=None,
                                 metrics_kws=None):
        return super(SpatioTemporalAnalysis,
                     self).compute_class_metrics_df(metrics=metrics,
                                                    classes=classes,
                                                    metrics_kws=metrics_kws)

    compute_class_metrics_df.__doc__ = \
        multilandscape._compute_class_metrics_df_doc.format(
            index_descr='multi-indexed by the class and date',
            index_return='class, date (multi-index)')

    def compute_landscape_metrics_df(self, metrics=None, metrics_kws=None):
        return super(SpatioTemporalAnalysis,
                     self).compute_landscape_metrics_df(
                         metrics=metrics, metrics_kws=metrics_kws)

    compute_landscape_metrics_df.__doc__ = \
        multilandscape._compute_landscape_metrics_df_doc.format(
            index_descr='indexed by the date', index_return='date (index)')

    # def plot_patch_metric(metric):
    #     # TODO: sns distplot?
    #     fig, ax = plt.subplots()
    #     ax.hist()


class SpatioTemporalBufferAnalysis(SpatioTemporalAnalysis):
    def __init__(self, landscapes, base_mask, buffer_dists, buffer_rings=False,
                 base_mask_crs=None, landscape_crs=None,
                 landscape_transform=None, dates=None):
        """
        Parameters
        ----------
        landscapes : list-like
            A list-like of `Landscape` objects or of strings/file objects/
            pathlib.Path objects so that each is passed as the `landscape`
            argument of `Landscape.__init__`
        base_mask : shapely geometry or geopandas GeoSeries
            Geometry that will serve as a base mask to buffer around
        buffer_rings : bool, default False
            If `False`, each buffer zone will consist of the whole region that
            lies within the respective buffer distance around the base mask.
            If `True`, buffer zones will take the form of rings around the
            base mask.
        base_mask_crs : str, dict or pyproj.CRS, optional
            The coordinate reference system of the base mask. Required if the
            base mask is a shapely geometry or a geopandas GeoSeries without
            the `crs` attribute set
        landscape_crs : str, dict or pyproj.CRS, optional
            The coordinate reference system of the landscapes. Required if the
            passed-in landscapes are `Landscape` objects, ignored if they are
            paths to GeoTiff rasters that already contain such information.
        landscape_transform : affine.Affine
            Transformation from pixel coordinates to coordinate reference
            system. Required if the passed-in landscapes are `Landscape`
            objects, ignored if they are paths to GeoTiff rasters that already
            contain such information.
        dates : list-like, optional
            A list-like of ints or strings that label the date of each
            snapshot of `landscapes` (for DataFrame indices and plot labels)
        """
        super(SpatioTemporalBufferAnalysis,
              self).__init__(landscapes, dates=dates)
        ba = zonal.BufferAnalysis(landscapes[0], base_mask=base_mask,
                                  buffer_dists=buffer_dists,
                                  buffer_rings=buffer_rings,
                                  base_mask_crs=base_mask_crs,
                                  landscape_crs=landscape_crs,
                                  landscape_transform=landscape_transform)
        # while `BufferAnalysis.__init__` will set the `buffer_dists`
        # attribute to the instantiated object (stored in the variable `ba`),
        # it will not set it to the current `SpatioTemporalBufferAnalysis`,
        # so we need to do it here
        self.buffer_dists = ba.buffer_dists

        # init the `SpatioTemporalAnalysis` objects
        self.stas = []
        for buffer_dist, mask_arr in zip(ba.buffer_dists, ba.masks_arr):
            self.stas.append(
                SpatioTemporalAnalysis([
                    pls_landscape.Landscape(
                        np.where(mask_arr, landscape.landscape_arr,
                                 landscape.nodata).astype(
                                     landscape.landscape_arr.dtype),
                        res=(landscape.cell_width, landscape.cell_height),
                        nodata=landscape.nodata, transform=landscape.transform)
                    for landscape in self.landscapes
                ], dates=dates))

        # the `self.present_classes` attribute will have been set by this
        # instance father's init (namely the `super` in the first line of this
        # method), however some of the classes may not actually be found in
        # any of buffer zones. We therefore need to get the union of the
        # classes found at the spatio-temporal analysis instance of each
        # `buffer_dist`
        self.present_classes = reduce(
            np.union1d, tuple(sta.present_classes for sta in self.stas))

        # the dates will be the same for all the `SpatioTemporalAnalysis`
        # instances stored in `self.stas`. We will just take them from the
        # first instance and store them as attribute of this
        # `SpatioTemporalBufferAnalysis` so that it can be used more
        # conveniently below.
        # ACHTUNG: we do it AFTER instantiating the `SpatioTemporalAnalysis`
        # objects of `self.stats` so that we let the `__init__` method of
        # `SpatioTemporalAnalysis.__init__` deal with the logic of what to do
        # with the `dates` argument
        self.dates = self.stas[0].dates

    def compute_class_metrics_df(self, metrics=None, classes=None,
                                 metrics_kws=None):
        if classes is None:
            classes = self.present_classes

        # get the columns to init the data frame
        if metrics is None:
            columns = pls_landscape.Landscape.CLASS_METRICS
        else:
            columns = metrics

        # IMPORTANT: since some classes might not be present for each date
        # and/or buffer distance, we will init the MultiIndex manually to
        # ensure that every class is present in the resulting data frame. If
        # some class does not appear for some some date/buffer distance, the
        # corresponding row will be nan. This probably preferable than having
        # a MultiIndex that can have different levels (i.e., the second level
        # `class_val`) for each buffer distance.
        # Note that this approach is likely slower since for each of the
        # `buffer_dists`, we have to iterate as in (see below):
        # `for class_val, date in class_metrics_df.loc[buffer_dist].index`
        class_metrics_df = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [self.buffer_dists, classes, self.dates]), columns=columns)
        class_metrics_df.index.names = 'buffer_dist', 'class_val', 'dates'
        class_metrics_df.columns.name = 'metric'

        for buffer_dist, sta in zip(self.buffer_dists, self.stas):
            # get the class metrics data frame for the
            # `SpatioTemporalAnalysis` instance that corresponds to this
            # `buffer_dist`
            df = sta.compute_class_metrics_df(metrics=metrics, classes=classes,
                                              metrics_kws=metrics_kws)
            # put the metrics data frame of the `SpatioTemporalAnalysis`
            # of this `buffer_dist` into the global metrics data frame of
            # the `SpatioTemporalBufferAnalysis`
            for class_val, date in class_metrics_df.loc[buffer_dist].index:
                # use `class_metrics_df.loc` for the first level (i.e.,
                # `buffer_dist`) again (we have already used it in the
                # iterator above) to avoid `SettingWithCopyWarning`
                try:
                    class_metrics_df.loc[buffer_dist, class_val,
                                         date] = df.loc[class_val, date]
                except KeyError:
                    # this means that `class_val` is not in `df`,
                    # therefore we do nothing and the corresponding row of
                    # `class_metrics_df` will stay as nan
                    pass

        return class_metrics_df

    compute_class_metrics_df.__doc__ = \
        multilandscape._compute_class_metrics_df_doc.format(
            index_descr='multi-indexed by the buffer distance, class and date',
            index_return='buffer distance, class, distance (multi-index)')

    def compute_landscape_metrics_df(self, metrics=None, metrics_kws=None):
        # we will create a dict where each key is a `buffer_dist`, and its
        # value is the corresponding metrics data frame of the
        # `SpatioTemporalAnalysis` instance
        df_dict = {
            buffer_dist:
            sta.compute_landscape_metrics_df(metrics=metrics,
                                             metrics_kws=metrics_kws)
            for buffer_dist, sta in zip(self.buffer_dists, self.stas)
        }

        # we concatenate each value of the dict dataframe using its respective
        # `buffer_dist` key to create an extra index level (i.e., using the
        # `keys` argument of `pd.concat`)
        landscape_metrics_df = pd.concat(df_dict.values(), keys=df_dict.keys())
        # now we set the name of each index and column level
        landscape_metrics_df.index.names = 'buffer_dist', 'dates'
        landscape_metrics_df.columns.name = 'metric'

        self._landscape_metrics_df = landscape_metrics_df

        return self._landscape_metrics_df

    compute_landscape_metrics_df.__doc__ = \
        multilandscape._compute_landscape_metrics_df_doc.format(
            index_descr='multi-indexed by the buffer distance and date',
            index_return='buffer distance, date (multi-index)')

    def plot_metric(self, metric, class_val=None, ax=None, metric_legend=True,
                    metric_label=None, buffer_dist_legend=True, fmt='--o',
                    plot_kws=None, subplots_kws=None):
        """
        Parameters
        ----------
        metric : str
            A string indicating the name of the metric to plot
        class_val : int, optional
            If provided, the metric will be plotted at the level of the
            corresponding class, otherwise it will be plotted at the landscape
            level
        ax : axis object, optional
            Plot in given axis; if None creates a new figure
        metric_legend : bool, default True
            Whether the metric label should be displayed within the plot (as
            label of the y-axis)
        metric_label : str, optional
            Label of the y-axis to be displayed if `metric_legend` is `True`.
            If the provided value is `None`, the label will be taken from the
            `settings` module
        buffer_dist_legend : bool, default True
            Whether a legend linking each plotted line to a buffer distance
            should be displayed within the plot
        fmt : str, default '--o'
            A format string for `plt.plot`
        plot_kws : dict, default None
            Keyword arguments to be passed to `plt.plot`
        subplots_kws : dict, default None
            Keyword arguments to be passed to `plt.subplots`, only if no axis
            is given (through the `ax` argument)

        Returns
        -------
        ax : axis object
            Returns the Axes object with the plot drawn onto it
        """
        # TODO: refactor this method so that it uses `class_metrics_df` and
        # `landscape_metrics_df` properties?

        if ax is None:
            if subplots_kws is None:
                subplots_kws = {}
            fig, ax = plt.subplots(**subplots_kws)

        if plot_kws is None:
            plot_kws = {}

        if 'label' not in plot_kws:
            # avoid alias/refrence issues
            _plot_kws = plot_kws.copy()
            for buffer_dist, sta in zip(self.buffer_dists, self.stas):
                _plot_kws['label'] = buffer_dist
                ax = sta.plot_metric(metric, class_val=class_val, ax=ax,
                                     metric_legend=metric_legend,
                                     metric_label=metric_label, fmt=fmt,
                                     plot_kws=_plot_kws)
        else:
            for sta in self.stas:
                ax = sta.plot_metric(metric, class_val=class_val, ax=ax,
                                     metric_legend=metric_legend,
                                     metric_label=metric_label, fmt=fmt,
                                     plot_kws=plot_kws)

        if buffer_dist_legend:
            ax.legend()

        return ax

    def plot_landscapes(self, cmap=None, legend=True, subplots_kws=None,
                        show_kws=None, subplots_adjust_kws=None):
        """
        Plots each landscape snapshot in a dedicated matplotlib axis by means
        of the `Landscape.plot_landscape` method of each instance

        Parameters
        -------
        cmap : str or `~matplotlib.colors.Colormap`, optional
            A Colormap instance
        legend : bool, optional
            If ``True``, display the legend of the land use/cover color codes
        subplots_kws: dict, default None
            Keyword arguments to be passed to `plt.subplots`
        show_kws : dict, default None
            Keyword arguments to be passed to `rasterio.plot.show`
        subplots_adjust_kws: dict, default None
            Keyword arguments to be passed to `plt.subplots_adjust`

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The figure with its corresponding plots drawn into its axes
        """

        # the number of rows is the number of dates, which will be the same
        # for all the `SpatioTemporalAnalysis` instances of `self.stas`
        dates = self.stas[0].dates

        # avoid alias/refrence issues
        if subplots_kws is None:
            _subplots_kws = {}
        else:
            _subplots_kws = subplots_kws.copy()
        figsize = _subplots_kws.pop('figsize', None)
        if figsize is None:
            figwidth, figheight = plt.rcParams['figure.figsize']
            figsize = (figwidth * len(self.buffer_dists),
                       figheight * len(dates))

        fig, axes = plt.subplots(len(self.buffer_dists), len(dates),
                                 figsize=figsize, **_subplots_kws)

        if show_kws is None:
            show_kws = {}
        flat_axes = axes.flat
        for buffer_dist, sta in zip(self.buffer_dists, self.stas):
            for date, landscape in zip(sta.dates, sta.landscapes):
                ax = landscape.plot_landscape(cmap=cmap, ax=next(flat_axes),
                                              legend=legend, **show_kws)

        # labels in first row and column only
        for date, ax in zip(dates, axes[0]):
            ax.set_title(date)

        for buffer_dist, ax in zip(self.buffer_dists, axes[:, 0]):
            ax.set_ylabel(buffer_dist)

        # adjust spacing between axes
        if subplots_adjust_kws is not None:
            fig.subplots_adjust(**subplots_adjust_kws)

        return fig
