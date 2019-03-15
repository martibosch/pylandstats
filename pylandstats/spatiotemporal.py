from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .landscape import Landscape, read_geotiff

__all__ = ['SpatioTemporalAnalysis']


class SpatioTemporalAnalysis:
    def __init__(self, landscapes, metrics=None, classes=None, dates=None,
                 metrics_kws={}):
        """
        Parameters
        ----------
        landscapes : list-like
            A list-like of `Landscape` objects or of strings/file objects/
            pathlib.Path objects so that each is passed as the `fp` argument of
            `pls.read_geotiff`
        metrics : list-like, optional
            A list-like of strings with the names of the metrics that should
            be computed in the context of this analysis case
        classes : list-like, optional
            A list-like of ints or strings with the class values that should
            be considered in the context of this analysis case
        dates : list-like, optional
            A list-like of ints or strings that label the date of each
            snapshot of `landscapes` (for DataFrame indices and plot labels)
        metrics_kws : dict, optional
            Dictionary mapping the keyword arguments (values) that should be
            passed to each metric method (key), e.g., to exclude the boundary
            from the computation of `total_edge`, metric_kws should map the
            string 'total_edge' (method name) to {'count_boundary': False}.
            The default empty dictionary will compute each metric according to
            FRAGSTATS defaults.
        """

        if isinstance(landscapes[0], Landscape):
            self.landscapes = landscapes
        else:
            self.landscapes = list(map(read_geotiff, landscapes))

        if metrics is None:
            self.class_metrics = Landscape.CLASS_METRICS
            self.landscape_metrics = Landscape.LANDSCAPE_METRICS
        else:
            # TODO: how to handle `Landscape.PATCH_METRICS`
            implemented_metrics = np.union1d(Landscape.CLASS_METRICS,
                                             Landscape.LANDSCAPE_METRICS)
            inexistent_metrics = np.setdiff1d(metrics, implemented_metrics)
            if inexistent_metrics.size > 0:
                raise ValueError(
                    "The metrics {} are not among the implemented metrics ".
                    format(inexistent_metrics) +
                    "(that is {})".format(implemented_metrics))
            else:
                self.class_metrics = np.intersect1d(metrics,
                                                    Landscape.CLASS_METRICS)
                self.landscape_metrics = np.intersect1d(
                    metrics, Landscape.LANDSCAPE_METRICS)

        present_classes = reduce(
            np.union1d,
            tuple(landscape.classes for landscape in self.landscapes))
        if classes is None:
            self.classes = present_classes
        else:
            inexistent_classes = np.setdiff1d(classes, present_classes)
            if inexistent_classes.size > 0:
                raise ValueError(
                    "The classes {} are not among the classes present on the ".
                    format(inexistent_classes) +
                    "landscapes (that is {})".format(present_classes))
            else:
                self.classes = classes

        if dates is None:
            self.dates = ['t{}'.format(i) for i in range(len(self.landscapes))]
        else:
            if len(landscapes) == len(dates):
                self.dates = dates
            else:
                raise ValueError(
                    "The lengths of `landscapes` and `dates` (if provided) "
                    "must coincide")

        self.metrics_kws = metrics_kws

    def __len__(self):
        return len(self.landscapes)

    @property
    def class_metrics_df(self):
        try:
            return self._class_metrics_df
        except AttributeError:
            # TODO: one-level index if only one class?
            class_metrics_df = pd.DataFrame(
                index=pd.MultiIndex.from_product([self.classes, self.dates]),
                columns=self.class_metrics)
            class_metrics_df.index.names = 'class_val', 'date'
            class_metrics_df.columns.name = 'metric'

            for date, landscape in zip(self.dates, self.landscapes):
                # get the class metrics DataFrame for the landscape snapshot
                # at this particular date
                df = landscape.compute_class_metrics_df(
                    metrics=self.class_metrics, metrics_kws=self.metrics_kws)
                # filter so we only check the classes considered in this
                # spatiotemporal analysis
                df = df.loc[df.index.intersection(self.classes)]
                # put every row of the filtered DataFrame of this particular
                # date in this spatiotemporal corresponding
                for class_val, row in df.iterrows():
                    class_metrics_df.loc[class_val, date] = row

            self._class_metrics_df = class_metrics_df

            return self._class_metrics_df

    @property
    def landscape_metrics_df(self):
        try:
            return self._landscape_metrics_df
        except AttributeError:
            landscape_metrics_df = pd.DataFrame(index=self.dates,
                                                columns=self.landscape_metrics)
            landscape_metrics_df.index.name = 'date'
            landscape_metrics_df.columns.name = 'metric'

            for date, landscape in zip(self.dates, self.landscapes):
                landscape_metrics_df.loc[
                    date] = landscape.compute_landscape_metrics_df(
                        self.landscape_metrics,
                        metrics_kws=self.metrics_kws).iloc[0]

            self._landscape_metrics_df = landscape_metrics_df

            return self._landscape_metrics_df

    # def plot_patch_metric(metric):
    #     # TODO: sns distplot?
    #     fig, ax = plt.subplots()
    #     ax.hist()

    def plot_metric(self, metric, class_val=None, ax=None, metric_legend=True,
                    fmt='--o', plt_kws={}, subplots_kws={}):
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
        fmt : str, default '--o'
            A format string for the `plt.plot`
        plt_kws : dict
            Keyword arguments to be passed to `plt.plot`
        subplots_kws : dict
            Keyword arguments to be passed to `plt.subplots`, only if no axis
            is given (through the `ax` argument)

        Returns
        -------
        ax : axis object
            Returns the Axes object with the plot drawn onto it
        """

        # TODO: metric_legend parameter acepting a set of str values
        # indicating, e.g., whether the metric label should appear as legend
        # or as yaxis label
        # TODO: if we use seaborn in the future, we can use the pd.Series
        # directly, since its index corresponds to this SpatioTemporalAnalysis
        # dates
        try:
            if class_val is None:
                metric_values = self.landscape_metrics_df[metric].values
            else:
                metric_values = self.class_metrics_df.loc[class_val,
                                                          metric].values
        except KeyError:
            if class_val is None:
                raise ValueError(
                    "Metric '{metric}' is not among {metrics}".format(
                        metric=metric, metrics=self.landscape_metrics))
            else:
                if class_val not in self.classes:
                    raise ValueError(
                        "Class '{class_val}' is not among {classes}".format(
                            class_val=class_val, classes=self.classes))
                else:
                    raise ValueError(
                        "Metric '{metric}' is not among {metrics}".format(
                            metric=metric, metrics=self.class_metrics))

        if ax is None:
            fig, ax = plt.subplots(**subplots_kws)

        ax.plot(self.dates, metric_values, fmt, **plt_kws)

        if metric_legend:
            ax.set_ylabel(metric)

        return ax

    def plot_metrics(self, class_val=None, metrics=None, num_cols=3,
                     metric_legend=True, xtick_labelbottom=False, plt_kws={},
                     subplots_adjust_kws={}):
        """
        Parameters
        ----------
        class_val : int, optional
            If provided, the metrics will be plotted at the level of the
            corresponding class, otherwise it will be plotted at the landscape
            level
        metrics : list-like, optional
            A list-like of strings with the names of the metrics that should
            be plotted. The metrics should have been passed within the
            initialization of this `SpatioTemporalAnalysis` instance
        num_cols : int, default 3
            Number of columns for the figure; the rows will be deduced
            accordingly
        metric_legend : bool, default True
            Whether the metric label should be displayed within the plot (as
            label of the y-axis)
        xtick_labelbottom : bool, default False
            If True, the label ticks (dates) in the xaxis will be displayed at
            each row. Otherwise they will only be displayed at the bottom row
        plt_kws : dict
            Keyword arguments to be passed to `plt.plot`
        subplots_adjust_kws: dict, optional
            Keyword arguments to be passed to `plt.subplots_adjust`

        Returns
        -------
        fig, ax : tuple
            - figure object
            - axis object with the plot drawn onto it
        """

        if metrics is None:
            if class_val is None:
                metrics = self.landscape_metrics
            else:
                metrics = self.class_metrics

        if len(metrics) < num_cols:
            num_cols = len(metrics)
            num_rows = 1
        else:
            num_rows = int(np.ceil(len(metrics) / num_cols))

        figwidth, figlength = plt.rcParams['figure.figsize']
        fig, axes = plt.subplots(
            num_rows, num_cols, sharex=True, figsize=(figwidth * num_cols,
                                                      figlength * num_rows))

        if num_rows == 1 and num_cols == 1:
            flat_axes = [axes]
        else:
            flat_axes = axes.flatten()

        for metric, ax in zip(metrics, flat_axes):
            self.plot_metric(metric, class_val=class_val, ax=ax,
                             metric_legend=metric_legend, plt_kws=plt_kws)

        if xtick_labelbottom:
            for ax in flat_axes:
                # this requires matplotlib >= 2.2
                ax.xaxis.set_tick_params(labelbottom=True)

        # disable axis for the latest cells that correspond to no metric (i.e.,
        # when len(metrics) < num_rows * num_cols)
        for i in range(len(metrics), len(flat_axes)):
            flat_axes[i].axis('off')

        # adjust spacing between axes
        if subplots_adjust_kws:
            fig.subplots_adjust(**subplots_adjust_kws)

        return fig, axes

    def plot_landscapes(self, cmap=None, legend=True, imshow_kws={},
                        subplots_adjust_kws={}):
        """
        Plots each landscape snapshot in a dedicated matplotlib axis

        Parameters
        -------
        cmap : str or `~matplotlib.colors.Colormap`, optional
            A Colormap instance
        legend : bool, optional
            If ``True``, display the legend of the land use/cover color codes
        imshow_kws : dict, optional
            Keyword arguments to be passed to `plt.imshow`
        subplots_adjust_kws: dict, optional
            Keyword arguments to be passed to `plt.subplots_adjust`

        Returns
        -------
        fig, ax : tuple
            - figure object
            - axis object with the plot drawn onto it
        """

        figwidth, figlength = plt.rcParams['figure.figsize']
        fig, axes = plt.subplots(
            1, len(self.dates), figsize=(figwidth * len(self.dates),
                                         figlength))

        for date, landscape, ax in zip(self.dates, self.landscapes, axes):
            ax.imshow(landscape.landscape_arr, cmap=cmap, **imshow_kws)
            ax.set_title(date)

        # adjust spacing between axes
        if subplots_adjust_kws:
            fig.subplots_adjust(**subplots_adjust_kws)

        return fig, axes
