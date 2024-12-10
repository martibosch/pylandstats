"""Multi-landscape analysis."""

import abc
import functools

import dask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask import diagnostics

from pylandstats import settings
from pylandstats.landscape import Landscape

_compute_class_metrics_df_doc = """
Compute the data frame of class-level metrics, which is {index_descr}.

Parameters
----------
metrics : list-like, optional
    A list-like of strings with the names of the metrics that should be computed in the
    context of this analysis case.
classes : list-like, optional
    A list-like of ints or strings with the class values that should be considered in
    the context of this analysis case.
metrics_kwargs : dict, optional
    Dictionary mapping the keyword arguments (values) that should be passed to each
    metric method (key), e.g., to exclude the boundary from the computation of
    `total_edge`, metric_kwargs should map the string 'total_edge' (method name) to
    {{'count_boundary': False}}. The default empty dictionary will compute each metric
    according to FRAGSTATS defaults.
fillna : bool, optional
    Whether `NaN` values representing landscapes with no occurrences of patches of the
    provided class should be replaced by zero when appropriate, e.g., area and edge
    metrics (no occurrences mean zero area/edge). If the provided value is `None`
    (default), the value will be taken from `settings.CLASS_METRICS_DF_FILLNA`.

Returns
-------
df : pandas.DataFrame
    Dataframe with the values computed for each {index_return} and metric (columns).
"""

_compute_landscape_metrics_df_doc = """
Computes the data frame of landscape-level metrics, which is {index_descr}.

Parameters
----------
metrics : list-like, optional
    A list-like of strings with the names of the metrics that should be computed. If
    `None`, all the implemented landscape-level metrics will be computed.
metrics_kwargs : dict, optional
    Dictionary mapping the keyword arguments (values) that should be passed to each
    metric method (key), e.g., to exclude the boundary from the computation of
    `total_edge`, metric_kwargs should map the string 'total_edge' (method name) to
    {{'count_boundary': False}}. The default empty dictionary will compute each metric
    according to FRAGSTATS defaults.

Returns
-------
df : pandas.DataFrame
    Dataframe with the values computed at the landscape level for each {index_return}
    and metric (columns).
"""


class MultiLandscape(abc.ABC):
    """Multi-landscape base abstract class."""

    @abc.abstractmethod
    def __init__(
        self, landscapes, attribute_name, attribute_values, **landscape_kwargs
    ):
        """Initialize the multi-landscape instance.

        Parameters
        ----------
        landscapes : list-like
            A list-like of `Landscape` instances or of strings/file-like/pathlib.Path
            objects so that each is passed as the `landscape` argument of
            `Landscape.__init__`.
        attribute_name : str
            Name of the attribute that will distinguish each landscape.
        attribute_values : list-like
            Values of the attribute that are characteristic to each landscape.
        landscape_kwargs : dict, optional
            Keyword arguments to be passed to the instantiation of
            `pylandstats.Landscape` for each element of `landscapes`. Ignored if the
            elements of `landscapes` are already instances of `pylandstats.Landcape`.
        """
        if not isinstance(landscapes[0], Landscape):
            # we assume that landscapes is a list of strings/file-like/path-like objects
            landscapes = [
                Landscape(landscape, **landscape_kwargs) for landscape in landscapes
            ]
        if len(landscapes) != len(attribute_values):
            raise ValueError(
                "The lengths of `landscapes` and `{}` must coincide".format(
                    attribute_name
                )
            )

        # at this point, landscapes is a list of pylandstats.Landscape instances
        self.landscape_ser = pd.Series(landscapes, index=attribute_values).rename_axis(
            attribute_name
        )

        # get the all classes present in the provided landscapes
        self.present_classes = functools.reduce(
            np.union1d,
            tuple(landscape.classes for landscape in self.landscape_ser),
        )

    # fillna for metrics in class metrics dataframes. Since some classes might not
    # appear in some of the landscapes (e.g., zones or temporal snapshots without any
    # pixel of a particular class type), they will appear as `NaN` in the data frame. We
    # can, however, infer the meaning of this situation for certain metrics, e.g,
    # non-occurence of a given class in a landscape means a number of patches, total
    # area, proportion of landscape, total edge... of the class of 0
    METRIC_FILLNA_DICT = {
        metric: 0
        for metric in [
            patch_metric + "_" + suffix
            for patch_metric in ["area", "perimeter", "core_area"]
            for suffix in ["mn", "am", "md", "ra", "sd"]
        ]
        + [
            "total_area",
            "proportion_of_landscape",
            "number_of_patches",
            "patch_density",
            "largest_patch_index",
            "total_edge",
            "edge_density",
            "total_core_area",
        ]
    }

    def __len__(self):  # noqa: D105
        return len(self.landscape_ser)

    def compute_class_metrics_df(  # noqa: D102
        self, *, metrics=None, classes=None, metrics_kwargs=None, fillna=None
    ):
        # if the classes kwarg is not provided, get the classes present in the
        # landscapes
        if classes is None:
            classes = self.present_classes
        # to avoid issues with mutable defaults
        if metrics_kwargs is None:
            metrics_kwargs = {}
        # to avoid setting the same default keyword argument in multiple methods, use
        # the settings module
        if fillna is None:
            fillna = settings.CLASS_METRICS_DF_FILLNA

        tasks = [
            dask.delayed(landscape.compute_class_metrics_df)(
                metrics=metrics,
                classes=np.intersect1d(classes, landscape.classes),
                metrics_kwargs=metrics_kwargs,
            )
            for landscape in self.landscape_ser
        ]
        with diagnostics.ProgressBar():
            dfs = dask.compute(*tasks)

        names = self.landscape_ser.index.names
        # get the landscape series index and if not a multi-index, reshape it so that it
        # the list comprehensions below work for both one-dimensional and multi index
        landscape_index = self.landscape_ser.index.values
        if len(names) == 1:
            landscape_index = landscape_index.reshape(-1, 1)
        class_metrics_df = (
            pd.concat(
                [
                    df.assign(
                        **{
                            name: val if isinstance(i, tuple) else i[0]
                            for name, val in zip(names, i)
                        }
                    )
                    for i, df in zip(landscape_index, dfs)
                    if not df.empty
                ]
            )
            .set_index(names, append=True)
            # only sort the first level, i.e., class val
            .sort_index(level="class_val")
        )
        # then reindex to sort the other indices as they were originally sorted
        # TODO: this is probably only needed for "zones" - not for dates, since we
        # probably do not want to alphabetically sort zone labels but we probably want
        # to sort dates. In any case, avoid premature optimization: we assume that the
        # costs of sorting the metrics data frames are negligible
        for name in self.landscape_ser.index.names:
            class_metrics_df = class_metrics_df.reindex(
                self.landscape_ser.index.get_level_values(name).unique(), level=name
            )

        # ensure numeric types and fillna
        class_metrics_df = class_metrics_df.apply(pd.to_numeric)
        if fillna:
            class_metrics_df = class_metrics_df.fillna(
                MultiLandscape.METRIC_FILLNA_DICT
            )
        return class_metrics_df

    compute_class_metrics_df.__doc__ = _compute_class_metrics_df_doc.format(
        index_descr="multi-indexed by the class and attribute value",
        index_return="class, attribute value (multi-index)",
    )

    def compute_landscape_metrics_df(  # noqa: D102
        self, *, metrics=None, metrics_kwargs=None
    ):
        # to avoid issues with mutable defaults
        if metrics_kwargs is None:
            metrics_kwargs = {}

        tasks = [
            dask.delayed(landscape.compute_landscape_metrics_df)(
                metrics=metrics, metrics_kwargs=metrics_kwargs
            )
            for landscape in self.landscape_ser
        ]
        with diagnostics.ProgressBar():
            dfs = dask.compute(*tasks)

        names = self.landscape_ser.index.names
        # get the landscape series index and if not a multi-index, reshape it so that it
        # the list comprehensions below work for both one-dimensional and multi index
        landscape_index = self.landscape_ser.index.values
        if len(names) == 1:
            landscape_index = landscape_index.reshape(-1, 1)
        landscape_metrics_df = (
            pd.concat(
                [
                    df.assign(
                        **{
                            name: val if isinstance(i, tuple) else i[0]
                            for name, val in zip(names, i)
                        }
                    )
                    for i, df in zip(landscape_index, dfs)
                ]
            ).set_index(names)
            # there is no need to sort here
            # .sort_index()
        )

        return landscape_metrics_df.apply(pd.to_numeric)

    compute_landscape_metrics_df.__doc__ = _compute_landscape_metrics_df_doc.format(
        index_descr="indexed by the attribute value",
        index_return="attribute value (index)",
    )

    def plot_metric(
        self,
        metric,
        *,
        class_val=None,
        ax=None,
        metric_legend=True,
        metric_label=None,
        fmt="--o",
        plot_kwargs=None,
        subplots_kwargs=None,
        metric_kwargs=None,
    ):
        """Plot the metric.

        Parameters
        ----------
        metric : str
            A string indicating the name of the metric to plot.
        class_val : int, optional
            If provided, the metric will be plotted at the level of the corresponding
            class, otherwise it will be plotted at the landscape level.
        ax : axis object, optional
            Plot in given axis; if None creates a new figure.
        metric_legend : bool, default True
            Whether the metric label should be displayed within the plot (as label of
            the y-axis).
        metric_label : str, optional
            Label of the y-axis to be displayed if `metric_legend` is `True`. If the
            provided value is `None`, the label will be taken from the `settings`
            module.
        fmt : str, default '--o'
            A format string for `matplotlib.pyplot.plot`.
        plot_kwargs : dict, default None
            Keyword arguments to be passed to `matplotlib.pyplot.plot`.
        subplots_kwargs : dict, default None
            Keyword arguments to be passed to `matplotlib.pyplot.plot.subplots` only if
            no axis is given (through the `ax` argument).
        metric_kwargs : dict, default None
            Keyword arguments to be passed to the method that computes the metric
            (specified in the `metric` argument) for each landscape.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Returns the `Axes` object with the plot drawn onto it.
        """
        # TODO: metric_legend parameter accepting a set of str values indicating, e.g.,
        # whether the metric label should appear as legend or as yaxis label
        # TODO: if we use seaborn in the future, we can use the pd.Series directly,
        # since its index corresponds to this SpatioTemporalAnalysis dates
        if metric_kwargs is None:
            metric_kwargs = {}
        # since we are using the compute data frame methods even though we are just
        # computing a single metric (so that error management regarding the computation
        # of metrics is defined in a single place), we need to provide the
        # `metrics_kwargs` (mapping a metric to its keyword-arguments `metric_kwargs`).
        metrics_kwargs = {metric: metric_kwargs}
        metrics = [metric]
        if class_val is None:
            metric_values = self.compute_landscape_metrics_df(
                metrics=metrics, metrics_kwargs=metrics_kwargs
            ).values
        else:
            metric_values = self.compute_class_metrics_df(
                metrics=metrics, classes=[class_val], metrics_kwargs=metrics_kwargs
            ).values

        if ax is None:
            if subplots_kwargs is None:
                subplots_kwargs = {}
            fig, ax = plt.subplots(**subplots_kwargs)

        if plot_kwargs is None:
            plot_kwargs = {}

        ax.plot(self.landscape_ser.index, metric_values, fmt, **plot_kwargs)

        if metric_legend:
            if metric_label is None:
                # get the metric label from the settings, otherwise use the metric
                # method name, i.e., metric name in camel-case
                metric_label = settings.metric_label_dict.get(metric, metric)

            ax.set_ylabel(metric_label)

        return ax

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

        Uses the `Landscape.plot_landscape` method of each instance.

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
        num_landscapes = len(self.landscape_ser)

        # avoid alias/reference issues
        if subplots_kwargs is None:
            _subplots_kwargs = {}
        else:
            _subplots_kwargs = subplots_kwargs.copy()
        figsize = _subplots_kwargs.pop("figsize", None)
        if figsize is None:
            figwidth, figheight = plt.rcParams["figure.figsize"]
            figsize = (figwidth * num_landscapes, figheight)

        fig, axes = plt.subplots(1, num_landscapes, figsize=figsize, **_subplots_kwargs)
        if len(axes) == 1:  # len(attribute_values) == 1
            axes = [axes]
        if show_kwargs is None:
            show_kwargs = {}
        for (attribute_value, landscape), ax in zip(self.landscape_ser.items(), axes):
            ax = landscape.plot_landscape(
                cmap=cmap, ax=ax, legend=legend, **show_kwargs
            )
            ax.set_title(attribute_value)

        # adjust spacing between axes
        if subplots_adjust_kwargs is not None:
            fig.subplots_adjust(**subplots_adjust_kwargs)

        return fig
