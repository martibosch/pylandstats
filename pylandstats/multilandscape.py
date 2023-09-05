"""Multi-landscape analysis."""
import abc
import functools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import settings
from .landscape import Landscape

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
metrics_kws : dict, optional
    Dictionary mapping the keyword arguments (values) that should be passed to each
    metric method (key), e.g., to exclude the boundary from the computation of
    `total_edge`, metric_kws should map the string 'total_edge' (method name) to
    {{'count_boundary': False}}. The default empty dictionary will compute each metric
    according to FRAGSTATS defaults.
fillna : bool, optional
    Whether `NaN` values representing landscapes with no occurrences of patches of the
    provided class should be replaced by zero when appropriate, e.g., area and edge
    metrics (no ocurrences mean zero area/edge). If the provided value is `None`
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
metrics_kws : dict, optional
    Dictionary mapping the keyword arguments (values) that should be passed to each
    metric method (key), e.g., to exclude the boundary from the computation of
    `total_edge`, metric_kws should map the string 'total_edge' (method name) to
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
    def __init__(self, landscapes, attribute_name, attribute_values, **landscape_kws):
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
        landscape_kws : dict, optional
            Keyword arguments to be passed to the instantiation of
            `pylandstats.Landscape` for each element of `landscapes`. Ignored if the
            elements of `landscapes` are already instances of `pylandstats.Landcape`.
        """
        if isinstance(landscapes[0], Landscape):
            self.landscapes = landscapes
        else:
            self.landscapes = [
                Landscape(landscape, **landscape_kws) for landscape in landscapes
            ]

        if len(self.landscapes) != len(attribute_values):
            raise ValueError(
                "The lengths of `landscapes` and `{}` must coincide".format(
                    attribute_name
                )
            )

        # set a `attribute_name` attribute with the value `attribute_values`, so that
        # children classes can access it (e.g., for `SpatioTemporalAnalysis`,
        # `attribute_name` will be 'dates' and `attribute_values` will be a list of
        # dates that will therefore be accessible as an attribute as in `instance.dates`
        setattr(self, attribute_name, attribute_values)
        # also set a `attribute_name` attribute so that the methods of this class know
        # how to access such attribute, i.e., as in `getattr(self, self.attribute_name)`
        setattr(self, "attribute_name", attribute_name)

        # get the all classes present in the provided landscapes
        self.present_classes = functools.reduce(
            np.union1d,
            tuple(landscape.classes for landscape in self.landscapes),
        )

    # fillna for metrics in class metrics dataframes. Since some classes might not
    # apprear in some of the landscapes (e.g., zones or temporal snapshots without any
    # pixel of a particular class type), they will appear as `NaN` in the data frame. We
    # can, however, infer the meaning of this situation for certain metrics, e.g,
    # non-occurence of a given class in a landscape means a number of patches, total
    # area, proportion of landscape, total edge... of the class of 0
    METRIC_FILLNA_DICT = {
        metric: 0
        for metric in [
            patch_metric + "_" + suffix
            for patch_metric in ["area", "perimeter"]
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
        ]
    }

    def __len__(self):  # noqa: D105
        return len(self.landscapes)

    def compute_class_metrics_df(  # noqa: D102
        self, *, metrics=None, classes=None, metrics_kws=None, fillna=None
    ):
        attribute_values = getattr(self, self.attribute_name)

        # get the columns to init the data frame
        if metrics is None:
            columns = Landscape.CLASS_METRICS
        else:
            columns = metrics
        # if the classes kwarg is not provided, get the classes present in the
        # landscapes
        if classes is None:
            classes = self.present_classes
        # to avoid issues with mutable defaults
        if metrics_kws is None:
            metrics_kws = {}
        # to avoid setting the same default keyword argument in multiple methods, use
        # the settings module
        if fillna is None:
            fillna = settings.CLASS_METRICS_DF_FILLNA

        # IMPORTANT: here we need this approach (uglier when compared to the
        # `compute_landscape_metrics_df` method below) because we need to filter each
        # class metrics data frame so that we only include the classes considered in
        # this `MultiLandscape` instance. We need to do it like this because the
        # `Landcape.compute_class_metrics_df` does not have a `classes` argument that
        # allows computing the data frame only for a custom set of classes. Should such
        # `classes` argument be added at some point, we could use the approach of the
        # `compute_landscape_metrics_df` method below.
        # TODO: one-level index if only one class?
        class_metrics_df = pd.DataFrame(
            index=pd.MultiIndex.from_product([classes, attribute_values]),
            columns=columns,
        )

        class_metrics_df.index.names = "class_val", self.attribute_name
        class_metrics_df.columns.name = "metric"

        for attribute_value, landscape in zip(attribute_values, self.landscapes):
            # get the class metrics DataFrame for the landscape that corresponds to this
            # attribute value
            df = landscape.compute_class_metrics_df(
                metrics=metrics, metrics_kws=metrics_kws
            )
            # filter so we only check the classes considered in this `MultiLandscape`
            # instance
            df = df.loc[df.index.intersection(classes)]
            # put every row of the filtered DataFrame of this particular attribute value
            for class_val, row in df.iterrows():
                class_metrics_df.loc[(class_val, attribute_value), columns] = row

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
        self, *, metrics=None, metrics_kws=None
    ):
        attribute_values = getattr(self, self.attribute_name)

        # get the columns to init the data frame
        if metrics is None:
            columns = Landscape.LANDSCAPE_METRICS
        else:
            columns = metrics
        # to avoid issues with mutable defaults
        if metrics_kws is None:
            metrics_kws = {}

        if isinstance(attribute_values[0], tuple):
            # for the zonal statistics analysis mainly
            index = pd.MultiIndex.from_tuples(attribute_values)
        else:
            index = attribute_values
        landscape_metrics_df = pd.DataFrame(index=index, columns=columns)
        landscape_metrics_df.index.name = self.attribute_name
        landscape_metrics_df.columns.name = "metric"

        for attribute_value, landscape in zip(attribute_values, self.landscapes):
            landscape_metrics_df.loc[
                attribute_value, columns
            ] = landscape.compute_landscape_metrics_df(
                metrics=metrics, metrics_kws=metrics_kws
            ).iloc[
                0
            ]

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
        plot_kws=None,
        subplots_kws=None,
        metric_kws=None,
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
        plot_kws : dict, default None
            Keyword arguments to be passed to `matplotlib.pyplot.plot`.
        subplots_kws : dict, default None
            Keyword arguments to be passed to `matplotlib.pyplot.plot.subplots` only if
            no axis is given (through the `ax` argument).
        metric_kws : dict, default None
            Keyword arguments to be passed to the method that computes the metric
            (specified in the `metric` argument) for each landscape.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Returns the `Axes` object with the plot drawn onto it.
        """
        # TODO: metric_legend parameter acepting a set of str values indicating, e.g.,
        # whether the metric label should appear as legend or as yaxis label
        # TODO: if we use seaborn in the future, we can use the pd.Series directly,
        # since its index corresponds to this SpatioTemporalAnalysis dates
        if metric_kws is None:
            metric_kws = {}
        # since we are using the compute data frame methods even though we are just
        # computing a single metric (so that error management regarding the computation
        # of metrics is defined in a single place), we need to provide the `metrics_kws`
        # (mapping a metric to its keyword-arguments `metric_kws`).
        metrics_kws = {metric: metric_kws}
        metrics = [metric]
        if class_val is None:
            metric_values = self.compute_landscape_metrics_df(
                metrics=metrics, metrics_kws=metrics_kws
            ).values
        else:
            metric_values = self.compute_class_metrics_df(
                metrics=metrics, classes=[class_val], metrics_kws=metrics_kws
            ).values

        if ax is None:
            if subplots_kws is None:
                subplots_kws = {}
            fig, ax = plt.subplots(**subplots_kws)

        # for `SpatioTemporalAnalysis`, `attribute_values` will be `dates`; for
        # `BufferAnalysis`, `attribute_values` will be `buffer_dists`
        attribute_values = getattr(self, self.attribute_name)

        if plot_kws is None:
            plot_kws = {}

        ax.plot(attribute_values, metric_values, fmt, **plot_kws)

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
        subplots_kws=None,
        show_kws=None,
        subplots_adjust_kws=None,
    ):
        """Plot each landscape snapshot in a dedicated matplotlib axis.

        Uses the `Landscape.plot_landscape` method of each instance.

        Parameters
        ----------
        cmap : str or `~matplotlib.colors.Colormap`, optional
            A Colormap instance.
        legend : bool, optional
            If ``True``, display the legend of the land use/cover color codes.
        subplots_kws : dict, default None
            Keyword arguments to be passed to `matplotlib.pyplot.subplots`.
        show_kws : dict, default None
            Keyword arguments to be passed to `rasterio.plot.show`.
        subplots_adjust_kws : dict, default None
            Keyword arguments to be passed to `matplotlib.pyplot.subplots_adjust`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure with its corresponding plots drawn into its axes.
        """
        attribute_values = getattr(self, self.attribute_name)

        # avoid alias/refrence issues
        if subplots_kws is None:
            _subplots_kws = {}
        else:
            _subplots_kws = subplots_kws.copy()
        figsize = _subplots_kws.pop("figsize", None)
        if figsize is None:
            figwidth, figheight = plt.rcParams["figure.figsize"]
            figsize = (figwidth * len(attribute_values), figheight)

        fig, axes = plt.subplots(
            1, len(attribute_values), figsize=figsize, **_subplots_kws
        )
        if len(axes) == 1:  # len(attribute_values) == 1
            axes = [axes]
        if show_kws is None:
            show_kws = {}
        for attribute_value, landscape, ax in zip(
            attribute_values, self.landscapes, axes
        ):
            ax = landscape.plot_landscape(cmap=cmap, ax=ax, legend=legend, **show_kws)
            ax.set_title(attribute_value)

        # adjust spacing between axes
        if subplots_adjust_kws is not None:
            fig.subplots_adjust(**subplots_adjust_kws)

        return fig
