import abc
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import six

from . import settings
from .landscape import Landscape


@six.add_metaclass(abc.ABCMeta)
class MultiLandscape:
    @abc.abstractmethod
    def __init__(self, landscapes, attribute_name, attribute_values,
                 metrics=None, classes=None, metrics_kws={}):
        """
        Parameters
        ----------
        landscapes : list-like
            A list-like of `Landscape` objects or of strings/file objects/
            pathlib.Path objects so that each is passed as the `landscape`
            argument of `Landscape.__init__`
        attribute_name : str
            Name of the attribute that will distinguish each landscape
        attribute_values : list-like
            Values of the attribute that are characteristic to each landscape
        metrics : list-like, optional
            A list-like of strings with the names of the metrics that should
            be computed in the context of this analysis case
        classes : list-like, optional
            A list-like of ints or strings with the class values that should
            be considered in the context of this analysis case
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
            self.landscapes = list(map(Landscape, landscapes))

        if len(self.landscapes) != len(attribute_values):
            raise ValueError(
                "The lengths of `landscapes` and `{}` must coincide".format(
                    attribute_name))

        # set a `attribute_name` attribute with the value `attribute_values`,
        # so that children classes can access it (e.g., for
        # `SpatioTemporalAnalysis`, `attribute_name` will be 'dates' and
        # `attribute_values` will be a list of dates that will therefore be
        # accessible as an attribute as in `instance.dates`
        setattr(self, attribute_name, attribute_values)
        # also set a `attribute_name` attribute so that the methods of this
        # class know how to access such attribute, i.e., as in
        # `getattr(self, self.attribute_name)`
        setattr(self, 'attribute_name', attribute_name)

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

            self.classes = classes

        self.metrics_kws = metrics_kws

    def __len__(self):
        return len(self.landscapes)

    @property
    def class_metrics_df(self):
        """
        Property that computes the data frame of class-level metrics, which
        is multi-indexed by the class and attribute value. Once computed, the
        data frame is cached so further calls to the property just access an
        attribute and therefore run in constant time.
        """
        try:
            return self._class_metrics_df
        except AttributeError:
            attribute_values = getattr(self, self.attribute_name)
            # IMPORTANT: here we need this approach (uglier when compared to
            # the `landscape_metrics_df` property below) because we need to
            # filter each class metrics data frame so that we only include the
            # classes considered in this `MultiLandscape` instance. We need to
            # do it like this because the `Landcape.compute_class_metrics_df`
            # does not have a `classes` argument that allows computing the
            # data frame only for a custom set of classes. Should such
            # `classes` argument be added at some point, we could use the
            # approach of the `landscape_metrics_df` property below.
            # TODO: one-level index if only one class?
            class_metrics_df = pd.DataFrame(
                index=pd.MultiIndex.from_product(
                    [self.classes, attribute_values]),
                columns=self.class_metrics)
            class_metrics_df.index.names = 'class_val', self.attribute_name
            class_metrics_df.columns.name = 'metric'

            for attribute_value, landscape in zip(attribute_values,
                                                  self.landscapes):
                # get the class metrics DataFrame for the landscape that
                # corresponds to this attribute value
                df = landscape.compute_class_metrics_df(
                    metrics=self.class_metrics, metrics_kws=self.metrics_kws)
                # filter so we only check the classes considered in this
                # `MultiLandscape` instance
                df = df.loc[df.index.intersection(self.classes)]
                # put every row of the filtered DataFrame of this particular
                # attribute value
                for class_val, row in df.iterrows():
                    class_metrics_df.loc[class_val, attribute_value] = row

            self._class_metrics_df = class_metrics_df

            return self._class_metrics_df

    @property
    def landscape_metrics_df(self):
        """
        Property that computes the data frame of landscape-level metrics, which
        is indexed by the attribute value. Once computed, the data frame is
        cached so further calls to the property just access an attribute and
        therefore run in constant time.
        """
        try:
            return self._landscape_metrics_df
        except AttributeError:
            attribute_values = getattr(self, self.attribute_name)
            # PREVIOUS APPROACH
            landscape_metrics_df = pd.DataFrame(index=attribute_values,
                                                columns=self.landscape_metrics)
            landscape_metrics_df.index.name = self.attribute_name
            landscape_metrics_df.columns.name = 'metric'

            for attribute_value, landscape in zip(attribute_values,
                                                  self.landscapes):
                landscape_metrics_df.loc[attribute_value] = \
                    landscape.compute_landscape_metrics_df(
                        self.landscape_metrics,
                        metrics_kws=self.metrics_kws).iloc[0]

            # # NEW APPROACH
            # # we will create a dict where each key is an `attribute_value`,
            # # and its value is the series of landscape-level `metrics of the
            # # corresponding `Landscape` instance
            # ser_dict = {
            #     attribute_value: landscape.compute_landscape_metrics_df(
            #         self.landscape_metrics,
            #         metrics_kws=self.metrics_kws).iloc[0]
            #     for attribute_value, landscape in zip(attribute_values,
            #                                           self.landscapes)
            # }

            # # we concatenate each value of the dict dataframe using its
            # # respective `buffer_dist` key to create an extra index level
            # # (i.e., using the `keys` argument of `pd.concat`)
            # landscape_metrics_df = pd.concat(ser_dict.values(),
            #                                  keys=ser_dict.keys())
            # # now we set the name of each index and column level
            # landscape_metrics_df.index.name = self.attribute_name
            # landscape_metrics_df.columns.name = 'metric'

            self._landscape_metrics_df = landscape_metrics_df

            return self._landscape_metrics_df

    def plot_metric(self, metric, class_val=None, ax=None, metric_legend=True,
                    metric_label=None, fmt='--o', plot_kws={},
                    subplots_kws={}):
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
        fmt : str, default '--o'
            A format string for `plt.plot`
        plot_kws : dict
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

            if class_val not in self.classes:
                raise ValueError(
                    "Class '{class_val}' is not among {classes}".format(
                        class_val=class_val, classes=self.classes))

            # if `class_val` is provided and is in `self.classes`, the
            # captured `KeyError` comes from an inexistent `metric` column
            raise ValueError("Metric '{metric}' is not among {metrics}".format(
                metric=metric, metrics=self.class_metrics))

        if ax is None:
            fig, ax = plt.subplots(**subplots_kws)

        # for `SpatioTemporalAnalysis`, `attribute_values` will be `dates`;
        # for `BufferAnalysis`, `attribute_values` will be `buffer_dists`
        attribute_values = getattr(self, self.attribute_name)

        ax.plot(attribute_values, metric_values, fmt, **plot_kws)

        if metric_legend:
            if metric_label is None:
                # get the metric label from the settings, otherwise use the
                # metric method name, i.e., metric name in camel-case
                metric_label = settings.metric_label_dict.get(metric, metric)

            ax.set_ylabel(metric_label)

        return ax

    def plot_landscapes(self, cmap=None, legend=True, subplots_kws={},
                        show_kws={}, subplots_adjust_kws={}):
        """
        Plots each landscape snapshot in a dedicated matplotlib axis by means
        of the `Landscape.plot_landscape` method of each instance

        Parameters
        -------
        cmap : str or `~matplotlib.colors.Colormap`, optional
            A Colormap instance
        legend : bool, optional
            If ``True``, display the legend of the land use/cover color codes
        subplots_kws: dict, optional
            Keyword arguments to be passed to `plt.subplots`
        show_kws : dict, optional
            Keyword arguments to be passed to `rasterio.plot.show`
        subplots_adjust_kws: dict, optional
            Keyword arguments to be passed to `plt.subplots_adjust`

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The figure with its corresponding plots drawn into its axes
        """

        attribute_values = getattr(self, self.attribute_name)

        # avoid alias/refrence issues
        _subplots_kws = subplots_kws.copy()
        figsize = _subplots_kws.pop('figsize', None)
        if figsize is None:
            figwidth, figheight = plt.rcParams['figure.figsize']
            figsize = (figwidth * len(attribute_values), figheight)

        fig, axes = plt.subplots(1, len(attribute_values), figsize=figsize,
                                 **_subplots_kws)

        for attribute_value, landscape, ax in zip(attribute_values,
                                                  self.landscapes, axes):
            ax = landscape.plot_landscape(cmap=cmap, ax=ax, legend=legend,
                                          **show_kws)
            ax.set_title(attribute_value)

        # adjust spacing between axes
        if subplots_adjust_kws:
            fig.subplots_adjust(**subplots_adjust_kws)

        return fig
