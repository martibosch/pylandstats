from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .landscape import Landscape, read_geotiff

__all__ = ['SpatioTemporalAnalysis']


class SpatioTemporalAnalysis:
    def __init__(self, landscapes, metrics=None, classes=None, dates=None):
        if isinstance(landscapes[0], Landscape):
            self.landscapes = landscapes
        else:
            self.landscapes = list(map(read_geotiff, landscapes))

        if metrics is not None:
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
        else:
            self.class_metrics = Landscape.CLASS_METRICS
            self.landscape_metrics = Landscape.LANDSCAPE_METRICS

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

        if dates:
            if len(landscapes) == len(dates):
                self.dates = dates
            else:
                raise ValueError(
                    "The lengths of `landscapes` and `dates` (if provided) "
                    "must coincide")
        else:
            self.dates = ['t{}'.format(i) for i in range(len(self.landscapes))]

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
                df = landscape.class_metrics_df(metrics=self.class_metrics)
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
                    date] = landscape.landscape_metrics_df(
                        self.landscape_metrics).iloc[0]

            self._landscape_metrics_df = landscape_metrics_df

            return self._landscape_metrics_df

    # def plot_patch_metric(metric):
    #     # TODO: sns distplot?
    #     fig, ax = plt.subplots()
    #     ax.hist()

    def plot_metric(self, metric, class_val=None, ax=None, legend=False,
                    figsize=None):
        metric_vals = [
            getattr(landscape, metric)(class_val)
            for landscape in self.landscapes
        ]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # if self.dates:
        #     ax.plot(self.dates, metric_vals, '--o', label=metric)
        # else:
        #     ax.plot(metric_vals, '--o', label=metric)
        ax.plot(self.dates, metric_vals, '--o', label=metric)

        if legend:
            ax.legend()

        return ax

    def plot_metrics(self, metrics, class_val=None, num_cols=1):
        num_rows = int(np.ceil(len(metrics) / num_cols))
        # TODO: base figsize?
        fig, axes = plt.subplots(num_rows, num_cols, sharex=True,
                                 figsize=(6 * num_cols, 6 * num_rows))
        flat_axes = axes.flatten()
        for metric, ax in zip(metrics, flat_axes):
            self.plot_metric(metric, class_val=class_val, ax=ax)

        # disable axis for the latest cells that correspond to no metric (i.e.,
        # when len(metrics) < num_rows * num_cols)
        for i in range(len(metrics), len(flat_axes)):
            flat_axes[i].axis('off')

        return fig, axes
