import matplotlib.pyplot as plt
import numpy as np

from .landscape import Landscape, read_geotiff

__all__ = ['SpatioTemporalAnalysis']


class SpatioTemporalAnalysis:
    def __init__(self, landscapes, dates=None):
        if isinstance(landscapes[0], Landscape):
            self.landscapes = landscapes
        else:
            self.landscapes = list(map(read_geotiff, landscapes))

        if dates:
            self.dates = dates

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

        if self.dates:
            ax.plot(self.dates, metric_vals, '--o', label=metric)
        else:
            ax.plot(metric_vals, '--o', label=metric)

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
