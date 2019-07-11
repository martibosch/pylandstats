import matplotlib.pyplot as plt
import numpy as np

from .gradient import BufferAnalysis
from .landscape import Landscape
from .multilandscape import MultiLandscape

__all__ = ['SpatioTemporalAnalysis', 'SpatioTemporalBufferAnalysis']


class SpatioTemporalAnalysis(MultiLandscape):
    def __init__(self, landscapes, metrics=None, classes=None, dates=None,
                 metrics_kws={}):
        """
        Parameters
        ----------
        landscapes : list-like
            A list-like of `Landscape` objects or of strings/file objects/
            pathlib.Path objects so that each is passed as the `landscape`
            argument of `Landscape.__init__`
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

        if dates is None:
            dates = ['t{}'.format(i) for i in range(len(landscapes))]

        # Call the parent's init
        super(SpatioTemporalAnalysis,
              self).__init__(landscapes, 'dates', dates, metrics=metrics,
                             classes=classes, metrics_kws=metrics_kws)

    # def plot_patch_metric(metric):
    #     # TODO: sns distplot?
    #     fig, ax = plt.subplots()
    #     ax.hist()


class SpatioTemporalBufferAnalysis(SpatioTemporalAnalysis):
    def __init__(self, landscapes, base_mask, buffer_dists, buffer_rings=False,
                 base_mask_crs=None, landscape_crs=None,
                 landscape_transform=None, metrics=None, classes=None,
                 dates=None, metrics_kws={}):
        """
        Parameters
        ----------
        landscapes : list-like
            A list-like of `Landscape` objects or of strings/file objects/
            pathlib.Path objects so that each is passed as the `landscape`
            argument of `Landscape.__init__`
        base_mask : 
        buffer_rings : 
        base_mask_crs : 
        landscape_crs : 
        landscape_transform :
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
        super(SpatioTemporalBufferAnalysis,
              self).__init__(landscapes, metrics=metrics, classes=classes,
                             dates=dates, metrics_kws=metrics_kws)
        ba = BufferAnalysis(
            landscapes[0], base_mask=base_mask, buffer_dists=buffer_dists,
            buffer_rings=buffer_rings, base_mask_crs=base_mask_crs,
            landscape_crs=landscape_crs,
            landscape_transform=landscape_transform, metrics=metrics,
            classes=classes, metrics_kws=metrics_kws)
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
                    Landscape(
                        np.where(mask_arr, landscape.landscape_arr,
                                 landscape.nodata),
                        res=(landscape.cell_width, landscape.cell_height),
                        nodata=landscape.nodata, transform=landscape.transform)
                    for landscape in self.landscapes
                ], metrics=metrics, classes=classes, dates=dates,
                                       metrics_kws=metrics_kws))

    def plot_metric(self, metric, class_val=None, ax=None, metric_legend=True,
                    metric_label=None, fmt='--o', plot_kws={},
                    subplots_kws={}):
        # for buffer_analysis in self.buffer_analyses
        if ax is None:
            fig, ax = plt.subplots(**subplots_kws)

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

        return ax
