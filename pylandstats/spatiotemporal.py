from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

        # the `self.classes` attribute will have been set by this instance
        # father's init (namely the `super` in the first line of this method),
        # however some of the classes may not actually be found in any of
        # buffer zones. We therefore need to get the union of the classes
        # found at the spatio-temporal analysis instance of each `buffer_dist`
        self.classes = reduce(np.union1d,
                              tuple(sta.classes for sta in self.stas))

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

    @property
    def class_metrics_df(self):
        try:
            return self._class_metrics_df
        except AttributeError:
            # IMPORTANT: since some classes might not be present for each date
            # and/or buffer distance, we will init the MultiIndex manually to
            # ensure that every class is present in the resulting data frame.
            # If some class does not appear for some some date/buffer distance,
            # the corresponding row will be nan. This probably preferable than
            # having a MultiIndex that can have different levels (i.e., the
            # second level `class_val`) for each buffer distance.
            # Note that this approach is likely slower since for each of
            # the `buffer_dists`, we have to iterate as in (see below):
            # `for class_val, date inclass_metrics_df.loc[buffer_dist].index`
            class_metrics_df = pd.DataFrame(
                index=pd.MultiIndex.from_product(
                    [self.buffer_dists, self.classes, self.dates]),
                columns=self.class_metrics)
            class_metrics_df.index.names = 'buffer_dist', 'class_val', 'dates'
            class_metrics_df.columns.name = 'metric'

            for buffer_dist, sta in zip(self.buffer_dists, self.stas):
                # get the class metrics data frame for the
                # `SpatioTemporalAnalysis` instance that corresponds to this
                # `buffer_dist`
                df = sta.class_metrics_df
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

            # # ALTERNATIVE (POTENTIALLY FASTER) APPROACH
            # # we will create a dict where each key is a `buffer_dist`, and
            # # its value is the corresponding metrics data frame of the
            # # `SpatioTemporalAnalysis` instance
            # df_dict = {
            #     buffer_dist: sta.class_metrics_df
            #     for buffer_dist, sta in zip(self.buffer_dists, self.stas)
            # }

            # # we concatenate each value of the dict dataframe using its
            # # respective `buffer_dist` key to create an extra index level
            # # (i.e., using the `keys` argument of `pd.concat`)
            # class_metrics_df = pd.concat(
            #     df_dict.values(), keys=df_dict.keys())
            # # now we set the name of each index and column level
            # class_metrics_df.index.names = \
            #     'buffer_dist', 'class_val', 'dates'
            # class_metrics_df.columns.name = 'metric'

            self._class_metrics_df = class_metrics_df

            return self._class_metrics_df

    @property
    def landscape_metrics_df(self):
        try:
            return self._landscape_metrics_df
        except AttributeError:
            # PREVIOUS APPROACH
            # landscape_metrics_df = pd.DataFrame(
            #     index=pd.MultiIndex.from_product(
            #         [self.buffer_dists, self.dates]),
            #     columns=self.landscape_metrics)
            # landscape_metrics_df.index.name = 'buffer_dist', 'dates'
            # landscape_metrics_df.columns.name = 'metric'

            # for buffer_dist, sta in zip(self.buffer_dists, self.stas):
            #     # TODO: find out why the `.loc` below does not allocate the
            #     # values correctly when we remove the `.values` suffix of the
            #     # right-hand side (although the resulting dataframes on both
            #     # sides are perfectly aligned)
            #     landscape_metrics_df.loc[
            #         buffer_dist] = sta.landscape_metrics_df.values

            # NEW APPROACH
            # we will create a dict where each key is a `buffer_dist`, and its
            # value is the corresponding metrics data frame of the
            # `SpatioTemporalAnalysis` instance
            df_dict = {
                buffer_dist: sta.landscape_metrics_df
                for buffer_dist, sta in zip(self.buffer_dists, self.stas)
            }

            # we concatenate each value of the dict dataframe using its
            # respective `buffer_dist` key to create an extra index level
            # (i.e., using the `keys` argument of `pd.concat`)
            landscape_metrics_df = pd.concat(df_dict.values(),
                                             keys=df_dict.keys())
            # now we set the name of each index and column level
            landscape_metrics_df.index.names = 'buffer_dist', 'dates'
            landscape_metrics_df.columns.name = 'metric'

            self._landscape_metrics_df = landscape_metrics_df

            return self._landscape_metrics_df

    def plot_metric(self, metric, class_val=None, ax=None, metric_legend=True,
                    metric_label=None, fmt='--o', plot_kws={},
                    subplots_kws={}):
        # TODO: refactor this method so that it uses `class_metrics_df` and
        # `landscape_metrics_df` properties?

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
