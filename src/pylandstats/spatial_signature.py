"""Spatial signature analysis."""

import warnings

import clustergram
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pylandstats import settings
from pylandstats.multilandscape import MultiLandscape

__all__ = ["SpatialSignatureAnalysis"]


def _compute_metrics_df(
    landscapes,
    class_metrics,
    classes,
    class_metrics_kwargs,
    class_metrics_fillna,
    landscape_metrics,
    landscape_metrics_kwargs,
):
    flat_metrics_dfs = []
    if len(class_metrics) != 0:
        class_metrics_df = landscapes.compute_class_metrics_df(
            metrics=class_metrics,
            classes=classes,
            metrics_kwargs=class_metrics_kwargs,
            fillna=class_metrics_fillna,
        )
        flat_class_metrics_df = class_metrics_df.unstack(level="class_val")
        flat_class_metrics_df.columns = [
            f"{metric}_{class_val}"
            for metric, class_val in flat_class_metrics_df.columns.values
        ]

        flat_metrics_dfs.append(flat_class_metrics_df)
    if len(landscape_metrics) != 0:
        flat_metrics_dfs.append(
            landscapes.compute_landscape_metrics_df(
                metrics=landscape_metrics, metrics_kwargs=landscape_metrics_kwargs
            )
        )
    return pd.concat(flat_metrics_dfs, axis=1)


def _fit_transform(X, transformer, **transformer_kwargs):
    # ACHTUNG: do not modify X in place to avoid side effects
    _X = transformer(**transformer_kwargs).fit_transform(X)
    if isinstance(X, pd.DataFrame):
        _X = pd.DataFrame(_X, index=X.index, columns=X.columns)

    return _X


class SpatialSignatureAnalysis:
    """Pattern-based analysis of landscapes based on spatial signatures."""

    def __init__(
        self,
        landscapes,
        *,
        class_metrics=None,
        landscape_metrics=None,
        classes=None,
        class_metrics_fillna=None,
        class_metrics_kwargs=None,
        landscape_metrics_kwargs=None,
    ):
        """Initialize the spatial signature analysis.

        Parameters
        ----------
        landscapes : pylandstats.MultiLandscape or list-like of pylandstats.Landscape
            A MultiLandscape object or list of the landscapes to be analyzed.
        class_metrics : list-like of str, optional
            A list-like of strings with the names of the metrics that should be
            computed. If `None`, no class-level metric will be computed.
        classes : list-like, optional
            A list-like of ints or strings with the class values that should be
            considered in the context of this analysis case. If `None` and class-level
            metrics are computed, all unique class values will be considered. Ignored if
            no class-level metrics are computed.
        class_metrics_fillna : bool, optional
            Whether `NaN` values representing landscapes with no occurrences of patches
            of the provided class should be replaced by zero when appropriate, e.g.,
            area and edge metrics (no occurrences mean zero area/edge). If the provided
            value is `None` (default), the value will be taken from
            `settings.CLASS_METRICS_DF_FILLNA`.
        class_metrics_kwargs, landscape_metrics_kwargs : dict, optional
            Dictionary mapping the keyword arguments (values) that should be passed to
            each metric method (key) for the class and landscape-level metrics
            respectively. For instance, to exclude the boundary from the computation of
            `total_edge`, metric_kwargs should map the string 'total_edge' (method name)
            to {'count_boundary': False}. If `None`, each metric will be computed
            according to FRAGSTATS defaults.
        """
        # overall idea: we only store the landscapes (and potentially zone_gser) as
        # attributes. Then, any metric can be computed at any point through the
        # `compute_metrics_df` method. This way, the key landscape attributes are cached
        # so that the cost of computing new metrics is dramatically reduced (almost
        # constant/access time in many cases).
        if isinstance(landscapes, MultiLandscape):
            self.landscapes = landscapes
        else:
            self.landscapes = MultiLandscape(
                landscapes, "landscape_id", np.arange(len(landscapes))
            )
        if hasattr(landscapes, "zone_gser"):
            self.zone_gser = landscapes.zone_gser

        if hasattr(landscapes, "dates"):
            self.dates = landscapes.dates

        if class_metrics is None:
            # class_metrics = Landscape.CLASS_METRICS
            class_metrics = []

        if landscape_metrics is None:
            # landscape_metrics = Landscape.LANDSCAPE_METRICS
            landscape_metrics = []

        self.metrics_df = _compute_metrics_df(
            self.landscapes,
            class_metrics,
            classes,
            class_metrics_kwargs,
            class_metrics_fillna,
            landscape_metrics,
            landscape_metrics_kwargs,
        )

    def decompose(
        self,
        *,
        decomposer=None,
        preprocessor=None,
        preprocessor_kwargs=None,
        imputer=None,
        imputer_kwargs=None,
        **decomposer_kwargs,
    ):
        """Factorize the spatial signature matrix into components.

        Parameters
        ----------
        decomposer : class, optional
            A class that implements the decomposition algorithm. It can be any
            scikit-learn like transformer that implements the `fit`, `transform` and
            `fit_transform` methods and with the `components_` and `n_components`
            attributes. If no value is provided, the default value set in
            `settings.DEFAULT_DECOMPOSER` will be taken.
        preprocessor : class, optional
            A class that implements the preprocessing algorithm. It can be any
            scikit-learn like transformer that implements the `fit_transform` method.
            If no value is provided, the default value set in
            `settings.DEFAULT_PREPROCESSOR` will be taken.
        preprocessor_kwargs : dict, optional
            Keyword arguments to be passed to the initializationof `preprocessor`.
        imputer : class, optional
            A class that implements the imputation algorithm. It can be any scikit-learn
            like transformer that implements the `fit_transform` method. If no value is
            provided, no imputation will be performed.
        imputer_kwargs : dict, optional
            Keyword arguments to be passed to the initialization of `imputer`. Ignored
            if `imputer` is `None`.
        **decomposer_kwargs : dict, optional
            Keyword arguments to be passed to the initialization of `decomposer`.

        Returns
        -------
        components_df : pandas.DataFrame
            A DataFrame with the components of the decomposition. Each row corresponds
            to a landscape and each column to a component.
        decomposer_model : object
            The fitted decomposer model.
        """
        # ACHTUNG: using a copy to avoid modifying the original metrics_df
        X = self.metrics_df.copy()

        if preprocessor is None:
            preprocessor = settings.DEFAULT_PREPROCESSOR

        if preprocessor:  # user can provide `preprocessor=False` to skip this step
            if preprocessor_kwargs is None:
                preprocessor_kwargs = {}
            X = _fit_transform(X, preprocessor, **preprocessor_kwargs)

        if imputer is not None:
            if imputer_kwargs is None:
                imputer_kwargs = {}
            X = _fit_transform(X, imputer, **imputer_kwargs)

        if decomposer is None:
            decomposer = settings.DEFAULT_DECOMPOSER
        try:
            # try if the model accepts nan values
            decompose_model = decomposer(**decomposer_kwargs).fit(X)
        except ValueError:
            warnings.warn(
                "The provided spatial signatures contain NaN values which are not "
                "supported by the decomposition model. In order to proceed, the NaN "
                "values will be dropped. However, you may consider either (i) changing "
                "the chosen metrics or (ii) imputing the NaN values by providing the "
                "`imputer` and `imputer_kwargs` arguments.",
                RuntimeWarning,
            )
            X = X.dropna()
            decompose_model = decomposer(**decomposer_kwargs).fit(X)
        # set X to the reduced matrix but as a data frame with the same index as the
        # original metrics' matrix (taking into account the dropped rows if any)
        return pd.DataFrame(
            decompose_model.transform(X), index=X.index
        ), decompose_model

    def get_loading_df(self, decompose_model, *, columns=None, index=None, **df_kwargs):
        """Get components loadings for each metric.

        Parameters
        ----------
        decompose_model : object
            The decomposition model fitted to the spatial signature matrix.
        columns : list-like, optional
            Column names for the components. If no value is provided, an integer range
            from 0 to `n_components_ - 1` will be used.
        index : list-like, optional
            Index names for the metrics. If no value is provided, the column names of
            `metrics_df` will be used.
        **df_kwargs : dict, optional
            Keyword arguments to be passed to the initialization of `pandas.DataFrame`

        Returns
        -------
        loading_df : pandas.DataFrame
            A DataFrame with the loadings of the components. Each row corresponds to a
            metric and each column to a component.
        """
        if df_kwargs is None:
            _df_kwargs = {}
        else:
            _df_kwargs = df_kwargs.copy()
        if columns is None:
            columns = _df_kwargs.pop(
                "columns",
                range(decompose_model.n_components_),
            )
        if index is None:
            index = _df_kwargs.pop("index", self.metrics_df.columns)
        return pd.DataFrame(
            decompose_model.components_.T, columns=columns, index=index, **_df_kwargs
        )

    def get_cgram(
        self,
        *,
        k_range=None,
        decomposer=None,
        decomposer_kwargs=None,
        preprocessor=None,
        preprocessor_kwargs=None,
        imputer=None,
        imputer_kwargs=None,
        **clustergram_kwargs,
    ):
        """
        Get the clustergram of the spatial signature matrix.

        Parameters
        ----------
        k_range : list-like, optional
            A list-like of integer values with the number of clusters to be tested. If
            no value is provided, the default value set in `settings.DEFAULT_K_RANGE`
            will be taken.
        decomposer : class, optional
            A class that implements the decomposition algorithm. It can be any
            scikit-learn like transformer that implements the `fit`, `transform` and
            `fit_transform` methods and with the `components_` and `n_components`
            attributes. If no value is provided, the clustering algorithm will be
            applied to the original spatial signature matrix.
        decomposer_kwargs : dict, optional
            Keyword arguments to be passed to the initialization of `decomposer`.
            Ignored if `decomposer` is `None`.
        preprocessor : class, optional
            A class that implements the preprocessing algorithm. It can be any
            scikit-learn like transformer that implements the `fit_transform` method.
            If no value is provided, the default value set in
            `settings.DEFAULT_PREPROCESSOR` will be taken.
        preprocessor_kwargs : dict, optional
            Keyword arguments to be passed to the initializationof `preprocessor`.
        imputer : class, optional
            A class that implements the imputation algorithm. It can be any scikit-learn
            like transformer that implements the `fit_transform` method. If no value is
            provided, no imputation will be performed.
        imputer_kwargs : dict, optional
            Keyword arguments to be passed to the initialization of `imputer`. Ignored
            if `imputer` is `None`.
        **clustergram_kwargs : dict, optional
            Keyword arguments to be passed to the initialization of `Clustergram`.

        Returns
        -------
        cgram : clustergram.Clustergram
            The fitted clustergram object.
        """
        # TODO
        # if cluster_traj:
        #     self.flat_metrics_df = self.flat_metrics_df.unstack(level="dates")
        #     self.flat_metrics_df.columns = [
        #         f"{metric}_{year}"
        #         for metric, year in self.flat_metrics_df.columns.values
        #     ]
        if decomposer is not None:
            if decomposer_kwargs is None:
                decomposer_kwargs = {}
            X, _ = self.decompose(
                decomposer=decomposer,
                preprocessor=preprocessor,
                preprocessor_kwargs=preprocessor_kwargs,
                imputer=imputer,
                imputer_kwargs=imputer_kwargs,
                **decomposer_kwargs,
            )
        else:
            # if no decomposer is provided, we can still preprocess and impute the data
            X = self.metrics_df.copy()
            if preprocessor is None:
                preprocessor = settings.DEFAULT_PREPROCESSOR
            if preprocessor_kwargs is None:
                preprocessor_kwargs = {}
            X = _fit_transform(X, preprocessor, **preprocessor_kwargs)
            if imputer is not None:
                if imputer_kwargs is None:
                    imputer_kwargs = {}
                X = _fit_transform(X, imputer, **imputer_kwargs)

        if clustergram_kwargs is None:
            clustergram_kwargs = {}
        if k_range is None:
            # TODO: use settings for the defaults
            # no need to pop k_range from the kwargs since it is already a keyword-only
            # argument of this method
            # k_range = _clustergram_kwargs.pop("k_range", settings.DEFAULT_K_RANGE)
            k_range = settings.DEFAULT_K_RANGE
        try:
            cgram = clustergram.Clustergram(k_range=k_range, **clustergram_kwargs).fit(
                X
            )
        except ValueError:
            warnings.warn(
                "The provided spatial signatures contain NaN values which are not "
                "supported by the clustering model. In order to proceed, the NaN values"
                " will be dropped. However, you may consider either (i) changing the "
                "chosen metrics or (ii) imputing the NaN values by providing the "
                "`imputer` and `imputer_kwargs` arguments.",
                RuntimeWarning,
            )
            X = X.dropna()
            cgram = clustergram.Clustergram(k_range=k_range, **clustergram_kwargs).fit(
                X
            )
        # update cgram.data in place so that it is a data frame (instead of a numpy
        # ndarray) with the same indices and columns as `self.metrics_df`. This ensures
        # that we properly match a cluster label to a landscape (crucial if we have
        # dropped the rows of landscapes with NaN values)
        cgram.data = pd.DataFrame(cgram.data, index=X.index, columns=X.columns)
        return cgram

    @staticmethod
    def scatterplot_cluster_metrics(
        cgram,
        n_clusters,
        metric_x,
        metric_y,
        *,
        ax=None,
        palette_name=None,
        center_marker="x",
        center_plot_kwargs=None,
        **scatterplot_kwargs,
    ):
        """Scatterplot the landscape samples colored by their cluster.

        Parameters
        ----------
        cgram : clustergram.Clustergram
            Fitted clustergram object.
        n_clusters : int
            Number of clusters to use.
        metric_x, metric_y : str
            Strings with the names of the metrics to be plotted on the x and y axes
            respectively. Should be a column of `cgram.data`. Note thus that for
            class-level metrics, the passed string should be of the form
            "{metric}_{class_val}".
        ax : matplotlib.axes.Axes, optional
            Axes object to draw the plot onto, otherwise create a new figure.
        palette_name : str, optional
            Name of palette or None to return current palette.
        center_marker : str, optional, default "x"
            Marker to be used for the cluster centers.
        center_plot_kwargs : dict, optional
            Keyword arguments to plot the cluster centers, which will be passed to
            `matplotlib.axes.Axes.plot`.
        scatterplot_kwargs : dict, optional
            Keyword arguments to be passed to `seaborn.scatterplot`.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Returns the `Axes` object with the plot drawn onto it.

        """
        if scatterplot_kwargs is None:
            _scatterplot_kwargs = {}
        else:
            _scatterplot_kwargs = scatterplot_kwargs.copy()
        if center_plot_kwargs is None:
            _center_plot_kwargs = {}
        else:
            _center_plot_kwargs = center_plot_kwargs.copy()
            center_marker = _center_plot_kwargs.pop("marker", center_marker)
        _center_plot_kwargs["marker"] = center_marker
        if ax is None:
            ax = _scatterplot_kwargs.pop("ax", None)
        if ax is None:
            _, ax = plt.subplots()

        cluster_centers = cgram.cluster_centers[n_clusters][
            :,
            [cgram.data.columns.get_loc(metric) for metric in [metric_x, metric_y]],
        ]
        n_clusters = cluster_centers.shape[0]
        palette = _scatterplot_kwargs.pop(
            "palette", sns.color_palette(palette_name, n_colors=n_clusters)
        )
        sns.scatterplot(
            x=metric_x,
            y=metric_y,
            hue=pd.Series(cgram.labels_[n_clusters], name="cluster").values,
            data=cgram.data,
            palette=palette,
            ax=ax,
            **_scatterplot_kwargs,
        )

        for color, cluster_center in zip(palette, cluster_centers):
            ax.plot(
                cluster_center[0],
                cluster_center[1],
                color=color,
                **_center_plot_kwargs,
            )

        return ax

    def plot_cluster_landscapes(
        self,
        cgram,
        n_clusters,
        *,
        n_cluster_landscapes=4,
        n_cols=4,
        sample_kwargs=None,
        figure_kwargs=None,
        subfigures_kwargs=None,
        subplots_kwargs=None,
        supylabel_kwargs=None,
        **plot_landscape_kwargs,
    ):
        """Scatterplot the landscape samples colored by their cluster.

        Parameters
        ----------
        cgram : Clustergram
            Fitted clustergram object.
        n_clusters : int
            Number of clusters to use.
        n_cluster_landscapes : int, optional, default 4
            Number of landscapes to plot for each cluster. Providing a value of None
            will plot all landscapes.
        n_cols : int, optional, default 4
            Number of columns for the figure.
        sample_kwargs, figure_kwargs, subfigures_kwargs, subplots_kwargs, \
            supylabel_kwargs, plot_landscape_kwargs : dict, optional
            Keyword arguments to be passed to `pandas.DataFrame.sample`,
            `matplotlib.pyplot.figure`,
            `matplotlib.figure.Figure.subfigures`,
            `matplotlib.figure.SubFigure.subplots`,
            `matplotlib.figure.SubFigure.supylabel` and
            `pylandstats.Landscape.plot_landscape` respectively.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure with its corresponding plots drawn into its axes.
        """
        if figure_kwargs is None:
            _figure_kwargs = {}
        else:
            _figure_kwargs = figure_kwargs.copy()
        if subfigures_kwargs is None:
            subfigures_kwargs = {}
        if subplots_kwargs is None:
            subplots_kwargs = {}
        if supylabel_kwargs is None:
            supylabel_kwargs = {}
        if plot_landscape_kwargs is None:
            _plot_landscape_kwargs = {}
        else:
            _plot_landscape_kwargs = plot_landscape_kwargs.copy()
            _ = _plot_landscape_kwargs.pop("ax", None)

        labels = cgram.labels_[n_clusters]
        metrics_df = self.metrics_df.loc[cgram.data.index]
        if n_cluster_landscapes is None:
            # plot all landscapes
            n_rows_per_cluster = np.ceil(
                np.unique(labels, return_counts=True)[1] / n_cols
            ).astype(int)
        else:
            n_rows_per_cluster = np.full(
                n_clusters, np.ceil(n_cluster_landscapes / n_cols), dtype=int
            )
            if sample_kwargs is None:
                _sample_kwargs = {}
            else:
                _sample_kwargs = sample_kwargs.copy()
                _ = _sample_kwargs.pop("frac", None)
            # use sample to shuffle then groupby and head (because head does not raise
            # an error if the group has less elements than n_cluster_landscapes)
            metrics_df = (
                self.metrics_df.loc[cgram.data.index]
                .sample(frac=1, **_sample_kwargs)
                .groupby(labels)
                .head(n_cluster_landscapes)
            )
            # update labels to match the new metrics_df
            labels = pd.Series(labels, index=cgram.data.index)[metrics_df.index].astype(
                int
            )

        n_rows = n_rows_per_cluster.sum()

        figsize = _figure_kwargs.pop("figsize", None)
        if figsize is None:
            figwidth, figheight = plt.rcParams["figure.figsize"]
            figsize = (figwidth * n_cols, figheight * n_rows)
        fig = plt.figure(figsize=figsize, **_figure_kwargs)
        # gs = fig.add_gridspec(n_rows, n_cols)
        subfigs = fig.subfigures(
            n_clusters,
            1,
            height_ratios=n_rows_per_cluster,
            **subfigures_kwargs,
        )

        # TODO: manage nan metrics better in case or reduction or not
        for subfig, n_cluster_rows, (cluster_label, _cluster_df) in zip(
            subfigs,
            n_rows_per_cluster,
            metrics_df.groupby(labels.values),
        ):
            # n_cluster_rows = np.ceil(len(_cluster_df) / n_cols).astype(int)
            axes = subfig.subplots(n_cluster_rows, n_cols, **subplots_kwargs)
            try:
                axes = axes.flat
            except AttributeError:
                axes = [axes]
            subfig.supylabel(f"Cluster {cluster_label}", **supylabel_kwargs)

            for ls_i, ax in zip(_cluster_df.index, axes):
                self.landscapes.landscape_ser[ls_i].plot_landscape(
                    ax=ax, **plot_landscape_kwargs
                )
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            # if the cluster does not fill all the columns of the last row, set the
            # respective axes off
            n_empty_axes = len(axes) - len(_cluster_df) % n_cols
            if n_empty_axes > 0:
                for ax in axes[-n_empty_axes:]:
                    ax.set_axis_off()

        return fig

    def plot_cluster_zones(
        self,
        cgram,
        n_clusters,
        *,
        legend=True,
        categorical=True,
        ax=None,
        subplots_kwargs=None,
        **plot_kwargs,
    ):
        """Plot the landscape zones colored by their cluster.

        Only available if the `SpatialSignatureAnalysis` object has been initialized
        with a zonal analysis class (or subclass). If the `SpatialSignatureAnalysis`
        object has been initialized with a spatio-temporal zonal analysis class (or
        subclass), each date snapshot will be plotted separately.

        Parameters
        ----------
        cgram : Clustergram
            Fitted clustergram object.
        n_clusters : int
            Number of clusters to use.
        ax : matplotlib.axes.Axes, optional
            Axes object to draw the plot onto, otherwise create a new figure. Ignored
            if the `SpatialSignatureAnalysis` object has been initialized with a
            spatio-temporal zonal analysis class (or subclass).
        legend : bool, optional, default True
            Whether to show the legend.
        categorical : bool, optional, default True
            Whether the cluster color map should be categorical.
        subplots_kwargs, plot_kwargs : dict, default None
            Keyword arguments to be passed to `matplotlib.pyplot.subplots` and
            `geopandas.GeoDataFrame.plot` respectively.

        Returns
        -------
        matplotlib.figure.Figure or matplotlib.axes.Axes
            If the `SpatialSignatureAnalysis` object has been initialized with a
            spatio-temporal zonal analysis class (or subclass), returns the figure with
            its corresponding plots drawn into its axes. Otherwise, returns the axes
            object with the plot drawn onto it.
        """
        if not hasattr(self, "zone_gser"):
            warnings.warn(
                "The `plot_cluster_zones` method can only be called when"
                " `SpatialSignatureAnalysis` has been initialized with a zonal analysis"
                " class (or sub-class) as the `landscapes` argument. Returning None",
                RuntimeWarning,
            )
            return None

        def _plot(cluster_labels, cgram_data, ax):
            return gpd.GeoDataFrame(
                {"cluster": cluster_labels},
                geometry=cgram_data.index.map(self.zone_gser).values,
            ).plot(
                "cluster", legend=legend, categorical=categorical, ax=ax, **plot_kwargs
            )

        if ax is None or hasattr(self, "dates"):
            if subplots_kwargs is None:
                _subplots_kwargs = {}
            else:
                _subplots_kwargs = subplots_kwargs.copy()

            if hasattr(self, "dates"):
                figsize = _subplots_kwargs.pop(
                    "figsize",
                    (
                        plt.rcParams["figure.figsize"][0] * len(self.dates),
                        plt.rcParams["figure.figsize"][1],
                    ),
                )
                fig, axes = plt.subplots(
                    1, len(self.dates), figsize=figsize, **_subplots_kwargs
                )
                for date, ax in zip(self.dates, axes):
                    date_query = cgram.data.index.get_level_values("date") == date
                    _plot(
                        cgram.labels_[n_clusters][date_query],
                        cgram.data[date_query].droplevel("date"),
                        ax,
                    )
                    ax.set_title(date)
                return fig
            elif ax is None:
                figsize = _subplots_kwargs.pop(
                    "figsize", plt.rcParams["figure.figsize"]
                )
                _, ax = plt.subplots(figsize=figsize, **_subplots_kwargs)

        return _plot(cgram.labels_[n_clusters], cgram.data, ax)
