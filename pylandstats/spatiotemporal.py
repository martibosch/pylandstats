from .multilandscape import MultiLandscape

__all__ = ['SpatioTemporalAnalysis']


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
