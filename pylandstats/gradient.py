import numpy as np
import rasterio
from rasterio import features

from .landscape import Landscape
from .multilandscape import MultiLandscape

try:
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.geometry.base import BaseGeometry
    geo_imports = True
except ImportError:
    geo_imports = False

__all__ = ['GradientAnalysis', 'BufferAnalysis']


class GradientAnalysis(MultiLandscape):
    def __init__(self, landscape, masks_arr, attribute_name=None,
                 attribute_values=None, **kwargs):
        """
        Parameters
        ----------
        landscapes : list-like
            A list-like of `Landscape` objects or of strings/file objects/
            pathlib.Path objects so that each is passed as the `landscape`
            argument of `Landscape.__init__`
        masks_arr : list-like or np.ndarray
            A list-like of numpy arrays of shape (width, height), i.e., of the
            same shape as the landscape raster image. Each array will serve to
            mask the base landscape and define a region of study for which the
            metrics will be computed separately. The same information can also
            be provided as a single array of shape (num_masks, width, height).
        attribute_name : str, optional
            Name of the attribute that will distinguish each landscape
        attribute_values : str, optional
            Values of the attribute that correspond to each of the landscapes
        """

        if not isinstance(landscape, Landscape):
            landscape = Landscape(landscape)

        landscapes = [
            Landscape(
                np.where(mask_arr, landscape.landscape_arr, landscape.nodata),
                res=(landscape.cell_width, landscape.cell_height),
                nodata=landscape.nodata, transform=landscape.transform)
            for mask_arr in masks_arr
        ]

        # TODO: is it useful to store `masks_arr` as instance attribute?
        self.masks_arr = masks_arr

        # The attribute name will be `buffer_dists` for `BufferAnalysis` or
        # `transect_dist` for `TransectAnalysis`, but for any other custom use
        # of `GradientAnalysis`, the user might provide (or not) a custom name
        if attribute_name is None:
            attribute_name = 'attribute_values'

        # If the values for the distinguishing attribute are not provided, a
        # basic enumeration will be automatically generated
        if attribute_values is None:
            attribute_values = [i for i in range(len(masks_arr))]

        # now call the parent's init
        super(GradientAnalysis, self).__init__(landscapes, attribute_name,
                                               attribute_values, **kwargs)


class BufferAnalysis(GradientAnalysis):
    def __init__(self, landscape, base_mask, buffer_dists, buffer_rings=False,
                 base_mask_crs=None, landscape_crs=None,
                 landscape_transform=None, metrics=None, classes=None,
                 metrics_kws={}):
        """
        Parameters
        ----------
        landscapes : list-like
            A list-like of `Landscape` objects or of strings/file objects/
            pathlib.Path objects so that each is passed as the `landscape`
            argument of `Landscape.__init__`
        base_mask : shapely geometry or geopandas GeoSeries
            Geometry that will serve as a base mask to buffer around
        buffer_dists : list-like
            Buffer distances
        buffer_rings : bool, default False
            If `False`, each buffer zone will consist of the whole region that
            lies within the respective buffer distance around the base mask.
            If `True`, buffer zones will take the form of rings around the
            base mask.
        base_mask_crs : dict, optional
            The coordinate reference system of the base mask. Required if the
            base mask is a shapely geometry or a geopandas GeoSeries without
            the `crs` attribute set
        landscape_crs : dict, optional
            The coordinate reference system of the landscapes. Required if the
            passed-in landscapes are `Landscape` objects, ignored if they are
            paths to GeoTiff rasters that already contain such information.
        landscape_transform : affine.Affine
            Transformation from pixel coordinates to coordinate reference
            system. Required if the passed-in landscapes are `Landscape`
            objects, ignored if they are paths to GeoTiff rasters that already
            contain such information.
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

        # first check that we meet the package dependencies
        if not geo_imports:
            raise ImportError(
                "The `BufferAnalysis` class requires the geopandas package. "
                "For better performance, we strongly suggest that you install "
                "its cythonized version via conda-forge as in:\nconda install "
                "-c conda-forge/label/dev geopandas\n See "
                "https://github.com/geopandas/geopandas for more information "
                "about installing geopandas")

        # get `buffer_masks_arr` from a base geometry and a list of buffer
        # distances
        # 1. get a GeoSeries with the base mask geometry
        if isinstance(base_mask, BaseGeometry):
            if base_mask_crs is None:
                raise ValueError(
                    "If `base_mask` is a shapely geometry, `base_mask_crs` "
                    "must be provided")
            # BufferSpatioTemporalAnalysis.get_buffer_masks_gser(
            base_mask_gser = gpd.GeoSeries(base_mask, crs=base_mask_crs)
        else:
            # we assume that `base_mask` is a geopandas GeoSeries
            if base_mask.crs is None:
                if base_mask_crs is None:
                    raise ValueError(
                        "If `base_mask` is a naive geopandas GeoSeries (with "
                        "no crs set), `base_mask_crs` must be provided")

                base_mask_gser = base_mask.copy()  # avoid alias/ref problems
                base_mask_gser.crs = base_mask_crs
            else:
                base_mask_gser = base_mask

        # 2. get the crs, transform and shape of the landscapes
        if isinstance(landscape, Landscape):
            if landscape_crs is None:
                raise ValueError(
                    "If passing `Landscape` objects (instead of geotiff "
                    "filepaths), `landscapes_crs` must be provided")
            if landscape_transform is None:
                raise ValueError(
                    "If passing `Landscape` objects (instead of geotiff "
                    "filepaths), `landscapes_transform` must be provided")
            landscape_shape = landscape.landscape_arr.shape
        else:
            with rasterio.open(landscape) as src:
                landscape_crs = src.crs
                landscape_transform = src.transform
                landscape_shape = src.height, src.width

        # 3. buffer around base mask
        avg_longitude = base_mask_gser.to_crs(
            '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
        ).unary_union.centroid.x
        # trick from OSMnx to be able to buffer in meters
        utm_zone = int(np.floor((avg_longitude + 180) / 6.) + 1)
        utm_crs = {
            'datum': 'WGS84',
            'ellps': 'WGS84',
            'proj': 'utm',
            'zone': utm_zone,
            'units': 'm'
        }
        base_mask_geom = base_mask_gser.to_crs(utm_crs).iloc[0]
        if buffer_rings:
            if not isinstance(base_mask_geom, Point):
                raise ValueError(
                    "Buffer rings can only work when `base_mask_geom` is a "
                    "`Point`")
            _buffer_dists = np.concatenate([[0], buffer_dists])
            buffer_dists = list(
                map(lambda d: '{}-{}'.format(d[0], d[1]),
                    zip(_buffer_dists[:-1], _buffer_dists[1:])))
            masks_gser = gpd.GeoSeries([
                base_mask_geom.buffer(_buffer_dists[i + 1]) -
                base_mask_geom.buffer(_buffer_dists[i])
                for i in range(len(_buffer_dists) - 1)
            ], index=buffer_dists, crs=utm_crs).to_crs(landscape_crs)
        else:
            masks_gser = gpd.GeoSeries([
                base_mask_geom.buffer(buffer_dist)
                for buffer_dist in buffer_dists
            ], index=buffer_dists, crs=utm_crs).to_crs(landscape_crs)

        # 4. rasterize each mask
        num_rows, num_cols = landscape_shape
        buffer_masks_arr = np.zeros((len(buffer_dists), num_rows, num_cols),
                                    dtype=np.uint8)
        for i in range(len(masks_gser)):
            buffer_masks_arr[i] = features.rasterize(
                [masks_gser.iloc[i]], out_shape=landscape_shape,
                transform=landscape_transform, dtype=np.uint8)

        buffer_masks_arr = buffer_masks_arr.astype(bool)

        # now we can call the parent's init with the landscape and the
        # constructed buffer_masks_arr
        super(BufferAnalysis,
              self).__init__(landscape, buffer_masks_arr, 'buffer_dists',
                             buffer_dists, metrics=metrics, classes=classes,
                             metrics_kws=metrics_kws)

    @property
    def class_metrics_df(self):
        """
        Property that computes the data frame of class-level metrics, which
        is multi-indexed by the class and buffer distance. Once computed, the
        data frame is cached so further calls to the property just access an
        attribute and therefore run in constant time.
        """
        # override so that we can add an explicit docstring
        return super(BufferAnalysis, self).class_metrics_df

    @property
    def landscape_metrics_df(self):
        """
        Property that computes the data frame of landcape-level metrics, which
        is indexed by the buffer distance. Once computed, the data frame is
        cached so further calls to the property just access an attribute and
        therefore run in constant time.
        """
        # override so that we can add an explicit docstring
        return super(BufferAnalysis, self).landscape_metrics_df
