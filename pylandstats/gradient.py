import numpy as np
import rasterio
from rasterio import features

from . import landscape as pls_landscape
from . import multilandscape

try:
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.geometry.base import BaseGeometry
    geo_imports = True
except ImportError:
    geo_imports = False

__all__ = ['GradientAnalysis', 'BufferAnalysis']


class GradientAnalysis(multilandscape.MultiLandscape):
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

        if not isinstance(landscape, pls_landscape.Landscape):
            landscape = pls_landscape.Landscape(landscape)

        landscapes = [
            pls_landscape.Landscape(
                np.where(mask_arr, landscape.landscape_arr,
                         landscape.nodata).astype(
                             landscape.landscape_arr.dtype),
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
                 landscape_transform=None):
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
        if isinstance(landscape, pls_landscape.Landscape):
            if landscape_crs is None:
                raise ValueError(
                    "If passing `Landscape` objects (instead of geotiff "
                    "filepaths), `landscape_crs` must be provided")
            if landscape_transform is None:
                raise ValueError(
                    "If passing `Landscape` objects (instead of geotiff "
                    "filepaths), `landscape_transform` must be provided")
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
        super(BufferAnalysis, self).__init__(landscape, buffer_masks_arr,
                                             'buffer_dists', buffer_dists)

    # override docs
    def compute_class_metrics_df(self, metrics=None, classes=None,
                                 metrics_kws={}):
        return super(BufferAnalysis,
                     self).compute_class_metrics_df(metrics=metrics,
                                                    classes=classes,
                                                    metrics_kws=metrics_kws)

    compute_class_metrics_df.__doc__ = \
        multilandscape._compute_class_metrics_df_doc.format(
            index_descr='multi-indexed by the class and buffer distance',
            index_return='class, buffer distance (multi-index)')

    def compute_landscape_metrics_df(self, metrics=None, metrics_kws={}):
        return super(BufferAnalysis, self).compute_landscape_metrics_df(
            metrics=metrics, metrics_kws=metrics_kws)

    compute_landscape_metrics_df.__doc__ = \
        multilandscape._compute_landscape_metrics_df_doc.format(
            index_descr='indexed by the buffer distance',
            index_return='buffer distance (index)')
