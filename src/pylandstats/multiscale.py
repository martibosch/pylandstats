"""Multi-scale analysis."""

import geopandas as gpd
import pandas as pd

from pylandstats.zonal import ZonalAnalysis


class MultiScaleAnalysis(ZonalAnalysis):
    """Multi-scale analysis around multiple site locations."""

    def __init__(
        self,
        landscape_filepath,
        site_gser,
        buffer_dists,
        *,
        buffer_rings=False,
        neighborhood_rule=None,
    ):
        """Initialize the multi-scale analysis.

        Parameters
        ----------
        landscape_filepath : str, file-like object or pathlib.Path object
            A string/file-like object/pathlib.Path object with the landscape data.
        site_gser : geopandas.GeoSeries
            Geo-series with the site locations.
        buffer_dists : list-like
            Buffer distances, in units of the landscape CRS.
        buffer_rings : bool, default False
            If `False`, each buffer zone will consist of the whole region that lies
            within the respective buffer distance around the base geometry. If `True`,
            buffer zones will take the form of rings around the base geometry.
        neighborhood_rule : {'8', '4'}, optional
            Neighborhood rule to determine patch adjacencies, i.e: '8' (queen's
            case/Moore neighborhood) or '4' (rook's case/Von Neumann neighborhood).
            Ignored if the passed-in landscapes are `Landscape` instances. If no value
            is provided and the passed-in landscapes are file-like objects or paths, the
            default value set in `settings.DEFAULT_NEIGHBORHOOD_RULE` will be taken.
        """
        site_index_name = site_gser.index.name
        if not site_index_name:
            site_index_name = "site"

        zone_gser = gpd.GeoSeries(
            [
                site_geom.buffer(buffer_dist)
                for site_geom in site_gser
                for buffer_dist in buffer_dists
            ],
            index=pd.MultiIndex.from_product(
                [site_gser.index, buffer_dists],
                names=[site_index_name, "buffer_dist"],
            ),
            crs=site_gser.crs,
        )

        super().__init__(
            landscape_filepath, zone_gser, neighborhood_rule=neighborhood_rule
        )

        # change index to multi-level (site, buffer distance)
        self.landscape_ser.index = pd.MultiIndex.from_tuples(
            self.zone_gser.index, names=[site_index_name, "buffer_dist"]
        )

        def _compute_overlap(gser):
            gdf = gser.reset_index().drop("buffer_dist", axis="columns")
            union_gdf = gdf.overlay(gdf, how="union")
            # return (
            #     union_gdf[
            #         union_gdf[union_gdf.columns[0]] != union_gdf[union_gdf.columns[1]]
            #     ]
            #     .union_all()
            #     .area
            #     / gdf.union_all().area
            # )
            return union_gdf[
                union_gdf[union_gdf.columns[0]] != union_gdf[union_gdf.columns[1]]
            ]

        overlap_gdf = (
            self.zone_gser.groupby("buffer_dist").apply(_compute_overlap).dropna()
        )
        # site_cols = overlap_gdf.columns.drop("geometry")
        site_cols = [f"{site_index_name}_{i + 1}" for i in range(2)]
        overlap_gdf[site_cols] = overlap_gdf[site_cols].astype(
            self.zone_gser.index.dtypes[site_index_name]
        )
        self.overlap_gdf = overlap_gdf
