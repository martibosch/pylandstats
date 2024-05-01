"""pylandstats settings."""

from os import environ

try:
    import dotenv

    # load environment variables from a '.env' file of a parent directory
    dotenv.load_dotenv(dotenv.find_dotenv())
except ImportError:
    pass

# BASIC DEFINITIONS
fragstats_abbrev_dict = {
    # patch-level metrics
    "area": "AREA",
    "perimeter": "PERIM",
    "perimeter_area_ratio": "PARA",
    "shape_index": "SHAPE",
    "fractal_dimension": "FRAC",
    "core_area": "CORE",
    "number_of_core_areas": "NCORE",
    "core_area_index": "CAI",
    "euclidean_nearest_neighbor": "ENN",
    # class-level metrics (can also be landscape-level except for PLAND)
    # ACHTUNG: the 'total_area' metric might be 'CA' or 'TA' in FRAGSTATS (depending on
    # whether the metric is computed at the class or landscape level respectively).
    # Nevertheless, considering the implementation/functioning of PyLandStats, making
    # this distinction in the abbreviations of 'total_area' might be arduous. To
    # simplify, we will use 'TA' in all cases.
    "total_area": "TA",
    "proportion_of_landscape": "PLAND",
    "number_of_patches": "NP",
    "patch_density": "PD",
    "largest_patch_index": "LPI",
    "total_edge": "TE",
    "edge_density": "ED",
    "landscape_shape_index": "LSI",
    "effective_mesh_size": "MESH",
    # landscape-level metrics
    "contagion": "CONTAG",
    "shannon_diversity_index": "SHDI",
}
# add the class/landscape distribution statistics metrics to the fragstats abbreviation
# dictionary

for metric in [
    "area",
    "perimeter",
    "perimeter_area_ratio",
    "shape_index",
    "fractal_dimension",
    "core_area",
    "core_area_index",
    "euclidean_nearest_neighbor",
]:
    for suffix in ["mn", "am", "md", "ra", "sd", "cv"]:
        fragstats_abbrev_dict[f"{metric}_{suffix}"] = (
            f"{fragstats_abbrev_dict[metric]}_{suffix.upper()}"
        )
for suffix in ["mn", "am", "md", "ra", "sd", "cv"]:
    fragstats_abbrev_dict[f"disjunct_core_area_{suffix}"] = f"DCORE_{suffix}"

# SETTINGS
# TODO: is it worth integrating `metrics` and `metrics_kwargs` into the settings scheme?
# The main difficulty is that depending on the method, the `metrics` argument might
# concern only patch-level metrics, class-level metrics (or landscape-level metrics,
# e.g., see the methods of the form `landscape.Landscape.compute_{level}_metrics_df`,
# where 'level' can be `patch`, `class` or `landscape`. On the other hand, integrating
# `metrics_kwargs` should be more straight-forward.
metric_label_dict = environ.get("METRIC_LABEL_DICT", fragstats_abbrev_dict)

# OTHER
DEFAULT_LANDSCAPE_NODATA = 0
DEFAULT_NEIGHBORHOOD_RULE = "8"
CLASS_METRICS_DF_FILLNA = True
