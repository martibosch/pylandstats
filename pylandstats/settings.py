import os

try:
    from dotenv import find_dotenv, load_dotenv
    dotenv = True
except ImportError:
    dotenv = False

if dotenv:
    # load environment variables from a '.env' file of a parent directory
    load_dotenv(find_dotenv())

# BASIC DEFINITIONS
fragstats_abbrev_dict = {
    # patch-level metrics
    'area': 'AREA',
    'perimeter': 'PERIM',
    'perimeter_area_ratio': 'PARA',
    'shape_index': 'SHAPE',
    'fractal_dimension': 'FRAC',
    'euclidean_nearest_neighbor': 'ENN',
    # class-level metrics (can also be landscape-level except for PLAND)
    # ACHTUNG: the 'total_area' metric might be 'CA' or 'TA' in FRAGSTATS
    # (depending on whether the metric is computed at the class or landscape
    # level respectively). Nevertheless, considering the implementation/
    # functioning of PyLandStats, making this disctinction in the
    # abbreviations of 'total_area' might be arduous. To simplify, we will use
    # 'TA' in all cases.
    'total_area': 'TA',
    'proportion_of_landscape': 'PLAND',
    'number_of_patches': 'NP',
    'patch_density': 'PD',
    'largest_patch_index': 'LPI',
    'total_edge': 'TE',
    'edge_density': 'ED',
    'landscape_shape_index': 'LSI',
    # landscape-level metrics
    'contagion': 'CONTAG',
    'shannon_diversity_index': 'SHDI'
}
# add the class/landscape distribution statistics metrics to the fragstats
# abbreviation dictionary
for metric in [
        'area', 'perimeter_area_ratio', 'shape_index', 'fractal_dimension',
        'euclidean_nearest_neighbor'
]:
    for suffix in ['mn', 'am', 'md', 'ra', 'sd', 'cv']:
        fragstats_abbrev_dict['{}_{}'.format(metric, suffix)] = '{}_{}'.format(
            fragstats_abbrev_dict[metric], suffix.upper())

# SETTINGS
metric_label_dict = os.environ.get('METRIC_LABEL_DICT', fragstats_abbrev_dict)
