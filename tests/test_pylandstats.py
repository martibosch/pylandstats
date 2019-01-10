def test_base_imports():
    import numpy as np  # noqa: F401
    from scipy import ndimage  # noqa: F401


def test_io():
    import pylandstats as pls

    ls = pls.read_geotiff('tests/input_data/ls.tif')
    assert ls.cell_width == 250
    assert ls.cell_height == 250
    assert ls.cell_area == 250 * 250


def test_patch_level_metrics_parameters():
    import numpy as np
    import pandas as pd
    import pylandstats as pls

    ls_arr = np.load('tests/input_data/ls.npy')
    ls = pls.Landscape(ls_arr, res=(250, 250))

    for metric in [
            'area', 'perimeter', 'perimeter_area_ratio', 'shape_index',
            'fractal_dimension'
    ]:  # 'contiguity_index', 'euclidean_nearest_neighbor', 'proximity'
        method = getattr(ls, metric)
        assert isinstance(method(), pd.DataFrame)
        assert isinstance(method(class_val=ls.classes[0]), pd.Series)


def test_class_landscape_level_metrics_parameters():
    import numpy as np
    import pylandstats as pls

    ls_arr = np.load('tests/input_data/ls.npy')
    ls = pls.Landscape(ls_arr, res=(250, 250))

    _suffixes = ['mn', 'am', 'md', 'ra', 'sd', 'cv']

    class_metrics = [
        'total_area',
        'proportion_of_landscape',
        'number_of_patches',
        'patch_density',
        'largest_patch_index',
        'total_edge',
        'edge_density',
        'landscape_shape_index',
    ] + ['area_{}'.format(suffix) for suffix in _suffixes] + [
        'perimeter_area_ratio_{}'.format(suffix) for suffix in _suffixes
    ] + ['shape_index_{}'.format(suffix) for suffix in _suffixes
         ] + ['fractal_dimension_{}'.format(suffix) for suffix in _suffixes]

    for class_metric in class_metrics:
        assert np.isreal(getattr(ls, class_metric)(class_val=ls.classes[0]))

    # these are the metrics computable at the landscape level
    class_metrics.remove('proportion_of_landscape')

    for landscape_metric in class_metrics:
        assert np.isreal(getattr(ls, landscape_metric)())


def test_metric_dataframes():
    import numpy as np
    from pandas.api.types import is_string_dtype
    import pylandstats as pls

    ls_arr = np.load('tests/input_data/ls.npy')
    ls = pls.Landscape(ls_arr, res=(250, 250))

    patch_df = ls.patch_metrics_df()
    assert is_string_dtype(patch_df.columns)
    assert patch_df.index.name == 'patch_id'

    class_df = ls.class_metrics_df()
    assert is_string_dtype(class_df.columns)
    assert class_df.index.name == 'class_val'

    landscape_df = ls.landscape_metrics_df()
    assert is_string_dtype(landscape_df.columns)
    assert len(landscape_df.index) == 1


def test_landscape_metrics_value_ranges():
    import numpy as np
    import pylandstats as pls

    ls_arr = np.load('tests/input_data/ls.npy')
    ls = pls.Landscape(ls_arr, res=(250, 250))

    # basic tests of the `Landscape` class' attributes
    assert ls.nodata not in ls.classes
    assert ls.landscape_area > 0

    class_val = ls.classes[0]
    # label_arr = ls._get_label_arr(class_val)

    # patch-level metrics
    assert (ls.area()['area'] > 0).all()
    assert (ls.perimeter()['perimeter'] > 0).all()
    assert (ls.perimeter_area_ratio()['perimeter_area_ratio'] > 0).all()
    assert (ls.shape_index()['shape_index'] >= 1).all()
    _fractal_dimension_ser = ls.fractal_dimension()['fractal_dimension']
    assert (_fractal_dimension_ser >= 1).all() and (_fractal_dimension_ser <=
                                                    2).all()
    # TODO: assert 0 <= ls.contiguity_index(patch_arr) <= 1
    # TODO: assert 0 <= ls.euclidean_nearest_neighbor(patch_arr) <= 1
    # TODO: assert 0 <= ls.proximity(patch_arr) <= 1

    # class-level metrics
    assert ls.total_area(class_val) > 0
    assert 0 < ls.proportion_of_landscape(class_val) < 100
    assert ls.patch_density(class_val) > 0
    assert 0 < ls.largest_patch_index(class_val) < 100
    assert ls.total_edge(class_val) >= 0
    assert ls.edge_density(class_val) >= 0

    # the value ranges of mean and area-weighted mean aggregations are going
    # to be the same as their respective original metrics
    mean_suffixes = ['_mn', '_am']
    # the value ranges of the standard deviation and coefficient of variation
    # will always be nonnegative as long as the means are nonnegative as well
    # (which is the case of all of the metrics implemented so far)
    var_suffixes = ['_sd', '_cv']

    for mean_suffix in mean_suffixes:
        assert getattr(ls, 'area' + mean_suffix)(class_val) > 0
        assert getattr(ls, 'perimeter_area_ratio' + mean_suffix)(class_val) > 0
        assert getattr(ls, 'shape_index' + mean_suffix)(class_val) >= 1
        assert 1 <= getattr(ls,
                            'fractal_dimension' + mean_suffix)(class_val) <= 2
        # assert 0 <= getattr(
        #     ls, 'contiguity_index' + mean_suffix)(class_val) <= 1
        # assert getattr(ls, 'proximity' + mean_suffix)(class_val) >= 0
        # assert getattr(
        #     ls, 'euclidean_nearest_neighbor' + mean_suffix)(class_val) >

    for var_suffix in var_suffixes:
        assert getattr(ls, 'area' + mean_suffix)(class_val) >= 0
        assert getattr(ls, 'perimeter_area_ratio' + var_suffix)(class_val) >= 0
        assert getattr(ls, 'shape_index' + var_suffix)(class_val) >= 0
        assert getattr(ls, 'fractal_dimension' + var_suffix)(class_val) >= 0
        # assert getattr(ls, 'contiguity_index' + var_suffix)(class_val) >= 0
        # assert getattr(ls, 'proximity' + var_suffix)(class_val) >= 0
        # assert getattr(
        #     ls, 'euclidean_nearest_neighbor' + var_suffix)(class_val) >= 0

    # TODO: assert 0 < ls.interspersion_juxtaposition_index(class_val) <= 100
    assert ls.landscape_shape_index(class_val) >= 1

    # landscape-level metrics
    assert ls.total_area() > 0
    assert ls.patch_density() > 0
    assert 0 < ls.largest_patch_index() < 100
    assert ls.total_edge() >= 0
    assert ls.edge_density() >= 0
    assert 0 < ls.largest_patch_index() <= 100
    assert ls.total_edge() >= 0
    assert ls.edge_density() >= 0

    # for class_val in ls.classes:
    #     print('num_patches', class_val, ls._get_num_patches(class_val))
    #     print('patch_areas', len(ls._get_patch_areas(class_val)))

    # raise ValueError

    for mean_suffix in mean_suffixes:
        assert getattr(ls, 'area' + mean_suffix)() > 0
        assert getattr(ls, 'perimeter_area_ratio' + mean_suffix)() > 0
        assert getattr(ls, 'shape_index' + mean_suffix)() >= 1
        assert 1 <= getattr(ls, 'fractal_dimension' + mean_suffix)() <= 2
    #     assert 0 <= getattr(ls, 'contiguity_index' + mean_suffix)() <= 1
    #     assert getattr(ls, 'proximity' + mean_suffix)() >= 0
    #     assert getattr(ls, 'euclidean_nearest_neighbor' + mean_suffix)() > 0
    for var_suffix in var_suffixes:
        assert getattr(ls, 'area' + var_suffix)() > 0
        assert getattr(ls, 'perimeter_area_ratio' + var_suffix)() >= 0
        assert getattr(ls, 'shape_index' + var_suffix)() >= 0
        assert getattr(ls, 'fractal_dimension' + var_suffix)() >= 0
    #     assert getattr(ls, 'contiguity_index' + var_suffix)() >= 0
    #     assert getattr(ls, 'proximity' + var_suffix)() >= 0
    #     assert getattr(ls, 'euclidean_nearest_neighbor' + var_suffix)() >= 0

    # TODO: assert 0 < ls.contagion() <= 100
    # TODO: assert 0 < ls.interspersion_juxtaposition_index() <= 100
    # TODO: assert ls.shannon_diversity_index() >= 0


def test_spatiotemporalanalysis():
    import numpy as np
    import pylandstats as pls

    res = (250, 250)

    landscapes = [
        pls.Landscape(np.load(fp), res=res) for fp in
        ['tests/input_data/ls.npy', 'tests/input_data/ls_future.npy']
    ]

    sta = pls.SpatioTemporalAnalysis(landscapes, dates=[2012, 2018])

    # TODO: test legend and figsize

    ax = sta.plot_metric('patch_density', class_val=None)
    assert len(ax.lines) == 1
    ax = sta.plot_metric('patch_density', class_val=54, ax=ax)
    assert len(ax.lines) == 2

    fig, axes = sta.plot_metrics(['edge_density', 'patch_density'],
                                 class_val=54)
    assert len(axes) == 2
