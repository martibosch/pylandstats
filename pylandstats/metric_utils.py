import numpy as np
from scipy import ndimage

KERNEL_HORIZONTAL = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.int8)
KERNEL_VERTICAL = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.int8)
KERNEL_MOORE = ndimage.generate_binary_structure(2, 2)

# compute methods to obtain a scalar from an array


def compute_arr_perimeter(arr, cell_width, cell_height):
    return np.sum(arr[1:, :] != arr[:-1, :]) * cell_width + np.sum(
        arr[:, 1:] != arr[:, :-1]) * cell_height


def compute_arr_edge(arr, landscape_arr, cell_width, cell_height, nodata=0):
    """
    Computes the edge of a feature considering the landscape background in
    order to exclude the edges between the feature and nodata values
    """
    # check self.nodata in class_arr?
    class_cond = arr != nodata
    # class_with_bg_arr = np.copy(self.landscape_arr)
    # class_with_bg_arr[~class_cond] = self.landscape_arr[~class_cond]
    # get a 'boolean-like' integer array where one indicates that the cell
    # corresponds to some class value whereas zero indicates that the cell
    # corresponds to a nodata value
    data_arr = (landscape_arr != nodata).astype(np.int8)

    # use a convolution to determine which edges should be exluded from the
    # perimeter's width and height
    perimeter_width = np.sum(arr[1:, :] != arr[:-1, :]) + np.sum(
        ndimage.convolve(data_arr, KERNEL_VERTICAL)[class_cond] - 3)
    perimeter_height = np.sum(arr[:, 1:] != arr[:, :-1]) + np.sum(
        ndimage.convolve(data_arr, KERNEL_HORIZONTAL)[class_cond] - 3)

    return perimeter_width * cell_width + perimeter_height * cell_height


# compute methods to obtain patchwise scalars


def compute_patch_areas(label_arr, cell_area):
    # we could use `ndimage.find_objects`, but since we do not need to
    # preserve the feature shapes, `np.bincount` is much faster
    return np.bincount(label_arr.ravel())[1:] * cell_area


def compute_patch_perimeters(label_arr, cell_width, cell_height):
    # NOTE: performance comparison of `patch_perimeters` as np.array of
    # fixed size with `patch_perimeters[i] = ...` within the loop is
    # slower and less Pythonic but can lead to better performances if
    # optimized via Cython/numba
    patch_perimeters = []
    # `ndimage.find_objects` only finds the (rectangular) bounds; there
    # might be parts of other patches within such bounds, so we need to
    # check which pixels correspond to the patch of interest. Since
    # `ndimage.label` labels patches with an enumeration starting by 1, we
    # can use Python's built-in `enumerate`
    # NOTE: feature-wise iteration could this be done with
    # `ndimage.labeled_comprehension(
    #     label_arr, label_arr, np.arange(1, num_patches + 1),
    #     _compute_arr_perimeter, np.float, default=None)`
    # ?
    # I suspect no, because each feature array is flattened, which does
    # not allow for the computation of the perimeter or other shape metrics
    for i, patch_slice in enumerate(ndimage.find_objects(label_arr), start=1):
        patch_arr = np.pad(label_arr[patch_slice] == i, pad_width=1,
                           mode='constant',
                           constant_values=False)  # self.nodata

        patch_perimeters.append(
            compute_arr_perimeter(patch_arr, cell_width, cell_height))

    return patch_perimeters


def compute_perimeter_area_ratio(area_ser, perimeter_ser):
    return perimeter_ser / area_ser


def compute_shape_index(area_cells_ser, perimeter_cells_ser):
    n = np.floor(np.sqrt(area_cells_ser))
    m = area_cells_ser - n**2
    min_p = np.ones(len(area_cells_ser))
    min_p = np.where(m == 0, 4 * n, min_p)
    min_p = np.where((n**2 < area_cells_ser) & (area_cells_ser <= n * (n + 1)),
                     4 * n + 2, min_p)
    min_p = np.where(area_cells_ser > n * (n + 1), 4 * n + 4, min_p)

    return perimeter_cells_ser / min_p


def compute_fractal_dimension(area_ser, perimeter_ser):
    return 2 * np.log(.25 * perimeter_ser) / np.log(area_ser)
