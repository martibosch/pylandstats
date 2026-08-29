"""Pixel adjacencies between the classes of a landscape."""

import numpy as np

# the class pairs are counted in chunks so that the intermediate index arrays stay in
# cache, which is ~3x faster than counting them in a single pass over a large raster
ADJ_CHUNK_SIZE = 1 << 20


def compute_adjacency_arr(padded_arr, num_classes):
    """Compute the adjacency array.

    Parameters
    ----------
    padded_arr : numpy.ndarray
        Landscape array, reclassified so that the classes are integers from 0 to
        `num_classes - 1` and nodata is `num_classes`, padded with nodata.
    num_classes : int
        Number of classes of the landscape.

    Returns
    -------
    adjacency_arr : numpy.ndarray
        Array of shape (2, `num_classes` + 1, `num_classes` + 1) with the horizontal and
        vertical adjacency counts between each pair of classes.
    """
    num_cols_adjacency = num_classes + 1
    num_cols_pixel = padded_arr.shape[1]
    flat_arr = padded_arr.ravel()
    # ACHTUNG: the array is traversed flat, so the pixels of the left and right padding
    # columns are counted as neighbors of each other (wrapping around the rows). Since
    # both are nodata, this only affects the nodata-nodata count, which no metric reads
    start = num_cols_pixel + 1
    end = flat_arr.shape[0] - start
    size = num_cols_adjacency * num_cols_adjacency

    horizontal_adjacency_arr = np.zeros(size, dtype=np.int64)
    vertical_adjacency_arr = np.zeros(size, dtype=np.int64)

    for chunk_start in range(start, end, ADJ_CHUNK_SIZE):
        chunk_end = min(chunk_start + ADJ_CHUNK_SIZE, end)
        class_arr = flat_arr[chunk_start:chunk_end].astype(np.intp)
        for adjacency_arr, offsets in (
            (horizontal_adjacency_arr, (1, -1)),
            (vertical_adjacency_arr, (num_cols_pixel, -num_cols_pixel)),
        ):
            for offset in offsets:
                # ACHTUNG: the neighbor class is the row of the adjacency array, i.e.,
                # the flat index is `class_i + num_cols_adjacency * neighbor_class`
                neighbor_arr = flat_arr[
                    chunk_start + offset : chunk_end + offset
                ].astype(np.intp)
                adjacency_arr += np.bincount(
                    neighbor_arr * num_cols_adjacency + class_arr, minlength=size
                )

    return np.stack(
        (
            horizontal_adjacency_arr.astype(np.uint32),
            vertical_adjacency_arr.astype(np.uint32),
        )
    ).reshape((2, num_cols_adjacency, num_cols_adjacency))
