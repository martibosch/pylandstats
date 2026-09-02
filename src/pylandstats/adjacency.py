"""Pixel adjacencies between the classes of a landscape."""

import numpy as np

# number of rows whose class pairs are counted at a time, so that the intermediate index
# arrays stay in cache. The result does not depend on it, only the time it takes
ADJ_CHUNK_NUM_ROWS = 512


def _count_pairs(class_arr, neighbor_arr, num_cols_adjacency):
    """Count the occurrences of each (class, neighbor class) pair."""
    return np.bincount(
        neighbor_arr.ravel().astype(np.intp) * num_cols_adjacency
        + class_arr.ravel().astype(np.intp),
        minlength=num_cols_adjacency * num_cols_adjacency,
    )


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
    num_rows_pixel = padded_arr.shape[0]
    size = num_cols_adjacency * num_cols_adjacency

    horizontal_adjacency_arr = np.zeros(size, dtype=np.int64)
    vertical_adjacency_arr = np.zeros(size, dtype=np.int64)

    for start in range(0, num_rows_pixel, ADJ_CHUNK_NUM_ROWS):
        end = min(start + ADJ_CHUNK_NUM_ROWS, num_rows_pixel)
        # horizontal adjacencies, i.e., between the columns of each row
        chunk = padded_arr[start:end]
        horizontal_adjacency_arr += _count_pairs(
            chunk[:, :-1], chunk[:, 1:], num_cols_adjacency
        )
        # vertical adjacencies, i.e., between each row and the next one, hence the chunk
        # extends one row beyond its end
        chunk = padded_arr[start : end + 1]
        vertical_adjacency_arr += _count_pairs(
            chunk[:-1], chunk[1:], num_cols_adjacency
        )

    # each adjacency has been counted once, from the perspective of the left/upper
    # pixel of the pair, so add the transpose to also count it from that of the
    # right/lower one
    return np.stack(
        [
            (arr + arr.T).astype(np.uint32)
            for arr in (
                horizontal_adjacency_arr.reshape(
                    num_cols_adjacency, num_cols_adjacency
                ),
                vertical_adjacency_arr.reshape(num_cols_adjacency, num_cols_adjacency),
            )
        ]
    )
