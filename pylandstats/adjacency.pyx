# cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as np

# define a C-level array type
ctypedef np.uint32_t DTYPE_t  # data type for elements


def compute_adjacency_arr(np.ndarray[DTYPE_t, ndim=2] padded_arr, int num_classes):
    """Compute adjacency array."""

    # prepare adjacency array
    cdef int num_cols_adjacency = num_classes + 1
    cdef int num_cols_pixel = padded_arr.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] horizontal_adjacency_arr = np.zeros(
        num_cols_adjacency * num_cols_adjacency, dtype=np.uint32)
    cdef np.ndarray[DTYPE_t, ndim=1] vertical_adjacency_arr = np.zeros(
        num_cols_adjacency * num_cols_adjacency, dtype=np.uint32)

    # flattened array
    cdef np.ndarray[DTYPE_t, ndim=1] flat_arr = padded_arr.ravel()
    cdef int flat_length = flat_arr.shape[0]

    # define neighbor offsets
    cdef int horizontal_neighbors[2]
    horizontal_neighbors[0] = 1
    horizontal_neighbors[1] = -1
    cdef int vertical_neighbors[2]
    vertical_neighbors[0] = num_cols_pixel
    vertical_neighbors[1] = -num_cols_pixel

    cdef int start = num_cols_pixel + 1
    cdef int end = flat_length - start

    cdef int i, class_i, neighbor, neighbor_idx, target_idx

    for i in range(start, end):
        class_i = flat_arr[i]

        # update horizontal adjacency counts
        for neighbor in horizontal_neighbors:
            neighbor_idx = i + neighbor
            target_idx = class_i + num_cols_adjacency * flat_arr[neighbor_idx]
            horizontal_adjacency_arr[target_idx] += 1

        # update vertical adjacency counts
        for neighbor in vertical_neighbors:
            neighbor_idx = i + neighbor
            target_idx = class_i + num_cols_adjacency * flat_arr[neighbor_idx]
            vertical_adjacency_arr[target_idx] += 1

    return np.stack((horizontal_adjacency_arr, vertical_adjacency_arr)).reshape(
        (2, num_cols_adjacency, num_cols_adjacency)
    )
