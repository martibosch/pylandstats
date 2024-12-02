import numpy as np
cimport cython
cimport numpy as np

# define a C-level array type
ctypedef np.uint32_t DTYPE_t  # data type for elements


@cython.boundscheck(False)  # deactivate bounds checking
@cython.wraparound(False)  # deactivate negative indexing.
def compute_adjacency_arr(np.ndarray[DTYPE_t, ndim=2] padded_arr, int num_classes):
    """Compute adjacency array using memoryviews."""

    # prepare adjacency array with memoryviews
    cdef int num_cols_adjacency = num_classes + 1
    cdef int num_cols_pixel = padded_arr.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] horizontal_adjacency_arr = np.zeros(
        num_cols_adjacency * num_cols_adjacency, dtype=np.uint32)
    cdef np.ndarray[DTYPE_t, ndim=1] vertical_adjacency_arr = np.zeros(
        num_cols_adjacency * num_cols_adjacency, dtype=np.uint32)

    # memoryviews for efficient access
    cdef DTYPE_t[:] flat_arr = padded_arr.ravel()
    cdef DTYPE_t[:] horizontal_adj_mem = horizontal_adjacency_arr
    cdef DTYPE_t[:] vertical_adj_mem = vertical_adjacency_arr

    # define neighbor offsets
    cdef int horizontal_neighbors[2]
    horizontal_neighbors[0] = 1
    horizontal_neighbors[1] = -1
    cdef int vertical_neighbors[2]
    vertical_neighbors[0] = num_cols_pixel
    vertical_neighbors[1] = -num_cols_pixel

    cdef int start = num_cols_pixel + 1
    cdef int end = flat_arr.shape[0] - start

    cdef int i, class_i, neighbor, neighbor_idx, target_idx

    for i in range(start, end):
        class_i = flat_arr[i]

        # update horizontal adjacency counts
        for neighbor in horizontal_neighbors:
            neighbor_idx = i + neighbor
            target_idx = class_i + num_cols_adjacency * flat_arr[neighbor_idx]
            horizontal_adj_mem[target_idx] += 1

        # update vertical adjacency counts
        for neighbor in vertical_neighbors:
            neighbor_idx = i + neighbor
            target_idx = class_i + num_cols_adjacency * flat_arr[neighbor_idx]
            vertical_adj_mem[target_idx] += 1

    # return the result as a reshaped numpy array
    return np.stack((horizontal_adjacency_arr, vertical_adjacency_arr)).reshape(
        (2, num_cols_adjacency, num_cols_adjacency)
    )
