import numpy as np


# pythran export compute_adjacency_arr(uint32[:,:], int)
def compute_adjacency_arr(padded_arr, num_classes):
    # flat-array approach to pixel adjacency from link below:
    # https://ilovesymposia.com/2016/12/20/numba-in-the-real-world/
    # the first axis of `adjacency_arr` is of fixed size of 2 and serves to
    # distinguish between vertical and horizontal adjacencies (we could also
    # use a tuple of two 2-D arrays)
    # adjacency_arr = np.zeros((2, num_classes + 1, num_classes + 1),
    #                          dtype=np.uint32)
    num_cols_adjacency = num_classes + 1
    horizontal_adjacency_arr = np.zeros(
        num_cols_adjacency * num_cols_adjacency, dtype=np.uint32)
    vertical_adjacency_arr = np.zeros(num_cols_adjacency * num_cols_adjacency,
                                      dtype=np.uint32)
    num_cols_pixel = padded_arr.shape[1]
    flat_arr = padded_arr.ravel()
    # steps_to_neighbours as argument to distinguish between vertical/
    # horizontal adjacencies
    # steps_to_neighbours = [1, num_cols, -1, -num_cols]
    horizontal_neighbours = [1, -1]
    vertical_neighbours = [num_cols_pixel, -num_cols_pixel]
    start = num_cols_pixel + 1
    end = len(flat_arr) - start
    for i in range(start, end):
        class_i = flat_arr[i]
        # class_left = flat_arr[i - 1]
        # class_right = flat_arr[i + 1]
        # class_above = flat_arr[i - num_cols]
        # class_below = flat_arr[i + num_cols]
        # adjacency_arr[0, class_i, class_left] += 1
        # adjacency_arr[0, class_i, class_right] += 1
        # adjacency_arr[1, class_i, class_above] += 1
        # adjacency_arr[1, class_i, class_below] += 1
        for neighbour in horizontal_neighbours:
            # adjacency_arr[0, class_i, flat_arr[i + neighbour]] += 1
            horizontal_adjacency_arr[class_i + num_cols_adjacency *
                                     flat_arr[i + neighbour]] += 1
        for neighbour in vertical_neighbours:
            # adjacency_arr[1, class_i, flat_arr[i + neighbour]] += 1
            vertical_adjacency_arr[class_i + num_cols_adjacency *
                                   flat_arr[i + neighbour]] += 1

    return np.stack(
        (horizontal_adjacency_arr, vertical_adjacency_arr)).reshape(
            (2, num_cols_adjacency, num_cols_adjacency))
