"""Some helpers for working with arrays."""

import numpy as np


def check_strict_monotonic(array, list_dimensions=None):
    """Check that an array is strictly monotonic. Raise a
    ValueError if not.

    Input:
        array: a numpy array of any dimension.
        list_dimensions: the list of dimensions on which to do
            the check. Check all dimensions if None (default).

    Output: None.

    Can raise:
        a ValueError indicating the first non monotonic dimension.
    """

    if list_dimensions is None:
        n_dim = len(np.shape(array))
        list_dimensions = range(n_dim)
    else:
        assert isinstance(list_dimensions, list)

    for dim in list_dimensions:
        dim_diff = np.diff(array, axis=dim)
        if not (np.all(dim_diff < 0) or np.all(dim_diff > 0)):
            raise ValueError("Array non stricly monotonic on dim {}".format(dim))


def index_ranges_within_bounds(x_coords, y_coords, x_bounds, y_bounds, comp_epsilon=1e-12):
    """Select the index ranges on the x and y directions that contain all data having
    x and y coordinates with prescribed ranges.

    Input:
        x_coords: the 2D array containing the x coordinate of the points
        y_coords: idem, y coordinate
        x_bounds=[min_x, max_x]: the min and max desired values for x coordinates
        y_bounds: idem, y coordinate
        comp_epsilon: tolerance for performing floating comparisons

    Output:
        (lower_0_index, upper_0_index, lower_1_index, upper_1_index):
            the lower and upper bounds for ranges on the 0 and 1 axis indexes so that all data
            that have coordinates within the prescribed x and y bounds are captured.

            Note: these are Python bounds, to be used with the convention [lower:upper[,
            for example for index slicing.

    Can raise:
        This can raise an IndexError in case there are no points having coordinates
        within the prescsribed bounds.
    """

    assert x_bounds[0] < x_bounds[1]
    assert y_bounds[0] < y_bounds[1]
    check_strict_monotonic(x_coords, [1])
    check_strict_monotonic(y_coords, [0])

    valid_x_locs = np.logical_and(x_coords + comp_epsilon >= x_bounds[0],
                                  x_coords <= x_bounds[1] + comp_epsilon)

    valid_y_locs = np.logical_and(y_coords + comp_epsilon >= y_bounds[0],
                                  y_coords <= y_bounds[1] + comp_epsilon)

    valid_x_y_locs = np.logical_and(valid_x_locs,
                                    valid_y_locs)

    if not np.any(valid_x_y_locs):
        raise IndexError("There are no points within the x y range prescrived.")

    index_valid_range_0 = np.where(np.any(valid_x_y_locs, axis=1))
    index_valid_range_1 = np.where(np.any(valid_x_y_locs, axis=0))

    # Note: +1 on the maxima because these should be bounds for ranges,
    # i.e. [low_bound, upper_bound+1[ in python
    return(index_valid_range_0[0][0], index_valid_range_0[0][-1] + 1,
           index_valid_range_1[0][0], index_valid_range_1[0][-1] + 1
           )
