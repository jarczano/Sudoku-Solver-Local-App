import numpy as np


def sorted_squares(contours_square):
    """
    Function sort list of contours of squares from top left to bottom right
    :param contours_square: list of contours of squares
    :return: sorted list of contours of squares
    """

    mean_coordinates = []
    for j in range(len(contours_square)):
        x_m = np.mean([contours_square[j][k][0][0] for k in range(4)])
        y_m = np.mean([contours_square[j][k][0][1] for k in range(4)])
        mean_coordinates.append([x_m, y_m])

    contours_square_mean = zip(contours_square, mean_coordinates)
    contours_sorted = sorted(contours_square_mean, key=lambda x: x[1][1])
    result = np.zeros((9, 9), dtype=object)
    for i in range(9):
        nine = contours_sorted[i * 9: (i + 1) * 9]
        nine_sorted = sorted(nine, key=lambda x: x[1][0])
        tuples = zip(*nine_sorted)
        coor, mean = [list(tuple) for tuple in tuples]
        result[i, :] = coor
    return result