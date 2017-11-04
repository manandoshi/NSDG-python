import numpy as np


def opp(direction):
    if direction == 'N':
        return "S"
    elif direction == 'S':
        return "N"
    elif direction == 'E':
        return "W"
    elif direction == 'W':
        return "E"
    else:
        assert False


def edge(elements, direction):
    if direction == 'N':
        return elements[-1][:]
    if direction == 'S':
        return elements[0][:]
    if direction == 'E':
        return [element[-1] for element in elements]
    if direction == 'W':
        return [element[0] for element in elements]


def split(direction, array, num_splits_x, num_splits_y):
    if direction == 'N' or direction == 'S':
        num_splits = num_splits_x
    elif direction == 'E' or direction == 'W':
        num_splits = num_splits_y
    else:
        assert False
    return np.split(array, num_splits)
