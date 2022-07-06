import numpy as np
from parameters import args
from constants import SHAPE


def empty():
    return np.empty(SHAPE, dtype=args.DTYPE)


def builder(index, matrix):
    acc = np.array([1.])
    for i in range(index):
        acc = np.kron(acc, np.eye(2, dtype=args.DTYPE))
    acc = np.kron(acc, matrix)
    for i in range(index + 1, args.NUM_CELLS):
        acc = np.kron(acc, np.eye(2, dtype=args.DTYPE))
    return acc
