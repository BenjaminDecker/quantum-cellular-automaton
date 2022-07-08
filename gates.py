import numpy as np
from parameters import args
from constants import DIM

SWAP_GATE = np.array([
    [1., 0., 0., 0.],
    [0., 0., 1., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.]])

LOWERING_OPERATOR = np.array([[0., 1.], [0., 0.]])
RISING_OPERATOR = np.array([[0., 0.], [1., 0.]])
PROJECTION_KET_0 = np.array([[1., 0.], [0., 0.]])
PROJECTION_KET_1 = np.array([[0., 0.], [0., 1.]])


def reorder_rotate_gate():
    print("Calculating reorder rotate gate...")
    gate = np.eye(DIM)
    for i in range(args.NUM_CELLS - 1):
        swap = np.kron(np.eye(2**i), np.kron(SWAP_GATE,
                                             np.eye(2**(args.NUM_CELLS - (i + 2)))))
        gate = np.dot(swap, gate)
    return gate


REORDER_ROTATE_GATE = reorder_rotate_gate()


def Rx_gate(theta):
    return np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)]])
