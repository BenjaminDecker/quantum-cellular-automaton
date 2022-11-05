import numpy as np
from parameters import Parser

args = Parser.instance()

KET_0 = np.array([1., 0.])
KET_1 = np.array([0., 1.])
KET_PLUS = np.array([1., 1.]) / np.sqrt(2)

DIM = 2**args.rules.ncells

SWAP_GATE = np.array([
    [1., 0., 0., 0.],
    [0., 0., 1., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.]])

LOWERING_OPERATOR = np.array([[0., 1.], [0., 0.]])
RISING_OPERATOR = np.array([[0., 0.], [1., 0.]])
PROJECTION_KET_0 = np.array([[1., 0.], [0., 0.]])
PROJECTION_KET_1 = np.array([[0., 0.], [0., 1.]])
S_OPERATOR = LOWERING_OPERATOR + RISING_OPERATOR


def reorder_rotate_gate():
    print("Calculating reorder rotate gate...")
    gate = np.eye(DIM)
    for i in range(args.rules.ncells - 1):
        swap = np.kron(np.eye(2**i), np.kron(SWAP_GATE,
                                             np.eye(2**(args.rules.ncells - (i + 2)))))
        gate = np.dot(swap, gate)
    return gate


REORDER_ROTATE_GATE = reorder_rotate_gate()


def Rx_gate(theta):
    return np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)]
    ])
