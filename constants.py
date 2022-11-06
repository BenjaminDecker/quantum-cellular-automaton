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


def Rx_gate(theta):
    return np.array([
        [np.cos(theta / 2),         -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2),   np.cos(theta / 2)]
    ])
