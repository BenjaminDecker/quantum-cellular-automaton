import numpy as np
from parameters import *

KET_0 = np.array([1., 0.])
KET_1 = np.array([0., 1.])
KET_PLUS = np.array([1., 1.]) / np.sqrt(2)
SIZE = 2**NUM_CELLS
SHAPE = [NUM_CELLS, SIZE, SIZE]
LOWERING_OPERATOR = np.array([[0., 1.], [0., 0.]])
RISING_OPERATOR = np.array([[0., 0.], [1., 0.]])
PROJECTION_KET_0 = np.array([[1., 0.], [0., 0.]])
PROJECTION_KET_1 = np.array([[0., 0.], [0., 1.]])
SWAP_GATE = np.array([
    [1., 0., 0., 0.],
    [0., 0., 1., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.]])
STEP_RANGE = range(NUM_CELLS) if PERIODIC_BOUNDARIES else range(
    DISTANCE, NUM_CELLS - DISTANCE)
