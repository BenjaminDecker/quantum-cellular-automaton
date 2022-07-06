import numpy as np
from parameters import args

KET_0 = np.array([1., 0.])
KET_1 = np.array([0., 1.])
KET_PLUS = np.array([1., 1.]) / np.sqrt(2)
DIM = 2**args.NUM_CELLS
SHAPE = [args.NUM_CELLS, DIM, DIM]
