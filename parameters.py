import numpy as np

###Simulation parameters###
# number of cells
NUM_CELLS = 9
# distance to look for alive or dead neighbours
DISTANCE = 1
# number of alive neighbours required for a flip (range(2,4) means either 2 or 3 alive neighbours are required)
RULE = range(1, 2)
# time steps to simulate until the program exits
NUM_STEPS = 100
# data type for real numbers
DTYPE = np.float32
# True means periodic boundary conditions, False means constant boundary conditions
PERIODIC_BOUNDARIES = False
# Size of one time step in the heatmap. The time step size is calculated as TIME_STEP_SIZE * (np.pi / 2.)
TIME_STEP_SIZE = 1.
