import numpy as np
import argparse
import states


def tuple_type(s):
    try:
        s = s.replace("(", "").replace(")", "")
        lower, upper = map(int, s.split(','))
        return range(lower, upper)
    except:
        raise argparse.ArgumentTypeError(
            'RULE must be of the form "lower,upper" or "(lower,upper)"')


def get_state_vector(s):
    return eval()


parser = argparse.ArgumentParser(
    description='A classical simulation of the quantum game of life',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ncells', dest='NUM_CELLS', type=int, default=9,
                    help='The number of cells to use in the simulation. Computation running time scales exponentially with NUM_CELLS. Anything higher than 11 takes a lot of time and memory to compute.')
parser.add_argument('--distance', dest='DISTANCE', type=int, default=1,
                    help='The distance each cell looks for alive or dead neighbours')
parser.add_argument('--rule', dest='RULE', type=tuple_type, default='(1,2)',
                    help='Range of alive neighbours required for a flip, right index is excluded (Good choices are "(1,2)" and "(2,4)")')
parser.add_argument('--nsteps', dest='NUM_STEPS', type=int, default=100,
                    help='Number of time steps to simulate')
parser.add_argument('--stepsize', dest='STEP_SIZE', type=float, default=1.,
                    help='Size of one time step. The time step size is calculated as (STEP_SIZE * pi/2).')
parser.add_argument('--periodic', dest='PERIODIC_BOUNDARIES', action='store_const', const=True, default=False,
                    help="Use periodic boundary conditions instead of constant boundary conditions")
parser.add_argument('--show', dest='SHOW', action='store_const', const=False, default=True,
                    help="Show the output heatmaps immediately")
parser.add_argument('--write', dest='WRITE', action='store_const', const=False, default=True,
                    help="Write the heapmaps to html files")
parser.add_argument('--initial_states', dest='STATE_VECTORS', nargs='*', default='blinker',
                    choices=["blinker", "triple_blinker", "single", "single_bottom", "all_ket_1", "all_ket_1_but_outer",
                             "equal_superposition", "equal_superposition_but_outer", "gradient", "rand", "snake"],
                    help="List of initial states")
parser.add_argument('--dtype', dest='DTYPE', default="float32",
                    choices=["float16", "float32", "float64", "float128"])

args = parser.parse_args()
args.DTYPE = eval("np." + args.DTYPE)
if not isinstance(args.STATE_VECTORS, list):
    args.STATE_VECTORS = [args.STATE_VECTORS]
args.STATE_VECTORS = map(
    lambda x: eval("states." + x + "()"),
    args.STATE_VECTORS)
