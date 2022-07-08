import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="A classical simulation of the quantum game of life",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--ncells", dest="NUM_CELLS", type=int, default=9,
                    help="The number of cells to use in the simulation. Computation running time scales exponentially with NUM_CELLS. Anything higher than 11 takes a lot of time and memory to compute.")
parser.add_argument("--distance", dest="DISTANCE", type=int, default=1,
                    help="The distance each cell looks for alive or dead neighbours")
parser.add_argument("--rule", dest="RULE", metavar=("LOWER", "UPPER"), type=int, nargs=2, default=(1, 2),
                    help="Range of alive neighbours required for a flip, upper index is excluded")
parser.add_argument("--nsteps", dest="NUM_STEPS", type=int, default=100,
                    help="Number of time steps to simulate")
parser.add_argument("--stepsize", dest="STEP_SIZE", type=float, default=1.,
                    help="Size of one time step. The time step size is calculated as (STEP_SIZE * pi/2).")
parser.add_argument("--periodic", dest="PERIODIC_BOUNDARIES", action="store_true",
                    help="Use periodic instead of constant boundary conditions")
parser.add_argument("--show", dest="SHOW", action="store_true",
                    help="Try to open and show the output heatmaps immediately")
parser.add_argument("--file-prefix", dest="PREFIX", default="plot",
                    help="Write to files with specified filename-prefix")
parser.add_argument("--file-format", dest="FORMAT", default="html",
                    help="Specify which file format to write to",
                    choices=["html", "eps", "jpeg", "jpg", "pdf", "pgf", "png", "ps", "raw", "rgba", "svg", "svgz", "tif", "tiff"])
parser.add_argument("--no-sse", dest="NOSSE", action="store_true",
                    help="Do not calculate and plot the single site entrpy (Speeds up simulation significantly)")
parser.add_argument("--initial-states", dest="STATE_VECTORS", nargs="*", default=["blinker"],
                    choices=["blinker", "triple_blinker", "single", "single_bottom", "all_ket_1", "all_ket_1_but_outer",
                             "equal_superposition", "equal_superposition_but_outer", "gradient", "rand", "snake"],
                    help="List of initial states")
parser.add_argument("--dtype", dest="DTYPE", default="float32",
                    choices=["float16", "float32", "float64", "float128"])

args = parser.parse_args()
args.RULE = range(args.RULE[0], args.RULE[1])
args.DTYPE = getattr(np, args.DTYPE)
if not isinstance(args.STATE_VECTORS, list):
    args.STATE_VECTORS = [args.STATE_VECTORS]
