import numpy as np
import argparse


class Rules(object):
    def __init__(self, ncells: int, activation_interval: range, distance: int) -> None:
        self.ncells = ncells
        self.activation_interval = activation_interval
        self.distance = distance


class Parser(object):
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description="A classical simulation of the quantum game of life",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            "--num-cells",
            dest="NUM_CELLS",
            type=int,
            default=9,
            help="The number of cells to use in the simulation. Computation running time scales exponentially with NUM_CELLS. Anything higher than 11 takes a lot of time and memory to compute."
        )
        parser.add_argument(
            "--distance",
            dest="DISTANCE",
            type=int,
            default=1,
            help="The distance each cell looks for alive or dead neighbours"
        )
        parser.add_argument(
            "--activation-interval",
            dest="INTERVAL",
            metavar=("LOWER", "UPPER"),
            type=int,
            nargs=2,
            default=(1, 2),
            help="Range of alive neighbours required for a flip, upper index is excluded"
        )
        parser.add_argument(
            "--num-steps",
            dest="NUM_STEPS",
            type=int,
            default=100,
            help="Number of time steps to simulate"
        )
        parser.add_argument(
            "--step-size",
            dest="STEP_SIZE",
            type=float,
            default=1.,
            help="Size of one time step. The time step size is calculated as (STEP_SIZE * pi/2)."
        )
        parser.add_argument(
            "--periodic",
            dest="PERIODIC",
            action="store_true",
            help="Use periodic instead of constant boundary conditions"
        )
        parser.add_argument(
            "--show",
            dest="SHOW",
            action="store_true",
            help="Show the output heatmaps immediately"
        )
        parser.add_argument(
            "--file-prefix",
            dest="PREFIX",
            default="plot",
            help="Write to files with specified filename-prefix"
        )
        parser.add_argument(
            "--file-formats",
            dest="FORMATS",
            nargs='+',
            default=["html"],
            help="Specify which file formats to write to",
            choices=["html", "eps", "jpeg", "jpg", "pdf", "pgf",
                     "png", "ps", "raw", "rgba", "svg", "svgz", "tif", "tiff"]
        )
        parser.add_argument(
            "--sse",
            dest="SSE",
            action="store_true",
            help="Calculate and plot the single site entropy (Slows down simulation significantly)"
        )
        parser.add_argument(
            "--initial-states",
            dest="INITIAL_STATES",
            nargs="+",
            default=["blinker"],
            choices=["blinker", "triple_blinker", "single", "single_bottom", "all_ket_1", "all_ket_1_but_outer",
                     "equal_superposition", "equal_superposition_but_outer", "gradient", "rand", "snake"],
            help="List of initial states"
        )
        parser.add_argument(
            "--dtype",
            dest="DTYPE",
            default="float32",
            choices=["float16", "float32", "float64", "float128"]
        )

        args = parser.parse_args()

        self.rules = Rules(
            ncells=args.NUM_CELLS,
            activation_interval=range(args.INTERVAL[0], args.INTERVAL[1]),
            distance=args.DISTANCE
        )
        self.num_steps = args.NUM_STEPS
        self.step_size = args.STEP_SIZE
        self.periodic = args.PERIODIC
        self.show = args.SHOW
        self.file_prefix = args.PREFIX
        self.file_formats = args.FORMATS
        self.sse = args.SSE
        self.initial_states = args.INITIAL_STATES
        if not isinstance(self.initial_states, list):
            self.initial_states = [self.initial_states]
        self.dtype = getattr(np, args.DTYPE)

    @classmethod
    def instance(cls):
        try:
            return cls._instance
        except AttributeError:
            cls._instance = cls()
            return cls._instance
