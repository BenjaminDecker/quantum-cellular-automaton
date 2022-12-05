import argparse

from parameters import Rules


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
            help="The number of cells to use in the simulation. Computation running time scales exponentially with "
                 "NUM_CELLS. Anything higher than 11 takes a lot of time and memory to compute. "
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
            default=10000,
            help="Number of time steps to simulate"
        )
        parser.add_argument(
            "--periodic",
            dest="PERIODIC",
            action="store_true",
            help="Use periodic instead of constant boundary conditions (experimental)"
        )
        parser.add_argument(
            "--algorithm",
            dest="ALGORITHM",
            default="exact",
            choices=["exact", "tdvp"],
            help="The algorithm used for the time evolution. Use 'exact' for a small number of cells, otherwise 'tdvp'"
        )
        parser.add_argument(
            "--step-size",
            dest="STEP_SIZE",
            type=float,
            default=.01,
            help="Size of one time step. The time step size is calculated as (STEP_SIZE * pi/2)"
        )
        parser.add_argument(
            "--plot-step-interval",
            dest="PLOT_STEP_INTERVAL",
            type=int,
            default=100,
            help="Amount of time steps after which a step is plotted to the output"
        )
        parser.add_argument(
            "--plot-classical",
            dest="PLOT_CLASSICAL",
            action="store_true",
            help="Plot the classical non-quantum time evolution"
        )
        parser.add_argument(
            "--show",
            dest="SHOW",
            action="store_true",
            help="Show the output heatmaps after finishing"
        )
        parser.add_argument(
            "--plot-file-path",
            dest="PLOT_FILE_PATH",
            default="plots/",
            help="Write to files at specified relative location, including file prefix"
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
            "--initial-states",
            dest="INITIAL_STATES",
            nargs="+",
            default=["blinker"],
            choices=["blinker", "triple_blinker", "single", "single_bottom", "all_ket_0", "all_ket_1",
                     "all_ket_1_but_outer", "equal_superposition", "equal_superposition_but_outer", "gradient", "rand"],
            help="List of initial states"
        )
        parser.add_argument(
            "--initial-state-files",
            dest="INITIAL_STATE_FILES",
            nargs="*",
            help="List of .npz files containing mps initial states as created by np.savez(file, *Alist), where Alist is"
                 "a list of MPS tensors. The number of sites of the mps must be the same as specified with --num-cells."
        )
        parser.add_argument(
            "--bond-dim",
            dest="BOND_DIM",
            type=int,
            default=5,
            help="The bond dimension to use for the initial states. Only relevant for the tdvp algorithm."
        )

        args = parser.parse_args()

        self.rules = Rules(
            ncells=args.NUM_CELLS,
            activation_interval=range(args.INTERVAL[0], args.INTERVAL[1]),
            distance=args.DISTANCE,
            periodic=args.PERIODIC
        )
        self.num_steps = args.NUM_STEPS
        self.step_size = args.STEP_SIZE
        self.algorithm = args.ALGORITHM
        self.plot_classical = args.PLOT_CLASSICAL
        self.show = args.SHOW
        self.plot_step_interval = args.PLOT_STEP_INTERVAL
        self.plot_file_path = args.PLOT_FILE_PATH
        self.file_formats = args.FORMATS
        self.initial_states = args.INITIAL_STATES
        self.initial_state_files = args.INITIAL_STATE_FILES
        if self.initial_state_files is None:
            self.initial_state_files = []
        self.bond_dim = args.BOND_DIM
        self.plot_steps = self.num_steps // self.plot_step_interval
        if self.num_steps % self.plot_step_interval > 0:
            self.plot_steps += 1

    # TODO Pass a parser object around instead of using this global singleton
    @classmethod
    def instance(cls) -> 'Parser':
        try:
            return cls._instance
        except AttributeError:
            cls._instance = cls()
            return cls._instance
