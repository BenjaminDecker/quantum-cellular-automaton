import argparse

from parameters import Rules


class Parser(object):

    def check_positive(self, value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError("%s is not a positive int value" % value)
        return ivalue

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description="A classical simulation of the quantum game of life",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        rules_group = parser.add_argument_group('Rules')
        rules_group.add_argument(
            "--num-cells",
            dest="NUM_CELLS",
            type=int,
            default=9,
            help="The number of cells to use in the simulation. Computation running time scales exponentially with "
                 "NUM_CELLS. Anything higher than 11 takes a lot of time and memory to compute. "
        )
        rules_group.add_argument(
            "--distance",
            dest="DISTANCE",
            type=int,
            default=1,
            help="The distance each cell looks for alive or dead neighbours"
        )
        rules_group.add_argument(
            "--activation-interval",
            dest="INTERVAL",
            metavar=("LOWER", "UPPER"),
            type=int,
            nargs=2,
            default=(1, 2),
            help="Range of alive neighbours required for a flip, upper index is excluded"
        )
        rules_group.add_argument(
            "--num-steps",
            dest="NUM_STEPS",
            type=int,
            default=10000,
            help="Number of time steps to simulate"
        )
        # rules_group.add_argument(
        #     "--periodic",
        #     dest="PERIODIC",
        #     action="store_true",
        #     help="Use periodic instead of constant boundary conditions (experimental)"
        # )
        init_group = parser.add_argument_group('Initial States')
        init_group.add_argument(
            "--initial-states",
            dest="INITIAL_STATES",
            nargs="*",
            choices=["blinker", "triple_blinker", "full_blinker", "single", "single_bottom", "all_ket_0", "all_ket_1",
                     "only_outer", "all_ket_1_but_outer", "equal_superposition", "equal_superposition_but_outer",
                     "gradient", "rand"],
            help="List of initial states"
        )
        init_group.add_argument(
            "--initial-state-files",
            dest="INITIAL_STATE_FILES",
            nargs="*",
            help="List of .npz files containing mps initial states created by np.savez(). The number of sites of the "
                 "mps must be the same as specified with --num-cells."
        )
        algo_group = parser.add_argument_group('Algorithm')
        algo_group.add_argument(
            "--algorithm",
            dest="ALGORITHM",
            default="exact",
            choices=["exact", "1tdvp", "2tdvp", "a1tdvp"],
            help="The algorithm used for the time evolution. Use 'exact' for a small number of cells, otherwise some "
                 "version of 'tdvp'"
        )
        algo_group.add_argument(
            "--convergence-measure",
            dest="CONVERGENCE_MEASURE",
            default="taylor",
            choices=["taylor", "expm_multiply", "exact_exponential"],
            help="The convergence measure used to find new target bond dimensions in the a1tdvp algorithm. Is ignored "
                 "if the chosen algorithm is not 'a1tdvp'."
        )
        algo_group.add_argument(
            "--taylor-steps",
            dest="TAYLOR_STEPS",
            type=self.check_positive,
            default=5,
            help="The number of steps to use in the taylor expansion of the 'taylor' convergence measure. Is ignored "
                 "if the chosen convergence measure is not 'taylor'."
        )
        algo_group.add_argument(
            "--step-size",
            dest="STEP_SIZE",
            type=float,
            default=.005,
            help="Size of one time step. The time step size is calculated as (STEP_SIZE * pi/2)"
        )
        algo_group.add_argument(
            "--max-bond-dim",
            dest="MAX_BOND_DIM",
            type=int,
            default=32,
            help="The maximum that a bond of the MPS is allowed to grow to during simulation. Is ignored if the chosen "
                 "algorithm is not 'tdvp'."
        )
        algo_group.add_argument(
            "--svd-epsilon",
            dest="SVD_EPSILON",
            type=float,
            default=0.00005,
            help="A measure of accuracy for the truncation step after splitting a mps tensor. This parameter controls "
                 "how quickly the bond dimension of the mps grows during the simulation. Lower means more accurate, "
                 "but slower."
        )
        plot_group = parser.add_argument_group('Plot')
        plot_group.add_argument(
            "--plotting-frequency",
            dest="PLOTTING_FREQUENCY",
            type=float,
            default=1.0,
            help="Frequency at which time steps are plotted. Time between plot steps is calculated as "
                 "(pi/2 * 1/PLOT_FREQUENCY * 1/STEP_SIZE) "
        )
        plot_group.add_argument(
            "--plot-sse",
            dest="PLOT_SSE",
            action="store_true",
            help="Plot the single site entropy"
        )
        plot_group.add_argument(
            "--plot-bond-dims",
            dest="PLOT_BOND_DIMS",
            action="store_true",
            help="Plot the bond dimensions of the mps"
        )
        plot_group.add_argument(
            "--plot-rounded",
            dest="PLOT_ROUNDED",
            action="store_true",
            help="Plot a rounded version of the probability"
        )
        plot_group.add_argument(
            "--plot-classical",
            dest="PLOT_CLASSICAL",
            action="store_true",
            help="Plot the classical non-quantum time evolution"
        )
        plot_group.add_argument(
            "--show",
            dest="SHOW",
            action="store_true",
            help="Show the output heatmaps after finishing"
        )
        plot_group.add_argument(
            "--plot-file-path",
            dest="PLOT_FILE_PATH",
            default="plots/",
            help="Write to files at specified relative location, including file prefix"
        )
        plot_group.add_argument(
            "--file-formats",
            dest="FORMATS",
            nargs='+',
            default=["pdf"],
            help="Specify which file formats to write to",
            choices=["html", "eps", "jpeg", "jpg", "pdf", "pgf",
                     "png", "ps", "raw", "rgba", "svg", "svgz", "tif", "tiff"]
        )

        args = parser.parse_args()

        self.rules = Rules(
            ncells=args.NUM_CELLS,
            activation_interval=range(args.INTERVAL[0], args.INTERVAL[1]),
            distance=args.DISTANCE,
            periodic=False
        )
        self.num_steps = args.NUM_STEPS
        self.step_size = args.STEP_SIZE
        self.algorithm = args.ALGORITHM
        self.convergence_measure = args.CONVERGENCE_MEASURE
        self.taylor_steps = args.TAYLOR_STEPS
        self.plot_sse = args.PLOT_SSE
        self.plot_classical = args.PLOT_CLASSICAL
        self.plot_bond_dims = args.PLOT_BOND_DIMS
        self.plot_rounded = args.PLOT_ROUNDED
        self.show = args.SHOW
        self.plot_frequency = args.PLOTTING_FREQUENCY
        self.plot_file_path = args.PLOT_FILE_PATH
        self.file_formats = args.FORMATS
        self.initial_states = args.INITIAL_STATES
        if self.initial_states is None:
            self.initial_states = []
        self.initial_state_files = args.INITIAL_STATE_FILES
        if self.initial_state_files is None:
            self.initial_state_files = []
        self.max_bond_dim = args.MAX_BOND_DIM
        self.svd_epsilon = args.SVD_EPSILON
        self.plot_step_interval = int(1 / (self.plot_frequency * self.step_size))
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
