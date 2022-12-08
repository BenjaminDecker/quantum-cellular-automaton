import argparse

from parameters import Rules


class Parser(object):
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
            choices=["blinker", "triple_blinker", "single", "single_bottom", "all_ket_0", "all_ket_1",
                     "all_ket_1_but_outer", "equal_superposition", "equal_superposition_but_outer", "gradient", "rand"],
            help="List of initial states"
        )
        init_group.add_argument(
            "--initial-state-files",
            dest="INITIAL_STATE_FILES",
            nargs="*",
            help="List of .npz files containing mps initial states as created by np.savez(file, *Alist), where Alist is"
                 "a list of MPS tensors. The number of sites of the mps must be the same as specified with --num-cells."
        )
        init_group.add_argument(
            "--bond-dim",
            dest="BOND_DIM",
            type=int,
            default=5,
            help="The bond dimension to use for the initial states. Only relevant for the tdvp algorithm."
        )
        algo_group = parser.add_argument_group('Algorithm')
        algo_group.add_argument(
            "--algorithm",
            dest="ALGORITHM",
            default="exact",
            choices=["exact", "tdvp"],
            help="The algorithm used for the time evolution. Use 'exact' for a small number of cells, otherwise 'tdvp'"
        )
        algo_group.add_argument(
            "--step-size",
            dest="STEP_SIZE",
            type=float,
            default=.01,
            help="Size of one time step. The time step size is calculated as (STEP_SIZE * pi/2)"
        )
        plot_group = parser.add_argument_group('Plot')
        plot_group.add_argument(
            "--plot-frequency",
            dest="PLOT_FREQUENCY",
            type=float,
            default=1.0,
            help="Frequency at which time steps are plotted. Time between plot steps is calculated as "
                 "(pi/2 * 1/PLOT_FREQUENCY * 1/STEP_SIZE) "
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
            default=["html"],
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
        self.plot_classical = args.PLOT_CLASSICAL
        self.show = args.SHOW
        self.plot_frequency = args.PLOT_FREQUENCY
        self.plot_file_path = args.PLOT_FILE_PATH
        self.file_formats = args.FORMATS
        self.initial_states = args.INITIAL_STATES
        if self.initial_states is None:
            self.initial_states = []
        self.initial_state_files = args.INITIAL_STATE_FILES
        if self.initial_state_files is None:
            self.initial_state_files = []
        self.bond_dim = args.BOND_DIM
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
