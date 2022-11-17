from MPO import MPO
from MPS import MPS
from parameters import Parser
import numpy as np
from constants import PROJECTION_KET_1
from scipy.linalg import expm, logm
import warnings

args = Parser.instance()


class Result(object):
    def __init__(self, classical, population, d_population, single_site_entropy) -> None:
        self.classical = classical
        self.population = np.array(population)
        self.d_population = np.array(d_population)
        self.single_site_entropy = np.array(single_site_entropy)


class Time_Evolution(object):
    """
    Collection of different methods for calculating the time evolution of initial states according to a hamiltonian.
    """

    @classmethod
    def evolve(cls, states: list[MPS], hamiltonian: MPO, algorithm='exact') -> list[Result]:
        """
        Performs the time evolution of all states according to the given hamiltonian via the specified algorithm
        """
        cls.hamiltonian = hamiltonian
        population = np.empty(
            [args.plot_steps, args.rules.ncells]
        )
        d_population = np.empty(
            [args.plot_steps, args.rules.ncells]
        )
        single_site_entropy = np.empty(
            [args.plot_steps, args.rules.ncells]
        )

        print("Preparing data...")
        if algorithm == 'exact':
            cls.prepare_exact(args.step_size * args.plot_step_interval)

        results = []

        for state_index, state in enumerate(states):
            print("\nSimulating state " + str(state_index + 1) + "...")
            if algorithm == 'tdvp':
                cls.prepare_tdvp(state)

            site_canonical_hint = None
            for step in range(args.num_steps):
                if step % 10 == 0:
                    print("Step " + str(step) + " of " + str(args.num_steps))
                if step % args.plot_step_interval == 0:
                    plot_step = step // args.plot_step_interval
                    cls.measure(
                        state=state,
                        population=population[plot_step, :],
                        d_population=d_population[plot_step, :],
                        single_site_entropy=single_site_entropy[plot_step, :],
                        site_canonical_hint=site_canonical_hint
                    )
                if algorithm == 'exact':
                    if step % args.plot_step_interval == 0:
                        state = cls.exact_step(state)
                        # After the exact step, the mps will be in site canonical form with the center at the last site
                        site_canonical_hint = "last"
                if algorithm == 'tdvp':
                    state = cls.tdvp_step(state, args.step_size)
                    # After the tdvp step, the mps will be in site canonical form with the center at the first site
                    site_canonical_hint = "first"

            classical = None

            if args.plot_step_interval * args.plot_steps == args.num_steps:
                classical = cls.classical(first_column=population[0, :])
            results.append(Result(
                classical=classical,
                population=population,
                d_population=d_population,
                single_site_entropy=single_site_entropy
            ))

        return results

    @classmethod
    def measure(cls, state: MPS, population, d_population, single_site_entropy, site_canonical_hint=None):
        """
        Measures the population, rounded population and single-site entropy of the given state and writes the results into the given arrays
        """
        # If no site hint is given or if site hint is invalid, start with the center tensor at site 0
        if site_canonical_hint == None:
            state.make_site_canonical(0)
        backwards = site_canonical_hint == "last"
        site_range = range(len(state.A))
        first = True
        for site in reversed(site_range) if backwards else site_range:
            # In every iteration except the first, shift the center tensor one site to the left/right
            if first:
                first = False
            else:
                if backwards:
                    state.orthonormalize_right_qr(site + 1)
                else:
                    state.orthonormalize_left_qr(site - 1)

            A = state.A[site]

            # Calculate the population density
            density_matrix = np.tensordot(
                A,
                A.conj(),
                ((1, 2), (1, 2))
            )
            result = np.tensordot(
                density_matrix,
                PROJECTION_KET_1,
                ((0, 1), (0, 1))
            ).real
            population[site] = result
            d_population[site] = np.round(result)
            # Calculate the single-site entropy
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                single_site_entropy[site] = (-np.trace(np.dot(
                    density_matrix,
                    logm(density_matrix) / np.log(2)
                ))).real

    @classmethod
    def exact_step(cls, state: MPS) -> MPS:
        """
        Evolves the quantum state by calculating a time evolution operator matrix and using explicit matrix vector product. Does not use any tensor network optimizations.
        """
        psi = state.as_vector()
        psi = np.dot(cls.U, psi)
        return MPS.from_vector(psi)

    @classmethod
    def tdvp_step(cls, state: MPS, step_size) -> MPS:
        """
        Evolves the quantum state using the tdvp algorithm.
        """
        state.make_site_canonical(0)
        # ----------sweep-right----------
        for site in range(len(state.A) - 1):
            new_A = cls.evolve_site(state.A[site], site, step_size)
            new_A, bond = state.left_qr_tensors(new_A)
            state.A[site] = new_A
            cls.calculate_layer_left(new_A, site)
            new_bond = cls.evolve_bond(bond, site, step_size)
            state.A[site + 1] = np.transpose(np.tensordot(
                new_bond,
                state.A[site + 1],
                (1, 1)
            ), (1, 0, 2))
        state.A[len(state.A) - 1] = cls.evolve_site(
            state.A[len(state.A) - 1],
            len(state.A) - 1,
            step_size
        )
        # ----------sweep-right----------

        # ----------sweep-left-----------
        for site in reversed(range(1, len(state.A))):
            new_A = cls.evolve_site(state.A[site], site, step_size)
            new_A, bond = state.right_qr_tensors(new_A)
            state.A[site] = new_A
            cls.calculate_layer_right(new_A, site)
            new_bond = cls.evolve_bond(bond, site - 1, step_size)
            state.A[site - 1] = np.tensordot(
                state.A[site - 1],
                new_bond,
                (2, 0)
            )
        state.A[0] = cls.evolve_site(
            state.A[0],
            0,
            step_size
        )
        # ----------sweep-left---------

        return state

    @ classmethod
    def evolve_site(cls, A, site, step_size):
        layer_left = cls.get_layer_left(site - 1)
        layer_right = cls.get_layer_right(site + 1)
        H_eff = np.tensordot(
            cls.hamiltonian.A[site],
            layer_left,
            (2, 1)
        )
        H_eff = np.tensordot(
            H_eff,
            layer_right,
            (2, 1)
        )
        H_eff = np.transpose(H_eff, (0, 2, 4, 1, 3, 5))
        H_eff = np.reshape(H_eff, (
            H_eff.shape[0] * H_eff.shape[1] * H_eff.shape[2],
            H_eff.shape[3] * H_eff.shape[4] * H_eff.shape[5]
        ))
        t = ((np.pi / 2) * step_size) / 2
        U_eff = expm(-(1j) * t * H_eff)
        shape = A.shape
        new_A = np.reshape(A, -1)
        new_A = np.tensordot(new_A, U_eff, (0, 0))
        return np.reshape(new_A, shape)

    @ classmethod
    def evolve_bond(cls, bond, site, step_size):
        layer_left = cls.get_layer_left(site)
        layer_right = cls.get_layer_right(site + 1)
        H_eff = np.tensordot(
            layer_left,
            layer_right,
            (1, 1)
        )
        H_eff = np.transpose(H_eff, (0, 2, 1, 3))
        H_eff = np.reshape(H_eff, (
            H_eff.shape[0] * H_eff.shape[1],
            H_eff.shape[2] * H_eff.shape[3]
        ))
        t = -((np.pi / 2) * step_size) / 2
        U_eff = expm(-(1j) * t * H_eff)
        shape = bond.shape
        new_bond = np.reshape(bond, -1)
        new_bond = np.tensordot(new_bond, U_eff, (0, 0))
        return np.reshape(new_bond, shape)

    @ classmethod
    def get_layer_left(cls, site):
        if site == -1:
            return np.ones((1, 1, 1))
        else:
            return cls.L[site]

    @ classmethod
    def get_layer_right(cls, site):
        if site == args.rules.ncells:
            return np.ones((1, 1, 1))
        else:
            return cls.R[site]

    @ classmethod
    def form_new_layer(cls, A, site):
        # TODO optimize comtraction order
        """
        Form a new layer to be contracted with one of the sides of H_eff
        """
        # Contract physical leg of A with first physical leg of hamiltonian.A
        new_layer = np.tensordot(
            A,
            cls.hamiltonian.A[site],
            (0, 0)
        )
        # Contract second physical leg of hamiltonian.A with physical leg of A.conj()
        new_layer = np.tensordot(
            new_layer,
            A.conj(),
            (2, 0)
        )
        return new_layer

    @ classmethod
    def calculate_layer_left(cls, A, site):
        """
        Calculates H_eff on the left of the given site by adding one layer, given the left-orthogonal tensor A one to the left of the given site, and saving the result to cls.L
        """
        new_layer = cls.form_new_layer(A, site)
        # Contract the new layer with the rest of the left side of H_eff
        cls.L[site] = np.tensordot(
            new_layer,
            cls.get_layer_left(site - 1),
            ((0, 2, 4), (0, 1, 2))
        )

    @ classmethod
    def calculate_layer_right(cls, A, site):
        """
        Calculates H_eff on the right of the given site by adding one layer, given the right-orthogonal tensor A one to the right of the given site, and saving the result to cls.R
        """
        new_layer = cls.form_new_layer(A, site)
        # Contract the new layer with the rest of the right side of H_eff
        cls.R[site] = np.tensordot(
            new_layer,
            cls.get_layer_right(site + 1),
            ((1, 3, 5), (0, 1, 2))
        )

    @ classmethod
    def prepare_exact(cls, step_size):
        t = (np.pi / 2) * step_size
        cls.U = expm(-(1j) * t * cls.hamiltonian.asMatrix())

    @ classmethod
    def prepare_tdvp(cls, state: MPS):
        """
        Prepare the tdvp algorithm by calculating all right layers of H_eff for the first iteration
        """
        # Sweep once though the mps to fix the bond dimensions
        state.make_site_canonical(len(state.A) - 1)
        # Sweep back to ensure every tensor is in right canonical form
        state.make_site_canonical(0)

        cls.L = [None] * len(state.A)
        cls.R = [None] * len(state.A)

        for site in reversed(range(1, len(state.A))):
            cls.calculate_layer_right(state.A[site], site)

    @ classmethod
    def classical(cls, first_column):
        """
        Non-Quantum time evolution according to classical wolfram rules
        """
        classical = np.zeros([args.plot_steps, len(first_column)])
        classical[0, :] = first_column
        classical[:, 0] = first_column[0]
        classical[:, -1] = first_column[-1]
        for step in range(1, args.plot_steps):
            for site in range(args.rules.distance, len(first_column) - args.rules.distance):
                sum = 0.
                for offset in range(-args.rules.distance, args.rules.distance + 1):
                    if offset != 0:
                        sum += classical[step - 1, (site + offset)]
                if sum in args.rules.activation_interval:
                    classical[step, site] = 1. - classical[step - 1, site]
                else:
                    classical[step, site] = classical[step - 1, site]

        return classical
