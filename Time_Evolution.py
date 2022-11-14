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
            [args.num_steps, args.rules.ncells]
        )
        d_population = np.empty(
            [args.num_steps, args.rules.ncells]
        )
        single_site_entropy = np.empty(
            [args.num_steps, args.rules.ncells]
        )

        if algorithm == 'exact':
            print("Preparing data...")
            t = (np.pi / 2) * args.step_size
            cls.U = expm(-(1j) * t * cls.hamiltonian.asMatrix())

        results = []

        for state_index, state in enumerate(states):
            print("\nSimulating state " + str(state_index + 1) + "...")
            site_canonical_hint = None
            for step in range(args.num_steps):
                if step % 10 == 0:
                    print("Step " + str(step) + " of " + str(args.num_steps))
                cls.measure(
                    state=state,
                    population=population[step, :],
                    d_population=d_population[step, :],
                    single_site_entropy=single_site_entropy[step, :],
                    site_canonical_hint=site_canonical_hint
                )
                if algorithm == 'exact':
                    # After the exact step, the mps will be in site canonical form with the center at the last site
                    site_canonical_hint = "last"
                    state = cls.exact_step(state)
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
        site_range = range(args.rules.ncells)
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
            result = np.tensordot(
                A,
                A.conj(),
                ((1, 2), (1, 2))
            )
            result = np.tensordot(
                result,
                PROJECTION_KET_1,
                ((0, 1), (0, 1))
            ).real
            population[site] = result
            d_population[site] = np.round(result)

            # Calculate the single-site entropy
            partial_trace = np.tensordot(
                A.conj(),
                A,
                ((1, 2), (1, 2))
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                single_site_entropy[site] = (-np.trace(np.dot(
                    partial_trace,
                    logm(partial_trace) / np.log(2)
                ))).real

    @classmethod
    def tdvp_step(cls, state: MPS) -> list[list]:
        """
        Evolves the quantum state using the tdvp algorithm.
        """
        # TODO
        pass

    @classmethod
    def exact_step(cls, state: MPS) -> MPS:
        """
        Evolves the quantum state by calculating a time evolution operator matrix and using explicit matrix vector product. Does not use any tensor network optimizations.
        """
        psi = state.as_vector()
        psi = np.dot(cls.U, psi)
        return MPS.from_vector(psi)

    @classmethod
    def classical(cls, first_column):
        """
        Non-Quantum time evolution according to classical wolfram rules
        """
        classical = np.zeros(
            [args.num_steps, args.rules.ncells]
        )
        classical[0, :] = first_column
        classical[:, 0] = first_column[0]
        classical[:, -1] = first_column[-1]
        for step in range(1, args.num_steps):
            for site in range(args.rules.distance, args.rules.ncells - args.rules.distance):
                sum = 0.
                for offset in range(-args.rules.distance, args.rules.distance + 1):
                    if offset != 0:
                        sum += classical[step - 1, (site + offset)]
                if sum in args.rules.activation_interval:
                    classical[step, site] = 1. - classical[step - 1, site]
                else:
                    classical[step, site] = classical[step - 1, site]

        return classical
