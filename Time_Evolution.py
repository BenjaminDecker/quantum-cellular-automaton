from MPO import MPO
from MPS import MPS
from parameters import Parser
import numpy as np
from constants import DIM, SWAP_GATE, PROJECTION_KET_1
from scipy.linalg import expm, logm
import warnings

args = Parser.instance()


class Time_Evolution(object):
    """
    Collection of different methods for calculating the time evolution of initial states according to a hamiltonian.
    """

    @classmethod
    def classical(cls, states: list[MPS]) -> list:
        """
        Non-Quantum time evolution according to classical wolfram rules
        """
        # TODO
        return [np.empty([args.num_steps, args.rules.ncells]) for _ in range(len(states))]

    @classmethod
    def exact(cls, states: list[MPS], hamiltonian: MPO) -> list[list]:
        """
        Evolves the quantum state by calculating a time evolution operator matrix and using explicit matrix vector product. Does not use any tensor network optimizations.
        """
        ket_1_projectors = []
        for cell_index in range(args.rules.ncells):
            result = np.array([1.])
            for i in range(cell_index):
                result = np.kron(result, np.eye(2))
            result = np.kron(result, PROJECTION_KET_1)
            for i in range(cell_index + 1, args.rules.ncells):
                result = np.kron(result, np.eye(2))
            ket_1_projectors.append(result)

        # If sse is turned on, pre-compute the ror-gate
        if args.sse:
            cls.reorder_rotate_gate()

        print("Calculating unitary time evolution operator...")
        t = (np.pi / 2) * args.step_size
        U = expm(-(1j) * t * hamiltonian.asMatrix())

        result = []

        for state_index, state_vector in enumerate(states):
            state_vector = state_vector.as_vector()
            if len(states) > 1:
                print("\nSimulating state " + str(state_index + 1) + "...")
            else:
                print("\nSimulating state...")

            population = np.empty(
                [args.num_steps, args.rules.ncells]
            )
            d_population = np.empty(
                [args.num_steps, args.rules.ncells]
            )
            single_site_entropy = np.empty(
                [args.num_steps, args.rules.ncells]
            )
            for i in range(args.num_steps):
                if i % 10 == 0:
                    print("Step " + str(i) + " of " + str(args.num_steps))
                for j in range(args.rules.ncells):

                    # If the single-site-entropy is not calculated, part of computation is unnecessary
                    if not args.sse:
                        # Measure the j-th cell and save population and rounded population

                        pop_value = np.vdot(
                            state_vector,
                            np.dot(ket_1_projectors[j], state_vector)
                        ).real
                        population[i, j] = pop_value
                        d_population[i, j] = round(pop_value)
                    else:
                        # Measure the first cell and save population and rounded population
                        pop_value = np.vdot(
                            state_vector,
                            np.dot(ket_1_projectors[0], state_vector)
                        ).real
                        population[i, j] = pop_value
                        d_population[i, j] = round(pop_value)

                        # Calculate the single-site-entropy for the first cell
                        density_matrix = np.outer(
                            state_vector, state_vector.conj())
                        partial_trace = np.trace(
                            density_matrix.reshape(
                                2,
                                2**(args.rules.ncells - 1),
                                2,
                                2**(args.rules.ncells - 1)
                            ),
                            axis1=1,
                            axis2=3
                        )
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            single_site_entropy[i, j] = (
                                -np.trace(np.dot(
                                    partial_trace,
                                    logm(partial_trace) / np.log(2)
                                ))
                            ).real

                        # Rotate cells
                        # This is needed to have the next cell in the 0-th position in the next iteration, otherwise the calculation for the single-site-entropy would be more complicated
                        state_vector = np.dot(
                            cls.reorder_rotate_gate(),
                            state_vector
                        )

                state_vector = np.dot(U, state_vector)

            if args.sse:
                result.append([
                    population,
                    d_population,
                    single_site_entropy
                ])
            else:
                result.append([
                    population,
                    d_population
                ])
        return result

    @classmethod
    def reorder_rotate_gate(cls):
        """
        Singleton-method for the reorder rotate gate
        """
        try:
            return cls._rorgate
        except AttributeError:
            print("Calculating reorder rotate gate...")
            rorgate = np.eye(DIM)
            for i in range(args.rules.ncells - 1):
                swap = np.kron(
                    np.eye(2**i),
                    np.kron(SWAP_GATE, np.eye(
                        2**(args.rules.ncells - (i + 2))))
                )
                rorgate = np.dot(swap, rorgate)
            cls._rorgate = rorgate
            return cls._rorgate
