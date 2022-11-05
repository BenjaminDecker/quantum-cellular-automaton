from MPO import MPO
import numpy as np
from scipy.linalg import expm, logm
import states
from parameters import Parser
from constants import PROJECTION_KET_1, REORDER_ROTATE_GATE
import warnings
import webbrowser
import plot
import os

args = Parser.instance()

ket_1_projectors = []
for cell_index in range(args.rules.ncells):
    result = np.array([1.])
    for i in range(cell_index):
        result = np.kron(result, np.eye(2, dtype=args.dtype))
    result = np.kron(result, PROJECTION_KET_1)
    for i in range(cell_index + 1, args.rules.ncells):
        result = np.kron(result, np.eye(2, dtype=args.dtype))
    ket_1_projectors.append(result)


def measure(state_vector, basis_state):
    return np.vdot(state_vector, np.dot(ket_1_projectors[basis_state], state_vector)).real


mpo = MPO.hamiltonian_from_rules(args.rules)

hamiltonian = mpo.asMatrix()

step_range = range(args.rules.ncells) if args.periodic else range(
    args.rules.distance, args.rules.ncells - args.rules.distance)

print("\nCalculating unitary time evolution operator...")
t = (np.pi / 2) * args.step_size
U = expm(-(1j) * t * hamiltonian)

state_vectors = [getattr(states, name)().as_vector()
                 for name in args.initial_states]

for state_index, state_vector in enumerate(state_vectors):
    if len(state_vectors) > 1:
        print("\nSimulating state " + str(state_index + 1) + "...")
    else:
        print("\nSimulating state...")
    classical = np.empty([args.num_steps, args.rules.ncells], dtype=args.dtype)
    population = np.empty(
        [args.num_steps, args.rules.ncells], dtype=args.dtype)
    d_population = np.empty(
        [args.num_steps, args.rules.ncells], dtype=args.dtype)
    single_site_entropy = np.empty(
        [args.num_steps, args.rules.ncells], dtype=args.dtype)

    # ----------classical----------
    for i in range(args.rules.ncells):
        classical[:, i] = measure(state_vector, i)

    for i in range(1, args.num_steps):
        for j in step_range:
            sum = 0
            for offset in range(-args.rules.distance, args.rules.distance + 1):
                if offset != 0:
                    sum += classical[i - 1, (j + offset) % args.rules.ncells]
            if sum in args.rules.activation_interval:
                classical[i, j] = 0. if classical[i - 1, j] == 1. else 1.
            else:
                classical[i, j] = classical[i - 1, j]
    # ----------classical----------

    # ------------quantum-----------
    for i in range(args.num_steps):
        if i % 10 == 0:
            print("Step " + str(i) + " of " + str(args.num_steps))
        for j in range(args.rules.ncells):

            # If the single-site-entropy is not calculated, part of computation is unnecessary
            if not args.sse:
                # Measure the j-th cell and save population and rounded population
                pop_value = measure(state_vector, j)
                population[i, j] = pop_value
                d_population[i, j] = round(pop_value)
            else:
                # Measure the first cell and save population and rounded population
                pop_value = measure(state_vector, 0)
                population[i, j] = pop_value
                d_population[i, j] = round(pop_value)

                # Calculate the single-site-entropy for the first cell
                density_matrix = np.outer(
                    state_vector, state_vector.conj())
                partial_trace = np.trace(density_matrix.reshape(
                    2, 2**(args.rules.ncells - 1), 2, 2**(args.rules.ncells - 1)), axis1=1, axis2=3)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    single_site_entropy[i, j] = (
                        -np.trace(np.dot(partial_trace, logm(partial_trace) / np.log(2)))).real

                # Rotate cells
                # This is needed to have the next cell in the 0-th position in the next iteration,
                # otherwise the calculation for the single-site-entropy would be more complicated
                state_vector = np.dot(REORDER_ROTATE_GATE, state_vector)

        state_vector = np.dot(U, state_vector)
    # ------------quantum-----------

    # ----------visualization-------

    # Only print classical time evolution if it makes sense
    does_classical_make_sense = True
    does_classical_make_sense &= args.step_size == 1.
    for cell in classical[0]:
        does_classical_make_sense &= (cell == 0. or cell == 1.)

    # Choose which heatmaps to print
    heatmaps = []
    if does_classical_make_sense:
        heatmaps.append(classical)
    heatmaps.append(population)
    heatmaps.append(d_population)
    if args.sse:
        heatmaps.append(single_site_entropy)

    # Create one plot for each format specified
    for format in args.file_formats:
        path = os.path.join(os.getcwd(), args.file_prefix +
                            str(state_index) + "." + format)
        plot.plot(heatmaps=heatmaps, path=path)
        if args.show:
            webbrowser.open("file://" + path, new=2)

    # ----------visualization-------
