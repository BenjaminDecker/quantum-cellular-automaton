import numpy as np
from scipy.linalg import expm, logm
import states
from parameters import args
import builders
from constants import DIM
from gates import PROJECTION_KET_0, PROJECTION_KET_1, LOWERING_OPERATOR, RISING_OPERATOR, REORDER_ROTATE_GATE
import warnings
import webbrowser
import plot
import os


print("\nBuilding projection operators...")
dead_small_n_operators = builders.empty()
for i in range(args.NUM_CELLS):
    dead_small_n_operators[i] = builders.builder(i, PROJECTION_KET_0)
alive_small_n_operators = builders.empty()
for i in range(args.NUM_CELLS):
    alive_small_n_operators[i] = builders.builder(i, PROJECTION_KET_1)

# Add up all permutations for the specified rule


def recursive_big_n_calculator(index, offset, alive_count):
    if offset == 0:
        return recursive_big_n_calculator(index, 1, alive_count)
    if alive_count >= args.RULE.stop:
        return 0.
    if offset > args.DISTANCE:
        if alive_count in args.RULE:
            return np.eye(DIM)
        else:
            return 0.
    dead = np.dot(dead_small_n_operators[(
        index + offset) % args.NUM_CELLS], recursive_big_n_calculator(index, offset + 1, alive_count))
    alive = np.dot(alive_small_n_operators[(
        index + offset) % args.NUM_CELLS], recursive_big_n_calculator(index, offset + 1, alive_count + 1))
    return dead + alive


def measure(state_vector, basis_state):
    return np.vdot(state_vector, np.dot(alive_small_n_operators[basis_state], state_vector)).real


step_range = range(args.NUM_CELLS) if args.PERIODIC_BOUNDARIES else range(
    args.DISTANCE, args.NUM_CELLS - args.DISTANCE)

print("\nBuilding Hamiltonian...")
hamiltonian = np.zeros([DIM, DIM], dtype=args.DTYPE)
for i in step_range:
    if args.PERIODIC_BOUNDARIES:
        print("Step " + str(i + 1) + " of " + str(args.NUM_CELLS))
    else:
        print("Step " + str(i - args.DISTANCE + 1) +
              " of " + str(args.NUM_CELLS - 2 * args.DISTANCE))
    s_operator = builders.builder(
        i, LOWERING_OPERATOR) + builders.builder(i, RISING_OPERATOR)
    big_n_operator = recursive_big_n_calculator(i, -args.DISTANCE, 0)
    hamiltonian += np.matmul(s_operator, big_n_operator)

print("\nCalculating unitary time evolution operator...")
t = (np.pi / 2) * args.STEP_SIZE
U = expm(-(1j) * t * hamiltonian)

print("\nCreating state vectors...")
state_vectors = [getattr(states, name)() for name in args.STATE_VECTORS]

for state_index, state_vector in enumerate(state_vectors):
    print("\nSimulating state " + str(state_index) + "...")
    classical = np.empty([args.NUM_STEPS, args.NUM_CELLS], dtype=args.DTYPE)
    population = np.empty([args.NUM_STEPS, args.NUM_CELLS], dtype=args.DTYPE)
    d_population = np.empty([args.NUM_STEPS, args.NUM_CELLS], dtype=args.DTYPE)
    single_site_entropy = np.empty(
        [args.NUM_STEPS, args.NUM_CELLS], dtype=args.DTYPE)

    # ----------classical----------
    for i in range(args.NUM_CELLS):
        classical[:, i] = measure(state_vector, i)

    for i in range(1, args.NUM_STEPS):
        for j in step_range:
            sum = 0
            for offset in range(-args.DISTANCE, args.DISTANCE + 1):
                if offset != 0:
                    sum += classical[i - 1, (j + offset) % args.NUM_CELLS]
            if sum in args.RULE:
                classical[i, j] = 0. if classical[i - 1, j] == 1. else 1.
            else:
                classical[i, j] = classical[i - 1, j]
    # ----------classical----------

    # ------------quantum-----------
    for i in range(args.NUM_STEPS):
        if i % 10 == 0:
            print("Step " + str(i) + " of " + str(args.NUM_STEPS))
        for j in range(args.NUM_CELLS):
            if args.NOSSE:
                # If the single-site-entropy is not calculated, part of computation is unnecessary
                pop_value = measure(state_vector, j)
                population[i, j] = pop_value
                d_population[i, j] = round(pop_value)
            else:
                pop_value = measure(state_vector, 0)
                population[i, j] = pop_value
                d_population[i, j] = round(pop_value)

                if not args.NOSSE:
                    density_matrix = np.outer(
                        state_vector, state_vector.conj())
                    partial_trace = np.trace(density_matrix.reshape(
                        2, 2**(args.NUM_CELLS - 1), 2, 2**(args.NUM_CELLS - 1)), axis1=1, axis2=3)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        single_site_entropy[i, j] = (
                            -np.trace(np.dot(partial_trace, logm(partial_trace) / np.log(2)))).real

                state_vector = np.dot(REORDER_ROTATE_GATE, state_vector)

        state_vector = np.dot(U, state_vector)
    # ------------quantum-----------

    # ----------visualization-------
    # Choose which visualizations to write
    heatmaps = [
        classical,
        population,
        d_population,
    ]
    if not args.NOSSE:
        heatmaps.append(single_site_entropy)

    for format in args.FORMATS:
        path = os.path.join(os.getcwd(), args.PREFIX +
                            str(state_index) + "." + format)
        plot.plot(heatmaps=heatmaps, path=path)
        if args.SHOW:
            webbrowser.open("file://" + path, new=2)

    # ----------visualization-------
