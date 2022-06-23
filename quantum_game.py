import numpy as np
from scipy.linalg import expm, logm
from plotly.subplots import make_subplots
import states
import plotly.graph_objects as go
from parameters import args
import builders
from constants import DIM
from gates import PROJECTION_KET_0, PROJECTION_KET_1, LOWERING_OPERATOR, RISING_OPERATOR, REORDER_ROTATE_GATE

# Choose which state vectors you want. All options are defined in functions.py
state_vectors = [
    states.blinker_state(width=1),
    # triple_blinker(),
    # single_state(),
    # single_state(position=1),
    # all_ket_1_state(),
    # all_ket_1_but_outer(),
    # equal_superposition_state_but_outer(),
    # gradient_state(reversed=True),
    # random_state(.2),
    # random_state(.2),
    # random_state(.2),
    # random_state(.2),
]

print("Building dead small n operators...")
dead_small_n_operators = builders.empty()
for i in range(args.NUM_CELLS):
    dead_small_n_operators[i] = builders.builder(i, PROJECTION_KET_0)

print("Building alive small n operators...")
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

print("Building hamiltonian...")
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

print("Building U...")
t = (np.pi / 2) * args.STEP_SIZE
U = expm(-(1j) * t * hamiltonian)

for state_index, state_vector in enumerate(state_vectors):
    print("Calculating state " + str(state_index))
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
        for j in range(args.NUM_CELLS):
            pop_value = measure(state_vector, 0)
            population[i, j] = pop_value
            d_population[i, j] = round(pop_value)

            density_matrix = np.outer(state_vector, state_vector.conj())
            partial_trace = np.trace(density_matrix.reshape(
                2, 2**(args.NUM_CELLS - 1), 2, 2**(args.NUM_CELLS - 1)), axis1=1, axis2=3)
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
        single_site_entropy,
    ]

    fig = make_subplots(rows=len(heatmaps))

    for index, heatmap in enumerate(heatmaps):
        fig.add_trace(go.Heatmap(
            z=heatmap.T, coloraxis="coloraxis"), index + 1, 1)
        fig.update_yaxes(scaleanchor=("x" + str(index + 1)), row=(index + 1))

    fig.update_layout(
        coloraxis={'colorscale': 'inferno', 'cmax': 1.0, 'cmin': 0.0})

    fig.show()
    fig.write_html("plot" + str(state_index) + ".html")
    # ----------visualization-------
