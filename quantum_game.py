import numpy as np
from scipy.linalg import expm, logm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from parameters import *
from constants import *
from functions import *

# print("Building lowering operators...")
# lowering_operators = empty()
# for i in range(NUM_CELLS):
#     lowering_operators[i] = builder(i, LOWERING_OPERATOR)

# print("Building rising operators...")
# rising_operators = empty()
# for i in range(NUM_CELLS):
#     rising_operators[i] = builder(i, RISING_OPERATOR)

# print("Building pauli operators...")
# pauli_operators = empty()
# for i in range(NUM_CELLS):
#     pauli_operators[i] = np.dot(lowering_operators[i], rising_operators[i]) - np.dot(rising_operators[i], lowering_operators[i])

# print("Building s operators...")
# s_operators = empty()
# for i in range(NUM_CELLS):
#     s_operators[i] = lowering_operators[i] + rising_operators[i]

# #Free some memory
# lowering_operators = []
# rising_operators = []

print("Building dead small n operators...")
dead_small_n_operators = empty()
for i in range(NUM_CELLS):
    dead_small_n_operators[i] = builder(i, PROJECTION_KET_0)

print("Building alive small n operators...")
alive_small_n_operators = empty()
for i in range(NUM_CELLS):
    alive_small_n_operators[i] = builder(i, PROJECTION_KET_1)

def recursive_big_n_calculator(index, offset, alive_count):
    if offset == 0:
        return recursive_big_n_calculator(index, 1, alive_count)
    if alive_count >= RULE.stop:
        return 0.
    if offset > DISTANCE:
        if alive_count in RULE:
            return np.eye(SIZE)
        else:
            return 0.
    dead = np.dot(dead_small_n_operators[(index + offset) % NUM_CELLS], recursive_big_n_calculator(index, offset + 1, alive_count))
    alive = np.dot(alive_small_n_operators[(index + offset) % NUM_CELLS], recursive_big_n_calculator(index, offset + 1, alive_count + 1))
    return dead + alive

def measure(state_vector, basis_state):
    return np.vdot(state_vector, np.dot(alive_small_n_operators[basis_state], state_vector)).real

# print("Building big N operators...")
# big_n_operators = empty()
# for i in range(DISTANCE, NUM_CELLS - DISTANCE):
#     print(i)
#     big_n_operators[i] = recursive_big_n_calculator(i, -DISTANCE, 0)

print("Building hamiltonian...")
hamiltonian = np.zeros([SIZE, SIZE], dtype=DTYPE)
for i in STEP_RANGE:
    if PERIODIC_BOUNDARIES:
        print("Step " + str(i + 1) + " of " + str(NUM_CELLS))
    else:
        print("Step " + str(i - DISTANCE + 1) + " of " + str(NUM_CELLS - 2 * DISTANCE))
    s_operator = builder(i, LOWERING_OPERATOR) + builder(i, RISING_OPERATOR)
    big_n_operator = recursive_big_n_calculator(i, -DISTANCE, 0)
    hamiltonian += np.matmul(s_operator, big_n_operator)

print("Building U...")
t = (np.pi / 2) * TIME_STEP_SIZE
U = expm(-(1j) * t * hamiltonian)

reorder_gate = np.eye(SIZE)
for i in range(NUM_CELLS - 1):
    swap = np.kron(np.eye(2**i), np.kron(SWAP_GATE, np.eye(2**(NUM_CELLS - (i + 2)))))
    reorder_gate = np.dot(swap, reorder_gate)

state_vectors = [
    # blinker_state(),
    # single_state(),
    random_state(),
    random_state(),
    random_state(),
    random_state(),
    # gradient_state(reversed=True),
]

for index, state_vector in enumerate(state_vectors):
    classical = np.empty([NUM_STEPS, NUM_CELLS], dtype=DTYPE)
    population = np.empty([NUM_STEPS, NUM_CELLS], dtype=DTYPE)
    d_population = np.empty([NUM_STEPS, NUM_CELLS], dtype=DTYPE)
    single_site_entropy = np.empty([NUM_STEPS, NUM_CELLS], dtype=DTYPE)
    # norm = np.empty([NUM_STEPS, 1], dtype=DTYPE)
    # U_pow = np.eye(SIZE)

    # ----------classical----------
    for i in range(NUM_CELLS):
        classical[:, i] = measure(state_vector, i)
    
    for i in range(1, NUM_STEPS):
        for j in STEP_RANGE:
            sum = 0
            for offset in range (-DISTANCE, DISTANCE + 1):
                sum += classical[i - 1, (j + offset) % NUM_CELLS]
            if sum in RULE:
                classical[i, j] = 0. if classical[i, j] == 1. else 1.
            else:
                classical[i, j] = classical[i - 1, j]
    # ----------classical----------


    #------------quantum-----------
    for i in range(NUM_STEPS):
        for j in range(NUM_CELLS):
            pop_value = measure(state_vector, 0)
            population[i, j] = pop_value
            d_population[i, j] = round(pop_value)

            density_matrix = np.outer(state_vector, state_vector.conj())
            partial_trace = np.trace(density_matrix.reshape(2, 2**(NUM_CELLS -1), 2, 2**(NUM_CELLS -1)), axis1=1, axis2=3)
            single_site_entropy[i, j] = (-np.trace(np.dot(partial_trace, logm(partial_trace) / np.log(2)))).real

            state_vector = np.dot(reorder_gate, state_vector)

        # norm[i, 0] = np.linalg.norm(np.eye(SIZE) - U_pow, 2) / 2
        # print(norm[i, 0])
        # U_pow = np.matmul(U, U_pow)

        state_vector = np.dot(U, state_vector)
    #------------quantum-----------

    #----------visualization-------
    heatmaps = [
        classical,
        population,
        d_population,
        single_site_entropy,
        # norm,
    ]

    fig = make_subplots(rows=len(heatmaps))

    for index, heatmap in enumerate(heatmaps):
        fig.add_trace(go.Heatmap(z=heatmap.T, coloraxis = "coloraxis"), index + 1, 1)
        fig.update_yaxes(scaleanchor = ("x" + str(index + 1)), row=(index + 1))

    fig.update_layout(coloraxis = {'colorscale':'magma'})

    fig.show()
    # fig.write_html("plot" + str(index) + ".html")
    #----------visualization-------
