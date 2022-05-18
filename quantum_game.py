import numpy as np
from scipy.linalg import expm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import random

###Simulation parameters###
#number of cells
NUM_CELLS = 9
#distance to look for alive or dead neighbours
DISTANCE = 1
#number of alive neighbours required for a flip (range(2,4) means either 2 or 3 alive neighbours are required)
RULE = range(1, 2)
#time steps to simulate until the program exits
NUM_STEPS = 500
#data type for real numbers
DTYPE = np.float64
#True means periodic boundary conditions, False means constant boundary conditions
PERIODIC_BOUNDARIES = False
#Size of one time step in the heatmap. A time step size of 1 means one time step per cell
TIME_STEP_SIZE = .1

###Constants, do not change###
KET_0 = np.array([1., 0.])
KET_1 = np.array([0., 1.])
KET_PLUS = np.array([1., 1.]) / np.sqrt(2)
SIZE = 2**NUM_CELLS
SHAPE = [ NUM_CELLS, SIZE, SIZE ]
LOWERING_OPERATOR = np.array([[0., 1.], [0., 0.]])
RISING_OPERATOR = np.array([[0., 0.], [1., 0.]])
PROJECTION_KET_0 = np.array([[1., 0.], [0., 0.]])
PROJECTION_KET_1 = np.array([[0., 0.], [0., 1.]])

def Rx_gate(theta):
    return np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)]])

def blinker_state():
    n = int((NUM_CELLS - 3) / 2)
    state = np.array([1.])
    for i in range(n):
        state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    for i in range(n + 3, NUM_CELLS):
        state = np.kron(state, KET_0)
    return state

def single_state():
    n = int((NUM_CELLS - 1) / 2)
    state = np.array([1.])
    for i in range(n):
        state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    for i in range(n + 1, NUM_CELLS):
        state = np.kron(state, KET_0)
    return state

def all_ket_1_state():
    state = np.array([1.])
    for i in range(NUM_CELLS):
        state = np.kron(state, KET_1)
    return state

def equal_superposition_state():
    state = np.array([1.])
    for i in range(NUM_CELLS):
        state = np.kron(state, KET_PLUS)
    return state

def gradient_state(reversed=False):
    state = np.array([1.])
    for i in range(NUM_CELLS-1,-1,-1) if reversed else range(NUM_CELLS):
        state = np.kron(state, np.dot(Rx_gate(np.pi * i / (NUM_CELLS - 1)), KET_0))
    return state

def random_state(p=.5):
    state = np.array([1.])
    for i in range(NUM_CELLS):
        state = np.kron(state, KET_1 if random.random() > p else KET_0)
    return state

def snake_state():
    n = int((NUM_CELLS - 6) / 2)
    state = np.array([1.])
    for i in range(n):
        state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_0)
    state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_1)
    for i in range(n + 6, NUM_CELLS):
        state = np.kron(state, KET_0)
    return state

def empty():
    return np.empty(SHAPE, dtype=DTYPE)

def builder(index, matrix):
    acc = np.array([1.])
    for i in range(index):
        acc = np.kron(acc, np.eye(2, dtype=DTYPE))
    acc = np.kron(acc, matrix)
    for i in range(index + 1, NUM_CELLS):
        acc = np.kron(acc, np.eye(2, dtype=DTYPE))
    return acc

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

# print("Building big N operators...")
# big_n_operators = empty()
# for i in range(DISTANCE, NUM_CELLS - DISTANCE):
#     print(i)
#     big_n_operators[i] = recursive_big_n_calculator(i, -DISTANCE, 0)

print("Building hamiltonian...")
hamiltonian = np.zeros([SIZE, SIZE], dtype=DTYPE)
step_range = range(NUM_CELLS) if PERIODIC_BOUNDARIES else range(DISTANCE, NUM_CELLS - DISTANCE)
for i in step_range:
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

state_vectors = [
    blinker_state(),
    single_state(),
    gradient_state(reversed=True)]

for index, state_vector in enumerate(state_vectors):
    fig = make_subplots(rows=2)
    population = np.empty([NUM_STEPS, NUM_CELLS], dtype=DTYPE)
    d_population = np.empty([NUM_STEPS, NUM_CELLS], dtype=DTYPE)

    for i in range(NUM_STEPS):
        for j in range(NUM_CELLS):
            pop_value = np.dot(state_vector.conj().T, np.dot(alive_small_n_operators[j], state_vector)).real
            population[i, j] = pop_value
            d_population[i, j] = round(pop_value)
        state_vector = np.dot(state_vector, U)

    fig.add_trace(go.Heatmap(z=population.T, coloraxis = "coloraxis"), 1, 1)
    fig.update_yaxes(scaleanchor = ("x1"), row=1)

    fig.add_trace(go.Heatmap(z=d_population.T, coloraxis = "coloraxis"), 2, 1)
    fig.update_yaxes(scaleanchor = ("x2"), row=2)

    fig.update_layout(coloraxis = {'colorscale':'magma'})

    fig.show()
    # fig.write_html("plot" + str(index) + ".html")

