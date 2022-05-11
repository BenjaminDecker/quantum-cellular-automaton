import numpy as np
from scipy.linalg import expm
import plotly.express as px

###Simulation parameters###
#number of cells
NUM_CELLS = 11
#distance to look for alive or dead neighbours
DISTANCE = 2
#number of alive neighbours required for a flip (range(2,4) means either 2 or 3 alive neighbours are required)
RULE = range(2, 4)
#time steps to simulate until the program exits
NUM_STEPS = 50
#data type for real numbers
DTYPE = np.float32

###Constants, do not change###
KET_0 = np.array([1., 0.])
KET_1 = np.array([0., 1.])
SIZE = 2**NUM_CELLS
SHAPE = [ NUM_CELLS, SIZE, SIZE ]
LOWERING_OPERATOR = np.array([[0., 1.], [0., 0.]])
RISING_OPERATOR = np.array([[0., 0.], [1., 0.]])
PROJECTION_KET_0 = np.array([[1., 0.], [0., 0.]])
PROJECTION_KET_1 = np.array([[0., 0.], [0., 1.]])

def blinker_state():
    state = np.array([1], dtype=DTYPE)
    state = np.kron(state, KET_0)
    state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_0)
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

print("Building lowering operators...")
lowering_operators = empty()
for i in range(NUM_CELLS):
    lowering_operators[i] = builder(i, LOWERING_OPERATOR)

print("Building rising operators...")
rising_operators = empty()
for i in range(NUM_CELLS):
    rising_operators[i] = builder(i, RISING_OPERATOR)

print("Building pauli operators...")
pauli_operators = empty()
for i in range(NUM_CELLS):
    pauli_operators[i] = np.dot(lowering_operators[i], rising_operators[i]) - np.dot(rising_operators[i], lowering_operators[i])

print("Building s operators...")
s_operators = empty()
for i in range(NUM_CELLS):
    s_operators[i] = lowering_operators[i] + rising_operators[i]

#Free some memory
lowering_operators = []
rising_operators = []

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
        return 0
    if offset > DISTANCE:
        if alive_count in RULE:
            return np.eye(SIZE)
        else:
            return 0
    dead = np.dot(dead_small_n_operators[index + offset], recursive_big_n_calculator(index, offset + 1, alive_count))
    alive = np.dot(alive_small_n_operators[index + offset], recursive_big_n_calculator(index, offset + 1, alive_count + 1))
    return dead + alive

print("Building big N operators...")
big_n_operators = empty()
for i in range(DISTANCE, NUM_CELLS - DISTANCE):
    print(i)
    big_n_operators[i] = recursive_big_n_calculator(i, -DISTANCE, 0)

print("Building hamiltonian...")
hamiltonian = np.zeros([SIZE, SIZE], dtype=DTYPE)
for i in range(DISTANCE, NUM_CELLS - DISTANCE):
    hamiltonian += np.dot(s_operators[i], big_n_operators[i])

print("Building U...")
t = np.pi / 2.
U = expm(-(1j) * t * hamiltonian)

state_vector = blinker_state()
heatmap = np.empty([NUM_STEPS, NUM_CELLS], dtype=DTYPE)

for i in range(NUM_STEPS):
    for j in range(NUM_CELLS):
        heatmap[i, j] = np.dot(state_vector.conj().T, np.dot(alive_small_n_operators[j], state_vector)).real
    state_vector = np.dot(state_vector, U)

fig = px.imshow(heatmap)
fig.write_html("quantum_blinker.html")
fig.show()