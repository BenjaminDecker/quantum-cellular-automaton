import numpy as np
import random
from parameters import *
from constants import *

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
