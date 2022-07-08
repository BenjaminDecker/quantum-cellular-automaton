from typing_extensions import Self
import numpy as np
from random import random
from constants import KET_0, KET_1, KET_PLUS
from gates import Rx_gate
from parameters import args


def blinker(width=1):
    n = int((args.NUM_CELLS - (2 + width)) / 2)
    state = np.array([1.])
    for i in range(n):
        state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    for i in range(width):
        state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    for i in range(n + (2 + width), args.NUM_CELLS):
        state = np.kron(state, KET_0)
    return state


def triple_blinker():
    n = int((args.NUM_CELLS - 5) / 2)
    state = np.array([1.])
    for i in range(n):
        state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    for i in range(n + 5, args.NUM_CELLS):
        state = np.kron(state, KET_0)
    return state


def single(position=int((args.NUM_CELLS - 1) / 2)):
    state = np.array([1.])
    for i in range(position):
        state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    for i in range(position + 1, args.NUM_CELLS):
        state = np.kron(state, KET_0)
    return state


def single_bottom():
    return single(args.DISTANCE)


def all_ket_1():
    state = np.array([1.])
    for i in range(args.NUM_CELLS):
        state = np.kron(state, KET_1)
    return state


def all_ket_1_but_outer():
    state = np.array([1.])
    for i in range(args.DISTANCE):
        state = np.kron(state, KET_0)
    for i in range(args.NUM_CELLS - args.DISTANCE * 2):
        state = np.kron(state, KET_1)
    for i in range(args.DISTANCE):
        state = np.kron(state, KET_0)
    return state


def equal_superposition():
    state = np.array([1.])
    for i in range(args.NUM_CELLS):
        state = np.kron(state, KET_PLUS)
    return state


def equal_superposition_but_outer():
    state = np.array([1.])
    for i in range(args.DISTANCE):
        state = np.kron(state, KET_0)
    for i in range(args.NUM_CELLS - args.DISTANCE * 2):
        state = np.kron(state, KET_PLUS)
    for i in range(args.DISTANCE):
        state = np.kron(state, KET_0)
    return state


def gradient(reversed=False):
    state = np.array([1.])
    for i in range(args.NUM_CELLS):
        state = np.kron(state, np.dot(
            Rx_gate(np.pi * i / (args.NUM_CELLS - 1)), KET_1 if reversed else KET_0))
    return state


def rand(p=.5):
    state = np.array([1.])
    for i in range(args.NUM_CELLS):
        state = np.kron(state, KET_0 if random() > p else KET_1)
    return state


def snake():
    n = int((args.NUM_CELLS - 6) / 2)
    state = np.array([1.])
    for i in range(n):
        state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_0)
    state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_1)
    for i in range(n + 6, args.NUM_CELLS):
        state = np.kron(state, KET_0)
    return state
