import numpy as np
from random import random
from constants import KET_0, KET_1, KET_PLUS, Rx_gate
from parameters import Parser

args = Parser.instance()


def blinker(width=1):
    n = int((args.rules.ncells - (2 + width)) / 2)
    state = np.array([1.])
    for i in range(n):
        state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    for i in range(width):
        state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    for i in range(n + (2 + width), args.rules.ncells):
        state = np.kron(state, KET_0)
    return state


def triple_blinker():
    n = int((args.rules.ncells - 5) / 2)
    state = np.array([1.])
    for i in range(n):
        state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    for i in range(n + 5, args.rules.ncells):
        state = np.kron(state, KET_0)
    return state


def single(position=int((args.rules.ncells - 1) / 2)):
    state = np.array([1.])
    for i in range(position):
        state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    for i in range(position + 1, args.rules.ncells):
        state = np.kron(state, KET_0)
    return state


def single_bottom():
    return single(args.rules.distance)


def all_ket_1():
    state = np.array([1.])
    for i in range(args.rules.ncells):
        state = np.kron(state, KET_1)
    return state


def all_ket_1_but_outer():
    state = np.array([1.])
    for i in range(args.rules.distance):
        state = np.kron(state, KET_0)
    for i in range(args.rules.ncells - args.rules.distance * 2):
        state = np.kron(state, KET_1)
    for i in range(args.rules.distance):
        state = np.kron(state, KET_0)
    return state


def equal_superposition():
    state = np.array([1.])
    for i in range(args.rules.ncells):
        state = np.kron(state, KET_PLUS)
    return state


def equal_superposition_but_outer():
    state = np.array([1.])
    for i in range(args.rules.distance):
        state = np.kron(state, KET_0)
    for i in range(args.rules.ncells - args.rules.distance * 2):
        state = np.kron(state, KET_PLUS)
    for i in range(args.rules.distance):
        state = np.kron(state, KET_0)
    return state


def gradient(reversed=False):
    state = np.array([1.])
    for i in range(args.rules.ncells):
        state = np.kron(state, np.dot(
            Rx_gate(np.pi * i / (args.rules.ncells - 1)), KET_1 if reversed else KET_0))
    return state


def rand(p=.5):
    state = np.array([1.])
    for i in range(args.rules.ncells):
        state = np.kron(state, KET_0 if random() > p else KET_1)
    return state


def snake():
    n = int((args.rules.ncells - 6) / 2)
    state = np.array([1.])
    for i in range(n):
        state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_0)
    state = np.kron(state, KET_0)
    state = np.kron(state, KET_1)
    state = np.kron(state, KET_1)
    for i in range(n + 6, args.rules.ncells):
        state = np.kron(state, KET_0)
    return state
