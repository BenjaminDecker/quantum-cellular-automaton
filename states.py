from random import random

import numpy as np

from parameters import Parser
from tensor_networks import MPS

args = Parser.instance()


def blinker(width=1) -> MPS:
    plist = [0.] * args.rules.ncells
    mid = int(len(plist) / 2)
    plist[mid - width] = plist[mid + width] = 1.
    return MPS.from_density_distribution(plist=plist)


def triple_blinker() -> MPS:
    plist = [0.] * args.rules.ncells
    mid = int(len(plist) / 2)
    plist[mid - 2] = plist[mid] = plist[mid + 2] = 1.
    return MPS.from_density_distribution(plist=plist)


def single(position=int(args.rules.ncells / 2)) -> MPS:
    plist = [0.] * args.rules.ncells
    plist[position] = 1.
    return MPS.from_density_distribution(plist=plist)


def single_bottom() -> MPS:
    return single(0)


def all_ket_0() -> MPS:
    plist = [0.] * args.rules.ncells
    return MPS.from_density_distribution(plist=plist)


def all_ket_1() -> MPS:
    plist = [1.] * args.rules.ncells
    return MPS.from_density_distribution(plist=plist)


def all_ket_1_but_outer() -> MPS:
    plist = [0.] * args.rules.distance
    plist += [1.] * (args.rules.ncells - 2 * args.rules.distance)
    plist += [0.] * args.rules.distance
    return MPS.from_density_distribution(plist=plist)


def equal_superposition() -> MPS:
    plist = [.5] * args.rules.ncells
    return MPS.from_density_distribution(plist=plist)


def equal_superposition_but_outer() -> MPS:
    plist = [0.] * args.rules.distance
    plist += [.5] * (args.rules.ncells - 2 * args.rules.distance)
    plist += [0.] * args.rules.distance
    return MPS.from_density_distribution(plist=plist)


def gradient(reverse=False) -> MPS:
    plist = [
        np.sin(np.pi * i / (args.rules.ncells - 1) / 2) for i in range(args.rules.ncells)
    ]
    return MPS.from_density_distribution(plist=list(reversed(plist)) if reverse else plist)


def rand(p=.5) -> MPS:
    plist = [0. if random() > p else 1. for _ in range(args.rules.ncells)]
    return MPS.from_density_distribution(plist=plist)
