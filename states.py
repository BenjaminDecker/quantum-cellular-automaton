from random import random

import numpy as np

from MPS import MPS
from parameters import Parser

args = Parser.instance()


def blinker(width=1):
    plist = [0.] * args.rules.ncells
    mid = int(len(plist) / 2)
    plist[mid - width] = plist[mid + width] = 1.
    return MPS.from_density_distribution(plist=plist)


def triple_blinker():
    plist = [0.] * args.rules.ncells
    mid = int(len(plist) / 2)
    plist[mid - 2] = plist[mid] = plist[mid + 2] = 1.
    return MPS.from_density_distribution(plist=plist)


def single(position=int(args.rules.ncells / 2)):
    plist = [0.] * args.rules.ncells
    plist[position] = 1.
    return MPS.from_density_distribution(plist=plist)


def single_bottom():
    return single(args.rules.distance)


def all_ket_0():
    plist = [0.] * args.rules.ncells
    return MPS.from_density_distribution(plist=plist)


def all_ket_1():
    plist = [1.] * args.rules.ncells
    return MPS.from_density_distribution(plist=plist)


def all_ket_1_but_outer():
    plist = [0.] * args.rules.distance
    plist += [1.] * (args.rules.ncells - 2 * args.rules.distance)
    plist += [0.] * args.rules.distance
    return MPS.from_density_distribution(plist=plist)


def equal_superposition():
    plist = [.5] * args.rules.ncells
    return MPS.from_density_distribution(plist=plist)


def equal_superposition_but_outer():
    plist = [0.] * args.rules.distance
    plist += [.5] * (args.rules.ncells - 2 * args.rules.distance)
    plist += [0.] * args.rules.distance
    return MPS.from_density_distribution(plist=plist)


def gradient(reverse=False):
    plist = [
        np.sin(np.pi * i / (args.rules.ncells - 1) / 2) for i in range(args.rules.ncells)
    ]
    return MPS.from_density_distribution(plist=list(reversed(plist)) if reverse else plist)


def rand(p=.5):
    plist = [0. if random() > p else 1. for _ in range(args.rules.ncells)]
    return MPS.from_density_distribution(plist=plist)
