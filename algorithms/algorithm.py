from abc import ABC, abstractmethod

import numpy as np

from parameters import Parser
from parameters import Rules
from tensor_networks import MPS, MPO


class Algorithm(ABC):
    _H: MPO
    args: Parser

    @abstractmethod
    def __init__(self, psi_0: MPS, H: MPO, args: Parser) -> None:
        self.psi = psi_0
        self._H = H
        self.args = args

    @abstractmethod
    def do_time_step(self) -> None:
        pass

    @property
    @abstractmethod
    def psi(self) -> MPS:
        pass

    @psi.setter
    @abstractmethod
    def psi(self, value: MPS) -> None:
        pass

    @classmethod
    def classical_evolution(cls, first_column: np.ndarray, rules: Rules, plot_steps: int) -> np.ndarray:
        """
        Non-Quantum time evolution governed by classical wolfram rules
        """
        heatmap = np.zeros([plot_steps, len(first_column)])
        heatmap[0, :] = first_column
        heatmap[:, 0] = first_column[0]
        heatmap[:, -1] = first_column[-1]
        for step in range(1, plot_steps):
            for site in range(len(first_column)):
                sum = 0.
                for offset in range(-rules.distance, rules.distance + 1):
                    if offset != 0:
                        index = site + offset
                        if 0 <= index < len(first_column):
                            sum += heatmap[step - 1, index]
                if sum in rules.activation_interval:
                    heatmap[step, site] = 1. - heatmap[step - 1, site]
                else:
                    heatmap[step, site] = heatmap[step - 1, site]

        return heatmap

    def measure(self, population, d_population, single_site_entropy, bond_dims) -> None:
        """
        Measures the population, rounded population and single-site entropy of the given state and writes the results
        into the given arrays
        """
        self.psi.measure(population, d_population, single_site_entropy, bond_dims)
