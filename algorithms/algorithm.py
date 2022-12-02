from abc import ABC, abstractmethod

import numpy as np

from parameters import Rules
from tensor_networks import MPS, MPO


class Algorithm(ABC):
    _psi: MPS
    _H: MPO
    _step_size: float

    @abstractmethod
    def __init__(self, psi_0: MPS, H: MPO, step_size: float):
        self.psi = psi_0
        self._H = H
        self._step_size = step_size

    @abstractmethod
    def do_time_step(self):
        pass

    @property
    def psi(self):
        return self._psi

    @psi.setter
    def psi(self, value):
        self._psi = value

    @classmethod
    def calculate_U(cls, H_matrix: np.ndarray, step_size: float) -> np.ndarray:
        w, v = np.linalg.eigh(H_matrix)
        t = -1j * (np.pi / 2) * step_size
        w = np.exp(t * w)
        return (w * v) @ v.conj().T

    @classmethod
    def classical_evolution(cls, first_column: np.ndarray, rules: Rules, plot_steps: int) -> np.ndarray:
        """
        Non-Quantum time evolution according to classical wolfram rules
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

    def measure(self, population, d_population, single_site_entropy):
        """
        Measures the population, rounded population and single-site entropy of the given state and writes the results
        into the given arrays
        """
        self._psi.measure(population, d_population, single_site_entropy)

    def write_to_file(self, path: str):
        self._psi.write_to_file(path)
