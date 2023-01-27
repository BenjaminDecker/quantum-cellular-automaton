import numpy as np

from algorithms import Algorithm
from tensor_networks import MPS, MPO


class Exact(Algorithm):
    """
    Evolves the quantum state by calculating a time evolution operator matrix and using explicit matrix vector product.
    Does not use any tensor network optimizations.
    """

    def __init__(self, psi_0: MPS, H: MPO, step_size: float, max_bond_dim: int) -> None:
        super().__init__(psi_0, H, step_size, max_bond_dim)
        self._U = self.calculate_U(self._H.as_matrix(), self._step_size)

    @property
    def psi(self) -> MPS:
        return MPS.from_vector(self._psi)

    @psi.setter
    def psi(self, value: MPS) -> None:
        self._psi = value.as_vector()

    def do_time_step(self) -> None:
        self._psi = np.dot(self._U, self._psi)
