import numpy as np

from algorithms import Algorithm
from tensor_networks import MPS, MPO


class Exact(Algorithm):
    """
    Evolves the quantum state by calculating a time evolution operator matrix and using explicit matrix vector product.
    Does not use any tensor network optimizations.
    """

    def __init__(self, psi_0: MPS, H: MPO, step_size: float) -> None:
        super().__init__(psi_0, H, step_size)
        self._U = self.calculate_U(self._H.as_matrix(), self._step_size)

    def do_time_step(self) -> None:
        vec = self._psi.as_vector()
        vec_new = np.dot(self._U, vec)
        self._psi = MPS.from_vector(vec_new)
