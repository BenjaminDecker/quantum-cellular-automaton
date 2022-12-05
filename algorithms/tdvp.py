import numpy as np

from algorithms import Algorithm
from tensor_networks import MPS, MPO


class TDVP(Algorithm):

    def __init__(self, psi_0: MPS, H: MPO, step_size: float) -> None:
        """
        Prepare the tdvp algorithm by setting variables and calculating all right layers of H_eff for the first iteration
        """
        super().__init__(psi_0, H, step_size)

        num_sites = len(self._psi.A)
        # Sweep once though the mps to fix the bond dimensions
        self._psi.make_site_canonical(num_sites - 1)
        # Sweep back to ensure every tensor is in right canonical form
        self._psi.make_site_canonical(0)

        self._H_eff: list[np.ndarray] = [np.array(0)] * num_sites

        for site in reversed(range(1, num_sites)):
            self._calculate_layer_right(site)

    @property
    def psi(self) -> MPS:
        return self._psi

    @psi.setter
    def psi(self, value: MPS) -> None:
        self._psi = value

    def do_time_step(self) -> None:
        self._psi.make_site_canonical(0)
        A = self._psi.A
        for site in range(len(A)):
            self._sweep_step_right(site)
        for site in reversed(range(len(A))):
            self._sweep_step_left(site)

    def _sweep_step_right(self, site: int) -> None:
        new_A = self._evolve_site(site)
        if site == len(self._psi.A) - 1:
            self._psi.A[site] = new_A
        else:
            left_orthonormal_A, C = MPS.left_qr_tensors(new_A)
            self._psi.A[site] = left_orthonormal_A
            self._calculate_layer_left(site)
            new_C = self._evolve_bond(
                C,
                self._get_layer_H_eff(site),
                self._get_layer_H_eff(site + 1)
            )
            self._psi.A[site + 1] = np.transpose(np.tensordot(
                new_C,
                self._psi.A[site + 1],
                (1, 1)
            ), (1, 0, 2))

    def _sweep_step_left(self, site: int) -> None:
        new_A = self._evolve_site(site)
        if site == 0:
            self._psi.A[site] = new_A
        else:

            right_orthonormal_A, C = MPS.right_qr_tensors(new_A)

            self._psi.A[site] = right_orthonormal_A
            self._calculate_layer_right(site)
            new_C = self._evolve_bond(
                C,
                self._get_layer_H_eff(site - 1),
                self._get_layer_H_eff(site)
            )
            self._psi.A[site - 1] = np.tensordot(
                self._psi.A[site - 1],
                new_C,
                (2, 0)
            )

    def _get_layer_H_eff(self, site) -> np.ndarray:
        if site < 0 or site >= len(self._psi.A):
            return np.ones((1, 1, 1))
        else:
            return self._H_eff[site]

    def _calculate_layer_left(self, site: int) -> None:
        """
        Calculates the left part of H_eff from the first site to the given site and saves the result to self._H_eff
        """
        A = self._psi.A[site]
        # Contract the new layer with the rest of the left side of H_eff
        new_layer = np.tensordot(
            A.conj(),
            self._get_layer_H_eff(site - 1),
            (1, 2)
        )
        new_layer = np.tensordot(
            self._H.W[site],
            new_layer,
            ((1, 2), (0, 3))
        )
        new_layer = np.tensordot(
            A,
            new_layer,
            ((0, 1), (0, 3))
        )
        self._H_eff[site] = new_layer

    def _calculate_layer_right(self, site: int) -> None:
        """
        Calculates the right part of H_eff from the given site to the last site and saves the result to self._H_eff
        """
        A = self._psi.A[site]
        # Contract the new layer with the rest of the right side of H_eff
        new_layer = np.tensordot(
            A.conj(),
            self._get_layer_H_eff(site + 1),
            (2, 2)
        )
        new_layer = np.tensordot(
            self._H.W[site],
            new_layer,
            ((1, 3), (0, 3))
        )
        new_layer = np.tensordot(
            A,
            new_layer,
            ((0, 2), (0, 3))
        )
        self._H_eff[site] = new_layer

    def _evolve_site(self, site: int) -> np.ndarray:
        """
        Calculates the time evolution over the given time step for site tensor A at the given site
        """
        H_eff = np.tensordot(
            self._H.W[site],
            self._get_layer_H_eff(site - 1),
            (2, 1)
        )
        H_eff = np.tensordot(
            H_eff,
            self._get_layer_H_eff(site + 1),
            (2, 1)
        )
        H_eff = np.transpose(H_eff, (0, 2, 4, 1, 3, 5))
        H_eff = np.reshape(H_eff, (
            H_eff.shape[0] * H_eff.shape[1] * H_eff.shape[2],
            H_eff.shape[3] * H_eff.shape[4] * H_eff.shape[5]
        ))
        U_eff = self.calculate_U(H_eff, self._step_size / 2)
        A = self.psi.A[site]
        shape = A.shape
        new_A = np.reshape(A, -1)
        new_A = np.tensordot(new_A, U_eff, (0, 0))
        return np.reshape(new_A, shape)

    def _evolve_bond(self, C: np.ndarray, H_eff_left: np.ndarray, H_eff_right: np.ndarray) -> np.ndarray:
        """
        Calculates the time evolution over the given time step for bond tensor C between the given site and the previous
        """
        H_eff = np.tensordot(
            H_eff_left,
            H_eff_right,
            (1, 1)
        )
        H_eff = np.transpose(H_eff, (0, 2, 1, 3))
        H_eff = np.reshape(H_eff, (
            H_eff.shape[0] * H_eff.shape[1],
            H_eff.shape[2] * H_eff.shape[3]
        ))
        U_eff = self.calculate_U(H_eff, -self._step_size / 2)
        shape = C.shape
        new_C = np.reshape(C, -1)
        new_C = np.tensordot(new_C, U_eff, (0, 0))
        return np.reshape(new_C, shape)
