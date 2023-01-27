import random

import numpy as np

from algorithms import Algorithm
from tensor_networks import MPS, MPO


class TDVP(Algorithm):

    def __init__(self, psi_0: MPS, H: MPO, step_size: float, max_bond_dim: int) -> None:
        """
        Prepare the tdvp algorithm by setting variables and calculating all right layers of H_eff for the first
        iteration
        """
        super().__init__(psi_0, H, step_size, max_bond_dim)

        num_sites = len(self._psi.A)
        # Sweep once though the mps to fix the bond dimensions
        self._psi.make_site_canonical(num_sites - 1)
        # Sweep back to bring every tensor into right orthonormal form
        self._psi.make_site_canonical(0)

        self._H_eff_left: list[np.ndarray] = [np.array(0)] * num_sites
        self._H_eff_right: list[np.ndarray] = [np.array(0)] * num_sites

        self._target_bond_dims: list[int] = [
            min(2 ** i, 2 ** (len(self._psi.A) - i), 5) for i in range(len(self._psi.A) + 1)
        ]

        for site in reversed(range(1, num_sites)):
            self._psi.make_site_canonical(site - 1)
            self._calculate_new_right_H_eff(site, self._psi.A[site])
        self.start_counter = 100

    @property
    def psi(self) -> MPS:
        return self._psi

    @psi.setter
    def psi(self, value: MPS) -> None:
        self._psi = value

    def do_time_step(self) -> None:
        self._calculate_new_target_bond_dimensions()
        self._psi.make_site_canonical(0)
        A = self._psi.A
        for site in range(len(A)):
            self._sweep_step_right(site)
        for site in reversed(range(len(A))):
            self._sweep_step_left(site)

    @classmethod
    def _calculate_convergence_measure(
            cls,
            A_left: np.ndarray,
            A_right: np.ndarray,
            C: np.ndarray,
            H_left: np.ndarray,
            H_right: np.ndarray,
            K: np.ndarray
    ) -> float:
        H_left = np.reshape(
            H_left,
            (H_left.shape[0] * H_left.shape[1] * H_left.shape[2],
             H_left.shape[3] * H_left.shape[4] * H_left.shape[5])
        )
        H_right = np.reshape(
            H_right,
            (H_right.shape[0] * H_right.shape[1] * H_right.shape[2],
             H_right.shape[3] * H_right.shape[4] * H_right.shape[5])
        )
        K = np.reshape(
            K,
            (K.shape[0] * K.shape[1], K.shape[2] * K.shape[3])
        )
        A_left = np.reshape(A_left, -1)
        A_right = np.reshape(A_right, -1)
        C = np.reshape(C, -1)
        return (np.linalg.norm(np.tensordot(H_left, A_left, (0, 0))) ** 2 +
                np.linalg.norm(np.tensordot(H_right, A_right, (0, 0))) ** 2 +
                np.linalg.norm(np.tensordot(K, C, (0, 0))) ** 2)

    @classmethod
    def _assemble_H_eff(cls, H_eff_left: np.ndarray, H_eff_right: np.ndarray, W: np.ndarray) -> np.ndarray:
        H_eff = np.tensordot(W, H_eff_left, (2, 1))
        H_eff = np.tensordot(H_eff, H_eff_right, (2, 1))
        H_eff = np.transpose(H_eff, (0, 2, 4, 1, 3, 5))
        return H_eff

    @classmethod
    def _assemble_K_eff(cls, H_eff_left: np.ndarray, H_eff_right: np.ndarray) -> np.ndarray:
        K_eff = np.tensordot(
            H_eff_left,
            H_eff_right,
            (1, 1)
        )
        K_eff = np.transpose(K_eff, (0, 2, 1, 3))
        return K_eff

    def _calculate_new_target_bond_dimensions(self, accuracy: float = 0.000001):
        self._psi.make_site_canonical(0)
        for site in range(len(self._psi.A) - 1):
            self._psi.make_site_canonical(site)
            left_orthonormal_A, _ = MPS.left_qr_tensors(self._psi.A[site])
            self._calculate_new_left_H_eff(site, left_orthonormal_A)

        for bond in reversed(range(1, len(self._psi.A))):
            A_right = self._psi.A[bond].copy()
            _, C = MPS.right_qr_tensors(A_right, reduced=False)
            self._psi.make_site_canonical(bond - 1)
            A_left = self._psi.A[bond - 1].copy()
            C = MPS.truncate_and_pad_into_shape(C, (A_left.shape[2], A_right.shape[1]))

            W_left = self._H.W[bond - 1]
            W_right = self._H.W[bond]
            H_eff_left_left = self._get_layer_H_eff(
                site=bond - 2,
                side="left",
                shape=(A_left.shape[1], A_left.shape[1])
            )
            H_eff_left_middle = self._get_layer_H_eff(
                site=bond - 1,
                side="left",
                shape=(A_right.shape[1], self._max_bond_dim)
            )
            H_eff_right_middle = self._get_layer_H_eff(
                site=bond,
                side="right",
                shape=(A_left.shape[2], self._max_bond_dim)
            )
            H_eff_right_right = self._get_layer_H_eff(
                site=bond + 1,
                side="right",
                shape=(A_right.shape[2], A_right.shape[2])
            )

            H_left = self._assemble_H_eff(H_eff_left_left, H_eff_right_middle, W_left)
            H_right = self._assemble_H_eff(H_eff_left_middle, H_eff_right_right, W_right)
            K = self._assemble_K_eff(H_eff_left_middle, H_eff_right_middle)

            previous_convergence_measure = self._calculate_convergence_measure(
                A_left,
                A_right,
                C,
                H_left[:, :, :, :, :, :1],
                H_right[:, :, :, :, :1, :],
                K[:, :, :1, :1]
            )
            if previous_convergence_measure == 0.0:
                self._target_bond_dims[bond] = 1
                continue
            self._target_bond_dims[bond] = self._max_bond_dim
            for bond_dim in range(2, self._max_bond_dim):
                new_convergence_measure = self._calculate_convergence_measure(
                    A_left,
                    A_right,
                    C,
                    H_left[:, :, :, :, :, :bond_dim],
                    H_right[:, :, :, :, :bond_dim, :],
                    K[:, :, :bond_dim, :bond_dim]
                )
                assert new_convergence_measure >= previous_convergence_measure
                if ((new_convergence_measure / previous_convergence_measure) - 1) <= accuracy:
                    self._target_bond_dims[bond] = bond_dim
                    break
                previous_convergence_measure = new_convergence_measure

    def _sweep_step_right(self, site: int) -> None:
        target_shape = (2, self._target_bond_dims[site], self._target_bond_dims[site + 1])
        new_A = self._evolve_site(site, target_shape)
        assert new_A.shape == target_shape

        if site == len(self._psi.A) - 1:
            self._psi.A[site] = new_A
        else:
            left_orthonormal_A, C = MPS.left_qr_tensors(new_A, reduced=False)
            self._psi.A[site] = MPS.truncate_and_pad_into_shape(left_orthonormal_A, target_shape)
            assert self._psi.A[site].shape == target_shape
            self._calculate_new_left_H_eff(site, left_orthonormal_A)
            new_C = self._evolve_bond(site + 1, C, (target_shape[2], self._psi.A[site + 1].shape[1]))
            assert new_C.shape == (target_shape[2], self._psi.A[site + 1].shape[1])
            self._psi.A[site + 1] = np.transpose(np.tensordot(
                new_C,
                self._psi.A[site + 1],
                (1, 1)
            ), (1, 0, 2))

    def _sweep_step_left(self, site: int) -> None:
        target_shape = (2, self._target_bond_dims[site], self._target_bond_dims[site + 1])
        new_A = self._evolve_site(site, target_shape)
        assert new_A.shape == target_shape

        if site == 0:
            self._psi.A[site] = new_A
        else:
            right_orthonormal_A, C = MPS.right_qr_tensors(new_A, reduced=False)
            self._psi.A[site] = MPS.truncate_and_pad_into_shape(right_orthonormal_A, target_shape)
            assert self._psi.A[site].shape == target_shape
            self._calculate_new_right_H_eff(site, right_orthonormal_A)
            new_C = self._evolve_bond(site, C, (self._psi.A[site - 1].shape[2], target_shape[1]))
            assert new_C.shape == (self._psi.A[site - 1].shape[2], target_shape[1])
            self._psi.A[site - 1] = np.tensordot(
                self._psi.A[site - 1],
                new_C,
                (2, 0)
            )

    def _get_layer_H_eff(self, site: int, side: str, shape: tuple[int, ...]) -> np.ndarray:
        assert len(shape) == 2
        assert side == "left" or side == "right"
        if side == "left" and site < 0 or side == "right" and site >= len(self._psi.A):
            assert shape == (1, 1)
            return np.ones((1, 1, 1))
        else:
            H_eff = self._H_eff_left[site] if side == "left" else self._H_eff_right[site]
            shape = (shape[0], H_eff.shape[1], shape[1])
            if shape[0] > H_eff.shape[0] or shape[1] > H_eff.shape[1] or shape[2] > H_eff.shape[2]:
                return MPS.truncate_and_pad_into_shape(H_eff, shape)
            return H_eff[:shape[0], :shape[1], :shape[2]]

    def _calculate_new_left_H_eff(self, site: int, A: np.ndarray) -> None:
        new_left_H_eff = np.tensordot(
            A.conj(),
            self._get_layer_H_eff(site=site - 1, side="left", shape=(A.shape[1], A.shape[1])),
            (1, 2)
        )
        new_left_H_eff = np.tensordot(
            self._H.W[site],
            new_left_H_eff,
            ((1, 2), (0, 3))
        )
        new_left_H_eff = np.tensordot(
            A,
            new_left_H_eff,
            ((0, 1), (0, 3))
        )
        self._H_eff_left[site] = new_left_H_eff

    def _calculate_new_right_H_eff(self, site: int, A: np.ndarray) -> None:
        new_right_H_eff = np.tensordot(
            A.conj(),
            self._get_layer_H_eff(site=site + 1, side="right", shape=(A.shape[2], A.shape[2])),
            (2, 2)
        )
        new_right_H_eff = np.tensordot(
            self._H.W[site],
            new_right_H_eff,
            ((1, 3), (0, 3))
        )
        new_right_H_eff = np.tensordot(
            A,
            new_right_H_eff,
            ((0, 2), (0, 3))
        )
        self._H_eff_right[site] = new_right_H_eff

    def _evolve_site(self, site: int, target_shape: tuple[int, ...]) -> np.ndarray:
        """
        Calculates the time evolution of the site tensor at the given site
        """
        A = self._psi.A[site]
        W = self._H.W[site]

        calculation_shape = (2, max(A.shape[1], target_shape[1]), max(A.shape[2], target_shape[2]))

        H_eff_left = self._get_layer_H_eff(
            site=site - 1,
            side="left",
            shape=(calculation_shape[1], calculation_shape[1])
        )
        H_eff_right = self._get_layer_H_eff(
            site=site + 1,
            side="right",
            shape=(calculation_shape[2], calculation_shape[2])
        )
        H_eff = self._assemble_H_eff(H_eff_left, H_eff_right, W)
        assert H_eff.shape[1] == H_eff.shape[4]
        assert H_eff.shape[2] == H_eff.shape[5]
        H_eff = np.reshape(H_eff, (
            H_eff.shape[0] * H_eff.shape[1] * H_eff.shape[2],
            H_eff.shape[3] * H_eff.shape[4] * H_eff.shape[5]
        ))
        U_eff = self.calculate_U(H_eff, self._step_size / 2)

        # Pad A, if needed
        if A.shape[0] < calculation_shape[0] or A.shape[1] < calculation_shape[1] or A.shape[2] < calculation_shape[2]:
            A = MPS.truncate_and_pad_into_shape(A, calculation_shape)

        new_A = np.reshape(A, -1)
        new_A = np.tensordot(new_A, U_eff, (0, 0))
        new_A = np.reshape(new_A, calculation_shape)
        return new_A[:target_shape[0], :target_shape[1], :target_shape[2]]

    def _evolve_bond(self, bond: int, C: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
        """
        Calculates the time evolution of the given bond tensor based on the given halves of the effective hamiltonian
        """
        assert len(target_shape) == len(C.shape) == 2
        calculation_shape = (max(C.shape[0], target_shape[0]), max(C.shape[1], target_shape[1]))

        H_eff_left = self._get_layer_H_eff(
            site=bond - 1,
            side="left",
            shape=(calculation_shape[0], calculation_shape[0])
        )
        H_eff_right = self._get_layer_H_eff(
                site=bond,
                side="right",
                shape=(calculation_shape[1], calculation_shape[1])
            )
        H_eff = self._assemble_K_eff(H_eff_left, H_eff_right)
        assert H_eff.shape[0] == H_eff.shape[2]
        assert H_eff.shape[1] == H_eff.shape[3]
        H_eff = np.reshape(H_eff, (
            H_eff.shape[0] * H_eff.shape[1],
            H_eff.shape[2] * H_eff.shape[3]
        ))
        U_eff = self.calculate_U(H_eff, -self._step_size / 2)

        # Pad C, if needed
        if C.shape[0] < calculation_shape[0] or C.shape[1] < calculation_shape[1]:
            C = MPS.truncate_and_pad_into_shape(C, calculation_shape)

        new_C = np.reshape(C, -1)
        new_C = np.tensordot(new_C, U_eff, (0, 0))
        new_C = np.reshape(new_C, calculation_shape)
        return new_C[:target_shape[0], :target_shape[1]]
