import numpy as np

from algorithms import Algorithm
from lautils import timestep, normalize
from parameters import Parser
from tensor_networks import MPS, MPO


class TDVP(Algorithm):

    def __init__(self, psi_0: MPS, H: MPO, args: Parser) -> None:
        """
        Prepares the tdvp algorithm by setting variables and calculating all right layers of H_eff for the first
        iteration
        :param psi_0: The initial state
        :param H: The hamiltonian governing the time evolution
        :param args: Parameters
        """
        super().__init__(psi_0, H, args)

        num_sites = len(self._psi.A)
        # Sweep once though the mps to fix the bond dimensions
        self._psi.make_site_canonical(num_sites - 1)
        # Sweep back to bring every tensor into right orthonormal form
        self._psi.make_site_canonical(0)

        self._H_eff_left: list[np.ndarray] = [np.array(0)] * num_sites
        self._H_eff_right: list[np.ndarray] = [np.array(0)] * num_sites

        # self._max_bond_dims: list[int] = [
        #     min(2 ** i, 2 ** (len(self._psi.A) - i), self._max_bond_dim) for i in range(len(self._psi.A) + 1)
        # ]

        # self._target_bond_dims: list[int] = self._max_bond_dims.copy()

        for site in reversed(range(1, num_sites)):
            self._psi.make_site_canonical(site - 1)
            self._update_layer_H_eff(side="right", site=site)

    @property
    def psi(self) -> MPS:
        return self._psi

    @psi.setter
    def psi(self, value: MPS) -> None:
        self._psi = value

    def do_time_step(self) -> None:
        """
        Update psi by simulating one time step
        """
        self._psi.make_site_canonical(0)
        self._sweep_right_2tdvp()
        self._sweep_left_2tdvp()

    def _sweep_right(self) -> None:
        for site in range(len(self._psi.A)):
            target_shape = (2, self._target_bond_dims[site], self._target_bond_dims[site + 1])
            new_A = self._evolve_site(site, self._step_size / 2)
            assert new_A.shape[0] == target_shape[0]
            assert new_A.shape[1] == target_shape[1]
            if site == len(self._psi.A) - 1:
                self._psi.A[site] = new_A
            else:
                left_orthonormal_A, C = MPS.left_qr_tensors(new_A, reduced=False)
                self._psi.A[site] = MPS.truncate_and_pad_into_shape(left_orthonormal_A, target_shape)
                assert self._psi.A[site].shape == target_shape
                self._update_layer_H_eff(side="left", site=site)
                new_C = MPS.truncate_and_pad_into_shape(C, (target_shape[2], C.shape[1]))
                new_C = self._evolve_bond(bond=site + 1, C=new_C)
                self._psi.A[site + 1] = np.transpose(np.tensordot(
                    new_C,
                    self._psi.A[site + 1],
                    (1, 1)
                ), (1, 0, 2))

    def _sweep_left(self) -> None:
        for site in reversed(range(len(self._psi.A))):
            target_shape = (2, self._target_bond_dims[site], self._target_bond_dims[site + 1])
            new_A = self._evolve_site(site, self._step_size / 2)
            assert new_A.shape[0] == target_shape[0]
            assert new_A.shape[2] == target_shape[2]
            if site == 0:
                self._psi.A[site] = new_A
            else:
                right_orthonormal_A, C = MPS.right_qr_tensors(new_A, reduced=False)
                self._psi.A[site] = MPS.truncate_and_pad_into_shape(right_orthonormal_A, target_shape)
                assert self._psi.A[site].shape == target_shape
                self._update_layer_H_eff(side="right", site=site)
                new_C = MPS.truncate_and_pad_into_shape(C, (C.shape[0], target_shape[1]))
                new_C = self._evolve_bond(bond=site, C=new_C)
                self._psi.A[site - 1] = np.tensordot(
                    self._psi.A[site - 1],
                    new_C,
                    (2, 0)
                )

    def _sweep_right_2tdvp(self) -> None:
        for site in range(len(self._psi.A) - 1):
            new_A_left, s, new_A_right = self._evolve_split_and_truncate(
                A_left=self._psi.A[site],
                A_right=self._psi.A[site + 1],
                H_eff_left=self._get_layer_H_eff(side="left", site=site - 1),
                H_eff_right=self._get_layer_H_eff(side="right", site=site + 2),
                W_left=self._H.W[site],
                W_right=self._H.W[site + 1],
                max_bond_dim=self.args.max_bond_dim,
                step_size=(self.args.step_size / 2),
                epsilon=self.args.svd_epsilon
            )
            self._psi.A[site] = new_A_left
            self._psi.A[site + 1] = np.tensordot(np.diag(s), new_A_right, (1, 1)).transpose((1, 0, 2))
            if site < len(self._psi.A) - 2:
                self._update_layer_H_eff(side="left", site=site)
                self._psi.A[site + 1] = self._evolve_site(site + 1, -self.args.step_size / 2)

    def _sweep_left_2tdvp(self) -> None:
        for site in reversed(range(1, len(self._psi.A))):
            # print(site)
            new_A_left, s, new_A_right = self._evolve_split_and_truncate(
                A_left=self._psi.A[site - 1],
                A_right=self._psi.A[site],
                H_eff_left=self._get_layer_H_eff(side="left", site=site - 2),
                H_eff_right=self._get_layer_H_eff(side="right", site=site + 1),
                W_left=self._H.W[site - 1],
                W_right=self._H.W[site],
                max_bond_dim=self.args.max_bond_dim,
                step_size=(self.args.step_size / 2),
                epsilon=self.args.svd_epsilon
            )
            self._psi.A[site] = new_A_right
            self._psi.A[site - 1] = np.tensordot(new_A_left, np.diag(s), (2, 0))
            if site > 1:
                self._update_layer_H_eff(side="right", site=site)
                self._psi.A[site - 1] = self._evolve_site(site - 1, -self.args.step_size / 2)

    def _evolve_site(self, site: int, step_size: float) -> np.ndarray:
        """
        Calculates the time evolution of the site tensor at the given site
        """
        A = self._psi.A[site]
        W = self._H.W[site]

        H_eff_left = self._get_layer_H_eff(
            site=site - 1,
            side="left"
        )
        H_eff_right = self._get_layer_H_eff(
            site=site + 1,
            side="right"
        )
        return self._evolve_A(A, H_eff_left, H_eff_right, W, step_size)

    def _evolve_bond(self, bond: int, C: np.ndarray) -> np.ndarray:
        """
        Calculates the time evolution of the given bond tensor based on the given halves of the effective hamiltonian
        """
        H_eff_left = self._get_layer_H_eff(
            site=bond - 1,
            side="left"
        )
        H_eff_right = self._get_layer_H_eff(
            site=bond,
            side="right"
        )
        H_eff = self._assemble_K_eff(H_eff_left, H_eff_right)
        assert H_eff.shape[0] == H_eff.shape[2]
        assert H_eff.shape[1] == H_eff.shape[3]
        H_eff = np.reshape(H_eff, (
            H_eff.shape[0] * H_eff.shape[1],
            H_eff.shape[2] * H_eff.shape[3]
        ))

        shape = C.shape
        new_C = np.reshape(C, -1)
        new_C = timestep(H_eff, new_C, -self.args.step_size / 2)
        new_C = np.reshape(new_C, shape)
        return new_C

    def _get_layer_H_eff(self, side: str, site: int) -> np.ndarray:
        assert side == "left" or side == "right"
        if side == "left" and site < 0 or side == "right" and site >= len(self._psi.A):
            return np.ones((1, 1, 1))
        else:
            H_eff = self._H_eff_left[site] if side == "left" else self._H_eff_right[site]
            return H_eff

    def _update_layer_H_eff(self, side: str, site: int) -> None:
        assert side == "left" or side == "right"
        previous_site = (site - 1) if side == "left" else (site + 1)
        new_H_eff = self._assemble_new_layer_H_eff(
            side=side,
            previous_layer_H_eff=self._get_layer_H_eff(side=side, site=previous_site),
            A=self._psi.A[site],
            W=self._H.W[site]
        )
        if side == "left":
            self._H_eff_left[site] = new_H_eff
        else:
            self._H_eff_right[site] = new_H_eff

    @classmethod
    def _evolve_split_and_truncate(
            cls, A_left: np.ndarray, A_right: np.ndarray,
            H_eff_left: np.ndarray, H_eff_right: np.ndarray,
            W_left: np.ndarray, W_right: np.ndarray,
            step_size: float, max_bond_dim: int, epsilon: float
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        s_A_left, s_A_right = A_left.shape, A_right.shape
        A = (np.tensordot(A_left, A_right, (2, 1))
             .transpose((0, 2, 1, 3))
             .reshape((s_A_left[0] * s_A_right[0], s_A_left[1], s_A_right[2])))
        s_W_left, s_W_right = W_left.shape, W_right.shape
        W = (np.tensordot(W_left, W_right, (3, 2))
             .transpose((0, 3, 1, 4, 2, 5))
             .reshape((s_W_left[0] * s_W_right[0], s_W_left[1] * s_W_right[1], s_W_left[2], s_W_right[3])))
        new_A = cls._evolve_A(A, H_eff_left, H_eff_right, W, step_size)
        new_A = (new_A.reshape((s_A_left[0], s_A_right[0], s_A_left[1], s_A_right[2]))
                 .transpose((0, 2, 1, 3))
                 .reshape((s_A_left[0] * s_A_left[1], s_A_right[0] * s_A_right[2])))
        u, s, vh = np.linalg.svd(new_A, full_matrices=False)
        max_bond_dim = min(max_bond_dim, len(s))
        new_bond_dim = next((i for i, x in enumerate(s) if (np.linalg.norm(s[i:]) < epsilon)), max_bond_dim)
        new_A_left = u.reshape((s_A_left[0], s_A_left[1], -1))
        new_A_right = vh.reshape((-1, s_A_right[0], s_A_right[2])).transpose((1, 0, 2))
        return new_A_left[:, :, :new_bond_dim], normalize(s[:new_bond_dim]), new_A_right[:, :new_bond_dim, :]

    @classmethod
    def _assemble_H_eff(cls, H_eff_left: np.ndarray, H_eff_right: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Assembles the effective hamiltonian from two sides and an MPO tensor by contracting and transposing them
        :param H_eff_left: Left side of the effective hamiltonian
        :param H_eff_right: Right side of the effective hamiltonian
        :param W: An MPO tensor of the complete hamiltonian
        :return: The contracted effective hamiltonian
        """
        H_eff = np.tensordot(W, H_eff_left, (2, 1))
        H_eff = np.tensordot(H_eff, H_eff_right, (2, 1))
        H_eff = np.transpose(H_eff, (0, 2, 4, 1, 3, 5))
        return H_eff

    @classmethod
    def _assemble_K_eff(cls, H_eff_left: np.ndarray, H_eff_right: np.ndarray) -> np.ndarray:
        """
        Assembles the effective hamiltonian from two sides by contracting and transposing them
        :param H_eff_left: Left side of the effective hamiltonian
        :param H_eff_right: Right side of the effective hamiltonian
        :return: he contracted effective hamiltonian
        """
        K_eff = np.tensordot(
            H_eff_left,
            H_eff_right,
            (1, 1)
        )
        K_eff = np.transpose(K_eff, (0, 2, 1, 3))
        return K_eff

    @classmethod
    def _assemble_new_layer_H_eff(cls, side: str, previous_layer_H_eff: np.ndarray, A: np.ndarray,
                                  W: np.ndarray) -> np.ndarray:
        assert side == "left" or side == "right"
        new_layer_H_eff = np.tensordot(
            A.conj(),
            previous_layer_H_eff,
            (1, 2) if side == "left" else (2, 2)
        )
        new_layer_H_eff = np.tensordot(
            W,
            new_layer_H_eff,
            ((1, 2), (0, 3)) if side == "left" else ((1, 3), (0, 3))
        )
        new_layer_H_eff = np.tensordot(
            A,
            new_layer_H_eff,
            ((0, 1), (0, 3)) if side == "left" else ((0, 2), (0, 3))
        )
        return new_layer_H_eff

    @classmethod
    def _evolve_A(cls, A: np.ndarray, H_eff_left: np.ndarray, H_eff_right: np.ndarray, W: np.ndarray,
                  step_size: float) -> np.ndarray:
        H_eff = cls._assemble_H_eff(H_eff_left, H_eff_right, W)
        assert H_eff.shape[1] == H_eff.shape[4]
        assert H_eff.shape[2] == H_eff.shape[5]
        assert len(H_eff.shape) == 6
        H_eff = np.reshape(H_eff, (
            H_eff.shape[0] * H_eff.shape[1] * H_eff.shape[2],
            H_eff.shape[3] * H_eff.shape[4] * H_eff.shape[5]
        ))

        shape = A.shape
        new_A = np.reshape(A, -1)
        new_A = timestep(H_eff, new_A, step_size)
        new_A = np.reshape(new_A, shape)
        return new_A

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
