import warnings

import numpy as np
from scipy.linalg import logm

from constants import PROJECTION_KET_1
from parameters import Parser

# Most of the code of this file is adapted from the tensor network lecture by Prof. Christian Mendl

args = Parser.instance()


class MPS(object):
    """
    Matrix product state (MPS) class.

    The i-th MPS tensor has dimension `[d, D[i], D[i+1]]` with `d` the physical dimension at each site and `D` the
    list of virtual bond dimensions.
    """

    A: list[np.ndarray] = []

    def __init__(self, Alist: list[np.ndarray]) -> None:
        self.A = Alist

    @classmethod
    def from_tensors(cls, Alist) -> 'MPS':
        """
        Construct an MPS from a list of tensors.
        """
        return cls(Alist=[np.array(A) for A in Alist])

    @classmethod
    def from_density_distribution(cls, plist, bond_dim=args.bond_dim) -> 'MPS':
        """
        Constructs an MPS with the given bond-dimension from a list of density values describing the probability of
        each site to be in state ket-1.
        """
        left = np.zeros((2, 1, bond_dim))
        left[:, 0, 0] = np.array([(1. - plist[0]) ** .5, plist[0] ** .5])
        right = np.zeros((2, bond_dim, 1))
        right[:, 0, 0] = np.array([(1. - plist[-1]) ** .5, plist[-1] ** .5])
        if len(plist) == 1:
            return cls.from_tensors(Alist=[left[:, :, :]])
        Alist = [left]
        for i in range(1, len(plist) - 1):
            tensor = np.zeros((2, bond_dim, bond_dim))
            tensor[:, 0, 0] = np.array([(1. - plist[i]) ** .5, (plist[i]) ** .5])
            Alist.append(tensor)
        Alist.append(right)
        return cls.from_tensors(Alist=Alist)

    @classmethod
    def from_vector(cls, psi) -> 'MPS':
        """
        Creates an MPS from a full state vector array.
        """
        Alist = []
        psi = np.array(psi)
        psi = np.reshape(psi, (2, -1))

        while psi.shape[1] > 1:
            Q, R = np.linalg.qr(psi)
            Q = np.reshape(Q, (-1, 2, R.shape[0]))
            Q = np.transpose(Q, (1, 0, 2))
            Alist.append(Q)
            psi = np.reshape(R, (R.shape[0] * 2, -1))
        psi = np.reshape(psi, (-1, 2, 1))
        psi = np.transpose(psi, (1, 0, 2))
        Alist.append(psi)

        return cls.from_tensors(Alist=Alist)

    @classmethod
    def from_file(cls, path: str) -> 'MPS':
        with open(path, 'rb') as f:
            Adict = np.load(f)
            Alist = [Adict[F"arr_{i}"] for i in range(len(Adict.files))]
            return cls(Alist=Alist)

    # TODO find better name
    @classmethod
    def left_qr_tensors(cls, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = A.shape
        assert s[2] > 1
        Q, R = np.linalg.qr(np.reshape(A, (s[0] * s[1], s[2])))
        Q = np.reshape(Q, (s[0], s[1], -1))
        return Q, R

    @classmethod
    def right_qr_tensors(cls, A) -> tuple[np.ndarray, np.ndarray]:
        A_new, R_new = cls.left_qr_tensors(
            np.transpose(A, (0, 2, 1))
        )
        A_new = np.transpose(A_new, (0, 2, 1))
        R_new = np.transpose(R_new, (1, 0))
        return A_new, R_new

    def measure(self, population, d_population, single_site_entropy) -> None:
        """
        Measures the population, rounded population and single-site entropy of the given state and writes the results
        into the given arrays
        """
        # Start with the orthogonality center at site 0
        self.make_site_canonical(0)
        first = True
        for site in range(len(self.A)):
            # In every iteration except the first, shift the center tensor one site to the left/right
            if first:
                first = False
            else:
                self.orthonormalize_left_qr(site - 1)

            A = self.A[site]

            # Calculate the population density
            density_matrix = np.tensordot(
                A,
                A.conj(),
                ((1, 2), (1, 2))
            )
            result = np.tensordot(
                density_matrix,
                PROJECTION_KET_1,
                ((0, 1), (0, 1))
            ).real
            population[site] = result
            d_population[site] = np.round(result)
            # Calculate the single-site entropy
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                single_site_entropy[site] = (-np.trace(np.dot(
                    density_matrix,
                    logm(density_matrix) / np.log(2)
                ))).real

    def write_to_file(self, path: str) -> None:
        with open(path, 'wb') as f:
            np.savez(f, *self.A)

    def orthonormalize_left_qr(self, i) -> 'MPS':
        """
        Left-orthonormalize the MPS tensor at index i by a QR decomposition, and update tensor at next site.
        """
        assert i < len(self.A) - 1
        A = self.A[i]
        # perform QR decomposition and replace A by reshaped Q matrix
        A_new, R = MPS.left_qr_tensors(A)
        # update Anext tensor: multiply with R from left
        Aright = np.transpose(
            np.tensordot(R, self.A[i + 1], (1, 1)),
            (1, 0, 2)
        )
        self.A[i] = A_new
        self.A[i + 1] = Aright
        return self

    def orthonormalize_right_qr(self, i) -> 'MPS':
        """
        Right-orthonormalize the MPS tensor at index i by a QR decomposition, and update tensor at previous site.
        """
        assert i > 0
        A = self.A[i]
        # perform QR decomposition and replace A by reshaped Q matrix
        A_new, R = MPS.right_qr_tensors(A)
        # update left tensor: multiply with R from right
        Aleft = np.tensordot(self.A[i - 1], R, (2, 0))
        self.A[i] = A_new
        self.A[i - 1] = Aleft
        return self

    def make_site_canonical(self, i) -> 'MPS':
        """
        Brings the mps into site-canonical form with the center at site i
        """
        for j in range(i):
            self.orthonormalize_left_qr(j)
        for j in reversed(range(i + 1, len(self.A))):
            self.orthonormalize_right_qr(j)
        return self

    def as_vector(self) -> np.ndarray:
        """
        Merge all tensors to obtain the vector representation on the full Hilbert space.
        """
        psi = self.A[0]
        for i in range(1, len(self.A)):
            psi = np.tensordot(psi, self.A[i], (2, 1))
            psi = np.transpose(psi, (0, 2, 1, 3))
            psi = np.reshape(psi, (
                psi.shape[0] * psi.shape[1],
                psi.shape[2],
                psi.shape[3]
            ))
        # contract leftmost and rightmost virtual bond (has no influence if these virtual bond dimensions are 1)
        psi = np.trace(psi, axis1=1, axis2=2)
        return psi

    def print_shapes(self) -> None:
        for A in self.A:
            print(A.shape)
        print("")
