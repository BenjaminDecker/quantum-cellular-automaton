import numpy as np

# Most of the code of this file is adapted from the tensor network lecture by Prof. Christian Mendl


def crandn(size):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    # 1/sqrt(2) is a normalization factor
    return (np.random.normal(size=size) + 1j*np.random.normal(size=size)) / np.sqrt(2)


class MPS(object):
    """
    Matrix product state (MPS) class.

    The i-th MPS tensor has dimension `[d, D[i], D[i+1]]` with `d` the physical dimension at each site and `D` the list of virtual bond dimensions.
    """

    def __init__(self, d, D, fill='zero'):
        """
        Create a matrix product state.
        """
        self.d = d
        # leading and trailing bond dimensions must agree (typically 1)
        assert D[0] == D[-1]
        if fill == 'zero':
            self.A = [np.zeros((d, D[i], D[i+1])) for i in range(len(D)-1)]
        elif fill == 'random real':
            # random real entries
            self.A = [
                np.random.normal(size=(d, D[i], D[i+1])) / np.sqrt(d*D[i]*D[i+1]) for i in range(len(D)-1)
            ]
        elif fill == 'random complex':
            # random complex entries
            self.A = [
                crandn(size=(d, D[i], D[i+1])) / np.sqrt(d*D[i]*D[i+1]) for i in range(len(D)-1)
            ]
        else:
            raise ValueError('fill = {} invalid.'.format(fill))

    @classmethod
    def from_tensors(cls, Alist):
        """
        Construct a MPS from a list of tensors.
        """
        # create a MPS with dummy tensors
        s = cls(2, (len(Alist) + 1) * [1])
        # assign the actual tensors from `Alist`
        s.A = [np.array(A) for A in Alist]
        s.d = s.A[0].shape[0]
        return s

    @classmethod
    def from_density_distribution(cls, plist, bond_dim=5):
        """
        Construct a MPS with the given bond-dimension from a list of density values describing the probability of each site to be in state ket-1.
        """
        left = np.zeros((2, 1, bond_dim))
        left[:, 0, 0] = np.array([(1. - plist[0])**.5, plist[0]**.5])
        right = np.zeros((2, bond_dim, 1))
        right[:, 0, 0] = np.array([(1. - plist[-1])**.5, plist[-1]**.5])
        if len(plist) == 1:
            return cls.from_tensors(Alist=[left[:, :, :]])
        Alist = [left]
        for i in range(1, len(plist) - 1):
            tensor = np.zeros((2, bond_dim, bond_dim))
            tensor[:, 0, 0] = np.array([(1. - plist[i])**.5, (plist[i])**.5])
            Alist.append(tensor)
        Alist.append(right)
        return cls.from_tensors(Alist=Alist)

    @classmethod
    def from_vector(cls, psi):
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
    def merge_mps_tensor_pair(cls, A0, A1):
        """
        Merge two neighboring MPS tensors.
        """
        A = np.tensordot(A0, A1, (2, 1))
        # pair original physical dimensions of A0 and A1
        A = A.transpose((0, 2, 1, 3))
        # combine original physical dimensions
        A = A.reshape((A.shape[0]*A.shape[1], A.shape[2], A.shape[3]))
        return A

    def orthonormalize_left_qr(self, i):
        """
        Left-orthonormalize the MPS tensor at index i by a QR decomposition, and update tensor at next site.
        """
        assert i < len(self.A) - 1
        A = self.A[i]
        Anext = self.A[i + 1]
        # perform QR decomposition and replace A by reshaped Q matrix
        s = A.shape
        assert len(s) == 3
        Q, R = np.linalg.qr(np.reshape(A, (s[0]*s[1], s[2])))
        A = np.reshape(Q, (s[0], s[1], Q.shape[1]))
        # update Anext tensor: multiply with R from left
        Anext = np.transpose(np.tensordot(R, Anext, (1, 1)), (1, 0, 2))
        self.A[i] = A
        self.A[i + 1] = Anext
        return self

    def orthonormalize_right_qr(self, i):
        """
        Right-orthonormalize the MPS tensor at index i by a QR decomposition, and update tensor at previous site.
        """
        assert i > 0
        A = self.A[i]
        Aprev = self.A[i - 1]
        # flip left and right virtual bond dimensions
        A = np.transpose(A, (0, 2, 1))
        # perform QR decomposition and replace A by reshaped Q matrix
        s = A.shape
        assert len(s) == 3
        Q, R = np.linalg.qr(np.reshape(A, (s[0]*s[1], s[2])))
        A = np.transpose(np.reshape(Q, (s[0], s[1], Q.shape[1])), (0, 2, 1))
        # update Aprev tensor: multiply with R from right
        Aprev = np.tensordot(Aprev, R, (2, 1))
        self.A[i] = A
        self.A[i - 1] = Aprev
        return self

    def make_site_canonical(self, i):
        """
        Brings the mps into site-canonical form with the center at site i
        """
        for j in range(i):
            self.orthonormalize_left_qr(j)
        for j in reversed(range(i + 1, len(self.A))):
            self.orthonormalize_right_qr(j)
        return self

    # def make_bond_canonical(self, i):
    #     """
    #     Brings the mps into bond-canonical form with the center
    #     """
    #     # TODO
    #     pass

    def as_vector(self):
        """
        Merge all tensors to obtain the vector representation on the full Hilbert space.
        """
        psi = self.A[0]
        for i in range(1, len(self.A)):
            psi = self.merge_mps_tensor_pair(psi, self.A[i])
        # contract leftmost and rightmost virtual bond (has no influence if these virtual bond dimensions are 1)
        psi = np.trace(psi, axis1=1, axis2=2)
        return psi
