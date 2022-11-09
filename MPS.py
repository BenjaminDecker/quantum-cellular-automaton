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

    The i-th MPS tensor has dimension `[d, D[i], D[i+1]]` with `d` the physical
    dimension at each site and `D` the list of virtual bond dimensions.
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
            return cls.from_tensors(Alist=[left[:, :, 0:0]])
        if len(plist) == 2:
            return cls.from_tensors(Alist=[left, right])
        Alist = [left]
        for i in range(1, len(plist) - 1):
            tensor = np.zeros((2, bond_dim, bond_dim))
            tensor[:, 0, 0] = np.array([(1. - plist[i])**.5, (plist[i])**.5])
            Alist.append(tensor)
        Alist.append(right)
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

    def as_vector(self):
        """Merge all tensors to obtain the vector representation on the full Hilbert space."""
        psi = self.A[0]
        for i in range(1, len(self.A)):
            psi = self.merge_mps_tensor_pair(psi, self.A[i])
        # contract leftmost and rightmost virtual bond (has no influence if these virtual bond dimensions are 1)
        psi = np.trace(psi, axis1=1, axis2=2)
        return psi
