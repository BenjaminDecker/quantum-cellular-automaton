import numpy as np
import scipy as scp


def _orthonormalize(psi: np.ndarray, vectors: list[np.ndarray]) -> np.ndarray | None:
    """
    Orthonormalize `psi` against all `vectors`
    :param psi: A vector that should be orthonormalized
    :param vectors: A list of vectors to orthonormalize against
    :return: None, if `psi` was not linearly independent to all vectors in `vectors`, otherwise an orthonormalized `psi`
    """
    new_vector = psi.copy()
    for vector in vectors:
        new_vector -= np.inner(psi.conj(), vector) * vector
    length = np.linalg.norm(new_vector)
    if np.allclose(length, 0.0):
        return None
    return new_vector / length


def _apply_orthonormalize(H: np.ndarray, krylov_vectors: list[np.ndarray]) -> np.ndarray:
    """
    Create a new krylov vector by multiplying the hamiltonian matrix `H` with the last vector from `krylov_vectors` and
    then normalizing the result against all other vectors in `krylov_vectors`
    :param H: The hamiltonian matrix
    :param krylov_vectors: A list of krylov vectors
    :return: A new krylov vector that is orthonormalized against all other vectors in `krylov_vectors`
    """
    krylov_new = H @ krylov_vectors[-1]
    return _orthonormalize(krylov_new, krylov_vectors)


def _compute_effective_H(T_old: np.ndarray, H: np.ndarray, krylov_vectors: list[np.ndarray]) -> np.ndarray:
    T_new = np.empty((T_old.shape[0] + 1, T_old.shape[1] + 1), dtype=T_old.dtype)
    T_new[:-1, :-1] = T_old
    T_new[-1, :] = krylov_vectors[-1].conj() @ H @ np.array(krylov_vectors).T
    T_new[:, -1] = np.array(krylov_vectors).conj() @ H @ krylov_vectors[-1]
    return T_new


def calculate_U(H_matrix: np.ndarray, step_size: float) -> np.ndarray:
    """
    Calculate the matrix exponential of (-𝑖π/2) * `step_size` * `H_matrix`
    :param H_matrix: A hamiltonian in form of a hermitian matrix
    :param step_size: The step size of the time evolution
    :return: A unitary matrix, describing the time evolution governed by `H_matrix`
    """
    w, v = np.linalg.eigh(H_matrix)
    t = -1j * (np.pi / 2) * step_size
    w = np.exp(t * w)
    return (w * v) @ v.conj().T


def timestep(H: np.ndarray, psi: np.ndarray, delta: float) -> np.ndarray:
    """
    Calculate the result of multiplying the matrix exponential of ((-𝑖π/2) * `delta` * `H`) with `psi` using the global
    krylov time evolution
    :param H: A hamiltonian in form of a hermitian matrix
    :param psi: The vector to be multiplied with the matrix exponential
    :param delta: The step size of the time evolution
    :return: The result of the time evolution
    """
    krylov_vectors = [psi / np.linalg.norm(psi)]
    T = np.array([[krylov_vectors[0].conj().T @ H @ krylov_vectors[0]]])
    c = scp.linalg.expm(delta * T)[:, 0]
    while True:
        new_krylov = _apply_orthonormalize(H, krylov_vectors)
        if new_krylov is None:
            break
        krylov_vectors.append(new_krylov)
        T = _compute_effective_H(T, H, krylov_vectors)
        c = calculate_U(T, delta)[:, 0]
        if np.allclose(c[-1], 0.0):
            break
    return c @ np.array(krylov_vectors)
