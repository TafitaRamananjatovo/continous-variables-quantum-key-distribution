
import numpy as np
from numba import njit,jit
@njit(fastmath=True, parallel=True, cache=True)
def Omga(n):
    return np.kron(np.eye(int(n)), [[0, 1], [-1, 0]])

def symplectic_eigenvalue_calculation(V):

    if V.shape == (4, 4) and np.all(np.linalg.eigvals(V) > 0):
        A_mode = np.array([[V[0, 0], V[0, 1]], [V[1, 0], V[1, 1]]])
        B_mode = np.array([[V[2, 2], V[2, 3]], [V[3, 2], V[3, 3]]])
        C_mode = np.array([[V[0, 2], V[0, 3]], [V[1, 2], V[1, 3]]])
        Delta_V = np.linalg.det(A_mode) + np.linalg.det(B_mode) + 2 * np.linalg.det(C_mode)
        v_1 = np.sqrt((Delta_V + np.sqrt(Delta_V ** 2 - 4 * np.linalg.det(V))) / 2)
        v_2 = np.sqrt((Delta_V - np.sqrt(Delta_V ** 2 - 4 * np.linalg.det(V))) / 2)

        assert v_1 > 1
        assert v_2 > 1

        return v_1, v_2

    else:
        Om = Omga(V.shape[0]/2)
        eigsFull = np.linalg.eigvals(1j*np.dot(Om, V))
        eigsSort = np.sort(abs(eigsFull))
        symeigs = np.delete(eigsSort, [x for x in range(0, V.shape[0], 2)])
        
        assert np.all(symeigs > 1)

        return tuple(symeigs)

def h_f(v):
    return ((v + 1) / 2) * np.log2((v + 1) / 2) - ((v - 1) / 2) * np.log2((v - 1) / 2)
