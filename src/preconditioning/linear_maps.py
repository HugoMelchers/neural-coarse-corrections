import scipy
from scipy.sparse.linalg import LinearOperator
import numpy as np


class Zero(LinearOperator):
    def __init__(self, n, m, dtype):
        self.shape = (n, m)
        self.dtype = dtype

    def _matvec(self, x):
        return np.zeros((self.shape[0],), self.dtype)


class Identity(LinearOperator):
    def __init__(self, n, dtype):
        self.shape = (n, n)
        self.dtype = dtype

    def _matvec(self, x):
        return x


class Kronecker(LinearOperator):
    def __init__(self, A, B):
        self.A = A
        self.B = B
        m1, n1 = A.shape
        m2, n2 = B.shape
        self.shape = (m1 * m2, n1 * n2)
        self.dtype = A.dtype

    def _matvec(self, x):
        x = np.reshape(x, (self.A.shape[1], self.B.shape[1]))
        return (self.A @ x @ self.B.T).ravel()

    def _adjoint(self):
        return Kronecker(self.A.T, self.B.T)


class Chain(LinearOperator):
    def __init__(self, A, Ps):
        """
        Construct a chained preconditioner for Ax = b with preconditioners P0, P1, P2,..., that, given b, computes

        - x0 = P0 * b
        - x1 = x0 + P1 * b - A * x0
        - x2 = x1 + P2 * b - A * x1
        - ...
        """
        self.A = A
        self.Ps = Ps
        self.shape = A.shape
        self.dtype = A.dtype

    def _matvec(self, b):
        x = self.Ps[0] @ b
        for P in self.Ps[1:]:
            x = x + P @ (b - self.A @ x)

        return x
