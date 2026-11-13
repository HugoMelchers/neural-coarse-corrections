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
    def __init__(self, *factors):
        """
        Construct the Kronecker product A₁ ⊗ A₂ ⊗ ... of two or more matrices. Vectors are
        interpreted as row-major arrays of shape (n₁, n₂, ...), i.e. the index of the last factor
        varies fastest.
        """
        assert len(factors) >= 2, "Kronecker: expected at least two factors"
        self.factors = factors
        self.shape = (
            int(np.prod([F.shape[0] for F in factors])),
            int(np.prod([F.shape[1] for F in factors])),
        )
        self.dtype = factors[0].dtype

    def _matvec(self, x):
        x = np.reshape(x, [F.shape[1] for F in self.factors])
        # Apply each factor along its own axis
        for axis, F in enumerate(self.factors):
            x = np.moveaxis(np.tensordot(F, x, axes=(1, axis)), 0, axis)
        return x.ravel()

    def _adjoint(self):
        return Kronecker(*(F.T for F in self.factors))


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
