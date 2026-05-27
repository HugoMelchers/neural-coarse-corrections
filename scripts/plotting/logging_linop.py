from scipy.sparse.linalg import LinearOperator
from timeit import default_timer as timer


class Timed(LinearOperator):
    def __init__(self, inner):
        self.inner = inner
        self.ts = []
        self.shape = inner.shape
        self.dtype = inner.dtype

    def _matvec(self, x):
        t0 = timer()
        y = self.inner._matvec(x)
        self.ts.append(timer() - t0)
        return y
