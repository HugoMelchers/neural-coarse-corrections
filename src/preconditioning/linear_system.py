import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import scipy
from timeit import default_timer as timer

from scipy.sparse.linalg import aslinearoperator
from pyamg.krylov import fgmres


class LinearSystem:
    def __init__(
        self,
        A,
        b,
        *,
        P_left=None,
        P_right=None,
        P_inner=None,
        x0=None,
        solver="gmres",
        t_prec=0.0,
        kwargs={},
        callback=None,
    ):
        """
        Create an object corresponding to a linear system Ax = b with specific choices for the
        initial guess, preconditioner, tolerance, and other solver settings. The system that will be
        solved depends on which preconditioners are given:

        - A x = b            if P_left and P_right are None
        - Pl A x = Pl b      if P_left is not None and P_right = None
        - A Pr x b           if P_left is None and P_right is not None
        - Pl A Pr x = Pl b   if P_left and P_right are not None

        Note that apart from Pl and Pr, it is also possible to give P_inner, which is passed
        to the SciPy linear solver that is used. SciPy solvers like GMRES and BiCGSTAB use left
        preconditioning, which can also be represented by setting Pl, but CG uses a symmetric
        preconditioning that can't be represented by Pl and Pr. As such P_inner may be needed.

        Arguments:
        - A: the matrix of the linear system of the equations. This can also be a `LinearOperator`
        - b: the right-hand side vector of the linear system
        - P_left: an optional left preconditioner
        - P_right: an optional right preconditioner
        - P_inner: an optional preconditioner passed to the scipy linear solver
        - x0: an optional initial guess for the solution
        - solver: which solver to use. Right now only "cg", "gmres", and "bicgstab" are supported
        - t_prec: if relevant, the time it took to construct the preconditioner(s), which doesn't
          affect the linear solve but can be useful in computing effective speed-ups
        - kwargs: keyword arguments passed to the solver, such as "maxiter", "rtol", and "atol". See
          the SciPy documentation for an overview of arguments.
        - callback: an optional callback to call after each iteration of the solver. Note that this
          callback is passed to the solver, and therefore the behaviour depends on the solver. For
          example, by default GMRES will call the callback with the residual of the current iterate
          as argument, whereas CG and BiCGSTAB provide the current solution estimate.
        """
        self.A = A
        self.b = b
        self.Pl = P_left
        self.Pr = P_right
        self.P_inner = P_inner
        self.x0 = x0
        self.solver = solver
        self.t_prec = t_prec
        self.kwargs = dict(kwargs)
        self.user_callback = callback
        self.norm_b = np.linalg.norm(b)

    def solve(self):
        if self.x0 is not None:
            r0 = np.linalg.norm(self.A @ self.x0 - self.b) / self.norm_b
            self.norms = [r0]
        else:
            self.norms = [1.0]
        kwargs = self.kwargs
        if self.solver == "gmres":
            solve_algo = scipy.sparse.linalg.gmres
            kwargs = dict(kwargs, callback_type="legacy")
        elif self.solver == "gmresinf":
            solve_algo = scipy.spares.linalg.gmres
            kwargs = dict(kwargs, callback_type="legacy", restart=self.b.shape[0])
        elif self.solver == "fgmres":
            solve_algo = fgmres
            # PyAMG's F-GMRES only has a "tol" argument which measures relative tolerance
            # Also, it counts "maxiter" as the number of outer iterations, which differs from the
            # number of inner itations by a factor "restart"
            kwargs = dict(kwargs)
            kwargs["tol"] = max(kwargs["rtol"], kwargs["atol"] / self.norm_b)
            del kwargs["rtol"]
            del kwargs["atol"]
            kwargs["restart"] = 20
            kwargs["maxiter"] = (kwargs["maxiter"] - 1) // kwargs["restart"] + 1
        elif self.solver == "fgmresinf":
            solve_algo = fgmres
            kwargs = dict(kwargs)
            kwargs["tol"] = max(kwargs["rtol"], kwargs["atol"] / self.norm_b)
            del kwargs["rtol"]
            del kwargs["atol"]
            kwargs["restart"] = None
            kwargs["maxiter"] = self.b.shape[0]
        elif self.solver == "bicgstab":
            solve_algo = scipy.sparse.linalg.bicgstab
        elif self.solver == "cg":
            solve_algo = scipy.sparse.linalg.cg

        # Construct preconditioned system, including left and right preconditioners if present
        # Depending on which preconditioners exist, the preconditioned system is either
        # 1. A x = b
        # 2. Pl A x = Pl b
        # 3. A Pr x = b
        # 4. Pl A Pr x = b
        A_prec = aslinearoperator(self.A)
        rhs = self.b
        if self.Pl is not None:
            A_prec = self.Pl @ A_prec
            rhs = self.Pl @ rhs
        if self.Pr is not None:
            A_prec = A_prec @ self.Pr

        t0 = timer()
        sol, info = solve_algo(
            A_prec,
            rhs,
            M=self.P_inner,
            x0=self.x0,
            callback=self.callback,
            **kwargs,
        )
        t1 = timer()

        self.sol = self.Pr @ sol if self.Pr is not None else sol
        self.info = info
        self.log = np.array(self.norms)
        self.t = t1 - t0

    def callback(self, update):
        if self.user_callback is not None:
            self.user_callback(update)
        if self.solver == "gmres" or self.solver == "gmresinf":
            # GMRES passes the relative residual norm to the callback, i.e. |Ax - b| / |b|
            self.norms.append(update)
        elif (
            self.solver == "bicgstab"
            or self.solver == "cg"
            or self.solver == "fgmres"
            or self.solver == "fgmresinf"
        ):
            # BiCGStab passes the iterate, so we compute the residual ourselves
            # Note that this is the residual without the left preconditioner Pl
            if self.Pr is not None:
                update = self.Pr @ update

            res = np.linalg.norm(self.A @ update - self.b)
            self.norms.append(res / self.norm_b)
