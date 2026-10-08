"""
© 2026 Nokia
Licensed under the BSD 3-Clause License
SPDX-License-Identifier: BSD-3-Clause

Solve SOCP problems with IPOPT and record the iteration path.

JAX supplies the objective and constraint derivatives. ``solve_socp`` is the
entry point used by ``solver.py``.
"""

from jax import config
from jax import jit, grad, jacfwd
import jax.numpy as jnp
import numpy as np
import cyipopt

# -------- Jax settings ------
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")


class QCQPSolver(cyipopt.Problem):
    """
    Solve QCQP problems using IPOPT.

    Problem format:
        minimize    (1/2) x^T Q x + p^T x
        subject to  (1/2) x^T Qi[i] x + pi[i]^T x + b[i] <= 0  for i = 0,...,n_cons-1
    """

    def __init__(self, Q, p, Qi, pi, b, max_iter, **kwargs):
        """
        Args:
            Q: Objective quadratic matrix [n, n]
            p: Objective linear vector [n]
            Qi: Constraint quadratic matrices [n_cons, n, n]
            pi: Constraint linear vectors [n_cons, n]
            b: Constraint RHS values [n_cons]
            max_iter: Maximum number of iterations (required)
        """
        self.Q = jnp.array(Q)
        self.p = jnp.array(p)
        self.Qi = jnp.array(Qi)
        self.pi = jnp.array(pi)
        self.b = jnp.array(b)

        # Trace of iterations
        self.trace = []
        self.last_recorded_iter = (
            -1
        )  # Track last recorded iteration to avoid duplicates
        if max_iter is None:
            raise ValueError("max_iter is required and cannot be None")
        self.max_iter = max_iter

        self.obj_jit = jit(self._objective)
        self.con_jit = jit(self._constraints)

        self.obj_grad = jit(grad(self.obj_jit))
        self.con_jac = jit(jacfwd(self.con_jit))

        super().__init__(**kwargs)

    def _objective(self, x):
        """Objective function: (1/2) x^T Q x + p^T x"""
        return 0.5 * jnp.dot(x, self.Q @ x) + jnp.dot(self.p, x)

    def objective(self, x):
        return float(self.obj_jit(x))

    def gradient(self, x):
        return np.array(self.obj_grad(x))

    def _constraints(self, x):
        """Constraint functions: (1/2) x^T Qi[i] x + pi[i]^T x + b[i]"""
        constraints = []
        for i in range(len(self.b)):
            c_i = (
                0.5 * jnp.dot(x, self.Qi[i] @ x)
                + jnp.dot(self.pi[i], x)
                + self.b[i]
            )
            constraints.append(c_i)
        return jnp.array(constraints)

    def constraints(self, x):
        return np.array(self.con_jit(x))

    def jacobian(self, x):
        """Jacobian of constraints"""
        return np.array(self.con_jac(x)).flatten()

    def intermediate(
        self,
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
    ):
        """
        Keep one point per iteration: the last accepted point of that iteration.
        IPOPT may report iteration 0. Recording stops once iter_count exceeds
        max_iter, so the trace can hold iterations 0 through max_iter.
        """
        iterate = self.get_current_iterate()
        current_entry = {
            "x": np.array(iterate["x"]),
            "obj": obj_value,
            "iter": iter_count,
        }

        # Drop iterations past max_iter. Iteration 0 is kept when IPOPT reports it.
        if iter_count > self.max_iter:
            return  # Don't record beyond max_iter

        # If this is a new iteration, append a new entry
        if iter_count > self.last_recorded_iter:
            self.trace.append(current_entry)
            self.last_recorded_iter = iter_count
        else:
            # Same iteration - update the last entry (overwrite with final state after line search)
            # This ensures we keep the final accepted point for this iteration
            if len(self.trace) > 0:
                self.trace[-1] = current_entry


class SOCPSolver(cyipopt.Problem):
    """
    Solve SOCP problems using IPOPT.

    Problem format:
        minimize    f^T z
        subject to  ||A_k z + b_k||_2 <= c_k^T z + d_k  for k = 0,...,n_cons
                    F @ z <= g  (linear constraints, optional)

    z is the decision variable.

    We reformulate the second-order cone constraints as:
        ||A_k z + b_k||_2 - (c_k^T z + d_k) <= 0
    """

    def __init__(
        self,
        f_vec,
        A_indices,
        A_values,
        A_shape,
        b_vec,
        c_mat,
        d_vec,
        norm2cons,
        max_iter,
        F=None,
        g=None,
        **kwargs,
    ):
        """
        Args:
            f_vec: Variable objective coefficients [n_vars]
            A_indices: Sparse matrix indices [2, E] (COO format)
            A_values: Sparse matrix values [E]
            A_shape: Shape of A (K_tot, n_vars)
            b_vec: Norm constraint LHS values [K_tot]
            c_mat: RHS linear coefficients [n_cons, n_vars]
            d_vec: RHS constant terms [n_cons]
            norm2cons: Mapping from norms to constraints [K_tot]
            max_iter: Maximum number of iterations (required)
            F: Linear constraint matrix [n_linear, n_vars] (optional)
            g: Linear constraint RHS vector [n_linear] (optional)
        """
        self.f_vec = jnp.array(f_vec)
        self.b_vec = jnp.array(b_vec)
        self.c_mat = jnp.array(c_mat)
        self.d_vec = jnp.array(d_vec)
        self.norm2cons = np.array(norm2cons)
        
        # Linear constraints: F @ z <= g
        if F is not None and g is not None:
            self.F = jnp.array(F)
            self.g = jnp.array(g)
            self.n_linear = len(g)
        else:
            self.F = None
            self.g = None
            self.n_linear = 0

        # Build dense A matrix from sparse representation
        K_tot, n_vars = A_shape
        self.A_full = jnp.zeros(A_shape)
        for i in range(A_indices.shape[1]):
            row_idx = A_indices[0, i]
            col_idx = A_indices[1, i]
            self.A_full = self.A_full.at[row_idx, col_idx].set(A_values[i])

        # Number of SOCP constraints
        self.n_socp_cons = len(d_vec)
        self.n_vars = n_vars
        self.K_tot = K_tot

        # Trace of iterations
        self.trace = []
        self.last_recorded_iter = (
            -1
        )  # Track last recorded iteration to avoid duplicates
        if max_iter is None:
            raise ValueError("max_iter is required and cannot be None")
        self.max_iter = max_iter

        self.obj_jit = jit(self._objective)
        self.con_jit = jit(self._constraints)

        self.obj_grad = jit(grad(self.obj_jit))
        self.con_jac = jit(jacfwd(self.con_jit))

        super().__init__(**kwargs)

    def _objective(self, z):
        """Objective: f^T z"""
        return jnp.dot(self.f_vec, z)

    def objective(self, z):
        return float(self.obj_jit(z))

    def gradient(self, z):
        return np.array(self.obj_grad(z))

    def _constraints(self, z):
        """
        SOCP constraints reformulated as:
        ||A_k z + b_k||_2 - (c_k^T z + d_k) <= 0
        
        Plus linear constraints:
        F @ z - g <= 0  (i.e., F @ z <= g)

        This avoids squaring which can introduce sign issues.
        """
        constraints = []

        # SOCP constraints
        for k in range(self.n_socp_cons):
            # Find rows belonging to constraint k
            mask = self.norm2cons == k
            A_k = self.A_full[mask, :]  # [n_k, n_vars]
            b_k = self.b_vec[mask]  # [n_k]
            c_k = self.c_mat[k, :]  # [n_vars]
            d_k = self.d_vec[k]  # scalar

            # Compute ||A_k z + b_k||_2
            Az_plus_b = A_k @ z + b_k
            # Clamp to avoid negative values due to numerical precision issues
            norm_val = jnp.sqrt(jnp.maximum(jnp.sum(Az_plus_b**2), 0.0))

            # Compute c_k^T z + d_k
            linear_term = jnp.dot(c_k, z) + d_k

            # Constraint: norm_val - linear_term <= 0
            constraints.append(norm_val - linear_term)

        # Linear constraints: F @ z <= g  (reformulated as F @ z - g <= 0)
        if self.F is not None and self.g is not None:
            linear_cons = self.F @ z - self.g  # [n_linear]
            constraints.extend(linear_cons)

        return jnp.array(constraints)

    def constraints(self, z):
        return np.array(self.con_jit(z))

    def jacobian(self, z):
        """Jacobian of constraints"""
        return np.array(self.con_jac(z)).flatten()

    def intermediate(
        self,
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
    ):
        """
        Keep one point per iteration: the last accepted point of that iteration.
        IPOPT may report iteration 0. Recording stops once iter_count exceeds
        max_iter, so the trace can hold iterations 0 through max_iter.
        """
        iterate = self.get_current_iterate()
        current_entry = {
            "x": np.array(iterate["x"]),
            "obj": obj_value,
            "iter": iter_count,
        }

        # Drop iterations past max_iter. Iteration 0 is kept when IPOPT reports it.
        if iter_count > self.max_iter:
            return  # Don't record beyond max_iter

        # If this is a new iteration, append a new entry
        if iter_count > self.last_recorded_iter:
            self.trace.append(current_entry)
            self.last_recorded_iter = iter_count
        else:
            # Same iteration - update the last entry (overwrite with final state after line search)
            # This ensures we keep the final accepted point for this iteration
            if len(self.trace) > 0:
                self.trace[-1] = current_entry


def solve_qcqp(Q, p, Qi, pi, b, x0=None, lb=None, ub=None, print_level=0):
    """
    Solve a QCQP problem.

    Args:
        Q, p, Qi, pi, b: QCQP problem data
        x0: Initial guess (if None, uses zeros)
        lb: Lower bounds on variables (if None, unbounded)
        ub: Upper bounds on variables (if None, unbounded)
        print_level: IPOPT verbosity (0=silent, 5=very verbose)

    Returns:
        Tuple of (solution, info, trace)
    """
    n = Q.shape[0]
    n_cons = len(b)

    if x0 is None:
        x0 = np.zeros(n)

    # Variable bounds
    if lb is None:
        lb = [-np.inf] * n
    if ub is None:
        ub = [np.inf] * n

    # Constraint bounds: g(x) <= 0
    cl = [-np.inf] * n_cons
    cu = [0.0] * n_cons

    max_iter = 50
    problem = QCQPSolver(
        Q,
        p,
        Qi,
        pi,
        b,
        max_iter=max_iter,
        n=n,
        m=n_cons,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )
    problem.add_option("print_level", print_level)
    problem.add_option("tol", 1e-7)
    problem.add_option("max_iter", max_iter)
    problem.add_option("hessian_approximation", "limited-memory")
    problem.add_option("limited_memory_max_history", 50)

    x, info = problem.solve(x0)

    return x, info, problem.trace


def solve_socp(
    f_vec,
    A_indices,
    A_values,
    A_shape,
    b_vec,
    c_mat,
    d_vec,
    norm2cons,
    z0=None,
    l_bound=None,
    r_bound=None,
    F=None,
    g=None,
    print_level=0,
):
    """
    Solve an SOCP problem.

    Args:
        f_vec, A_indices, A_values, A_shape, b_vec, c_mat, d_vec, norm2cons: SOCP problem data
        z0: Initial guess (if None, uses zeros)
        l_bound: Lower bounds on variables (if None, uses -inf)
        r_bound: Upper bounds on variables (if None, uses +inf)
        F: Linear constraint matrix [n_linear, n_vars] (optional)
        g: Linear constraint RHS vector [n_linear] (optional)
        print_level: IPOPT verbosity (0=silent, 5=very verbose)

    Returns:
        Tuple of (solution, info, trace)
    """
    n_vars = len(f_vec)
    n_cons = len(d_vec)
    
    # Count total constraints (SOCP + linear)
    n_linear = len(g) if g is not None else 0
    total_cons = n_cons + n_linear

    # Variable bounds
    if l_bound is not None:
        lb = list(l_bound)
    else:
        lb = [-np.inf] * n_vars
    
    if r_bound is not None:
        ub = list(r_bound)
    else:
        ub = [np.inf] * n_vars

    # Initial guess: use midpoint of bounds if available, otherwise zeros
    if z0 is None:
        if l_bound is not None and r_bound is not None:
            z0 = (np.array(l_bound) + np.array(r_bound)) / 2.0
        else:
            z0 = np.zeros(n_vars)

    # Ensure initial guess is within bounds
    z0 = np.clip(z0, lb, ub)

    # Constraint bounds: g(z) <= 0 (for both SOCP and linear constraints)
    cl = [-np.inf] * total_cons
    cu = [0.0] * total_cons

    max_iter = 1000
    problem = SOCPSolver(
        f_vec,
        A_indices,
        A_values,
        A_shape,
        b_vec,
        c_mat,
        d_vec,
        norm2cons,
        max_iter=max_iter,
        F=F,
        g=g,
        n=n_vars,
        m=total_cons,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )
    problem.add_option("print_level", print_level)
    problem.add_option("tol", 1e-7)
    problem.add_option("max_iter", max_iter)
    problem.add_option("hessian_approximation", "limited-memory")
    problem.add_option("limited_memory_max_history", 50)
    z, info = problem.solve(z0)

    return z, info, problem.trace
