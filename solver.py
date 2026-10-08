"""
© 2026 Nokia
Licensed under the BSD 3-Clause License
SPDX-License-Identifier: BSD-3-Clause

Solve one SOCP instance in a subprocess and check feasibility.

``solve_in_subprocess`` is what ``generate_data.py`` calls. A crash in the
solver process does not kill the caller.
"""

import numpy as np
import pickle
import sys
import subprocess
import tempfile
import os
from pathlib import Path

from ipopt import solve_socp


def validate_socp_constraints(
    z, A_indices, A_values, A_shape, b_vec, c_mat, d_vec, norm2cons, tol=1e-5, F=None, g=None
):
    """
    Validate that solution z satisfies all SOCP constraints and linear constraints.

    SOCP constraints: ||A_k z + b_k||_2 <= c_k^T z + d_k  for k = 0,...,n_cons-1
    Linear constraints: F @ z <= g  (if F and g are provided)

    Args:
        z: Solution vector [n_vars]
        A_indices: Sparse matrix indices [2, E] (COO format)
        A_values: Sparse matrix values [E]
        A_shape: Shape of A (K_tot, n_vars)
        b_vec: Norm constraint LHS values [K_tot]
        c_mat: RHS linear coefficients [n_cons, n_vars]
        d_vec: RHS constant terms [n_cons]
        norm2cons: Mapping from norms to constraints [K_tot]
        tol: Tolerance for constraint violation (default: 1e-5)
        F: Linear constraint matrix [n_linear, n_vars] (optional)
        g: Linear constraint RHS vector [n_linear] (optional)

    Returns:
        (is_valid, violations): Tuple of (bool, list of violation amounts)
        violations[k] = max(0, ||A_k z + b_k||_2 - (c_k^T z + d_k)) for SOCP constraints
        violations[n_cons + i] = max(0, (F @ z)[i] - g[i]) for linear constraints
    """
    if z is None:
        return False, []

    # Build dense A matrix from sparse representation
    K_tot, n_vars = A_shape
    A_full = np.zeros(A_shape)
    for i in range(A_indices.shape[1]):
        row_idx = A_indices[0, i]
        col_idx = A_indices[1, i]
        A_full[row_idx, col_idx] = A_values[i]

    n_cons = len(d_vec)
    violations = []
    max_violation = 0.0

    # Check SOCP constraints
    for k in range(n_cons):
        # Find rows belonging to constraint k
        mask = norm2cons == k
        A_k = A_full[mask, :]  # [n_k, n_vars]
        b_k = b_vec[mask]  # [n_k]
        c_k = c_mat[k, :]  # [n_vars]
        d_k = d_vec[k]  # scalar

        # Compute ||A_k z + b_k||_2
        Az_plus_b = A_k @ z + b_k
        norm_val = np.sqrt(np.maximum(np.sum(Az_plus_b**2), 0.0))

        # Compute c_k^T z + d_k
        linear_term = np.dot(c_k, z) + d_k

        # Constraint violation: norm_val - linear_term (should be <= 0)
        violation = norm_val - linear_term
        violations.append(violation)
        max_violation = max(max_violation, violation)

    # Check linear constraints: F @ z <= g
    if F is not None and g is not None:
        linear_vals = F @ z  # [n_linear]
        for i in range(len(g)):
            violation = linear_vals[i] - g[i]  # Should be <= 0
            violations.append(violation)
            max_violation = max(max_violation, violation)

    # Solution is valid if all violations are within tolerance
    is_valid = max_violation <= tol

    return is_valid, violations


def solve_in_subprocess(
    f_vec,
    A_indices,
    A_values,
    A_shape,
    b_vec,
    c_mat,
    d_vec,
    norm2cons,
    l_bound=None,
    r_bound=None,
    F=None,
    g=None,
    timeout=600,
):
    """
    Solve one SOCP in a subprocess so a solver crash does not kill the caller.

    Returns ``(z, info, trace)``. On failure ``z`` is ``None``, ``info["status"]``
    is ``-1``, and ``info["status_msg"]`` describes the error.
    """
    # Create temporary files for input/output
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, suffix=".pkl"
    ) as tmp_in:
        pickle.dump(
            {
                "f_vec": f_vec,
                "A_indices": A_indices,
                "A_values": A_values,
                "A_shape": A_shape,
                "b_vec": b_vec,
                "c_mat": c_mat,
                "d_vec": d_vec,
                "norm2cons": norm2cons,
                "l_bound": l_bound,
                "r_bound": r_bound,
                "F": F,
                "g": g,
            },
            tmp_in,
        )
        tmp_input = tmp_in.name

    tmp_output = tmp_input.replace(".pkl", "_out.pkl")
    cwd = os.getcwd()

    # Imports live inside try so an import error is written to the result file.
    cwd_q = repr(cwd)
    tin_q = repr(tmp_input)
    tout_q = repr(tmp_output)
    solver_script = f"""
import os
import pickle
import sys
import traceback

cwd = {cwd_q}
tmp_input = {tin_q}
tmp_output = {tout_q}
os.chdir(cwd)
sys.path.insert(0, cwd)

result = {{"success": False, "error": "init"}}
try:
    import numpy as np  # noqa: F401
    from ipopt import solve_socp

    with open(tmp_input, "rb") as f:
        data = pickle.load(f)
    z, info, trace = solve_socp(
        data["f_vec"],
        data["A_indices"],
        data["A_values"],
        data["A_shape"],
        data["b_vec"],
        data["c_mat"],
        data["d_vec"],
        data["norm2cons"],
        z0=None,
        l_bound=data.get("l_bound"),
        r_bound=data.get("r_bound"),
        F=data.get("F"),
        g=data.get("g"),
        print_level=0,
    )
    result = {{"success": True, "z": z, "info": info, "trace": trace}}
except BaseException as e:
    result = {{
        "success": False,
        "error": str(e),
        "traceback": traceback.format_exc(),
    }}

with open(tmp_output, "wb") as f:
    pickle.dump(result, f)
"""

    script_file = tmp_input.replace(".pkl", "_solver.py")
    with open(script_file, "w") as f:
        f.write(solver_script)

    try:
        result = subprocess.run(
            [sys.executable, script_file],
            cwd=cwd,
            timeout=timeout,
            capture_output=True,
            text=True,
        )

        if os.path.exists(tmp_output):
            with open(tmp_output, "rb") as f:
                result_data = pickle.load(f)
            if result_data.get("success", False):
                return (
                    result_data["z"],
                    result_data["info"],
                    result_data["trace"],
                )
            else:
                err = result_data.get("error", "Unknown")
                tb = result_data.get("traceback")
                msg = f"Error: {err}"
                if tb:
                    msg += f" | {tb[-2500:]}"
                return (
                    None,
                    {
                        "status": -1,
                        "status_msg": msg,
                    },
                    [],
                )
        else:
            # No result file: the child was killed, or it died before writing one.
            rc = getattr(result, "returncode", None)
            tail = ""
            if result and result.stderr:
                tail = result.stderr.strip()
                if len(tail) > 2000:
                    tail = "... " + tail[-2000:]
            msg = f"Subprocess ended without result file (returncode={rc})"
            if tail:
                msg += f"; stderr_tail: {tail}"
            return None, {"status": -1, "status_msg": msg}, []
    except subprocess.TimeoutExpired:
        return None, {"status": -1, "status_msg": "Timeout"}, []
    except Exception as e:
        return None, {"status": -1, "status_msg": f"Exception: {str(e)}"}, []
    finally:
        for f in [tmp_input, tmp_output, script_file]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass


def _atomic_pickle_dump(obj, path: Path) -> None:
    """Write pickle atomically (crash mid-write leaves old path or missing tmp)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)
