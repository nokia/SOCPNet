"""
© 2026 Nokia
Licensed under the BSD 3-Clause License
SPDX-License-Identifier: BSD-3-Clause

Generate SOCP instances and record their IPOPT interior-point trajectories.

Two benchmarks, one per subcommand:

  unstructured  randomly structured instances following Li et al. (2025). A
          secret point x_s is drawn first and every constraint is built around
          it, so the instance is strictly feasible by construction and IPOPT
          solves essentially all of them.

  family  the three structured families of the generalization study
          (robust_ellipsoid, block_angular, portfolio_risk). These are *not*
          feasible by construction: the conic right-hand side gets a signed
          margin, so candidates are solved and only successful solves are kept
          until --n instances have been collected.

Problem form:

    min  f^T x
    s.t. ||A_k x + b_k||_2 <= c_k^T x + d_k,   k = 1..m
         F x <= g
         l <= x <= r

Sizes (paper Section 4):  small n=50, m=10, b=10   medium n=100, m=50, b=50

Output is a pickle of problem dicts, each carrying the IPOPT `trajectory`, the
optimum `solution`, and solver metadata. Feed it to preprocess.py.

Examples:
    python generate_data.py unstructured --scale small --n 5000 \
        --out outputs/data/small_unstructured.pkl
    python generate_data.py family --family portfolio_risk --scale small \
        --n 5000 --out outputs/data/small_portfolio_risk.pkl

Both subcommands checkpoint every 100 instances and accept --resume.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from solver import (  # noqa: E402
    _atomic_pickle_dump as atomic_pickle_dump,
    solve_in_subprocess,
    validate_socp_constraints,
)

# Feasibility tolerance and per-instance IPOPT solve timeout (seconds).
FEASIBILITY_TOL = 1e-4
SOLVE_TIMEOUT_S = 600


def check_solver_available() -> None:
    """Fail early with an actionable message if the IPOPT stack is missing."""
    missing = []
    try:
        import cyipopt  # noqa: F401
    except ImportError:
        missing.append("cyipopt")
    try:
        import jax  # noqa: F401
    except ImportError:
        missing.append("jax")
    if missing:
        raise SystemExit(
            "ERROR: data generation needs the IPOPT stack, missing: "
            + ", ".join(missing)
            + "\nInstall with:\n"
            "  conda install -c conda-forge cyipopt   # brings the IPOPT library\n"
            "  pip install 'jax[cpu]'\n"
            "See README.md section 'Installing IPOPT'."
        )

# n = variables, m = conic constraints, b = rows of F (linear constraints).
SCALES = {
    "small": dict(n=50, m=10, b=10),
    "medium": dict(n=100, m=50, b=50),
}

# Cone block dimensions n_k are drawn uniformly from this inclusive range.
M_I_MIN, M_I_MAX = 1, 7

# IPOPT status codes that count as a successful solve.
SUCCESS_STATUSES = {0, 1}

CHECKPOINT_EVERY = 100

# Family sampling: signed margin on the conic right-hand side, slack on linear rows.
MARGIN_SOC_STD = 0.35
MARGIN_LIN_STD = 0.4
LIN_SLACK_BIAS = 0.08
F_NONZERO_PROB = 0.5
F_ENTRY_STD = 0.1


# ---------------------------------------------------------------------------
# Packing helpers
# ---------------------------------------------------------------------------

def _pack_coo_from_blocks(A_list):
    """Stack per-cone A blocks into one sparse (COO) matrix."""
    n = A_list[0].shape[1]
    K_tot = sum(A.shape[0] for A in A_list)
    idx_list, val_list = [], []
    row_offset = 0
    for A_i in A_list:
        rows, cols = np.nonzero(A_i)
        if rows.size == 0:
            raise ValueError(
                "Empty A block; generators must leave >=1 nonzero per cone block."
            )
        idx_list.append(np.stack([rows + row_offset, cols]))
        val_list.append(A_i[rows, cols])
        row_offset += A_i.shape[0]
    return np.concatenate(idx_list, axis=1), np.concatenate(val_list), (K_tot, n)


def pack_problem_dict(seed, problem_family, f_vec, x_ref, l_bound, r_bound,
                      F, g, A_list, b_list, c_list, d_list):
    A_indices, A_values, A_shape = _pack_coo_from_blocks(A_list)
    norm2cons = np.concatenate(
        [np.full(A_i.shape[0], i, dtype=int) for i, A_i in enumerate(A_list)]
    )
    return {
        "seed": seed,
        "n_dim": f_vec.shape[0],
        "n_cons": len(A_list),
        "n_linear": F.shape[0],
        "f_vec": f_vec,
        "A_indices": A_indices,
        "A_values": A_values,
        "A_shape": A_shape,
        "b_vec": np.concatenate(b_list),
        "c_mat": np.asarray(c_list),
        "d_vec": np.asarray(d_list, dtype=float),
        "norm2cons": norm2cons,
        "l_bound": l_bound,
        "r_bound": r_bound,
        "F": F,
        "g": g,
        "problem_family": problem_family,
        "x_ref": x_ref,
    }


# ---------------------------------------------------------------------------
# Unstructured benchmark
# ---------------------------------------------------------------------------

def generate_unstructured(seed, n, m, b):
    """Sample one SOCP that is strictly feasible at a secret point x_s.

    Step I   secret point and objective
    Step II  box constraints around x_s
    Step III linear constraints F x <= g, slack at x_s
    Step IV  conic constraints, slack epsilon ~ U(0.5, 1.0) at x_s
    """
    rng = np.random.RandomState(seed)

    # Step I: x_s ~ N(0, I), objective f ~ N(0, 0.25 I)
    x_s = rng.randn(n)
    e = rng.randn(n) * 0.5  # std = sqrt(0.25)

    # Step II: bounds strictly straddling x_s
    delta_1 = rng.randn(n) * 0.5
    delta_2 = rng.randn(n) * 0.5
    l = x_s - np.abs(delta_1) - 0.1
    r = x_s + np.abs(delta_2) + 0.1

    # Step III: F entries ~ N(0, 0.01), each nonzero with probability 0.5
    std_F = np.sqrt(0.01)
    F = rng.randn(b, n) * std_F * (rng.rand(b, n) < 0.5)
    delta_3 = rng.randn(b) * 0.5
    g = F @ x_s + np.abs(delta_3) + 0.1

    # Step IV: conic constraints, entries ~ N(0, 0.0025), sparsified at 0.5
    A_list, b_list, c_list, d_list = [], [], [], []
    for _ in range(m):
        cone_dim = rng.randint(M_I_MIN, M_I_MAX + 1)
        std = np.sqrt(0.0025)

        A_i = rng.randn(cone_dim, n) * std * (rng.rand(cone_dim, n) < 0.5)
        b_i = rng.randn(cone_dim) * std * (rng.rand(cone_dim) < 0.5)
        c_i = rng.randn(n) * std * (rng.rand(n) < 0.5)

        epsilon = rng.uniform(0.5, 1.0)
        d_i = np.linalg.norm(A_i @ x_s + b_i, 2) - c_i.T @ x_s + epsilon

        A_list.append(A_i)
        b_list.append(b_i)
        c_list.append(c_i)
        d_list.append(d_i)

    problem = pack_problem_dict(seed, "unstructured", e, x_s, l, r, F, g,
                                A_list, b_list, c_list, d_list)
    problem["x_secret"] = x_s
    return problem


# ---------------------------------------------------------------------------
# Structured families
# ---------------------------------------------------------------------------

def sample_bounds_around_ref(rng, x_ref, pad_min=0.15):
    delta_1 = rng.randn(*x_ref.shape) * 0.5
    delta_2 = rng.randn(*x_ref.shape) * 0.5
    return x_ref - np.abs(delta_1) - pad_min, x_ref + np.abs(delta_2) + pad_min


def random_linear_inequalities(rng, n, b, x_ref):
    F = rng.randn(b, n) * F_ENTRY_STD * (rng.rand(b, n) < F_NONZERO_PROB)
    zeta = np.abs(rng.randn(b) * MARGIN_LIN_STD) + LIN_SLACK_BIAS + 0.05 * rng.randn(b)
    return F, F @ x_ref + zeta


def soc_rhs_from_ref(A, b, c, x_ref, rng):
    """Right-hand side d with a SIGNED margin: the cone may be violated at x_ref."""
    lhs_norm = np.linalg.norm(A @ x_ref + b)
    return float(lhs_norm - c @ x_ref + rng.randn() * MARGIN_SOC_STD)


def _cone_rows(rng):
    return int(rng.randint(M_I_MIN, M_I_MAX + 1))


def generate_robust_ellipsoid(seed, n, m, b, rng):
    """Robust LP with ellipsoidal uncertainty: each cone comes from P^T x."""
    x_ref = rng.randn(n)
    f_vec = rng.randn(n) * 0.5
    l_bound, r_bound = sample_bounds_around_ref(rng, x_ref)
    F, g = random_linear_inequalities(rng, n, b, x_ref)
    A_list, b_list, c_list, d_list = [], [], [], []
    std_p = 0.09
    for _ in range(m):
        k = _cone_rows(rng)
        P = rng.randn(n, k) * std_p * (rng.rand(n, k) < 0.45)
        A = P.T
        if not np.any(A):
            A[rng.randint(max(k, 1)), rng.randint(n)] = std_p
        b_i = rng.randn(k) * std_p * (rng.rand(k) < 0.5)
        a = rng.randn(n) * std_p * (rng.rand(n) < 0.55)
        c, d = -a, soc_rhs_from_ref(A, b_i, -a, x_ref, rng)
        A_list.append(A); b_list.append(b_i); c_list.append(c); d_list.append(d)
    return pack_problem_dict(seed, "robust_ellipsoid", f_vec, x_ref, l_bound,
                             r_bound, F, g, A_list, b_list, c_list, d_list)


def _random_partition(rng, n, n_blocks):
    perm = rng.permutation(n)
    cuts = np.linspace(0, n, n_blocks + 1).astype(int)
    return [perm[cuts[i]: cuts[i + 1]].tolist() for i in range(n_blocks)]


def generate_block_angular(seed, n, m, b, rng):
    """Block-separable cones over a random partition, coupled by the linear rows."""
    x_ref = rng.randn(n)
    f_vec = rng.randn(n) * 0.5
    l_bound, r_bound = sample_bounds_around_ref(rng, x_ref)
    F, g = random_linear_inequalities(rng, n, b, x_ref)
    n_blocks = int(np.clip(rng.randint(4, 11), 2, max(2, n // max(M_I_MIN, 1))))
    blocks = [bl for bl in _random_partition(rng, n, n_blocks)
              if len(bl) >= M_I_MIN] or [list(range(n))]
    A_list, b_list, c_list, d_list = [], [], [], []
    std = 0.09
    for _ in range(m):
        bl = blocks[rng.randint(len(blocks))]
        k_max = min(M_I_MAX, len(bl))
        k_min = min(M_I_MIN, len(bl))
        if k_min > k_max:
            bl, k_max, k_min = list(range(n)), min(M_I_MAX, n), M_I_MIN
        k = rng.randint(k_min, k_max + 1)
        subcols = bl if k >= len(bl) else list(rng.choice(bl, size=k, replace=False))
        k = len(subcols)
        A = np.zeros((k, n))
        A[:, subcols] = rng.randn(k, k) * std * (rng.rand(k, k) < 0.65)
        if not np.any(A):
            A[0, subcols[0]] = std
        b_i = rng.randn(k) * std * (rng.rand(k) < 0.55)
        c = rng.randn(n) * std * (rng.rand(n) < 0.4)
        d = soc_rhs_from_ref(A, b_i, c, x_ref, rng)
        A_list.append(A); b_list.append(b_i); c_list.append(c); d_list.append(d)
    return pack_problem_dict(seed, "block_angular", f_vec, x_ref, l_bound,
                             r_bound, F, g, A_list, b_list, c_list, d_list)


def generate_portfolio_risk(seed, n, m, b, rng):
    """Mean-variance risk cones from a Cholesky factor, plus a budget row in F."""
    x_ref = rng.randn(n) * 0.45
    f_vec = -np.abs(rng.randn(n)) * 0.4 - 0.05 * rng.randn(n)
    l_bound, r_bound = sample_bounds_around_ref(rng, x_ref, pad_min=0.2)
    M = rng.randn(n, n) * 0.14
    Sigma = M @ M.T + 0.06 * np.eye(n)
    A_list, b_list, c_list, d_list = [], [], [], []
    for _ in range(m):
        k = min(_cone_rows(rng), n)
        idx = np.sort(rng.choice(n, size=k, replace=False))
        Sig_sub = Sigma[np.ix_(idx, idx)] + 1e-4 * np.eye(len(idx))
        R = np.linalg.cholesky(Sig_sub)
        A = np.zeros((k, n))
        A[:, idx] = R
        b_i = rng.randn(k) * 0.025
        c = np.zeros(n)
        d = soc_rhs_from_ref(A, b_i, c, x_ref, rng)
        A_list.append(A); b_list.append(b_i); c_list.append(c); d_list.append(d)
    F, g = random_linear_inequalities(rng, n, b, x_ref)
    # Row 0 of F is a budget constraint with positive weights summing to one.
    w = np.abs(rng.randn(n)) + 0.1
    w /= w.sum()
    F[0, :] = w
    g[0] = float(w @ x_ref + rng.randn() * MARGIN_LIN_STD + LIN_SLACK_BIAS)
    return pack_problem_dict(seed, "portfolio_risk", f_vec, x_ref, l_bound,
                             r_bound, F, g, A_list, b_list, c_list, d_list)


FAMILIES = {
    "robust_ellipsoid": generate_robust_ellipsoid,
    "block_angular": generate_block_angular,
    "portfolio_risk": generate_portfolio_risk,
}


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------

def solve_one(problem, timeout):
    """Solve one instance; return it extended with trajectory and solver fields."""
    t0 = time.time()
    z, info, trace = solve_in_subprocess(
        problem["f_vec"],
        problem["A_indices"],
        problem["A_values"],
        problem["A_shape"],
        problem["b_vec"],
        problem["c_mat"],
        problem["d_vec"],
        problem["norm2cons"],
        l_bound=problem.get("l_bound"),
        r_bound=problem.get("r_bound"),
        F=problem.get("F"),
        g=problem.get("g"),
        timeout=timeout,
    )
    elapsed = time.time() - t0

    status = info.get("status", -1)
    feasible = False
    if status in SUCCESS_STATUSES and z is not None:
        feasible, _ = validate_socp_constraints(
            z,
            problem["A_indices"],
            problem["A_values"],
            problem["A_shape"],
            problem["b_vec"],
            problem["c_mat"],
            problem["d_vec"],
            problem["norm2cons"],
            tol=FEASIBILITY_TOL,
            F=problem.get("F"),
            g=problem.get("g"),
        )

    out = dict(problem)
    out.update(
        trajectory=trace,
        solution=z,
        feasibility="feasible" if feasible else "infeasible",
        solve_time=elapsed,
        n_iterations=len(trace),
        solver_status=status,
        solver_status_msg=info.get("status_msg", ""),
    )
    return out, feasible


def load_resume(out_path, resume):
    if resume and out_path.is_file():
        with open(out_path, "rb") as f:
            done = pickle.load(f)
        print(f"Resuming: {len(done)} instances already in {out_path}")
        return done
    return []


def run_unstructured(args):
    cfg = SCALES[args.scale]
    n, m, b = cfg["n"], cfg["m"], cfg["b"]
    print(f"Generating {args.n} {args.scale} unstructured instances (n={n}, m={m}, b={b})")

    solved = load_resume(args.out, args.resume)
    t_start = time.time()

    for i in range(len(solved), args.n):
        problem = generate_unstructured(args.seed + i, n, m, b)
        out, _ = solve_one(problem, args.timeout)
        solved.append(out)
        if len(solved) % CHECKPOINT_EVERY == 0:
            atomic_pickle_dump(solved, args.out)
            print(f"  {len(solved)}/{args.n} solved "
                  f"({time.time() - t_start:.0f}s)", flush=True)

    atomic_pickle_dump(solved, args.out)
    n_feasible = sum(1 for p in solved if p["feasibility"] == "feasible")
    print(f"Saved {len(solved)} instances to {args.out}")
    print(f"  feasible: {n_feasible} "
          f"({100 * n_feasible / max(len(solved), 1):.1f}%)")
    if n_feasible == 0:
        raise SystemExit(
            "ERROR: no instance solved successfully. Check the IPOPT installation."
        )


def run_family(args):
    cfg = SCALES[args.scale]
    n, m, b = cfg["n"], cfg["m"], cfg["b"]
    print(f"Collecting {args.n} feasible '{args.family}' instances "
          f"({args.scale}: n={n}, m={m}, b={b})")

    collected = load_resume(args.out, args.resume)
    rejected = 0
    t_start = time.time()

    for attempt in range(args.max_attempts):
        if len(collected) >= args.n:
            break
        seed = args.seed + attempt
        rng = np.random.RandomState(seed)
        try:
            problem = FAMILIES[args.family](seed, n, m, b, rng)
        except (ValueError, np.linalg.LinAlgError):
            rejected += 1
            continue

        out, feasible = solve_one(problem, args.timeout)
        if not feasible:
            rejected += 1
            continue
        collected.append(out)

        if len(collected) % CHECKPOINT_EVERY == 0:
            atomic_pickle_dump(collected, args.out)
            rate = len(collected) / max(attempt + 1, 1)
            print(f"  {len(collected)}/{args.n} feasible "
                  f"({rate:.1%} accept, {time.time() - t_start:.0f}s)", flush=True)

    atomic_pickle_dump(collected, args.out)
    print(f"Saved {len(collected)} feasible instances to {args.out}")
    print(f"  rejected: {rejected}")
    if not collected:
        raise SystemExit("ERROR: no feasible instance collected. Check IPOPT setup.")
    if len(collected) < args.n:
        print(f"WARNING: only {len(collected)}/{args.n} collected within "
              f"--max-attempts {args.max_attempts}.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--scale", choices=sorted(SCALES), required=True)
        p.add_argument("--out", type=Path, required=True, help="Output .pkl")
        p.add_argument("--n", type=int, default=5000, help="Instances to produce")
        p.add_argument("--seed", type=int, default=0, help="Base seed")
        p.add_argument("--timeout", type=int, default=SOLVE_TIMEOUT_S,
                       help="Per-instance solve timeout in seconds")
        p.add_argument("--resume", action="store_true",
                       help="Continue a previous run from --out")

    p_unstructured = sub.add_parser(
        "unstructured", help="Randomly structured benchmark")
    common(p_unstructured)

    p_family = sub.add_parser("family", help="One structured problem family")
    common(p_family)
    p_family.add_argument("--family", choices=sorted(FAMILIES), required=True)
    p_family.add_argument("--max-attempts", type=int, default=200000)

    args = ap.parse_args()

    check_solver_available()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.command == "unstructured":
        run_unstructured(args)
    else:
        run_family(args)
    print("Next: preprocess.py")


if __name__ == "__main__":
    main()
