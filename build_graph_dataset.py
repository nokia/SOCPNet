"""
© 2026 Nokia
Licensed under the BSD 3-Clause License
SPDX-License-Identifier: BSD-3-Clause

Convert a raw SOCP problem pickle (list of dicts) into a graph dataset
pickle consumable by preprocess.py.

Each raw item is a dict of problem tensors (f_vec, A_indices, ..., trajectory).
Output items are {'graph': HeteroData, 'trajectory': [x_0, ..., x_L]} where each
x is the iterate vector taken from trajectory[i]['x'].
"""
import argparse
import pickle

import numpy as np
import torch
from tqdm import tqdm

from socp import build_direct_graph


def _t(x, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def _build_one(p):
    try:
        graph = build_direct_graph(
            _t(p["f_vec"]),
            _t(p["b_vec"]),
            _t(p["d_vec"]),
            _t(p["c_mat"]),
            _t(p["A_indices"], torch.long),
            _t(p["A_values"]),
            tuple(p["A_shape"]),
            _t(p["norm2cons"], torch.long),
            _t(p["l_bound"]),
            _t(p["r_bound"]),
            _t(p["F"]),
            _t(p["g"]),
        )
    except Exception as e:  # skip malformed problems
        return None, repr(e)

    traj_raw = p["trajectory"]
    trajectory = [
        np.asarray(step["x"] if isinstance(step, dict) else step,
                   dtype=np.float32)
        for step in traj_raw
    ]
    return {"graph": graph, "trajectory": trajectory}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Raw problem pickle path")
    ap.add_argument("--output", required=True, help="Output graph pickle path")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on number of problems")
    args = ap.parse_args()

    print(f"Loading raw dataset: {args.input}")
    with open(args.input, "rb") as f:
        raw = pickle.load(f)
    if args.limit:
        raw = raw[: args.limit]
    print(f"Building graphs for {len(raw)} problems (serial)")

    out = []
    n_fail = 0
    first_err = None
    for p in tqdm(raw):
        item, err = _build_one(p)
        if item is None:
            n_fail += 1
            first_err = first_err or err
            continue
        out.append(item)

    print(f"Built {len(out)} graphs ({n_fail} failed)")
    if first_err:
        print(f"First failure: {first_err}")

    print(f"Saving to {args.output}")
    with open(args.output, "wb") as f:
        pickle.dump(out, f)
    print("Done.")


if __name__ == "__main__":
    main()
