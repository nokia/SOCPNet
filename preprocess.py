"""
© 2026 Nokia
Licensed under the BSD 3-Clause License
SPDX-License-Identifier: BSD-3-Clause
"""

import json
import pickle
import torch
import os
import argparse
from tqdm import tqdm

def load_normalization_stats_from_json(path):
    """Load precomputed feature normalization statistics from a JSON file."""
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for key, val in raw.items():
        if key.startswith("["):
            key = tuple(json.loads(key))
        mean, std = val["mean"], val["std"]
        out[key] = (
            None if mean is None else torch.tensor(mean, dtype=torch.float32),
            None if std is None else torch.tensor(std, dtype=torch.float32),
        )
    return out


def preprocess_graph_data(dataset_path, train_split, val_split, end2end_mode,
                          target_length, normalize, normalization_stats=None):
    """Load, split, process, and normalize data in memory."""
    print(f"Loading raw dataset from {dataset_path}...")
    with open(dataset_path, "rb") as f:
        feasible_graph_data = pickle.load(f)

    # Split data FIRST (before any processing)
    n_total = len(feasible_graph_data)
    train_size = int(train_split * n_total)
    val_size = int((train_split + val_split) * n_total)

    train_items = feasible_graph_data[:train_size]
    val_items = feasible_graph_data[train_size:val_size]
    test_items = feasible_graph_data[val_size:]

    train_data = _extract_graphs(train_items, end2end_mode)
    val_data = _extract_graphs(val_items, end2end_mode)
    test_data = _extract_graphs(test_items, end2end_mode)

    # Normalize features for all splits, trajectories only for train/val
    if normalize:
        print("Normalizing features based on training data statistics...")
        if normalization_stats is None:
            stats = _compute_norm_stats(train_data, end2end_mode)
        else:
            stats = normalization_stats
        _normalize_features(
            stats, train_data, val_data, test_data, end2end_mode)
        trajectory_mean, trajectory_std = stats["trajectory"]
        x_opt_mean, x_opt_std = stats["x_opt"]
        print(f"Trajectory mean (train): {trajectory_mean}")
        print(f"Trajectory std (train): {trajectory_std}")
        print(f"Optimal solution mean (train): {x_opt_mean}")
        print(f"Optimal solution std (train): {x_opt_std}")
    else:
        trajectory_mean = trajectory_std = x_opt_mean = x_opt_std = None

    if not end2end_mode:
        print(f"Using target trajectory length L = {target_length}")
    else:
        print("End-to-end mode: skipping trajectory processing (only using "
              "optimal solutions)")

    print(f"Padding training data to length {target_length}...")
    train_data = _process_items(train_data, target_length)
    print(f"Padding validation data to length {target_length}...")
    val_data = _process_items(val_data, target_length)

    # Log statistics
    if not end2end_mode:
        train_masks = torch.stack([g.trajectory_mask for g in train_data])
        real_steps = train_masks.sum(dim=1)
        print(f"Training set mask stats: "
              f"mean real steps = {real_steps.mean():.1f}, "
              f"min = {real_steps.min():.0f}, max = {real_steps.max():.0f}")
    else:
        print(f"End-to-end mode: {len(train_data)} training samples, "
              f"{len(val_data)} validation samples")
        # Log optimal solution stats
        all_x_opt = torch.cat([g.x_opt for g in train_data], dim=0)
        print(f"Optimal solution stats: mean={all_x_opt.mean():.4f}, "
              f"std={all_x_opt.std():.4f}, min={all_x_opt.min():.4f}, "
              f"max={all_x_opt.max():.4f}")

    test_lengths = [len(item["trajectory"]) for item in test_items]
    print(f"Test set (raw): {len(test_lengths)} samples, "
          f"lengths: mean={sum(test_lengths)/len(test_lengths):.1f}, "
          f"min={min(test_lengths)}, max={max(test_lengths)}")

    return train_data, val_data, test_data, test_items, \
        trajectory_mean, trajectory_std, x_opt_mean, x_opt_std


def _extract_graphs(items, end2end_mode):
    graphs = []
    for item in items:
        g = item["graph"]
        traj = item["trajectory"]
        g.trajectory_length = len(traj)

        # Create unpadded trajectory tensor
        g.trajectory = torch.stack(
            [torch.tensor(t, dtype=torch.float32) for t in traj], dim=1
        )

        if end2end_mode:
            # [n_dim]
            g.x_opt = torch.tensor(traj[-1], dtype=torch.float32)
        else:
            # [n_dim, 1]
            g.x_opt = torch.tensor(traj[-1], dtype=torch.float32).unsqueeze(1)

        graphs.append(g)
    return graphs


def _process_items(graphs, target_length):
    result = []
    for graph in tqdm(graphs):
        if graph["var"].x.shape[1] != 3:
            raise ValueError("Expected variable features with shape "
                                f"[N, 3], got {graph['var'].x.shape}")

        graph.trajectory, graph.trajectory_mask = \
            _process_trajectory(graph.trajectory, target_length)

        result.append(graph)
    return result


def _process_trajectory(traj_tensor, target_length):
    """
    Process a single trajectory: truncate or pad to target_length.

    Args:
        traj_tensor: torch.Tensor of shape [n_dim, L_orig]

    Returns:
        traj_tensors: torch.Tensor of shape [n_dim, L]
        mask: torch.Tensor of shape [L], 1 for real steps, 0 for padded
    """
    traj_len = traj_tensor.shape[1]

    if traj_len >= target_length:
        # Truncate: keep the LAST target_length steps (preserves convergence)
        traj_tensors = traj_tensor[:, -target_length:]
    else:
        # Pad: use the final (optimal) solution vector repeated
        final_solution = traj_tensor[:, -1:]
        padded = final_solution.repeat(1, target_length - traj_len)
        traj_tensors = torch.cat([traj_tensor, padded], dim=1)

    mask = torch.zeros(target_length, dtype=torch.float32)
    mask[:min(traj_len, target_length)] = 1.0

    return traj_tensors, mask


def _compute_norm_stats(train_data, end2end_mode):
    """
    Compute normalization statistics from training data only.

    Returns:
        stats: dict mapping feature keys to (mean, std) tensors for
            node types, trajectories or x_opt, and edge types.
            In end-to-end mode, ``stats["trajectory"]`` is ``(None, None)``.
            In trajectory mode, ``stats["x_opt"]`` is ``(None, None)`` and
            ``stats["trajectory"]`` uses 0-dim (scalar) mean/std tensors.
    """
    stats = {}
    for node_type in ["var", "norm", "cons"]:
        features = torch.cat([g[node_type].x for g in train_data], dim=0)
        stats[node_type] = (
            features.mean(dim=0, keepdim=True),
            features.std(dim=0, keepdim=True),
        )

    if end2end_mode:
        all_x_opt = torch.cat([g.x_opt for g in train_data], dim=0)
        x_opt_mean = all_x_opt.mean()
        x_opt_std = all_x_opt.std()
        stats["x_opt"] = (x_opt_mean, x_opt_std)
        stats["trajectory"] = (None, None)
    else:
        all_iterates = torch.cat(
            [g.trajectory.reshape(-1) for g in train_data], dim=0
        )
        traj_mean = all_iterates.mean()
        traj_std = all_iterates.std()

        stats["trajectory"] = (traj_mean, traj_std)
        stats["x_opt"] = (None, None)

    edge_stats = {}
    for g in train_data:
        for edge_type in g.edge_types:
            if (
                hasattr(g[edge_type], "edge_attr")
                and g[edge_type].edge_attr is not None
            ):
                edge_stats.setdefault(edge_type, []).append(
                    g[edge_type].edge_attr
                )

    for edge_type, attrs in edge_stats.items():
        edge_features = torch.cat(attrs, dim=0)
        stats[edge_type] = (
            edge_features.mean(dim=0, keepdim=True),
            edge_features.std(dim=0, keepdim=True),
        )

    return stats


def _normalize_features(stats, train_data, val_data, test_data, end2end_mode):
    """Apply normalization from precomputed stats (in place)."""
    for split in [train_data, val_data, test_data]:
        for g in split:
            for node_type in ["var", "norm", "cons"]:
                mean, std = stats[node_type]
                std = torch.clamp(std, min=1e-8)
                g[node_type].x_deprocessed = g[node_type].x
                g[node_type].x = (g[node_type].x - mean) / std

            for edge_type in g.edge_types:
                if (
                    edge_type in stats
                    and hasattr(g[edge_type], "edge_attr")
                    and g[edge_type].edge_attr is not None
                ):
                    mean, std = stats[edge_type]
                    std = torch.clamp(std, min=1e-8)
                    g[edge_type].edge_attr_deprocessed = g[edge_type].edge_attr
                    g[edge_type].edge_attr = (
                        (g[edge_type].edge_attr - mean) / std)

    if end2end_mode:
        mean, std = stats["x_opt"]
        std = torch.clamp(std, min=1e-8)
        for split in [train_data, val_data]:
            for g in split:
                g.x_opt = (g.x_opt - mean) / std
    else:
        for split in [train_data, val_data]:
            for g in split:
                mean, std = stats["trajectory"]
                std = torch.clamp(std, min=1e-8)
                g.trajectory = (g.trajectory - mean) / std


def _save_preprocessed_data(save_path, raw_dataset_path, train_split,
                            val_split, end2end_mode, target_length, normalize,
                            train_data, val_data, test_data, test_items,
                            trajectory_mean, trajectory_std, x_opt_mean,
                            x_opt_std):
    # Create folder and subfolders for splits
    os.makedirs(save_path, exist_ok=True)
    split_dirs = {
        "train": os.path.join(save_path, "train"),
        "val": os.path.join(save_path, "val"),
        "test": os.path.join(save_path, "test"),
        "test_items": os.path.join(save_path, "test_items"),
    }
    for dir_path in split_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Save each graph as a separate .pt file with its index as filename
    def save_split(items, split_name):
        filenames = []
        for idx, item in enumerate(items):
            filename = f"{idx:06d}.pt"
            file_path = os.path.join(split_dirs[split_name], filename)
            torch.save(item, file_path)
            filenames.append(filename)
        return filenames
    train_files = save_split(train_data or [], "train")
    val_files = save_split(val_data or [], "val")
    test_files = save_split(test_data or [], "test")
    test_item_files = save_split(test_items or [], "test_items")

    # Create info file with metadata, split filenames, and normalization stats
    metadata = {
        "raw_dataset_path": os.path.abspath(raw_dataset_path),
        "train_split": train_split,
        "val_split": val_split,
        "target_length": target_length,
        "normalize": normalize,
        "end2end_mode": end2end_mode,
    }
    info_file = {
        "metadata": metadata,
        "splits_filenames": {
            "train": train_files,
            "val": val_files,
            "test": test_files,
            "test_items": test_item_files,
        },
        "trajectory_mean": trajectory_mean,
        "trajectory_std": trajectory_std,
        "x_opt_mean": x_opt_mean,
        "x_opt_std": x_opt_std,
    }
    torch.save(info_file, os.path.join(save_path, "info.pt"))

    print("metadata saved in info.pt:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess graph dataset for SOCPNet training"
    )
    parser.add_argument(
        "--dataset_path",
        type=str, required=True,
        help="Path to the raw dataset pickle file",
    )
    parser.add_argument(
        "--save_path",
        type=str, required=True,
        help="Directory to save preprocessed data",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--end2end_mode",
        action="store_true",
        help="Use end-to-end mode (default: not applied)",
    )
    parser.add_argument(
        "--target_length",
        type=int, required=True,
        help="Target trajectory length L",
    )
    parser.add_argument(
        "--normalize",
        type=str2bool,
        default=True,
        help="Normalize features (default: True)",
    )
    parser.add_argument(
        "--normalization_stats",
        type=str,
        default=None,
        help="Path to JSON file with normalization statistics",
    )
    args = parser.parse_args()

    # Check if folder already exists and ask user to confirm overwrite
    if os.path.exists(args.save_path):
        response = input(
            f"Folder {args.save_path} already exists. "
            "Do you want to overwrite it? (y/n): "
        )
        if response.lower() != 'y':
            print("Aborting preprocessing.")
            exit(0)

    normalization_stats = None
    if args.normalization_stats:
        normalization_stats = load_normalization_stats_from_json(
            args.normalization_stats)

    train_data, val_data, test_data, test_items, \
        trajectory_mean, trajectory_std, x_opt_mean, x_opt_std = \
            preprocess_graph_data(args.dataset_path, args.train_split,
                                  args.val_split, args.end2end_mode,
                                  args.target_length, args.normalize,
                                  normalization_stats=normalization_stats)

    _save_preprocessed_data(
        args.save_path, args.dataset_path, args.train_split, args.val_split,
        args.end2end_mode, args.target_length, args.normalize, train_data,
        val_data, test_data, test_items, trajectory_mean, trajectory_std,
        x_opt_mean, x_opt_std)
