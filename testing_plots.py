"""
© 2026 Nokia
Licensed under the BSD 3-Clause License
SPDX-License-Identifier: BSD-3-Clause
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


# Canonical plot/metric keys, reused for filenames and per-plot y-axis limits.
METRIC_KEYS = (
    "paper_metric",
    "relative_error",
    "step_size",
    "relative_obj_error",
    "constraint_violation",
)
YMIN_ALL_KEY = "all"  # special key applied as a global fallback


def parse_ymin_map(ymin_args):
    """Parse ``METRIC=VALUE`` pairs into a dict of per-plot lower y-limits.

    The special key ``all`` sets a global fallback applied to every plot that
    does not have an explicit override.
    """
    ymin_map = {}
    for item in ymin_args:
        if "=" not in item:
            raise ValueError(
                f"Invalid --ymin entry '{item}'. Expected METRIC=VALUE, e.g. "
                f"paper_metric=1e-6."
            )
        key, val = item.split("=", 1)
        key = key.strip()
        if key not in METRIC_KEYS and key != YMIN_ALL_KEY:
            raise ValueError(
                f"Unknown --ymin metric '{key}'. Valid keys: "
                f"{', '.join((YMIN_ALL_KEY,) + METRIC_KEYS)}."
            )
        ymin_map[key] = float(val)
    return ymin_map


def get_ymin(ymin_map, metric_key):
    """Return the lower y-limit for ``metric_key`` (None means auto-scale)."""
    return ymin_map.get(metric_key, ymin_map.get(YMIN_ALL_KEY))


def compute_metrics(predictions, xopts):
    """
    Compute paper metric and relative error per iteration.

    Args:
        predictions: [num_samples, N, n_iterations+1]  (denormalized)
        xopts:       [num_samples, N]                  (optimal solution)

    Returns:
        paper_metrics:    [num_samples, n_iterations+1]
        relative_errors:  [num_samples, n_iterations+1]
    """
    xopt_sq_norm = (xopts ** 2).sum(dim=1).clamp(min=1.0)   # [S]
    xopt_norm    = xopts.norm(dim=1).clamp(min=1e-8)          # [S]

    n_iters_p1 = predictions.shape[2]
    pm_list, re_list = [], []
    for k in range(n_iters_p1):
        diff = predictions[:, :, k] - xopts                  # [S, N]
        pm_list.append((diff ** 2).sum(dim=1) / xopt_sq_norm)  # [S]
        re_list.append(diff.norm(dim=1) / xopt_norm)           # [S]

    return torch.stack(pm_list, dim=1), torch.stack(re_list, dim=1)


def compute_step_sizes(predictions):
    """
    Compute the step size between consecutive iterations.

    The step size at iteration k is the 2-norm of the difference between two
    consecutive variables: ||x_k - x_{k-1}||.

    Args:
        predictions: [num_samples, N, n_iterations+1]

    Returns:
        step_sizes: [num_samples, n_iterations]  (step k plotted at iteration k)
    """
    diffs = predictions[:, :, 1:] - predictions[:, :, :-1]  # [S, N, T]
    return diffs.norm(dim=1)                                  # [S, T]


def compute_relative_obj_error(obj_values, obj_opt):
    """
    Relative objective error per iteration: |obj_k - obj_opt| / |obj_opt|.

    Args:
        obj_values: [num_samples, n_iterations+1]  objective value per iterate
        obj_opt:    [num_samples]                  optimal objective value

    Returns:
        relative_obj_errors: [num_samples, n_iterations+1]
    """
    denom = obj_opt.abs().clamp(min=1e-8).unsqueeze(1)   # [S, 1]
    return (obj_values - obj_opt.unsqueeze(1)).abs() / denom


def plot_metric_curve(ax, metric_tensor, title, ylabel, color, label=None,
                      linestyle="-", marker="o", x_offset=0,
                      estimator=np.median, errorbar=("pi", 50)):
    """Plot a central-tendency curve + spread band over iterations using seaborn.

    Defaults to median + IQR (25th-75th percentile) band.
    """
    data = metric_tensor.detach().numpy()          # [S, T]
    S, T = data.shape
    iterations = np.tile(np.arange(T) + x_offset, S)
    values = data.flatten()

    sns.lineplot(
        x=iterations,
        y=values,
        ax=ax,
        estimator=estimator,
        errorbar=errorbar,
        color=color,
        label=label,
        linestyle=linestyle,
        marker=marker,
    )
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Plot SOCPNet test results")
    parser.add_argument(
        "--primary_results_path",
        type=str,
        required=True,
        help="Path to results.pt file produced by test_lightning.py",
    )
    parser.add_argument(
        "--secondary_results_paths",
        type=str,
        nargs="*",
        default=[],
        help="Paths to additional results.pt files to overlay (space-separated)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_plots",
        help="Directory to save output plots (default: test_plots)",
    )
    parser.add_argument(
        "--ymin",
        nargs="*",
        default=[],
        metavar="METRIC=VALUE",
        help=(
            "Per-plot lower y-axis limit as METRIC=VALUE pairs (default: auto). "
            "Use 'all=VALUE' for a global fallback. "
            "Valid metrics: " + ", ".join(METRIC_KEYS) + ". "
            "Example: --ymin all=1e-12 step_size=1e-3"
        ),
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=[],
        help="Optional labels for primary and secondary results"
    )
    parser.add_argument(
        "--constraint_violation_epsilon",
        type=float,
        default=1e-16,
        help="Clip the minimum constraint violation value (default: 1e-16)",
    )
    args = parser.parse_args()

    ymin_map = parse_ymin_map(args.ymin)

    assert len(args.labels) == 0 or len(args.labels) == 1 + len(args.secondary_results_paths), \
        "Number of labels must match number of result files (primary + secondary)"

    results = torch.load(args.primary_results_path, map_location="cpu", weights_only=False)
    print(f"Loaded results from {args.primary_results_path}")
    for dataset_name, data in results.items():
        print(f"  {dataset_name}: predictions={data['predictions'].shape}, "
              f"trajectories={data['trajectories'].shape}, "
              f"xopts={data['xopts'].shape}")

    secondary_res_list = []
    for i, path in enumerate(args.secondary_results_paths):
        sec = torch.load(path, map_location="cpu", weights_only=False)
        if args.labels:
            label = args.labels[i + 1]
        else:
            label = os.path.splitext(os.path.basename(path))[0]
        secondary_res_list.append((label, sec))
        print(f"Loaded secondary results from {path} (label: {label})")

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette()

    for dataset_name, data in results.items():
        predictions  = data["predictions"]   # [S, N, T+1]
        trajectories = data["trajectories"]  # [S, N, T+1]
        xopts        = data["xopts"]         # [S, N]

        pred_pm,  pred_re  = compute_metrics(predictions,  xopts)
        traj_pm,  traj_re  = compute_metrics(trajectories, xopts)

        pred_step = compute_step_sizes(predictions)
        traj_step = compute_step_sizes(trajectories)

        os.makedirs(args.output_dir, exist_ok=True)

        if args.labels:
            main_label = args.labels[0]
        else:
            main_label = "SOCPNet predictions"

        med_estimator, med_errorbar = np.median, None
        med_suffix, med_stat_label = "_median", "median"

        for metric_key, pred_metric, traj_metric, title, ylabel, filename in [
            (
                "paper_metric", pred_pm, traj_pm,
                r"Paper metric  $\|x-x^*\|^2 / \max(1, \|x^*\|^2)$",
                "Paper metric",
                f"{dataset_name}_paper_metric{med_suffix}.pdf",
            ),
            (
                "relative_error", pred_re, traj_re,
                r"Relative error  $\|x-x^*\| / \|x^*\|$",
                "Relative error",
                f"{dataset_name}_relative_error{med_suffix}.pdf",
            ),
        ]:
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.suptitle(f"Dataset: {dataset_name} ({med_stat_label})", fontsize=14)
            plot_metric_curve(ax, pred_metric, title=title, ylabel=ylabel,
                              color=palette[0], label=main_label,
                              linestyle="--", marker="s",
                              estimator=med_estimator, errorbar=med_errorbar,
                              )
            plot_metric_curve(ax, traj_metric, title=title, ylabel=ylabel,
                              color=palette[1], label="IPOPT trajectories",
                              linestyle=":", marker="^",
                              estimator=med_estimator, errorbar=med_errorbar)
            for i, (sec_label, sec_results) in enumerate(secondary_res_list):
                if dataset_name not in sec_results:
                    continue
                sec_data = sec_results[dataset_name]
                sec_xopts = sec_data["xopts"]
                sec_pm, sec_re = compute_metrics(sec_data["predictions"],
                                                 sec_xopts)
                sec_metric = sec_pm if "paper" in filename else sec_re
                plot_metric_curve(ax, sec_metric, title=title, ylabel=ylabel,
                                  color=palette[2 + i], label=sec_label,
                                  linestyle="-.", marker="D",
                                  estimator=med_estimator, errorbar=med_errorbar)
            ax.legend()
            ymin = get_ymin(ymin_map, metric_key)
            if ymin is not None:
                ax.set_ylim(bottom=ymin)
            plt.tight_layout()
            out_path = os.path.join(args.output_dir, filename)
            plt.savefig(out_path, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.close(fig)

        # Step size between consecutive iterations: ||x_k - x_{k-1}||
        step_title = r"Step size  $\|x_k - x_{k-1}\|$"
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.suptitle(f"Dataset: {dataset_name} ({med_stat_label})", fontsize=14)
        plot_metric_curve(ax, pred_step, title=step_title, ylabel="Step size",
                          color=palette[0], label=main_label,
                          linestyle="--", marker="s", x_offset=1,
                          estimator=med_estimator, errorbar=med_errorbar)
        plot_metric_curve(ax, traj_step, title=step_title, ylabel="Step size",
                          color=palette[1], label="IPOPT trajectories",
                          linestyle=":", marker="^", x_offset=1,
                          estimator=med_estimator, errorbar=med_errorbar)
        for i, (sec_label, sec_results) in enumerate(secondary_res_list):
            if dataset_name not in sec_results:
                continue
            sec_data = sec_results[dataset_name]
            sec_step = compute_step_sizes(sec_data["predictions"])
            plot_metric_curve(ax, sec_step, title=step_title, ylabel="Step size",
                              color=palette[2 + i], label=sec_label,
                              linestyle="-.", marker="D", x_offset=1,
                              estimator=med_estimator, errorbar=med_errorbar)
        ax.legend()
        ymin = get_ymin(ymin_map, "step_size")
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        plt.tight_layout()
        out_path = os.path.join(
            args.output_dir, f"{dataset_name}_step_size{med_suffix}.pdf")
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved {out_path}")
        plt.close(fig)

        # Relative objective error: |obj_k - obj_opt| / |obj_opt|
        if "obj_predictions" in data and "obj_trajectories" in data:
            obj_opt = data["obj_trajectories"][:, -1]  # optimal objective value
            pred_obj_re = compute_relative_obj_error(data["obj_predictions"],
                                                     obj_opt)
            traj_obj_re = compute_relative_obj_error(data["obj_trajectories"],
                                                     obj_opt)

            obj_title = r"Relative objective error  $|f^\top x_k - f^\top x^*| / |f^\top x^*|$"
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.suptitle(f"Dataset: {dataset_name} ({med_stat_label})", fontsize=14)
            plot_metric_curve(ax, pred_obj_re, title=obj_title,
                              ylabel="Relative objective error",
                              color=palette[0], label=main_label,
                              linestyle="--", marker="s",
                              estimator=med_estimator, errorbar=med_errorbar)
            plot_metric_curve(ax, traj_obj_re, title=obj_title,
                              ylabel="Relative objective error",
                              color=palette[1], label="IPOPT trajectories",
                              linestyle=":", marker="^",
                              estimator=med_estimator, errorbar=med_errorbar)
            for i, (sec_label, sec_results) in enumerate(secondary_res_list):
                if dataset_name not in sec_results:
                    continue
                sec_data = sec_results[dataset_name]
                if "obj_predictions" not in sec_data \
                        or "obj_trajectories" not in sec_data:
                    continue
                sec_obj_opt = sec_data["obj_trajectories"][:, -1]
                sec_obj_re = compute_relative_obj_error(
                    sec_data["obj_predictions"], sec_obj_opt)
                plot_metric_curve(ax, sec_obj_re, title=obj_title,
                                  ylabel="Relative objective error",
                                  color=palette[2 + i], label=sec_label,
                                  linestyle="-.", marker="D",
                                  estimator=med_estimator, errorbar=med_errorbar)
            ax.legend()
            ymin = get_ymin(ymin_map, "relative_obj_error")
            if ymin is not None:
                ax.set_ylim(bottom=ymin)
            plt.tight_layout()
            out_path = os.path.join(
                args.output_dir, f"{dataset_name}_relative_obj_error{med_suffix}.pdf")
            plt.savefig(out_path, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.close(fig)

        # Constraint violation (mean absolute violation over all constraints)
        if "viol_predictions" in data and "viol_trajectories" in data:
            eps = args.constraint_violation_epsilon
            viol_title = r"Constraint violation  (mean $|$violation$|$)"
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.suptitle(f"Dataset: {dataset_name} ({med_stat_label})", fontsize=14)
            plot_metric_curve(ax, data["viol_predictions"].clamp(min=eps),
                              title=viol_title,
                              ylabel="Constraint violation",
                              color=palette[0], label=main_label,
                              linestyle="--", marker="s",
                              estimator=med_estimator, errorbar=med_errorbar)
            plot_metric_curve(ax, data["viol_trajectories"].clamp(min=eps),
                              title=viol_title,
                              ylabel="Constraint violation",
                              color=palette[1], label="IPOPT trajectories",
                              linestyle=":", marker="^",
                              estimator=med_estimator, errorbar=med_errorbar)
            for i, (sec_label, sec_results) in enumerate(secondary_res_list):
                if dataset_name not in sec_results:
                    continue
                sec_data = sec_results[dataset_name]
                if "viol_predictions" not in sec_data:
                    continue
                plot_metric_curve(ax, sec_data["viol_predictions"].clamp(min=eps),
                                  title=viol_title,
                                  ylabel="Constraint violation",
                                  color=palette[2 + i], label=sec_label,
                                  linestyle="-.", marker="D",
                                  estimator=med_estimator, errorbar=med_errorbar)
            ax.legend()
            ymin = get_ymin(ymin_map, "constraint_violation")
            if ymin is not None:
                ax.set_ylim(bottom=ymin)
            plt.tight_layout()
            out_path = os.path.join(
                args.output_dir, f"{dataset_name}_constraint_violation{med_suffix}.pdf")
            plt.savefig(out_path, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.close(fig)


if __name__ == "__main__":
    main()
