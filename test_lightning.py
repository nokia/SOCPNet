"""
© 2026 Nokia
Licensed under the BSD 3-Clause License
SPDX-License-Identifier: BSD-3-Clause

SOCPNet Testing with PyTorch Lightning

This script runs test evaluation on a trained model checkpoint.
"""

import os
from datetime import datetime
import torch
from lightning_modules import (
    SOCPDataModule,
    SOCPNetLightning,
    SOCPNetend2endLightning,
    FCNNEnd2EndLightning,
)
import argparse
from tqdm import tqdm


# The three training regimes reported in the paper, and the Lightning class
# each checkpoint must be loaded with:
#   recursive  SOCPNet unrolled for --n_iterations steps via predict()
#   gnn_e2e    GNN single-shot prediction of x*
#   mlp_e2e    plain-MLP (FCNN) single-shot prediction of x*
REGIME_CLASSES = {
    "recursive": SOCPNetLightning,
    "gnn_e2e": SOCPNetend2endLightning,
    "mlp_e2e": FCNNEnd2EndLightning,
}


def detect_regime(checkpoint_path):
    """Infer the model type stored in a checkpoint.

    The training script records ``dataset_info["regime"]``. If that tag is
    missing, the type is inferred from the hyperparameters: ``n_vars`` marks
    the FCNN baseline, ``model_type == "SOCPNetend2end"`` marks the
    single-shot GNN, and everything else is the recursive SOCPNet.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    # Prefer the regime explicitly recorded at training time (train_lightning.py).
    di = hp.get("dataset_info", {})
    if isinstance(di, dict) and di.get("regime") in REGIME_CLASSES:
        return di["regime"]
    # No saved tag: infer the model type from the hyperparameters.
    if "n_vars" in hp:
        return "mlp_e2e"
    if hp.get("model_type") == "SOCPNetend2end":
        return "gnn_e2e"
    return "recursive"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def mean_constraint_violation(model, batch, x_true):
    """Mean absolute constraint violation (SOC + linear) for a single iterate.

    Returns a scalar tensor: the mean over all constraints of the (non-negative)
    violation magnitudes at ``x_true``.
    """
    # get_constraint_violations expects [N, T]; a single iterate is [N], so add
    # a trailing timestep dim and the returned violations carry one extra dim.
    violation_soc, violation_lin = model.get_constraint_violations(
        batch, x_true.unsqueeze(-1)
    )
    all_violations = torch.cat([violation_soc, violation_lin])
    if all_violations.numel() == 0:
        return torch.zeros((), device=x_true.device)
    return all_violations.mean()


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="Test SOCPNet from a trained checkpoint"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint file (.ckpt)",
    )
    parser.add_argument(
        "--dataset_paths",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more dataset roots (each must contain info.pt).",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        required=True,
        help="Number of test-time model unrolling steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_predictions",
        help=("Base directory to save test curves and results "
              "(default: test_predictions)"),
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="auto",
        choices=["auto"] + sorted(REGIME_CLASSES),
        help=("Training regime of the checkpoint. 'auto' (default) infers it "
              "from the checkpoint hyper-parameters."),
    )

    args = parser.parse_args()

    run_tests(
        checkpoint_path=args.checkpoint_path,
        dataset_paths=args.dataset_paths,
        output_dir=args.output_dir,
        n_iterations=args.n_iterations,
        regime=args.regime,
    )


def run_tests(checkpoint_path, dataset_paths, output_dir, n_iterations,
              regime="auto"):
    """
    Run test on a trained model checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint (.ckpt file)
        dataset_paths: Dataset roots (each must contain info.pt)
        output_dir: Base directory to save test curves and results
        n_iterations: Number of test-time model unrolling steps
        regime: One of {"auto", "recursive", "gnn_e2e", "mlp_e2e"}. "auto"
            infers the regime from the checkpoint hyper-parameters.
    """
    if regime == "auto":
        regime = detect_regime(checkpoint_path)
    single_shot = regime != "recursive"
    print(f"Loading model from checkpoint: {checkpoint_path}")
    print(f"Regime: {regime} ({'single-shot' if single_shot else 'recursive'})")

    # Load checkpoint with the Lightning class matching the regime.
    model = REGIME_CLASSES[regime].load_from_checkpoint(checkpoint_path)

    # Create datamodule for testing
    datamodule = SOCPDataModule(dataset_paths, batch_size=1)
    datamodule.setup()

    # Run test
    print("\nRunning tests...")
    model.eval()
    results = {}
    for dataloader_idx, test_dataset in enumerate(datamodule.test_dataloader()):
        list_predictions = []
        list_trajectories = []
        list_xopts = []
        list_obj_predictions = []
        list_obj_trajectories = []
        list_viol_predictions = []
        list_viol_trajectories = []
        dataset_name = datamodule.dataset_names[dataloader_idx]
        for test_data in tqdm(test_dataset, desc=f"Testing {dataset_name}"):
            test_data = test_data.to(model.device)
            with torch.no_grad():
                if single_shot:
                    # One forward pass -> x* prediction; denormalize with the
                    # x_opt statistics (single-shot models regress x* directly,
                    # not trajectory-normalized iterates).
                    x_pred = model._predict_trajectory(test_data).view(-1)
                    if (datamodule.normalize
                            and datamodule.x_opt_mean is not None
                            and datamodule.x_opt_std is not None):
                        x_pred = (x_pred * float(datamodule.x_opt_std)
                                  + float(datamodule.x_opt_mean))
                    predictions = x_pred.unsqueeze(1)  # [N, 1]  (T+1 == 1)
                else:
                    predictions = model.predict(
                        test_data, n_iterations=n_iterations
                    )
            # Pad or truncate trajectory to n_iterations+1 columns
            traj = test_data.trajectory  # [N, L]
            L = traj.shape[1]
            target_len = n_iterations + 1
            if L > target_len:
                traj = traj[:, :target_len]
            elif L < target_len:
                pad = traj[:, -1:].expand(-1, target_len - L)
                traj = torch.cat([traj, pad], dim=1)

            # Objective values: linear objective f^T x at every iterate.
            with torch.no_grad():
                obj_pred = torch.stack(
                    [model.get_batch_objective(test_data, predictions[:, k]).squeeze(0)
                     for k in range(predictions.shape[1])]
                )  # [T+1]
                obj_traj = torch.stack(
                    [model.get_batch_objective(test_data, traj[:, k]).squeeze(0)
                     for k in range(traj.shape[1])]
                )  # [T+1]

                # Mean constraint violation (SOC + linear) at every iterate.
                viol_pred = torch.stack(
                    [mean_constraint_violation(model, test_data, predictions[:, k])
                     for k in range(predictions.shape[1])]
                )  # [T+1]
                viol_traj = torch.stack(
                    [mean_constraint_violation(model, test_data, traj[:, k])
                     for k in range(traj.shape[1])]
                )  # [T+1]

            list_predictions.append(predictions.cpu())
            list_trajectories.append(traj.cpu())
            list_xopts.append(test_data.trajectory[:, -1].cpu())
            list_obj_predictions.append(obj_pred.cpu())
            list_obj_trajectories.append(obj_traj.cpu())
            list_viol_predictions.append(viol_pred.cpu())
            list_viol_trajectories.append(viol_traj.cpu())
        results[dataset_name] = {
            # [num_samples, N, n_iterations+1]
            "predictions": torch.stack(list_predictions, dim=0),
            # [num_samples, N, n_iterations+1]
            "trajectories": torch.stack(list_trajectories, dim=0),
            "xopts": torch.stack(list_xopts, dim=0),  # [num_samples, N]
            # [num_samples, n_iterations+1]
            "obj_predictions": torch.stack(list_obj_predictions, dim=0),
            # [num_samples, n_iterations+1]
            "obj_trajectories": torch.stack(list_obj_trajectories, dim=0),
            # [num_samples, n_iterations+1]
            "viol_predictions": torch.stack(list_viol_predictions, dim=0),
            # [num_samples, n_iterations+1]
            "viol_trajectories": torch.stack(list_viol_trajectories, dim=0),
        }

    os.makedirs(output_dir, exist_ok=True)
    # add datetime and regime to saved file name (regime disambiguates the two
    # single-shot models, which share the same model_type marker).
    filename = "results_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    filename += "_" + regime + ".pt"
    save_path = os.path.join(output_dir, filename)
    torch.save(results, save_path)
    print(f"Testing predictions saved to {save_path}.")


if __name__ == "__main__":
    main()
