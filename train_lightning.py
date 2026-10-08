"""
© 2026 Nokia
Licensed under the BSD 3-Clause License
SPDX-License-Identifier: BSD-3-Clause

SOCPNet Training with PyTorch Lightning

This script implements the SOCPNet training using PyTorch Lightning framework
with TensorBoard logging for better organization and experiment tracking. 
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from lightning_modules import (
    SOCPDataModule,
    SOCPNetLightning,
    SOCPNetend2endLightning,
    FCNNEnd2EndLightning,
)
import argparse


# Zero-padding cap for the "norm" node axis in FCNNEnd2End (SOC block dims vary
# per problem instance). Values cover the observed maximum number of norm nodes
# per dataset size, with margin. Unrecognized sizes fall back to a live scan.
_FCNN_NORM_PAD = {"_small": 64, "_medium": 256}


def _scan_max_n_norms(datamodule) -> int:
    """Live scan for the largest number of norm nodes across all splits."""
    datasets = []
    train = getattr(datamodule, "train_dataset", None)
    if train is not None:
        datasets.append(train)
    for attr in ("val_datasets", "test_datasets"):
        datasets.extend(getattr(datamodule, attr, []) or [])
    max_n = 0
    for ds in datasets:
        for g in ds:
            max_n = max(max_n, g["norm"].x.shape[0])
    return max_n


def _infer_max_n_norms(dataset_size_tag: str, datamodule=None) -> int:
    if dataset_size_tag in _FCNN_NORM_PAD:
        return _FCNN_NORM_PAD[dataset_size_tag]
    if datamodule is not None:
        return _scan_max_n_norms(datamodule)
    raise ValueError(
        f"No known norm-padding cap for dataset size tag '{dataset_size_tag}'. "
        f"Add an entry to _FCNN_NORM_PAD (scan train split for max n_norms)."
    )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train SOCPNet with PyTorch Lightning"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="SOCPNet",
        choices=[
            "SOCPNet",
            "FCNNEnd2End",
        ],
        help=(
            "Model architecture to use. 'FCNNEnd2End' is a flatten+MLP "
            "baseline (no graph structure); only valid together with "
            "--end2end."
        ),
        required=True,
    )
    parser.add_argument(
        "--fcnn_hidden_dims",
        type=int,
        nargs="+",
        default=[512, 256, 128],
        help="Hidden layer widths for --model FCNNEnd2End (default: 512 256 128).",
    )
    parser.add_argument(
        "-d",
        "--embedding_size",
        type=int,
        default=32,
        help="Model dimension in terms of embedding size",
    )
    parser.add_argument(
        "--target_length",
        type=int,
        default=None,
        help="Target trajectory length L (required for trajectory-based training, optional for end-to-end mode)",
    )
    parser.add_argument(
        "-t",
        "--max_iterations",
        type=int,
        default=None,
        help="Number of trajectory prediction steps T. Since trajectory includes x0, T = target_length - 1 by default",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=200,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "-l", "--learning_rate", type=float, default=5e-4, help="Learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=40, help="Batch size for training"
    )
    parser.add_argument(
        "--zero_everything",
        action="store_true",
        help="Zero out all input features",
    )
    parser.add_argument(
        "--overfitting_test",
        action="store_true",
        help="Run overfitting test with single batch",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=-1,
        help="Number of GPUs to use (-1 for all available)",
    )
    parser.add_argument(
        "--dataset_paths",
        nargs="+",
        required=True,
        metavar="PATH",
        help=(
            "One or more dataset roots (each must contain info.pt). "
            "Training uses all roots combined; val and test use one dataloader per root."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--message_passing_scheme",
        type=str,
        default="three_step",
        choices=["three_step"],
        help="Message passing scheme for SOCPNet model. "
             "Only 'three_step' (var→all, norm↔cons, all→var) is supported. "
             "Only used when model is 'SOCPNet'.",
    )
    parser.add_argument(
        "--teacher_forcing_p",
        type=float,
        default=0.0,
        help="Teacher forcing probability for trajectory mode (default: 0.0)",
    )
    parser.add_argument(
        "--end2end",
        action="store_true",
        help="Use end-to-end optimal solution prediction mode (2 MP layers, no trajectory unrolling)",
    )
    parser.add_argument(
        "--log_group",
        type=str,
        default="training_normal_diverse",
        help="Override the default log group folder name (e.g., standard-training2)",
    )
    parser.add_argument(
        "--learn_fixed_point",
        type=str2bool, default=True,
        help=("Whether to include two times the final solution at the end of "
              "the target trajectory. If used, the model can learn that the "
              "final solution is a fixed point of the optimization procedure")
    )
    parser.add_argument(
        "--traj_geometric_coef",
        type=float, default=None,
        help=("Applies geometric decay to the trajectory steps in the loss. "
              "The coefficient for iteration/step i is "
              "traj_geometric_coef ** (end_idx - 1 - i)), where end_idx is "
              "the index of the final step in the trajectory.")
    )
    parser.add_argument(
        "--full_tail_supervision", type=str2bool, default=False,
        help=(
            "Supervise the ENTIRE padded trajectory tail (all padded columns "
            "equal x*), so every unroll step is supervised to stay at the "
            "optimum. Default False."
        ),
    )
    parser.add_argument(
        "--lr_schedule", type=str, default="none", choices=["none", "cosine"],
        help="LR schedule: 'cosine' = linear warmup then cosine anneal to ~0.",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=0,
        help="Number of linear-warmup epochs when --lr_schedule cosine.",
    )
    parser.add_argument(
        "--aggr", type=str, default="mean",
        choices=["add", "mean", "max", "min"],
        help="Aggregation method for message passing (default: mean)"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--residual",
        type=str2bool,
        default=False,
        help="Residual readout: x_{t+1}=x_t+Δ. SOCPNet trajectory mode only.",
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=0,
        help="Optional multi-head self-attention on var nodes (0=off); embedding dim d must divide heads.",
    )
    parser.add_argument(
        "--message_passing_depth",
        type=int,
        default=1,
        help="Number of sequential message passing rounds per unroll step (default: 1)",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=["mse", "soft_dtw", "target_final", "target_k", "traj_logsre"],
        help=(
            "Main supervised trajectory loss: 'mse' (default, strict step-by-step MSE), "
            "'soft_dtw' (order-preserving elastic alignment via Soft-DTW; requires tslearn), "
            "'target_final' (every iterate compared to the final optimal solution x*), "
            "'target_k' (prediction at step t compared to ground truth at step t+k), or "
            "'traj_logsre' (log relative SRE of every iterate against x*)."
        ),
    )
    parser.add_argument(
        "--target_k_step",
        type=int,
        default=5,
        help="Number of steps k for 'target_k' loss type.",
    )
    parser.add_argument(
        "--soft_dtw_gamma",
        type=float,
        default=1.0,
        help=(
            "Smoothing parameter for Soft-DTW (only used when --loss_type soft_dtw). "
            "Smaller values approach hard DTW; larger values approach MSE (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--train_loss_weight",
        nargs="+",
        default=None,
        metavar=("MODE", "PARAM"),
        help=(
            "Per-timestep trajectory loss weighting applied before reducing over "
            "the trajectory. Modes: 'constant C' (every step weighted by C) or "
            "'linear START END' (weights linearly spaced from START to END over the "
            "T timesteps). Not supported with --loss_type soft_dtw. "
            "Examples: '--train_loss_weight constant 8', "
            "'--train_loss_weight linear 1 0', '--train_loss_weight linear 0 1'."
        ),
    )
    parser.add_argument(
        "--train_loss_constraint",
        nargs="+",
        default=None,
        metavar=("MODE", "PARAM"),
        help=(
            "Enable the constraint-violation trajectory loss and set its "
            "per-timestep weighting. Same modes as --train_loss_weight: "
            "'constant C' or 'linear START END'. Omit to disable (default). "
            "Examples: '--train_loss_constraint constant 8', "
            "'--train_loss_constraint linear 1 0'."
        ),
    )
    parser.add_argument(
        "--train_loss_objective",
        nargs="+",
        default=None,
        metavar=("MODE", "PARAM"),
        help=(
            "Enable the objective trajectory loss and set its per-timestep "
            "weighting. Same modes as --train_loss_weight: 'constant C' or "
            "'linear START END'. Omit to disable (default). "
            "Examples: '--train_loss_objective constant 8', "
            "'--train_loss_objective linear 1 0'."
        ),
    )
    parser.add_argument(
        "--train_loss_fixed_point",
        "--fixed_point_loss",
        dest="train_loss_fixed_point",
        type=float,
        default=None,
        metavar="WEIGHT",
        help=(
            "Enable the fixed-point consistency loss and set its (constant scalar) "
            "weight. Penalizes ||G(x*) - x*||^2, i.e. feeding the optimum back "
            "through one model step should return the optimum. Computed once per "
            "trajectory, so only a constant weight applies (no per-timestep "
            "schedule). Omit to disable (default). "
            "Examples: '--fixed_point_loss 1.0' or '--train_loss_fixed_point 1.0'."
        ),
    )
    parser.add_argument(
        "--picard_weight", type=float, default=0.0,
        help=(
            "Weight on the Picard/monotone-descent contractivity penalty "
            "relu(d_{t+1} - rho*d_t), d = relative distance to x*. Forces the "
            "unroll to contract so the step size collapses toward the optimum. "
            "0 disables."
        ),
    )
    parser.add_argument(
        "--picard_rho", type=float, default=0.9,
        help="Contraction factor rho (<1) in the Picard penalty.",
    )
    parser.add_argument(
        "--final_anchor_weight", type=float, default=0.0,
        help=(
            "Weight on the final-iterate relative-SRE anchor to the true x*. "
            "0 disables."
        ),
    )
    parser.add_argument(
        "--iterfeat_eps", type=float, default=1.0,
        help=(
            "Epsilon in the inverse-slack dual proxies of the iterate features. "
            "The default 1.0 saturates near active constraints; a smaller value "
            "keeps resolution there."
        ),
    )
    parser.add_argument(
        "--scale_direction_loss",
        nargs="+",
        default=None,
        metavar=("MODE", "PARAM"),
        help=(
            "Enable the scale/direction readout (residual step parametrized as "
            "exp(scale)*direction) together with its supervised log-step loss "
            "MSE(scale - log||x*_{t+1}-x*_t||), and set the loss per-timestep "
            "weighting. Same modes as --train_loss_constraint: 'constant C' or "
            "'linear START END'. SOCPNet trajectory mode only. Omit to "
            "disable (default). Examples: '--scale_direction_loss constant 1', "
            "'--scale_direction_loss linear 1 0'."
        ),
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=1.0,
        help=(
            "Global gradient-norm clipping threshold to tame Backpropagation "
            "Through Time (BPTT) loss spikes from the trajectory unroll. "
            "Set to 0 to disable (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--mp_decouple_soc_lc",
        type=str2bool,
        default=False,
        help=(
            "If True, the final message-passing step uses two separate aggregators "
            "(Phi_soc for the coupled norm+cons SOC signals, then Phi_lc for a linear-"
            "constraint correction). If False (default), a single joint Phi4 aggregates "
            "all incoming messages. SOCPNet trajectory mode only."
        ),
    )
    parser.add_argument(
        "--use_gru_update",
        type=str2bool,
        default=False,
        help=(
            "If True, a gated recurrent unit (GRU) is used for feature update at "
            "every unroll step. If False (default), a simple concat+Linear feature "
            "update is used instead. SOCPNet trajectory mode only."
        ),
    )
    parser.add_argument(
        "--use_step_encoding",
        type=str2bool,
        default=False,
        help=(
            "If True, a sinusoidal encoding of the unroll-step index is injected "
            "into the node embeddings at every iteration, giving the "
            "model a notion of position along the central path. If False "
            "(default), no step encoding is used. SOCPNet trajectory mode only."
        ),
    )
    parser.add_argument(
        "--use_iterate_features",
        type=str2bool,
        default=False,
        help=(
            "If True, the local KKT information at the current iterate (constraint/"
            "bound slacks and inverse-slack dual proxies) is computed from the raw "
            "problem data and injected into the node embeddings at every unroll step. "
            "If False (default), these features are not used. SOCPNet trajectory "
            "mode only."
        ),
    )

    args = parser.parse_args()

    assert not (args.zero_everything and args.overfitting_test), (
        "Cannot use both zero_everything and overfitting_test together."
    )
    if args.model == "FCNNEnd2End" and not args.end2end:
        raise ValueError("--model FCNNEnd2End requires --end2end.")

    # Gradient clipping (global norm) to suppress BPTT loss spikes; 0 disables.
    gradient_clip_val = args.gradient_clip_val if args.gradient_clip_val > 0 else None

    pl.seed_everything(args.seed, workers=True)

    # Validate target_length (optional for end-to-end mode)
    if not args.end2end and args.target_length is None:
        raise ValueError("--target_length is required for trajectory-based training")

    # Create datamodule
    datamodule = SOCPDataModule(
        dataset_paths=args.dataset_paths,
        batch_size=args.batch_size,
        overfitting_test=args.overfitting_test,
        end2end_mode=args.end2end,
        learn_fixed_point=args.learn_fixed_point,
        traj_geometric_coef=args.traj_geometric_coef,
        full_tail_supervision=args.full_tail_supervision,
    )
    datamodule.setup()

    # Collect dataset info to be stored in the checkpoint hyperparameters
    dataset_info_file = os.path.join(args.dataset_paths[0], "info.pt")
    tmp_dict = torch.load(dataset_info_file, map_location="cpu")
    dataset_info = {
        "dataset_path": args.dataset_paths,
        "batch_size": args.batch_size,
        "overfitting_test": args.overfitting_test,
        "learn_fixed_point": args.learn_fixed_point,
        "traj_geometric_coef": args.traj_geometric_coef
    }
    for key, value in tmp_dict["metadata"].items():
        if key == "raw_dataset_path":
            continue
        dataset_info[key] = value
    if tmp_dict["metadata"]["normalize"]:
        dataset_info["trajectory_mean"] = tmp_dict["trajectory_mean"].tolist()\
            if tmp_dict["trajectory_mean"] is not None else None
        dataset_info["trajectory_std"] = tmp_dict["trajectory_std"].tolist()\
            if tmp_dict["trajectory_std"] is not None else None
        dataset_info["x_opt_mean"] = tmp_dict["x_opt_mean"].tolist()\
            if tmp_dict["x_opt_mean"] is not None else None
        dataset_info["x_opt_std"] = tmp_dict["x_opt_std"].tolist()\
            if tmp_dict["x_opt_std"] is not None else None

    # Record the training regime in the checkpoint metadata so downstream
    # evaluation (test_lightning.py) can pick the right Lightning class without
    # guessing from hyper-parameters.
    if args.model == "FCNNEnd2End":
        dataset_info["regime"] = "mlp_e2e"
    elif args.end2end:
        dataset_info["regime"] = "gnn_e2e"
    else:
        dataset_info["regime"] = "recursive"

    if args.end2end:
        T = None
        print("End-to-end mode: T not used (no trajectory unrolling)")
    else:
        T = args.max_iterations if args.max_iterations is not None else args.target_length - 1
        if args.max_iterations is not None:
            print(f"Using user-specified max_iterations T = {T}")
        else:
            print(f"Using auto-computed T = {T} (target_length - 1)")

    # Extract dataset size from filename and add to experiment name
    dataset_basename = os.path.splitext(
        os.path.basename(args.dataset_paths[0])
    )[0].lower()
    dataset_size = ""
    for size in ["small", "medium", "large"]:
        if size in dataset_basename:
            dataset_size = f"_{size}"
            break

    # Define experiment name based on args
    if args.end2end:
        # End-to-end mode
        model_tag = "FCNNEnd2End" if args.model == "FCNNEnd2End" else "End2End"
        experiment = f"{model_tag}_d{args.embedding_size}{dataset_size}"
        if args.zero_everything:
            experiment = os.path.join("zeroed-test", experiment)
        elif args.overfitting_test:
            experiment = os.path.join("overfitting-test", experiment)
        else:
            log_group = args.log_group or "end2end-training"
            experiment = os.path.join(log_group, experiment)

        if args.model == "FCNNEnd2End":
            # Fixed dims (n_vars/n_cons/n_lin_cons) come from a single sample;
            # n_norms varies per instance (SOC block dims), so it is padded to
            # the dataset-wide max seen across train+val+test.
            sample0 = datamodule.train_dataset[0]
            n_vars = sample0["var"].x.shape[0]
            n_cons = sample0["cons"].x.shape[0]
            n_lin_cons = sample0["lin_cons"].x.shape[0]
            n_norms_max = _infer_max_n_norms(dataset_size, datamodule)
            print(
                f"FCNNEnd2End dims: n_vars={n_vars}, n_norms_max={n_norms_max}, "
                f"n_cons={n_cons}, n_lin_cons={n_lin_cons}"
            )
            model = FCNNEnd2EndLightning(
                d=args.embedding_size,
                lr=args.learning_rate,
                n_vars=n_vars,
                n_norms_max=n_norms_max,
                n_cons=n_cons,
                n_lin_cons=n_lin_cons,
                hidden_dims=tuple(args.fcnn_hidden_dims),
                zero_everything=args.zero_everything,
                gradient_clip_val=gradient_clip_val,
                train_loss_constraint=args.train_loss_constraint is not None,
                train_loss_objective=args.train_loss_objective is not None,
                dataset_info=dataset_info,
            )
        else:
            model = SOCPNetend2endLightning(
                d=args.embedding_size,
                lr=args.learning_rate,
                zero_everything=args.zero_everything,
                aggr=args.aggr,
                num_attention_heads=args.num_attention_heads,
                gradient_clip_val=gradient_clip_val,
                mp_decouple_soc_lc=args.mp_decouple_soc_lc,
                train_loss_constraint=args.train_loss_constraint is not None,
                train_loss_objective=args.train_loss_objective is not None,
                dataset_info=dataset_info,
            )
    else:
        # Trajectory-based mode
        experiment = args.model + f"_d{args.embedding_size}_t{T}{dataset_size}"
        if args.zero_everything:
            experiment = os.path.join("zeroed-test", experiment)
        elif args.overfitting_test:
            experiment = os.path.join("overfitting-test", experiment)
        else:
            log_group = args.log_group or "standard-training"
            experiment = os.path.join(log_group, "three_steps", experiment)

        model = SOCPNetLightning(
            d=args.embedding_size,
            T=T,
            lr=args.learning_rate,
            zero_everything=args.zero_everything,
            model_type=args.model,
            message_passing_scheme=args.message_passing_scheme,
            teacher_forcing_p=args.teacher_forcing_p,
            target_length=args.target_length,
            aggr=args.aggr,
            dataset_info=dataset_info,
            residual=args.residual,
            num_attention_heads=args.num_attention_heads,
            message_passing_depth=args.message_passing_depth,
            loss_type=args.loss_type,
            soft_dtw_gamma=args.soft_dtw_gamma,
            target_k_step=args.target_k_step,
            train_loss_weight=args.train_loss_weight,
            train_loss_constraint_weight=args.train_loss_constraint,
            train_loss_objective_weight=args.train_loss_objective,
            train_loss_constraint=args.train_loss_constraint is not None,
            train_loss_objective=args.train_loss_objective is not None,
            train_loss_fixed_point=args.train_loss_fixed_point,
            scale_direction=args.scale_direction_loss is not None,
            train_loss_scale_direction_weight=args.scale_direction_loss,
            gradient_clip_val=gradient_clip_val,
            mp_decouple_soc_lc=args.mp_decouple_soc_lc,
            use_gru_update=args.use_gru_update,
            use_step_encoding=args.use_step_encoding,
            use_iterate_features=args.use_iterate_features,
            full_tail_supervision=args.full_tail_supervision,
            picard_weight=args.picard_weight,
            picard_rho=args.picard_rho,
            final_anchor_weight=args.final_anchor_weight,
            iterfeat_eps=args.iterfeat_eps,
            lr_schedule=args.lr_schedule,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.max_epochs,
        )

    logger = TensorBoardLogger(save_dir="lightning_logs", name=experiment)
    vnames = datamodule.dataset_names
    monitor_metric = "average_val_loss"
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=200000000,
        mode="min",
        verbose=True,
    )

    # Configure DDP strategy
    if args.devices == 1:
        # Single GPU: use auto (no DDP overhead)
        strategy = "auto"
    else:
        # Multiple GPUs or auto-detect (-1): use DDP
        strategy = DDPStrategy(find_unused_parameters=False)

    trainer = pl.Trainer(
        logger=logger,
        devices=args.devices,
        accelerator="gpu",
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        strategy=strategy,
        deterministic=True,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model, datamodule, ckpt_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
