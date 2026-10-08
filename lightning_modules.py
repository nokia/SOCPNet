"""
© 2026 Nokia
Licensed under the BSD 3-Clause License
SPDX-License-Identifier: BSD-3-Clause

SOCPNet Lightning Modules

This module contains PyTorch Lightning modules for SOCPNet training:
- SOCPDataModule: Data handling and preprocessing
- SOCPNetLightning: Trajectory-based training module
- SOCPNetend2endLightning: End-to-end optimal solution prediction module
"""

import math
import os
import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader
from model import (
    SOCPNet,
    SOCPNetend2end,
    FCNNEnd2End,
)

class LazyDataset(Dataset):
    """Lazy-loading dataset. Accepts one root or a list of roots."""

    def __init__(
        self,
        roots,               # str or list[str]
        split_name,
        filenames,           # list[str] or list[list[str]]
        learn_fixed_point=False,
        traj_geometric_coef=None,
        full_tail_supervision=False,
    ):
        if isinstance(roots, str):
            roots, filenames = [roots], [filenames]
        assert len(roots) == len(filenames)
        self.filenames = [
            os.path.join(root, split_name,fn )
            for root, filename_list in zip(roots, filenames)
            for fn in filename_list
        ]

        self.learn_fixed_point = learn_fixed_point
        self.traj_geometric_coef = traj_geometric_coef
        # If True, supervise the whole padded trajectory tail (those columns
        # equal x*), so every unroll step is trained to stay at the optimum.
        self.full_tail_supervision = full_tail_supervision

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        graph_data = torch.load(
            filename, map_location="cpu", weights_only=False
        )
        # Only create mask if trajectory exists (train/val data), not for test data
        if (
            hasattr(graph_data, "trajectory")
            and graph_data.trajectory is not None
        ):
            graph_data.trajectory_mask = self._create_mask(graph_data)
        return graph_data

    def _create_mask(self, graph_data):
        """
        mask: torch.Tensor of shape [target_length],
              contains 1 for real steps, 0 for padded steps
        """
        target_length = graph_data.trajectory.shape[1]
        traj_len = graph_data.trajectory_length

        # Mask: 1 for real steps, 0 for padded steps
        mask = torch.zeros(
            target_length,
            dtype=torch.float32,
            device=graph_data.trajectory.device,
        )
        if self.full_tail_supervision:
            # Every column (padded tail = x*) is supervised => stay at optimum.
            end_idx = target_length
        elif self.learn_fixed_point:
            end_idx = min(traj_len + 1, target_length)
        else:
            end_idx = min(traj_len, target_length)
        mask[:end_idx] = 1.0

        # Apply geometric decay to trajectory steps
        end_idx = min(traj_len, target_length)
        if (
            self.traj_geometric_coef is not None
            and self.traj_geometric_coef < 1.0
        ):
            for i in range(end_idx):
                mask[i] *= self.traj_geometric_coef ** (end_idx - 1 - i)

        return mask


class SOCPDataModule(pl.LightningDataModule):
    """LightningDataModule for SOCPNet data handling."""

    def __init__(
        self,
        dataset_paths: str | list[str],
        batch_size: int,
        num_workers: int = 0,
        overfitting_test: bool = False,
        end2end_mode: bool = False,
        learn_fixed_point: bool = True,
        traj_geometric_coef: float | None = None,
        full_tail_supervision: bool = False,
    ):
        super().__init__()
        assert traj_geometric_coef is None or 0 <= traj_geometric_coef <= 1

        self._roots = (
            [dataset_paths] if isinstance(dataset_paths, str)
            else list(dataset_paths)
        )
        for r in self._roots:
            self._ensure_dataset_root(r)

        self.dataset_path = self._roots[0]   # primary root for stats
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.end2end_mode = end2end_mode
        self.learn_fixed_point = learn_fixed_point
        self.traj_geometric_coef = traj_geometric_coef
        self.full_tail_supervision = full_tail_supervision
        self.overfitting_test = overfitting_test

        self.dataset_names = [os.path.basename(os.path.normpath(r))
        for r in self._roots]

        if self.overfitting_test:
            print("Overfitting test mode: using only 1 batch for training!")

    @staticmethod
    def _ensure_dataset_root(path: str) -> None:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Dataset root must be a valid directory, got: {path}"
            )
        if not os.path.isfile(os.path.join(path, "info.pt")):
            raise FileNotFoundError(f"{path} does not contain info.pt")

    def denormalize_trajectory(self, normalized_trajectory):
        """Denormalize trajectory predictions [batch_size * n_dim, T]."""
        if self.trajectory_mean is None or self.trajectory_std is None:
            return normalized_trajectory
        dev = normalized_trajectory.device
        mean = self.trajectory_mean.to(dev)
        std = self.trajectory_std.to(dev)
        return normalized_trajectory * std + mean

    def setup(self, stage=None):
        infos = [
            torch.load(
                os.path.join(r, "info.pt"),
                map_location="cpu",
                weights_only=False,
            )
            for r in self._roots
        ]
        primary = infos[0]
        splits = [info["splits_filenames"] for info in infos]

        self.trajectory_mean = primary["trajectory_mean"]
        self.trajectory_std = primary["trajectory_std"]
        self.x_opt_mean = primary["x_opt_mean"]
        self.x_opt_std = primary["x_opt_std"]
        self.normalize = primary["metadata"]["normalize"]

        # Training: concatenation of all roots (shuffled in dataloader)
        self.train_dataset = LazyDataset(
            self._roots,
            "train",
            [s["train"] for s in splits],
            self.learn_fixed_point,
            self.traj_geometric_coef,
            self.full_tail_supervision,
        )

        # Val, test, and test_items: one dataset per root. Val uses the same
        # tail supervision so the val metric matches the training objective.
        def _make_split(split):
            per_root = [
                LazyDataset(r, split, s[split],
                            learn_fixed_point=self.learn_fixed_point,
                            full_tail_supervision=self.full_tail_supervision)
                for r, s in zip(self._roots, splits)
            ]
            return per_root

        self.val_datasets = _make_split("val")
        self.test_datasets = _make_split("test")
        self.test_items_list = _make_split("test_items")
        self.test_items = self.test_items_list[-1]  

        if len(self._roots) > 1:
            print("\n----- Loaded datasets with metadata -----")
            for idx, info in enumerate(infos):
                print(f"Dataset {idx + 1}: {self.dataset_names[idx]}")
                for key, value in info["metadata"].items():
                    print(f" - {key}: {value}")
        else:
            print("\n----- Loaded dataset with metadata -----")
            for key, value in infos[0]["metadata"].items():
                print(f" - {key}: {value}")
        print()

    def _create_dataloader(self, data, shuffle):
        if self.overfitting_test:
            data = Subset(data, list(range(min(self.batch_size, len(data)))))
        return DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return [
            self._create_dataloader(d, shuffle=False) for d in self.val_datasets
        ]

    def test_dataloader(self):
        def _make(d):
            if self.overfitting_test:
                d = Subset(d, [0])
            return DataLoader(
                d, batch_size=1, shuffle=False, num_workers=self.num_workers
            )
        return [_make(d) for d in self.test_datasets]


class SOCPNetLightning(pl.LightningModule):
    """
    LightningModule wrapper for SOCPNet model.

    Handles training, validation, testing, and optimization for trajectory-based training.
    """

    # Valid values for --loss_type (the supervised trajectory loss).
    SUPERVISED_LOSS_NAMES = (
        "mse", "soft_dtw", "target_final", "target_k", "traj_logsre"
    )

    def __init__(
        self,
        d: int,
        T: int,
        lr: float,
        zero_everything: bool = False,
        model_type: str = "SOCPNet",
        message_passing_scheme: str = "three_step",
        teacher_forcing_p: float = 0.0,
        target_length: int | None = None,
        aggr: str = "mean",
        residual: bool = False,
        num_attention_heads: int = 0,
        message_passing_depth: int = 1,
        loss_type: str = "mse",
        train_loss_constraint: bool = False,
        train_loss_objective: bool = False,
        soft_dtw_gamma: float = 1.0,
        target_k_step: int = 5,
        train_loss_weight: list | None = None,
        train_loss_constraint_weight: list | None = None,
        train_loss_objective_weight: list | None = None,
        train_loss_fixed_point: float | None = None,
        scale_direction: bool = False,
        train_loss_scale_direction_weight: list | None = None,
        mp_decouple_soc_lc: bool = False,
        use_gru_update: bool = False,
        use_step_encoding: bool = False,
        use_iterate_features: bool = False,
        full_tail_supervision: bool = False,
        picard_weight: float = 0.0,
        picard_rho: float = 0.9,
        final_anchor_weight: float = 0.0,
        iterfeat_eps: float = 1.0,
        lr_schedule: str = "none",
        warmup_epochs: int = 0,
        max_epochs: int = 200,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        paths = kwargs["dataset_info"]["dataset_path"]
        self.val_step_outputs = [[] for x in range(len(paths))]
        self.trajectory_mean = kwargs["dataset_info"]["trajectory_mean"]
        self.trajectory_std = kwargs["dataset_info"]["trajectory_std"]

        self.T = T
        self.lr = lr
        self.zero_everything = zero_everything
        self.target_length = target_length
        self.teacher_forcing_p = teacher_forcing_p
        self.aggr = aggr
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.message_passing_depth = message_passing_depth
        assert loss_type in self.SUPERVISED_LOSS_NAMES, (
            f"--loss_type must be one of {self.SUPERVISED_LOSS_NAMES}, got {loss_type!r}"
        )
        self.loss_type = loss_type
        self.train_loss_constraint = train_loss_constraint
        self.train_loss_objective = train_loss_objective
        # Fixed-point consistency loss weight (None disables the term). Enforces
        # G(x*) = x*, i.e. the optimum is a stable fixed point of the unroll.
        self.train_loss_fixed_point = train_loss_fixed_point
        # Scale/direction residual readout + supervised log-step loss. The loss
        # matches the model's per-graph predicted log-step to the IPOPT log-step;
        # it requires the scale/direction architecture (enabled together).
        self.scale_direction = scale_direction
        self.soft_dtw_gamma = soft_dtw_gamma
        self.target_k_step = target_k_step

        # Optional per-timestep weighting (spec: [mode, *params]) for the
        # supervised, constraint and objective trajectory losses. Each spec is
        # stored as a (mode, params) tuple (None when disabled).
        self.train_loss_weight = train_loss_weight
        self.train_loss_constraint_weight = train_loss_constraint_weight
        self.train_loss_objective_weight = train_loss_objective_weight
        self._loss_weight_spec = self._parse_weight_spec(
            train_loss_weight, "train_loss_weight"
        )
        self._cons_weight_spec = self._parse_weight_spec(
            train_loss_constraint_weight, "train_loss_constraint_weight"
        )
        self._obj_weight_spec = self._parse_weight_spec(
            train_loss_objective_weight, "train_loss_objective_weight"
        )
        self.train_loss_scale_direction_weight = train_loss_scale_direction_weight
        self._scale_dir_weight_spec = self._parse_weight_spec(
            train_loss_scale_direction_weight, "scale_direction_loss"
        )
        if self._loss_weight_spec[0] is not None and loss_type == "soft_dtw":
            raise ValueError(
                "--train_loss_weight is not supported with --loss_type "
                "soft_dtw (no per-timestep loss to weight)."
            )
        self.mp_decouple_soc_lc = mp_decouple_soc_lc
        self.use_gru_update = use_gru_update
        self.use_step_encoding = use_step_encoding
        self.use_iterate_features = use_iterate_features

        # Picard penalty: relu(d_{t+1} - rho*d_t), with
        # d_t = ||x_t - x*|| / ||x*||. Each step must shrink that distance
        # by a factor rho < 1. final_anchor is the relative squared error of
        # the last iterate against x*.
        self.picard_weight = picard_weight
        self.picard_rho = picard_rho
        self.final_anchor_weight = final_anchor_weight
        # Epsilon in the inverse-slack features. At the default 1.0 the proxy
        # saturates near active constraints.
        self.iterfeat_eps = iterfeat_eps
        self.lr_schedule = lr_schedule
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        if model_type == "SOCPNet":
            self.model = SOCPNet(
                d=d,
                out_dim=1,
                T=T,
                message_passing_scheme=message_passing_scheme,
                aggr=aggr,
                num_attention_heads=num_attention_heads,
                message_passing_depth=message_passing_depth,
                mp_decouple_soc_lc=mp_decouple_soc_lc,
                use_gru_update=use_gru_update,
                use_step_encoding=use_step_encoding,
                use_iterate_features=use_iterate_features,
                scale_direction=scale_direction,
            )

    def _zero_everything(self, batch):
        """Zero out all features in a batch."""
        batch["var"].x.zero_()
        batch["norm"].x.zero_()
        batch["cons"].x.zero_()

        for edge_type in batch.edge_types:
            if "edge_attr" in batch[edge_type]:
                batch[edge_type].edge_attr.zero_()

        return batch

    def _initialize_node_features_with_x0(self, batch, x0):
        """Initialize variable node features with [obj_coeff, l_bound, r_bound, x0]."""
        # Graph has [obj_coeff, l_bound, r_bound] (3D), add x0 to make it 4D
        if batch["var"].x.shape[1] != 3:
            raise ValueError(
                f"Expected variable features with shape [N, 3] before adding x0, "
                f"but got shape {batch['var'].x.shape}"
            )
        batch["var"].x = torch.cat([batch["var"].x, x0], dim=1)
        return batch

    def _predict_one_step(
        self,
        x_pred,
        x_dict,
        edge_index_dict,
        edge_attr_dict,
        batch_dict,
        x_gt_t=None,
        step_idx=None,
        batch=None,
    ):
        # SOCPNet.forward accepts these kwargs. Other models do not.
        model_kwargs = {}
        if self.use_step_encoding or self.scale_direction:
            # The scale head also consumes the (normalized) iteration index.
            model_kwargs["step_idx"] = step_idx
        if self.use_iterate_features and batch is not None:
            # Local KKT info at the current iterate, computed from the problem's
            # parameters and the (de-normalized) current iterate prediction.
            x_true = self._denormalize_prediction(x_pred)
            model_kwargs["iterate_feat_dict"] = self._compute_iterate_features(
                batch, x_true
            )
        model_out = self.model(
            x_dict,
            edge_index_dict,
            edge_attr_dict,
            batch_dict,
            **model_kwargs,
        )
        scale = None
        direction = None
        if self.scale_direction:
            # Model returns (scale, direction, x_dict) explicitly in this mode.
            # r = exp(clamp(scale)) * unit_direction, applied as a residual step.
            scale_raw, readout, x_dict = model_out
            var_batch = batch_dict["var"]
            d = readout if readout.dim() == 2 else readout.unsqueeze(1)  # [N, 1]
            G = int(var_batch.max().item()) + 1
            sq = torch.zeros(G, device=d.device, dtype=d.dtype)
            sq.index_add_(0, var_batch, (d.view(-1) ** 2))
            norm = torch.sqrt(sq + 1e-12)[var_batch].unsqueeze(1)  # [N, 1]
            direction = d / norm
            scale = scale_raw.clamp(-20.0, 5.0)  # [N, 1], for the log-step loss
            step = torch.exp(scale) * direction
            xp = x_pred.to(device=d.device, dtype=d.dtype)
            if xp.dim() == 1:
                xp = xp.unsqueeze(1)
            x_pred = xp + step
        else:
            readout, x_dict = model_out
            if self.residual:
                xp = x_pred.to(device=readout.device, dtype=readout.dtype)
                if xp.dim() == 1:
                    xp = xp.unsqueeze(1)
                x_pred = xp + readout
            else:
                x_pred = readout
        x_in = x_pred
        if self.teacher_forcing_p > 0.0 and self.training:
            if x_gt_t.dim() == 1:
                x_gt_t = x_gt_t.unsqueeze(1)
            use_teacher_mask = (
                torch.rand(1, device=x_pred.device) < self.teacher_forcing_p
            )
            x_in = torch.where(use_teacher_mask, x_gt_t, x_pred)

        # Cross-iteration variable-node embedding update defined in the model
        h_u = self.model.feature_update(x_dict["var"], x_in)

        # Store updated embeddings for next iteration
        x_dict["var"] = h_u
        if self.scale_direction:
            return x_pred, x_dict, scale, direction
        return x_pred, x_dict

    def _predict_trajectory(self, batch):
        """
        Unroll the model for T steps. Requires ``batch.trajectory`` with at least
        ``T + 1`` time columns: ``x0`` at index 0 and ground-truth states for
        optional teacher forcing at indices ``1 .. T``.
        """
        trajectory = batch.trajectory
        x0 = trajectory[:, 0:1]
        batch = self._initialize_node_features_with_x0(batch, x0)

        if self.zero_everything:
            batch = self._zero_everything(batch)

        # Get initial node embeddings and edge features
        x_dict = self.model.initialize_embeddings(batch)
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict
        batch_dict = {
            nt: getattr(batch[nt], "batch", None)
            if nt in batch.node_types
            else None
            for nt in ["var", "norm", "cons", "lin_cons"]
        }

        x_pred = trajectory[:, 0:1]
        pred_list = []
        scale_list = []
        direction_list = []
        for t in range(self.T):
            if self.teacher_forcing_p > 0.0 and self.training:
                x_gt_t = trajectory[:, t + 1 : t + 2]
                x_gt_t = x_gt_t.to(device=x_pred.device, dtype=x_pred.dtype)

            else:
                x_gt_t = None
            if self.scale_direction:
                x_pred, x_dict, scale, direction = self._predict_one_step(
                    x_pred,
                    x_dict,
                    edge_index_dict,
                    edge_attr_dict,
                    batch_dict,
                    x_gt_t=x_gt_t,
                    step_idx=t,
                    batch=batch,
                )
                scale_list.append(scale)
                direction_list.append(direction)
            else:
                x_pred, x_dict = self._predict_one_step(
                    x_pred,
                    x_dict,
                    edge_index_dict,
                    edge_attr_dict,
                    batch_dict,
                    x_gt_t=x_gt_t,
                    step_idx=t,
                    batch=batch,
                )
            pred_list.append(x_pred)

        if self.scale_direction:
            return (
                torch.cat(pred_list, dim=-1),
                torch.cat(scale_list, dim=-1),
                torch.cat(direction_list, dim=-1),
            )
        return torch.cat(pred_list, dim=-1)

    def _fixed_point_loss(self, batch):
        """One-step fixed-point consistency loss (normalized space).

        Feeds the optimum x* (last trajectory column) as the current iterate
        through a single model step and penalizes any movement away from it:
        ``||G(x*) - x*||^2``. With residual readout this reduces to ``||delta||^2``.
        The optimum is supervised here as an *input*, which the normal
        trajectory loss never sees (it only feeds the rollout's own iterates),
        so this term teaches the unroll to *stay* at x* once reached.

        The 3-D variable features are restored afterwards so the subsequent
        trajectory unroll (which re-appends x0) is unaffected.
        """
        x_star = batch.trajectory[:, -1:]  # [N, 1] normalized optimum
        var_x_3d = batch["var"].x  # [N, 3] raw problem features (pre-x0)
        batch = self._initialize_node_features_with_x0(batch, x_star)  # -> [N, 4]
        try:
            x_dict = self.model.initialize_embeddings(batch)
            edge_index_dict = batch.edge_index_dict
            edge_attr_dict = batch.edge_attr_dict
            batch_dict = {
                nt: getattr(batch[nt], "batch", None)
                if nt in batch.node_types
                else None
                for nt in ["var", "norm", "cons", "lin_cons"]
            }
            # Only the prediction is needed here; in scale/direction mode the
            # step also returns (scale, direction), which we ignore.
            step_out = self._predict_one_step(
                x_star,
                x_dict,
                edge_index_dict,
                edge_attr_dict,
                batch_dict,
                x_gt_t=x_star,
                step_idx=self.T,
                batch=batch,
            )
            x_pred = step_out[0]
        finally:
            batch["var"].x = var_x_3d  # restore for the trajectory rollout
        if x_pred.dim() == 1:
            x_pred = x_pred.unsqueeze(1)
        return ((x_pred - x_star) ** 2).mean()

    def _scale_direction_loss(self, batch, var_batch, traj_mask, traj_scales):
        """Per-[G, T] squared error between predicted and IPOPT log-step.

        Target ``s*_t = log || x*_{t+1} - x*_t ||`` (normalized space), computed
        per graph from ``batch.trajectory``. ``traj_scales`` ``[N, T]`` is the
        per-graph scale collected during the unroll (constant within a graph, so
        read off the first node of each graph via ``ptr``). Returns the raw
        per-step squared error ``[G, T]``; the optional per-timestep weighting
        and mask reduction happen once in the caller via ``_weighted_masked_mean``.
        """
        traj = batch.trajectory  # [N, L] normalized
        T = traj_scales.shape[1]
        deltas = traj[:, 1 : T + 1] - traj[:, 0:T]  # [N, T]
        G = traj_mask.shape[0]
        step_sq = torch.zeros(G, T, device=deltas.device, dtype=deltas.dtype)
        step_sq.index_add_(0, var_batch, deltas ** 2)  # per-graph ||.||^2
        s_star = 0.5 * torch.log(step_sq + 1e-12)  # [G, T]

        ptr = batch["var"].ptr
        s_pred = traj_scales[ptr[:-1]].squeeze(-1)  # [G, T]
        if s_pred.dim() == 1:
            s_pred = s_pred.unsqueeze(-1)

        return (s_pred - s_star) ** 2

    def _log_metrics(self, prefix, metrics, batch_size, prog_bar_metric=None):
        """Helper to log metrics consistently."""
        for name, value in metrics.items():
            if value is None:
                continue
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(
                    name == prog_bar_metric
                    and not str(prefix).startswith("Test")
                ),
                sync_dist=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )

    def _get_raw_node_features(self, node_store):
        """Return raw (deprocessed) node features when available."""
        return getattr(node_store, "x_deprocessed", node_store.x)

    def _get_raw_edge_attr(self, edge_store):
        """Return raw (deprocessed) edge attributes when available."""
        return getattr(
            edge_store, "edge_attr_deprocessed", edge_store.edge_attr
        )

    def _get_batch_vector(self, node_store):
        """Return batch vector; fall back to zeros for single-graph batches."""
        if hasattr(node_store, "batch") and node_store.batch is not None:
            return node_store.batch
        return torch.zeros(
            node_store.x.shape[0], dtype=torch.long, device=node_store.x.device
        )

    @staticmethod
    def _slog(v):
        """Signed log1p: tames the wide dynamic range of raw problem-scale residuals."""
        return torch.sign(v) * torch.log1p(torch.abs(v))

    def _compute_iterate_features(self, batch, x_true):
        """Per-iteration local KKT features at the current iterate (raw scale).

        Implements local Newton/residual features and the inverse-slack
        dual estimates. All quantities are deterministic functions of the current
        iterate ``x_true`` and the (raw) problem data.
        Returns a dict of per-node-type tensors matching ``model.ITERATE_FEAT_DIMS``.

        - var:      [slog(x - l), slog(r - x), 1/(|x-l|+1), 1/(|r-x|+1)]
        - norm:     [slog(A_row x + b)]
        - cons:     [slog(soc_slack), 1/(|soc_slack|+1)]  with soc_slack=(c^T x+d)-||A x+b||
        - lin_cons: [slog(g - F x), 1/(|g-F x|+1)]
        Inverse-slack channels are interior-point complementarity dual proxies
        (lambda_i ~ mu / slack_i); they are large where a constraint/bound is near-active.
        """
        device = x_true.device
        x = x_true.view(-1)  # [N]
        eps = self.iterfeat_eps

        def dual_proxy(slack):
            return 1.0 / (slack.abs() + eps)

        # --- Variable bound slacks + bound-multiplier proxies ---
        var_raw = self._get_raw_node_features(batch["var"])
        l_bound = var_raw[:, 1]
        r_bound = var_raw[:, 2]
        lower_slack = x - l_bound
        upper_slack = r_bound - x
        # Guard against infinite/absent bounds (treated as inactive -> no signal).
        lower_slack = torch.nan_to_num(lower_slack, nan=0.0, posinf=1e6, neginf=-1e6)
        upper_slack = torch.nan_to_num(upper_slack, nan=0.0, posinf=1e6, neginf=-1e6)
        var_feat = torch.stack(
            [
                self._slog(lower_slack),
                self._slog(upper_slack),
                dual_proxy(lower_slack),
                dual_proxy(upper_slack),
            ],
            dim=1,
        )

        # --- Per-norm row residual s = A_row x + b ---
        norm_x_raw = self._get_raw_node_features(batch["norm"]).squeeze(-1)
        v2n = batch[("var", "V2N", "norm")]
        v2n_vals = self._get_raw_edge_attr(v2n).view(-1)
        var_idx = v2n.edge_index[0]
        norm_idx = v2n.edge_index[1]
        ax = torch.scatter_reduce(
            torch.zeros(norm_x_raw.shape[0], device=device),
            0, norm_idx, v2n_vals * x[var_idx], "sum",
        )
        s_k = ax + norm_x_raw
        norm_feat = self._slog(s_k).unsqueeze(1)

        # --- Per-constraint SOC slack: (c^T x + d) - ||A x + b|| ---
        cons_x_raw = self._get_raw_node_features(batch["cons"]).squeeze(-1)
        n2c = batch[("norm", "N2C", "cons")].edge_index
        s_sq_sum = torch.scatter_reduce(
            torch.zeros(cons_x_raw.shape[0], device=device),
            0, n2c[1], s_k[n2c[0]] ** 2, "sum",
        )
        # Additive epsilon keeps the gradient finite at ||A x + b|| = 0
        # (sqrt'(0) = inf would otherwise produce NaN gradients).
        lhs = torch.sqrt(torch.clamp(s_sq_sum, min=0.0) + 1e-12)
        v2c = batch[("var", "V2C", "cons")]
        v2c_vals = self._get_raw_edge_attr(v2c).view(-1)
        cx = torch.scatter_reduce(
            torch.zeros(cons_x_raw.shape[0], device=device),
            0, v2c.edge_index[1], v2c_vals * x[v2c.edge_index[0]], "sum",
        )
        soc_slack = (cx + cons_x_raw) - lhs
        cons_feat = torch.stack(
            [self._slog(soc_slack), dual_proxy(soc_slack)], dim=1
        )

        # --- Per-linear-constraint slack: g - F x ---
        lin_x_raw = self._get_raw_node_features(batch["lin_cons"]).squeeze(-1)
        v2lc = batch[("var", "V2LC", "lin_cons")]
        v2lc_vals = self._get_raw_edge_attr(v2lc).view(-1)
        fx = torch.scatter_reduce(
            torch.zeros(lin_x_raw.shape[0], device=device),
            0, v2lc.edge_index[1], v2lc_vals * x[v2lc.edge_index[0]], "sum",
        )
        lin_slack = lin_x_raw - fx
        lin_feat = torch.stack(
            [self._slog(lin_slack), dual_proxy(lin_slack)], dim=1
        )

        return {
            "var": var_feat,
            "norm": norm_feat,
            "cons": cons_feat,
            "lin_cons": lin_feat,
        }

    def get_batch_objective(self, batch, x_true):
        """Compute the linear objective f^T x per graph in the batch (true scale).

        Args:
            batch: heterogeneous graph batch.
            x_true: per-var-node values, [N] (single point) or [N, T]
                (whole trajectory).

        Returns:
            objective: [B] for [N] input, or [B, T] for [N, T] input.
        """
        squeeze = x_true.dim() == 1
        x = x_true.unsqueeze(-1) if squeeze else x_true  # [N, T]
        T = x.shape[1]
        var_x_raw = self._get_raw_node_features(batch["var"])
        f = var_x_raw[:, 0].unsqueeze(-1)  # [N, 1]
        var_batch = self._get_batch_vector(batch["var"])
        batch_size = (
            int(var_batch.max().item()) + 1 if var_batch.numel() > 0 else 1
        )

        obj = torch.scatter_reduce(
            torch.zeros(batch_size, T, device=x.device),
            0,
            var_batch.unsqueeze(-1).expand(-1, T),
            f * x,
            "sum",
        )  # [B, T]
        return obj.squeeze(-1) if squeeze else obj

    def compute_objective_mse_loss(
        self, batch, x_pred_final_true, x_target_final_true
    ):
        """Compute objective loss on the final prediction point (true scale)."""
        obj_pred = self.get_batch_objective(batch, x_pred_final_true)
        obj_target = self.get_batch_objective(batch, x_target_final_true)
        return torch.mean((obj_pred - obj_target) ** 2)

    def get_constraint_violations(self, batch, x):
        """Per-constraint violations along a trajectory (true scale).

        Args:
            batch: heterogeneous graph batch.
            x: per-var-node values [N, T] (whole trajectory).

        Returns:
            violation_soc: [num_cons, T]      relu(||A x + b|| - (c^T x + d))
            violation_lin: [num_lin_cons, T]  relu(F x - g)
        """
        assert x.dim() == 2, "get_constraint_violations expects x of shape [N, T]"
        device = x.device
        T = x.shape[1]

        def _scatter(num_rows, index, src):  # index [E], src [E, T] -> [rows, T]
            return torch.scatter_reduce(
                torch.zeros(num_rows, T, device=device),
                0,
                index.unsqueeze(-1).expand(-1, T),
                src,
                "sum",
            )

        norm_x_raw = self._get_raw_node_features(batch["norm"]).squeeze(-1)
        cons_x_raw = self._get_raw_node_features(batch["cons"]).squeeze(-1)
        lin_cons_x_raw = self._get_raw_node_features(
            batch["lin_cons"]
        ).squeeze(-1)

        # A @ x + b (norm nodes)
        v2n = batch[("var", "V2N", "norm")]
        v2n_vals = self._get_raw_edge_attr(v2n).view(-1, 1)
        var_idx, norm_idx = v2n.edge_index
        ax = _scatter(norm_x_raw.shape[0], norm_idx, v2n_vals * x[var_idx])
        s_k = ax + norm_x_raw.unsqueeze(-1)  # [num_norm, T]

        # Map norms to constraints and compute SOC LHS per constraint
        norm_idx_n2c, cons_idx_n2c = batch[("norm", "N2C", "cons")].edge_index
        s_sq_sum = _scatter(
            cons_x_raw.shape[0], cons_idx_n2c, s_k[norm_idx_n2c] ** 2
        )
        # Additive epsilon keeps the gradient finite at ||A x + b|| = 0
        # (sqrt'(0) = inf would otherwise produce NaN gradients).
        lhs = torch.sqrt(torch.clamp(s_sq_sum, min=0.0) + 1e-12)

        # c^T x + d (constraint nodes)
        v2c = batch[("var", "V2C", "cons")]
        v2c_vals = self._get_raw_edge_attr(v2c).view(-1, 1)
        var_idx, cons_idx = v2c.edge_index
        cx = _scatter(cons_x_raw.shape[0], cons_idx, v2c_vals * x[var_idx])
        rhs = cx + cons_x_raw.unsqueeze(-1)
        violation_soc = torch.relu(lhs - rhs)  # [num_cons, T]

        # Linear constraints: F x <= g
        v2lc = batch[("var", "V2LC", "lin_cons")]
        v2lc_vals = self._get_raw_edge_attr(v2lc).view(-1, 1)
        var_idx, lc_idx = v2lc.edge_index
        fx = _scatter(lin_cons_x_raw.shape[0], lc_idx, v2lc_vals * x[var_idx])
        violation_lin = torch.relu(fx - lin_cons_x_raw.unsqueeze(-1))

        return violation_soc, violation_lin

    def _signed_constraints(self, batch, x):
        """Signed constraint values, feasible when g(x) <= 0. x is [N, T].

        Same quantities as ``get_constraint_violations``, without the relu.
        Returns g_soc [num_cons, T] = ||Ax+b|| - (c^T x + d) and
        g_lin [num_lin_cons, T] = Fx - g.
        """
        device = x.device
        T = x.shape[1]

        def _scatter(num_rows, index, src):
            return torch.scatter_reduce(
                torch.zeros(num_rows, T, device=device, dtype=x.dtype),
                0, index.unsqueeze(-1).expand(-1, T), src, "sum",
            )

        norm_x_raw = self._get_raw_node_features(batch["norm"]).squeeze(-1)
        cons_x_raw = self._get_raw_node_features(batch["cons"]).squeeze(-1)
        lin_cons_x_raw = self._get_raw_node_features(batch["lin_cons"]).squeeze(-1)

        v2n = batch[("var", "V2N", "norm")]
        v2n_vals = self._get_raw_edge_attr(v2n).view(-1, 1)
        var_idx, norm_idx = v2n.edge_index
        ax = _scatter(norm_x_raw.shape[0], norm_idx, v2n_vals * x[var_idx])
        s_k = ax + norm_x_raw.unsqueeze(-1)

        norm_idx_n2c, cons_idx_n2c = batch[("norm", "N2C", "cons")].edge_index
        s_sq_sum = _scatter(cons_x_raw.shape[0], cons_idx_n2c, s_k[norm_idx_n2c] ** 2)
        lhs = torch.sqrt(torch.clamp(s_sq_sum, min=0.0) + 1e-12)

        v2c = batch[("var", "V2C", "cons")]
        v2c_vals = self._get_raw_edge_attr(v2c).view(-1, 1)
        var_idx, cons_idx = v2c.edge_index
        cx = _scatter(cons_x_raw.shape[0], cons_idx, v2c_vals * x[var_idx])
        g_soc = lhs - (cx + cons_x_raw.unsqueeze(-1))  # [num_cons, T]

        v2lc = batch[("var", "V2LC", "lin_cons")]
        v2lc_vals = self._get_raw_edge_attr(v2lc).view(-1, 1)
        var_idx, lc_idx = v2lc.edge_index
        fx = _scatter(lin_cons_x_raw.shape[0], lc_idx, v2lc_vals * x[var_idx])
        g_lin = fx - lin_cons_x_raw.unsqueeze(-1)  # [num_lin, T]

        return g_soc, g_lin

    def _feasibility_restore(self, batch, x_norm, steps, alpha, clip, collect=False):
        """Gradient steps on the final iterate that reduce constraint violation.

        Minimizes V(x) = sum relu(g_soc)^2 + sum relu(g_lin)^2 by
        x <- x - clip(alpha * grad V), with g <= 0 feasible. A point that is
        already feasible (grad V = 0) is left unchanged. Runs in true scale
        and is detached from the training graph.

        Returns (restored_norm [N, 1], list of normalized iterates when collect).
        """
        std, mean = self.trajectory_std, self.trajectory_mean
        x_true = self._denormalize_prediction(x_norm).view(-1, 1)
        iterates = []
        for _ in range(steps):
            with torch.enable_grad():
                x_leaf = x_true.detach().requires_grad_(True)
                g_soc, g_lin = self._signed_constraints(batch, x_leaf)
                V = (torch.relu(g_soc) ** 2).sum() + (torch.relu(g_lin) ** 2).sum()
                grad = torch.autograd.grad(V, x_leaf)[0]
            step = torch.clamp(alpha * grad, -clip, clip)
            x_true = x_true - step
            if collect:
                iterates.append((x_true - mean) / std)
        return (x_true - mean) / std, iterates

    def compute_constraint_violation_loss(self, batch, x_pred_true):
        """Mean per-constraint violation (true scale).

        x_pred_true: [N] -> scalar (mean over graphs); [N, T] -> per-graph,
        per-timestep violations [B, T] (left unreduced for masking/weighting).
        """
        device = x_pred_true.device
        squeeze = x_pred_true.dim() == 1
        x = x_pred_true.unsqueeze(-1) if squeeze else x_pred_true  # [N, T]
        violation_soc, violation_lin = self.get_constraint_violations(batch, x)
        T = x.shape[1]

        cons_batch = self._get_batch_vector(batch["cons"])
        lin_cons_batch = self._get_batch_vector(batch["lin_cons"])
        batch_size = (
            int(
                max(
                    cons_batch.max().item() if cons_batch.numel() > 0 else 0,
                    lin_cons_batch.max().item()
                    if lin_cons_batch.numel() > 0
                    else 0,
                )
            )
            + 1
        )

        def _sum(index, src):  # index [E], src [E, T] -> [batch_size, T]
            return torch.scatter_reduce(
                torch.zeros(batch_size, T, device=device),
                0,
                index.unsqueeze(-1).expand(-1, T),
                src,
                "sum",
            )

        cons_sum = _sum(cons_batch, violation_soc)
        lin_sum = _sum(lin_cons_batch, violation_lin)
        cons_count = _sum(cons_batch, torch.ones_like(violation_soc))
        lin_count = _sum(lin_cons_batch, torch.ones_like(violation_lin))
        total_count = torch.clamp(cons_count + lin_count, min=1.0)
        per_graph_violation = (cons_sum + lin_sum) / total_count  # [B, T]
        return per_graph_violation.mean() if squeeze else per_graph_violation

    def _compute_soft_dtw_loss(self, pred, targets, traj_mask, batch_size, var_batch):
        """
        Soft-DTW loss between predicted and ground-truth trajectories.

        Order-preserving elastic alignment: the predicted trajectory may lag or
        lead the ground-truth by a few steps without being penalised for timing
        mismatch, while temporal ordering is still respected.

        Each graph's trajectory is treated as a multivariate sequence in R^n_dim
        (one iterate = one point), and Soft-DTW is computed per graph then averaged.

        Args:
            pred:        [N, T] predicted trajectory (normalised)
            targets:     [N, T] ground-truth trajectory (normalised)
            traj_mask:   [B, T] float mask (1 = real step, 0 = padded)
            batch_size:  int B
            var_batch:   [N] batch assignment for variable nodes

        Returns:
            Scalar Soft-DTW loss.
        """
        try:
            from tslearn.metrics import SoftDTWLossPyTorch
        except ImportError:
            raise ImportError(
                "tslearn is required for loss_type='soft_dtw'. "
                "Install with: pip install tslearn"
            )

        sdtw = SoftDTWLossPyTorch(gamma=self.soft_dtw_gamma, normalize=False)
        losses = []
        for b in range(batch_size):
            var_mask = var_batch == b
            if var_mask.sum() == 0:
                continue
            real_len = int((traj_mask[b] > 0).sum().item())
            if real_len == 0:
                continue
            # [n_dim, T_real] -> transpose to [T_real, n_dim] -> add batch dim
            pred_b = pred[var_mask, :real_len].T.unsqueeze(0)    # [1, T_real, n_dim]
            target_b = targets[var_mask, :real_len].T.unsqueeze(0)  # [1, T_real, n_dim]
            losses.append(sdtw(pred_b, target_b).squeeze())
        if not losses:
            return pred.sum() * 0.0  # differentiable zero
        return torch.stack(losses).mean()

    def _objective_gap_loss(self, batch, pred_true, x_star_true):
        """Scale-invariant squared objective gap per graph and timestep.

            (c^T x_t - c^T x*)^2 / max(1, (c^T x*)^2)

        Normalising by the optimal objective makes the term scale-invariant
        across problems (==0 exactly at the optimal objective value).

        Returns:
            [B, T] unreduced gap (masking/weighting/reduction done by caller).
        """
        obj_star = self.get_batch_objective(batch, x_star_true)  # [B]
        denom = torch.clamp(obj_star**2, min=1.0)  # [B]
        obj_pred = self.get_batch_objective(batch, pred_true)  # [B, T]
        return (obj_pred - obj_star.unsqueeze(-1)) ** 2 / denom.unsqueeze(-1)

    @staticmethod
    def _parse_weight_spec(spec, name):
        """Parse a [mode, *params] weight spec into (mode, params) or (None, None)."""
        if not spec:
            return None, None
        mode, *params = spec
        params = [float(p) for p in params]
        if mode == "constant":
            assert len(params) == 1, (
                f"--{name} constant expects 1 value, got {len(params)}"
            )
        elif mode == "linear":
            assert len(params) == 2, (
                f"--{name} linear expects 2 values (start end), got {len(params)}"
            )
        else:
            raise ValueError(
                f"Unknown --{name} mode {mode!r} (expected 'constant' or 'linear')"
            )
        return mode, params

    def _weighted_masked_mean(self, per_step, spec, mask):
        """Masked mean of a [*, T] loss with optional per-timestep weighting.

        spec is a (mode, params) tuple; when mode is None no weighting is
        applied. The weight vector [1, T] multiplies the mask before the
        sum/normalize, so the reduction happens once at the end.
        """
        mode, params = spec
        weight = mask
        if mode is not None:
            T = per_step.shape[-1]
            kw = {"device": per_step.device, "dtype": per_step.dtype}
            if mode == "constant":
                w = torch.full((T,), params[0], **kw)
            else:  # "linear"
                start, end = params
                w = torch.linspace(start, end, T, **kw)
            weight = mask * w.unsqueeze(0)
        return (per_step * weight).sum() / mask.sum()

    def _common_step(self, batch, prefix):
        """Common step for train/val/test with masked loss computation."""

        # Fixed-point consistency must run before the unroll mutates batch.var.x.
        fixed_point_loss = (
            self._fixed_point_loss(batch)
            if (
                self.train_loss_fixed_point is not None
                and prefix in ["Train", "Validation"]
            )
            else None
        )

        if self.scale_direction:
            pred, traj_scales, _ = self._predict_trajectory(batch)
        else:
            pred = self._predict_trajectory(batch)
            traj_scales = None
        targets = batch.trajectory[
            :, 1:
        ]  # [N, T] - skip x0 (N = batch_size * n_dim)
        batch_size = (
            batch["var"].ptr.shape[0] - 1
            if hasattr(batch["var"], "ptr")
            else 1
        )

        # Batch assignment for each variable node (used throughout).
        var_batch = batch["var"].batch  # [N]

        # Get trajectory mask and expand to node level
        # PyG concatenates trajectory_mask into [B*L], reshape to [B, L]
        L = batch.trajectory.shape[1]  # target_length
        traj_mask_full = batch.trajectory_mask.view(batch_size, L)  # [B, L]
        traj_mask = traj_mask_full[:, 1:]  # [B, T] - skip x0
        # Expand graph-level mask [B, T] to node-level [N, T] using batch assignment
        node_mask = traj_mask[var_batch]  # [N, T]

        # Last real timestep of each sample (last index where traj_mask is 1).
        last_real_timestep = torch.zeros(
            batch_size, dtype=torch.long, device=traj_mask.device
        )
        for b in range(batch_size):
            real_indices = torch.where(traj_mask[b] > 0)[0]
            last_real_timestep[b] = real_indices[-1].item()

        # Compute masked MSE/MAE loss (normalized by number of real steps)
        diff = pred - targets
        diff_sq = diff**2
        num_real_elements = node_mask.sum()
        mse_loss = (diff_sq * node_mask).sum() / num_real_elements
        mae_loss = (torch.abs(diff) * node_mask).sum() / num_real_elements

        # Denormalize the trajectories
        if self.trainer.datamodule.normalize:
            pred_true = self.trainer.datamodule.denormalize_trajectory(pred)
            targets_true = self.trainer.datamodule.denormalize_trajectory(
                targets
            )
        else:
            pred_true, targets_true = pred, targets
        diff_true = pred_true - targets_true

        # Paper metric: ||x̂ - x*||₂² / max(1, ||x*||₂²)
        # Use ONLY the final prediction point (last timestep), no averaging over iterations.
        diff_true_last = diff_true[:, -1]
        targets_true_last = targets_true[:, -1]
        pred_true_last = targets_true_last + diff_true_last
        zeros = torch.zeros((batch_size,), device=pred.device)
        diff_sq_norm = torch.scatter_reduce(
            zeros, 0, var_batch, diff_true_last**2, "sum"
        )
        target_sq_norm = torch.scatter_reduce(
            zeros, 0, var_batch, targets_true_last**2, "sum"
        )
        paper_metric_per_graph = diff_sq_norm / torch.clamp(
            target_sq_norm, min=1.0
        )
        paper_metric = paper_metric_per_graph.mean()

        # Objective/constraint losses on final point (true scale)
        objective_loss_final = self.compute_objective_mse_loss(
            batch, pred_true_last, targets_true_last
        )
        constraint_loss_final = self.compute_constraint_violation_loss(
            batch, pred_true_last
        )

        # Constraint loss at the last real timestep of each sample.
        pred_last_real = torch.zeros_like(pred_true[:, 0])
        for b in range(batch_size):
            var_mask = var_batch == b
            if var_mask.sum() == 0:
                continue
            last_t = last_real_timestep[b].item()
            pred_last_real[var_mask] = pred_true[var_mask, last_t]

        # Compute constraint loss on last real timestep predictions
        constraint_loss_last_real = self.compute_constraint_violation_loss(
            batch, pred_last_real
        )

        # Constraint violation across the whole trajectory (true scale), per
        # graph and timestep [B, T], with optional per-timestep weighting then
        # masked-mean over real steps.
        cons_per_step = self.compute_constraint_violation_loss(batch, pred_true)
        constraint_loss_traj = self._weighted_masked_mean(
            cons_per_step, self._cons_weight_spec, traj_mask
        )

        # Objective gap across the whole trajectory (true scale), per graph and
        # timestep [B, T], with optional per-timestep weighting then masked-mean
        # over real steps.
        obj_per_step = self._objective_gap_loss(
            batch, pred_true, targets_true_last
        )
        objective_loss_traj = self._weighted_masked_mean(
            obj_per_step, self._obj_weight_spec, traj_mask
        )

        # Per-graph, per-step relative squared error SRE^2 = ||x_t-x*||^2/||x*||^2,
        # reused by both the log-SRE loss and the Picard contractivity penalty.
        T_steps = pred_true.shape[1]
        x_star_rel = targets_true_last  # [N]
        d_rel = pred_true - x_star_rel.unsqueeze(1)  # [N, T]
        idx_bt = var_batch.unsqueeze(-1).expand(-1, T_steps)
        num_rel = torch.scatter_reduce(
            torch.zeros(batch_size, T_steps, device=pred.device, dtype=pred.dtype),
            0, idx_bt, d_rel ** 2, "sum",
        )  # [B, T]
        den_rel = torch.scatter_reduce(
            torch.zeros(batch_size, device=pred.device, dtype=pred.dtype),
            0, var_batch, x_star_rel ** 2, "sum",
        )
        den_rel = torch.clamp(den_rel, min=1.0).unsqueeze(-1)  # [B, 1]
        rel_per_step = num_rel / den_rel  # [B, T] == SRE^2

        # Supervised trajectory loss.
        if self.loss_type == "soft_dtw":
            traj_loss = self._compute_soft_dtw_loss(
                pred, targets, traj_mask, batch_size, var_batch
            )
        else:
            if self.loss_type == "mse":
                sq_err = diff_sq
            elif self.loss_type == "target_final":
                # Every predicted iterate compared to the final optimal solution x*.
                x_star = targets_true_last
                if self.trainer.datamodule.normalize:
                    mean = torch.tensor(
                        self.trajectory_mean, device=x_star.device, dtype=x_star.dtype
                    )
                    std = torch.tensor(
                        self.trajectory_std, device=x_star.device, dtype=x_star.dtype
                    )
                    x_star = (x_star - mean) / std
                sq_err = (pred - x_star.unsqueeze(1).expand_as(pred)) ** 2
            elif self.loss_type == "target_k":
                # Prediction at step t compared to the ground-truth iterate at t+k.
                idx = torch.arange(self.T, device=pred.device) + self.target_k_step
                idx = idx.clamp(max=self.T - 1)
                sq_err = (pred - targets[:, idx]) ** 2

            if self.loss_type == "traj_logsre":
                # Per-iterate log relative SRE: 0.5*log(SRE^2). Log makes each order of
                # magnitude of error weigh equally => gradient stays alive below 1e-2.
                logsre_per_step = 0.5 * torch.log(rel_per_step + 1e-12)  # [B, T]
                traj_loss = self._weighted_masked_mean(
                    logsre_per_step, self._loss_weight_spec, traj_mask
                )
            else:
                # Optional per-timestep weighting, then mask + reduce once.
                traj_loss = self._weighted_masked_mean(
                    sq_err, self._loss_weight_spec, node_mask
                )

        # Scale/direction log-step supervision: match the predicted per-graph
        # log-step to the IPOPT log-step magnitude. Per-step error [G, T] with
        # optional constant/linear per-timestep weighting, masked-mean reduced.
        scale_direction_loss = (
            self._weighted_masked_mean(
                self._scale_direction_loss(batch, var_batch, traj_mask, traj_scales),
                self._scale_dir_weight_spec,
                traj_mask,
            )
            if (
                self.scale_direction
                and self._scale_dir_weight_spec[0] is not None
                and prefix in ["Train", "Validation"]
            )
            else None
        )

        # Optional additive loss terms.
        total_loss = traj_loss
        if self.train_loss_constraint:
            total_loss = total_loss + constraint_loss_traj
        if self.train_loss_objective:
            total_loss = total_loss + objective_loss_traj
        if fixed_point_loss is not None:
            total_loss = total_loss + self.train_loss_fixed_point * fixed_point_loss
        if scale_direction_loss is not None:
            total_loss = total_loss + scale_direction_loss

        # ---- Contractivity (Picard) penalty ----
        # d_t = sqrt(SRE^2_t) = relative distance to x*. Penalize any step that
        # fails to shrink the distance by factor rho: relu(d_{t+1} - rho*d_t).
        # Forces a contraction => distance (and thus step size) collapses toward
        # the optimum, mirroring IPOPT's rapidly vanishing steps.
        picard_loss = None
        if self.picard_weight > 0.0:
            dist = torch.sqrt(rel_per_step + 1e-12)  # [B, T]
            gap = torch.relu(
                dist[:, 1:] - self.picard_rho * dist[:, :-1]
            )  # [B, T-1]
            step_mask = traj_mask[:, 1:] * traj_mask[:, :-1]
            picard_loss = (gap * step_mask).sum() / torch.clamp(
                step_mask.sum(), min=1.0
            )
            total_loss = total_loss + self.picard_weight * picard_loss

        # Relative squared error of the last real iterate against x*.
        final_anchor_loss = None
        if self.final_anchor_weight > 0.0:
            last_idx = (traj_mask.sum(dim=1).long() - 1).clamp(min=0)  # [B]
            final_anchor_loss = rel_per_step.gather(
                1, last_idx.unsqueeze(1)
            ).mean()
            total_loss = total_loss + self.final_anchor_weight * final_anchor_loss

        if prefix not in ["Train", "Validation"]:
            total_loss = None

        # Log metrics. Some are on the normalized scale, some on the true scale.
        metrics = {
            "loss_traj": traj_loss,
            "loss_mse_normalized": mse_loss,
            "loss_mae_normalized": mae_loss,
            "paper_metric_true": paper_metric,  # Always on true scale
            "loss_objective_final_true": objective_loss_final,
            "loss_constraint_final_true": constraint_loss_final,
            "loss_constraint_last_real": constraint_loss_last_real,
            "loss_constraint_traj": constraint_loss_traj,
            "loss_objective_traj": objective_loss_traj,
            "loss_fixed_point": fixed_point_loss,
            "loss_scale_direction": scale_direction_loss,
            "loss_picard": picard_loss,
            "loss_final_anchor": final_anchor_loss,
            "total_loss": total_loss
                if prefix in ["Train", "Validation"]
                else None,
        }

        # Denormalized metrics for validation/test
        if prefix in ["Validation", "Test"]:
            # Masked metrics
            mse_true = ((diff_true**2) * node_mask).sum() / num_real_elements
            mae_true = (
                torch.abs(diff_true) * node_mask
            ).sum() / num_real_elements
            rel_true = (
                (
                    torch.abs(diff_true)
                    / torch.clamp(torch.abs(targets_true), min=1e-8)
                )
                * node_mask
            ).sum() / num_real_elements

            metrics.update(
                {
                    "loss_mse_true": mse_true,
                    "loss_mae_true": mae_true,
                    "loss_relative_true": rel_true,
                }
            )

        return metrics, batch_size 

    def training_step(self, batch, batch_idx):
        metrics, batch_size = self._common_step(batch, "Train")
        self._log_metrics(
            "Train",
            metrics,
            batch_size,
            prog_bar_metric="loss_traj",
        )

        return metrics["total_loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        names = self.trainer.datamodule.dataset_names
        subdir = names[dataloader_idx]
        metrics, batch_size = self._common_step(batch, "Validation")
        val_loss = metrics["total_loss"]
        self._log_metrics(
            f"Validation/{subdir}",
            metrics,
            batch_size,
            prog_bar_metric="paper_metric_true",
        )
        kw = dict(
            on_step=False, on_epoch=True, sync_dist=True,
            batch_size=batch_size, add_dataloader_idx=False,
        )
        self.log(f"val_loss_{subdir}", val_loss, **kw)
        # Collect paper_metric_true (true-scale relative error) for on_validation_epoch_end()
        self.val_step_outputs[dataloader_idx].append(
            metrics["paper_metric_true"].detach()
        )
        return val_loss

    def on_validation_epoch_end(self):
        flat_list = []
        for idx in range(len(self.val_step_outputs)):
            flat_list.extend(self.val_step_outputs[idx])
            self.val_step_outputs[idx].clear()
        if not flat_list:
            return
        avg_loss = torch.stack(flat_list).mean()
        self.log(
            "average_val_loss",  # average paper_metric_true across all val datasets
            avg_loss,
            prog_bar=True,
            sync_dist=True,
        )

    def _normalize_x0(self, x0_raw):
        """Normalize x0 (scalar trajectory mean/std from preprocess)."""
        return (x0_raw - self.trajectory_mean) / self.trajectory_std

    def _denormalize_prediction(self, pred_k):
        """Denormalize one prediction step (same scalar mean/std every step)."""
        return pred_k * self.trajectory_std + self.trajectory_mean

    def predict(self, batch, n_iterations):
        """
        Predict optimization trajectory by running the model iteratively for a
        fixed number of iterations.

        For each test sample:
        1. Extract x0 and optimal solution from raw trajectory
        2. Run model iteratively for n_iterations steps
        3. Store x (optimization variable) iterates
        """

        # Initialize model with normalized x0
        x0_raw = batch.trajectory[:, :1]
        x0 = self._normalize_x0(x0_raw)
        batch = self._initialize_node_features_with_x0(batch, x0)
        if self.zero_everything:
            batch = self._zero_everything(batch)

        x_dict = self.model.initialize_embeddings(batch)
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict
        batch_dict = {
            nt: getattr(batch[nt], "batch", None)
            for nt in ["var", "norm", "cons", "lin_cons"]
        }

        # Iterate for n_iterations steps, starting from x0
        x_pred = x0
        N = x0_raw.shape[0]  # number of variables
        pred_trajectory = torch.zeros(N, n_iterations + 1, device=x0_raw.device, dtype=x0_raw.dtype)
        pred_trajectory[:, 0] = x0_raw.squeeze(1)
        for k in range(n_iterations):
            # In scale/direction mode the step also returns (scale, direction);
            # only the prediction and updated embeddings are needed here.
            step_out = self._predict_one_step(
                x_pred, x_dict, edge_index_dict, edge_attr_dict, batch_dict,
                step_idx=k,
                batch=batch,
            )
            x_pred, x_dict = step_out[0], step_out[1]
            pred_trajectory[:, k + 1] = self._denormalize_prediction(x_pred).squeeze(1)

        return pred_trajectory

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.lr)
        if self.lr_schedule != "cosine":
            return opt
        # Linear warmup for warmup_epochs, then cosine anneal to ~0 over the rest.
        warmup = max(int(self.warmup_epochs), 0)
        total = max(int(self.max_epochs), warmup + 1)

        def lr_lambda(epoch):
            if warmup > 0 and epoch < warmup:
                return float(epoch + 1) / float(warmup)
            progress = (epoch - warmup) / max(1, total - warmup)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }


class SOCPNetend2endLightning(SOCPNetLightning):
    """
    End-to-end optimal solution prediction Lightning module.

    Inherits from SOCPNetLightning but overrides:
    - _predict_trajectory(): Uses 2 MP layers + readout (no trajectory unrolling)
    - _common_step(): Computes loss only against final optimal solution
    """

    def __init__(
        self,
        d: int,
        lr: float,
        zero_everything: bool = False,
        aggr: str = "mean",
        num_attention_heads: int = 0,
        mp_decouple_soc_lc: bool = False,
        **kwargs,
    ):
        # Remove conflicting arguments from kwargs (from checkpoint) to avoid duplicate argument errors
        kwargs.pop("T", None)
        kwargs.pop("model_type", None)
        kwargs.pop("message_passing_scheme", None)
        kwargs.pop("target_length", None)
        # The parent requires T. End-to-end mode does not unroll, so the value is unused.
        super().__init__(
            d=d,
            T=1,
            lr=lr,
            zero_everything=zero_everything,
            model_type="SOCPNetend2end",
            message_passing_scheme="three_step",  # Not used but required
            target_length=None,
            aggr=aggr,
            num_attention_heads=num_attention_heads,
            mp_decouple_soc_lc=mp_decouple_soc_lc,
            **kwargs,
        )

        # Replace model with end-to-end version
        self.model = SOCPNetend2end(
            d=d,
            out_dim=1,
            aggr=aggr,
            num_attention_heads=num_attention_heads,
            mp_decouple_soc_lc=mp_decouple_soc_lc,
        )

    def _predict_trajectory(self, batch):
        """
        End-to-end prediction: 2 MP layers + readout.

        Does NOT unroll trajectory, does NOT add x0, does NOT feed predictions back.
        """
        # Do NOT add x0 - use graph features as-is (3D: [obj_coeff, l_bound, r_bound])
        if self.zero_everything:
            batch = self._zero_everything(batch)

        # Get initial node embeddings and edge features
        x_dict = self.model.initialize_embeddings(batch)
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict
        batch_dict = {
            nt: getattr(batch[nt], "batch", None)
            if nt in batch.node_types
            else None
            for nt in ["var", "norm", "cons", "lin_cons"]
        }

        # Run model: 2 MP layers + readout (all in one forward pass)
        x_pred = self.model(
            x_dict, edge_index_dict, edge_attr_dict, batch_dict
        )

        # Return predictions [N_total, 1] - squeeze to [N_total] for consistency
        return x_pred.squeeze(-1)  # [N_total]

    def _common_step(self, batch, prefix, validation_subdir=None):
        """
        End-to-end training step: compute loss only against final optimal solution.

        Uses paper metric: ||x̂ - x*||₂² / max(1, ||x*||₂²)
        """
        if prefix == "Validation" and validation_subdir:
            log_prefix = f"Validation/{validation_subdir}"
        else:
            log_prefix = prefix

        # Get predictions [N_total]
        x_pred = self._predict_trajectory(batch)

        # Get optimal solution labels
        var_batch = batch["var"].batch

        # x_opt is a graph attribute. Trajectory preprocessing stores it as
        # [N, 1]; the end-to-end pass stores it as [N].
        if hasattr(batch, "x_opt") and batch.x_opt is not None:
            x_opt = batch.x_opt
            if x_opt.dim() == 2 and x_opt.shape[-1] == 1:
                x_opt = x_opt.squeeze(-1)
            # One value per graph: expand to one value per variable node.
            if x_opt.shape[0] < len(var_batch):
                x_opt = x_opt[var_batch]
        else:
            raise ValueError("x_opt is not on the batch.")

        # Denormalize predictions and targets for paper metric (always use true scale)
        batch_size = (
            batch["var"].ptr.shape[0] - 1
            if hasattr(batch["var"], "ptr")
            else 1
        )

        if self.trainer.datamodule.normalize:
            # Denormalize for paper metric (always compute on true scale)
            datamodule = self.trainer.datamodule
            if (
                datamodule.x_opt_mean is not None
                and datamodule.x_opt_std is not None
            ):
                x_pred_true = (
                    x_pred * datamodule.x_opt_std + datamodule.x_opt_mean
                )
                x_opt_true = (
                    x_opt * datamodule.x_opt_std + datamodule.x_opt_mean
                )
            else:
                x_pred_true = x_pred
                x_opt_true = x_opt
        else:
            x_pred_true = x_pred
            x_opt_true = x_opt

        # Compute paper metric on true scale: ||x̂ - x*||₂² / max(1, ||x*||₂²)
        # Per graph, then average
        diff_true = x_pred_true - x_opt_true  # [N_total]
        diff_sq_true = diff_true**2  # [N_total]
        x_opt_sq_true = x_opt_true**2  # [N_total]

        # Aggregate per graph
        zeros = torch.zeros(batch_size, device=x_pred.device)
        diff_sq_norm = torch.scatter_reduce(
            zeros, 0, var_batch, diff_sq_true, "sum"
        )  # [B]
        x_opt_sq_norm = torch.scatter_reduce(
            zeros, 0, var_batch, x_opt_sq_true, "sum"
        )  # [B]

        # Paper metric per graph (always on true scale)
        paper_metric_per_graph = diff_sq_norm / torch.clamp(
            x_opt_sq_norm, min=1.0
        )  # [B]
        paper_metric = paper_metric_per_graph.mean()  # Scalar

        # Objective/constraint losses on final point (true scale)
        objective_loss_final = self.compute_objective_mse_loss(
            batch, x_pred_true, x_opt_true
        )
        constraint_loss_final = self.compute_constraint_violation_loss(
            batch, x_pred_true
        )

        # Compute MSE/MAE loss for training (use normalized for training stability)
        diff = x_pred - x_opt  # [N_total] - normalized
        diff_sq = diff**2  # [N_total]
        mse_loss = torch.mean(diff_sq)
        mae_loss = torch.mean(torch.abs(diff))

        # End-to-end is single-shot (no iterates): supervised term is MSE, and
        # the optional objective term reduces to the final-point objective gap.
        total_loss = mse_loss
        if prefix in ["Train", "Validation"]:
            if self.train_loss_constraint:
                total_loss = total_loss + constraint_loss_final
            if self.train_loss_objective:
                total_loss = total_loss + objective_loss_final

        # Log metrics
        metrics = {
            "loss_mse_normalized": mse_loss,  # Normalized MSE for training
            "loss_mae_normalized": mae_loss,
            "paper_metric_true": paper_metric,  # Always on true scale
            "loss_objective_final_true": objective_loss_final,
            "loss_constraint_final_true": constraint_loss_final,
            "total_loss": total_loss
            if prefix in ["Train", "Validation"]
            else mse_loss,
        }

        # Additional metrics for validation/test (all on true scale)
        if prefix in ["Validation", "Test"]:
            mse_true = torch.mean(diff_sq_true)
            mae_true = torch.mean(torch.abs(diff_true))
            rel_error_true = torch.mean(
                torch.abs(diff_true)
                / torch.clamp(torch.abs(x_opt_true), min=1e-8)
            )
            metrics.update(
                {
                    "loss_mse_true": mse_true,
                    "loss_mae_true": mae_true,
                    "loss_relative_true": rel_error_true,
                }
            )

        self.last_paper_metric = paper_metric.detach().item()

        return metrics, batch_size


class FCNNEnd2EndLightning(SOCPNetend2endLightning):
    """
    Plain FCNN (no graph structure) end-to-end baseline.

    Reuses all of SOCPNetend2endLightning's loss/logging logic (paper metric,
    optional objective/constraint terms), only swapping the underlying model
    for a flatten-and-MLP regressor and adapting how predictions are produced
    from a batch (via ``Batch.to_data_list()`` instead of the graph pipeline).
    """

    def __init__(
        self,
        d: int,
        lr: float,
        n_vars: int,
        n_norms_max: int,
        n_cons: int,
        n_lin_cons: int,
        zero_everything: bool = False,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        **kwargs,
    ):
        kwargs.pop("aggr", None)
        kwargs.pop("num_attention_heads", None)
        kwargs.pop("mp_decouple_soc_lc", None)
        super().__init__(
            d=d,
            lr=lr,
            zero_everything=zero_everything,
            **kwargs,
        )
        # Replace the graph-based end-to-end model with the flatten+MLP baseline.
        self.model = FCNNEnd2End(
            n_vars=n_vars,
            n_norms_max=n_norms_max,
            n_cons=n_cons,
            n_lin_cons=n_lin_cons,
            hidden_dims=hidden_dims,
        )

    def _predict_trajectory(self, batch):
        if self.zero_everything:
            batch = self._zero_everything(batch)
        data_list = batch.to_data_list()
        x_pred = self.model(data_list)  # [B, n_vars]
        return x_pred.reshape(-1)  # [N_total], graph order matches batch['var'].batch
