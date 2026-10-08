# SOCPNet

This repository contains the code of our paper "Learning SOCP Solver Dynamics with Graph Neural Networks", in the *Learning on Graphs Conference* (LoG), 2026.

© 2026 Nokia\
Licensed under the BSD 3-Clause License\
SPDX-License-Identifier: BSD-3-Clause

## Introduction

SOCPNet is a recursive graph neural network that learns the trajectory and final solution of an interior-point method (IPM) solver on Second-Order Cone Programs (SOCP):

```
min  fᵀx   s.t.  ‖A_k x + b_k‖₂ ≤ c_kᵀx + d_k,  k = 1..m
                 F x ≤ g,   l ≤ x ≤ r
```

Each problem is encoded as a heterogeneous graph (variable / norm / conic-constraint / linear-constraint nodes) and the network predicts the sequence of iterates x₀, x₁, …, x*.

## Layout

```
model.py                  neural networks: three-step message passing GNN + MLP baseline
lightning_modules.py      training and validation logic, losses, data module
generate_data.py          writes the problem pickle and its trajectories. Uses the utility files:
  - solver.py             solves one SOCP instance in a subprocess and checks feasibility
  - ipopt.py              IPOPT solver backend and JAX derivatives
build_graph_dataset.py    converts the problem pickle file to a graph pickle file. Uses the utility file:
  - socp.py               turns one SOCP instance into a heterogeneous graph
preprocess.py             splits and normalizes the graph pickle for training
train_lightning.py        training script
test_lightning.py         evaluates a checkpoint, saves its results to results.pt
testing_plots.py          loads results.pt and plots SRE, objective, constraint-violation curve
```

## Install

Create the environment and install all requirements.

```bash
# Create conda environment only if cyipopt is needed to generate data
conda create -n socpnet python=3.12 -y
conda activate socpnet
conda install -c conda-forge cyipopt
# Install all other dependencies
pip install -r requirements.txt
```

## Quickstart
 
 Here are instructions to prepare the data, train, and test as done in the paper. The commands below are written for the small dataset. The medium dataset can be processed similarly.

### 1. Generate data (records solver trajectories)

```bash
python generate_data.py unstructured --scale small --n 5000 --out data/small_unstructured.pkl
```

### 2. Build the graph dataset

```bash
python build_graph_dataset.py --input data/small_unstructured.pkl \
    --output data/small_unstructured_graphs.pkl
```

### 3. Preprocess (split + normalize)

Split the graphs into train, validation, and test, and normalize them. The
saved trajectory keeps every iterate, with x* being the last term.

```bash
python preprocess.py --dataset_path data/small_unstructured_graphs.pkl \
    --save_path preprocessed_data/small_unstructured --target_length 26
```

### 4. Train

S9 is the main model presented in the paper (`d=256`). S8 is the same procedure at `d=64`. Both use batch size 16, learning rate `5e-4`, and the Small dataset's training horizon is `target_length 26`. Checkpoints and TensorBoard logs land under `lightning_logs/`.

```bash
# S9  main model, d=256
python train_lightning.py --model SOCPNet \
    -d 256 --batch_size 16 --max_epochs 160 \
    --dataset_paths preprocessed_data/small_unstructured --target_length 26 \
    --log_group s9_d256 \
    --residual true --use_step_encoding true --use_iterate_features true \
    --loss_type traj_logsre \
    --train_loss_fixed_point 10 --picard_weight 10 --picard_rho 0.9 \
    --final_anchor_weight 15 \
    --train_loss_objective constant 2 --train_loss_constraint constant 2 \
    --scale_direction_loss constant 0.5 \
    --lr_schedule cosine --warmup_epochs 5 \
    --full_tail_supervision true

# S8  same procedure, d=64
python train_lightning.py --model SOCPNet \
    -d 64 --batch_size 16 --max_epochs 120 \
    --dataset_paths preprocessed_data/small_unstructured --target_length 26 \
    --log_group s8_d64 \
    --residual true --use_step_encoding true --use_iterate_features true \
    --loss_type traj_logsre \
    --train_loss_fixed_point 10 --picard_weight 10 --picard_rho 0.9 \
    --final_anchor_weight 15 \
    --train_loss_objective constant 2 --train_loss_constraint constant 2 \
    --scale_direction_loss constant 0.5 \
    --lr_schedule cosine --warmup_epochs 5 \
    --full_tail_supervision true
```

Every option (iterate features, soft-DTW loss, scale/direction readout,
message-passing depth, …) is listed by
`python train_lightning.py --help`.

### 5. Test and plot

Test unrolls past the training horizon, e.g., `h=30` steps on the small dataset. The model type is read from the checkpoint.

The plots saved in `test_plots/` are median curves of the paper metric, relative error, step size, relative objective error, and constraint violation (`{dataset}_<metric>_median.pdf`). The IPOPT trajectory is drawn on each plot.

```bash
python test_lightning.py --checkpoint_path <run>/checkpoints/last.ckpt \
    --dataset_paths preprocessed_data/small_unstructured --n_iterations 30 \
    --output_dir test_predictions

python testing_plots.py --primary_results_path test_predictions/<s9>.pt \
    --secondary_results_paths test_predictions/<s8>.pt test_predictions/<s1>.pt \
    --labels S9 S8 S1 --output_dir test_plots
```

| model type  | `--model` / flags           | prediction     |
|-------------|--------------------------------|----------------|
| `recursive` | `SOCPNet`                      | unrolled x₀…x* |
| `gnn_e2e`   | `SOCPNet` + `--end2end`        | single-shot x* |
| `mlp_e2e`   | `FCNNEnd2End` + `--end2end`    | single-shot x* |

Structured families (`robust_ellipsoid`, `block_angular`, `portfolio_risk`) are
generated with `python generate_data.py family --family <name> --scale small`.

## Ablation study

Same small dataset as the quickstart. Each version keeps the previous setup and changes one thing.

### S1

A plain MLP. It ignores the graph and predicts x* in one pass.

```bash
python train_lightning.py --end2end --model FCNNEnd2End \
    -d 64 --batch_size 16 --max_epochs 80 \
    --dataset_paths preprocessed_data/small_unstructured --target_length 26 \
    --log_group s1_mlp
```

### S2

Same one-shot prediction of x*, now with the GNN reading the graph.

```bash
python train_lightning.py --end2end --model SOCPNet \
    -d 64 --batch_size 16 --max_epochs 80 \
    --dataset_paths preprocessed_data/small_unstructured --target_length 26 \
    --log_group s2_gnn
```

### S3

The GNN is applied once per solver step and trained to match the trajectory with MSE.

```bash
python train_lightning.py --model SOCPNet \
    -d 64 --batch_size 16 --max_epochs 80 \
    --dataset_paths preprocessed_data/small_unstructured --target_length 26 \
    --log_group s3_mse \
    --residual false --loss_type mse
```

### S4

The next iterate is the current one plus a correction predicted by the network.

```bash
python train_lightning.py --model SOCPNet \
    -d 64 --batch_size 16 --max_epochs 80 \
    --dataset_paths preprocessed_data/small_unstructured --target_length 26 \
    --log_group s4_residual \
    --residual true --loss_type mse
```

### S5

Soft-DTW replaces MSE, so the predicted path can move at a different pace from the solver's trajectory.

```bash
python train_lightning.py --model SOCPNet \
    -d 64 --batch_size 16 --max_epochs 80 \
    --dataset_paths preprocessed_data/small_unstructured --target_length 26 \
    --log_group s5_softdtw \
    --residual true --loss_type soft_dtw --soft_dtw_gamma 0.1
```

### S6

A log-SRE loss is applied at every step: the mean of ½ log(‖x̂_t − x*‖² / max(1, ‖x*‖²)).

```bash
python train_lightning.py --model SOCPNet \
    -d 64 --batch_size 16 --max_epochs 80 \
    --dataset_paths preprocessed_data/small_unstructured --target_length 26 \
    --log_group s6_logsre \
    --residual true --loss_type traj_logsre
```

### S7

Adds the following losses: fixed-point (stay at x* once reached), Picard (shrink the distance to x* by 0.9 each step), anchor (fix the last supervised step), step-size (match the solver's log step length), objective (squared gap to fᵀx*), constraint (squared cone, linear, and box violations). Also adds KKT features (slacks of the current iterate) and a cosine learning-rate schedule.

```bash
python train_lightning.py --model SOCPNet \
    -d 64 --batch_size 16 --max_epochs 120 \
    --dataset_paths preprocessed_data/small_unstructured --target_length 26 \
    --log_group s7_recipe \
    --residual true --use_step_encoding true --use_iterate_features true \
    --loss_type traj_logsre \
    --train_loss_fixed_point 10 --picard_weight 10 --picard_rho 0.9 \
    --final_anchor_weight 15 \
    --train_loss_objective constant 2 --train_loss_constraint constant 2 \
    --scale_direction_loss constant 0.5 \
    --lr_schedule cosine --warmup_epochs 5
```
