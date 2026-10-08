"""
© 2026 Nokia
Licensed under the BSD 3-Clause License
SPDX-License-Identifier: BSD-3-Clause

SOCPNet Model Implementation

This module contains the core SOCPNet (Second-Order Cone Programming Graph Neural Network)
implementation using PyTorch Geometric. The model implements message passing schemes for
solving multi-constraint SOCP problems with both conic and linear constraints.

Architecture:
- One forward call is one solver step
- Node features: [obj_coeff, l_bound, r_bound, x_current], with x_current starting at x0
- The caller updates the features and calls the model again for the next step
- Message passing uses the three-step scheme
"""

import math
from typing import Optional

import torch
from torch import nn, Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import LayerNorm


# Per-node-type dimensionality of the per-iteration "physics" features that
# encode the local KKT picture at the current iterate:
#   var:      [lower_slack, upper_slack, dual_lower, dual_upper]
#   norm:     [row_residual = (A x + b) for that row]
#   cons:     [soc_slack, dual_soc]
#   lin_cons: [lin_slack, dual_lin]
ITERATE_FEAT_DIMS = {"var": 4, "norm": 1, "cons": 2, "lin_cons": 2}


class LinearFeatureUpdate(nn.Module):
    """Concat+Linear cross-iteration update of the variable state.

    Concatenates the current variable embedding ``h_u`` with the new prediction
    ``x_in`` (d+1 dims) and maps it back to d dims.

    Args:
        d (int): Hidden dimension size.
    """

    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 1, d), nn.ReLU())

    def forward(self, h_u: Tensor, x_in: Tensor) -> Tensor:
        return self.net(torch.cat([h_u, x_in], dim=1))


class GRUFeatureUpdate(nn.Module):
    """GRU cross-iteration update of the variable state.

    Treats the current variable embedding ``h_u`` as the GRU hidden state and the
    encoded new prediction ``x_in`` as the input, letting the embedding act as a
    latent observer of IPOPT's hidden state (barrier parameter / duals) across
    iterations.

    Args:
        d (int): Hidden dimension size.
    """

    def __init__(self, d: int):
        super().__init__()
        self.var_obs_enc = nn.Linear(1, d)
        self.var_gru = nn.GRUCell(d, d)

    def forward(self, h_u: Tensor, x_in: Tensor) -> Tensor:
        return self.var_gru(self.var_obs_enc(x_in), h_u)


class SimpleF(nn.Module):
    """Simple F function: single hidden layer with 128 neurons + ReLU (d -> 128 -> d)."""

    def __init__(self, d: int):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, d))

    def forward(self, x: Tensor) -> Tensor:
        return self.f(x)


# ============================================================================
# THREE-STEP MESSAGE PASSING SCHEME (Classes 1-3)
# ============================================================================


class ThreeStepStep1(MessagePassing):
    """
    Three-Step Scheme - Step 1: Variables → All (Norms, Constraints, Linear Constraints)

    This layer implements the first step of the three-step message passing scheme,
    where variables send messages to all other node types independently.

    Architecture:
    - F1: Transformation function for variable features → norms (default: SimpleF d -> 128 -> d)
    - F2: Transformation function for variable features → constraints (default: SimpleF d -> 128 -> d)
    - F8: Transformation function for variable features → linear constraints (default: SimpleF d -> 128 -> d)
    - Phi1, Phi, Phi_lc: Simple linear layers for message aggregation (2d -> d)

    Args:
        d (int): Hidden dimension size for all neural networks
        F1 (nn.Module, optional): Transformation function for variable features (for norms)
        F2 (nn.Module, optional): Transformation function for variable features (for constraints)
        F8 (nn.Module, optional): Transformation function for variable features (for lin_cons)
    """

    def __init__(
        self,
        d: int,
        F1: Optional[nn.Module] = None,
        F2: Optional[nn.Module] = None,
        F8: Optional[nn.Module] = None,
        aggr: str = "mean",
    ):
        super().__init__(aggr=aggr)
        self.F1 = F1 if F1 is not None else SimpleF(d=d)
        self.F2 = F2 if F2 is not None else SimpleF(d=d)
        self.F8 = F8 if F8 is not None else SimpleF(d=d)
        self.Phi1 = nn.Linear(2 * d, d)
        self.Phi = nn.Linear(2 * d, d)
        self.Phi_lc = nn.Linear(2 * d, d)

    def forward(
        self,
        h_u: Tensor,
        h_v: Tensor,
        h_c: Tensor,
        h_lc: Tensor,
        edge_index_vn: Tensor,
        w_A: Tensor,
        edge_index_vc: Tensor,
        w_c: Tensor,
        edge_index_vlc: Tensor,
        w_F: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass for Step 1: Variables → All.

        Args:
            h_u (Tensor): Variable node features [N, d]
            h_v (Tensor): Norm node features [K_tot, d]
            h_c (Tensor): Constraint node features [C, d]
            h_lc (Tensor): Linear constraint node features [b, d]
            edge_index_vn (Tensor): Edge indices for var->norm connections [2, E_vn]
            w_A (Tensor): Edge weights A[k,j] for var->norm connections [E_vn, 1]
            edge_index_vc (Tensor): Edge indices for var->cons connections [2, E_vc]
            w_c (Tensor): Edge weights c[m,j] for var->cons connections [E_vc, 1]
            edge_index_vlc (Tensor): Edge indices for var->lin_cons connections [2, E_vlc]
            w_F (Tensor): Edge weights F[k,j] for var->lin_cons connections [E_vlc, 1]

        Returns:
            tuple[Tensor, Tensor, Tensor]: Updated norm, constraint, and linear constraint features
        """
        # Variables → Norms
        u1 = self.F1(h_u)
        s_v = self.propagate(edge_index=edge_index_vn, x=(u1, h_v), w=w_A)
        h_v_updated = h_v + self.Phi1(torch.cat([h_v, s_v], dim=1))

        # Variables → Constraints
        u2 = self.F2(h_u)
        s_c = self.propagate(edge_index=edge_index_vc, x=(u2, h_c), w=w_c)
        h_c_updated = h_c + self.Phi(torch.cat([h_c, s_c], dim=1))

        # Variables → Linear Constraints
        u8 = self.F8(h_u)
        s_lc = self.propagate(edge_index=edge_index_vlc, x=(u8, h_lc), w=w_F)
        h_lc_updated = h_lc + self.Phi_lc(torch.cat([h_lc, s_lc], dim=1))

        return h_v_updated, h_c_updated, h_lc_updated

    def message(self, x_j: Tensor, w: Tensor) -> Tensor:
        return x_j * w.view(-1, 1)


class ThreeStepStep2(MessagePassing):
    """
    Three-Step Scheme - Step 2: Norm ↔ Constraint Interaction

    This layer implements the second step of the three-step message passing scheme,
    where norms and constraints interact bidirectionally.

    Architecture:
    - F7: Transformation function for norm features → constraints (default: SimpleF d -> 128 -> d)
    - F4: Transformation function for constraint features → norms (default: SimpleF d -> 128 -> d)
    - Phi: Simple linear layer for message aggregation (2d -> d)
    - Phi3: Simple linear layer for message aggregation (2d -> d)

    Args:
        d (int): Hidden dimension size for all neural networks
        F7 (nn.Module, optional): Transformation function for norm features
        F4 (nn.Module, optional): Transformation function for constraint features
    """

    def __init__(
        self,
        d: int,
        F7: Optional[nn.Module] = None,
        F4: Optional[nn.Module] = None,
        aggr: str = "mean",
    ):
        super().__init__(aggr=aggr)
        self.F7 = F7 if F7 is not None else SimpleF(d=d)
        self.F4 = F4 if F4 is not None else SimpleF(d=d)
        self.Phi = nn.Linear(2 * d, d)
        self.Phi3 = nn.Linear(2 * d, d)

    def forward(
        self,
        h_v: Tensor,
        h_c: Tensor,
        edge_index_nc: Tensor,
        edge_index_cn: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass for Step 2: Norm ↔ Constraint.

        Args:
            h_v (Tensor): Norm node features [K_tot, d]
            h_c (Tensor): Constraint node features [C, d]
            edge_index_nc (Tensor): Edge indices for norm->cons connections [2, E_nc]
            edge_index_cn (Tensor): Edge indices for cons->norm connections [2, E_cn]

        Returns:
            tuple[Tensor, Tensor]: Updated constraint and norm features
        """
        # Norms → Constraints
        v7 = self.F7(h_v)
        s_c = self.propagate(edge_index=edge_index_nc, x=(v7, h_c), w=None)
        h_c_updated = h_c + self.Phi(torch.cat([h_c, s_c], dim=1))

        # Constraints → Norms
        c4 = self.F4(h_c_updated)
        s_v = self.propagate(edge_index=edge_index_cn, x=(c4, h_v), w=None)
        h_v_updated = h_v + self.Phi3(torch.cat([h_v, s_v], dim=1))

        return h_c_updated, h_v_updated

    def message(self, x_j: Tensor, w: Optional[Tensor] = None) -> Tensor:
        return x_j


class ThreeStepStep3(MessagePassing):
    """
    Three-Step Scheme - Step 3: All → Variables

    This layer implements the third step of the three-step message passing scheme,
    where all node types send messages back to variables.

    Architecture:
    - F5: Transformation function for constraint features (default: SimpleF d -> 128 -> d)
    - F6: Transformation function for norm features (default: SimpleF d -> 128 -> d)
    - F9: Transformation function for linear constraint features (default: SimpleF d -> 128 -> d)
    - mp_decouple_soc_lc:
      If True, the final message-passing step uses two separate aggregators Phi_soc and Phi_lc:
        - Phi_soc: Simple linear layer for coupled SOC aggregation (3d -> d) — norm + cons processed first
        - Phi_lc: Simple linear layer for linear constraint correction (2d -> d) — applied after SOC stage
      If False (default), a single joint Phi4 aggregates all incoming messages:
        - Phi4: Simple linear layer for message aggregation (4d -> d) — all incoming messages combined

    Args:
        d (int): Hidden dimension size for all neural networks
        F5 (nn.Module, optional): Transformation function for constraint features
        F6 (nn.Module, optional): Transformation function for norm features
        F9 (nn.Module, optional): Transformation function for linear constraint features
    """

    def __init__(
        self,
        d: int,
        F5: Optional[nn.Module] = None,
        F6: Optional[nn.Module] = None,
        F9: Optional[nn.Module] = None,
        aggr: str = "mean",
        mp_decouple_soc_lc: bool = False,
    ):
        super().__init__(aggr=aggr)
        self.F5 = F5 if F5 is not None else SimpleF(d=d)
        self.F6 = F6 if F6 is not None else SimpleF(d=d)
        self.F9 = F9 if F9 is not None else SimpleF(d=d)
        self.mp_decouple_soc_lc = mp_decouple_soc_lc
        if mp_decouple_soc_lc:
            # Two-stage: coupled SOC (norm+cons) then a separate linear-constraint correction.
            self.Phi_soc = nn.Linear(3 * d, d)
            self.Phi_lc = nn.Linear(2 * d, d)
        else:
            # Single joint aggregation over all incoming messages.
            self.Phi4 = nn.Linear(4 * d, d)

    def forward(
        self,
        h_u: Tensor,
        h_v: Tensor,
        h_c: Tensor,
        h_lc: Tensor,
        edge_index_cv: Tensor,
        w_c: Tensor,
        edge_index_nv: Tensor,
        w_A: Tensor,
        edge_index_lcv: Tensor,
        w_F: Tensor,
    ) -> Tensor:
        """
        Forward pass for Step 3: All → Variables.

        Args:
            h_u (Tensor): Variable node features [N, d]
            h_v (Tensor): Norm node features [K_tot, d]
            h_c (Tensor): Constraint node features [C, d]
            h_lc (Tensor): Linear constraint node features [b, d]
            edge_index_cv (Tensor): Edge indices for cons->var connections [2, E_cv]
            w_c (Tensor): Edge weights c[m,j] for cons->var connections [E_cv, 1]
            edge_index_nv (Tensor): Edge indices for norm->var connections [2, E_nv]
            w_A (Tensor): Edge weights A[k,j] for norm->var connections [E_nv, 1]
            edge_index_lcv (Tensor): Edge indices for lin_cons->var connections [2, E_lcv]
            w_F (Tensor): Edge weights F[k,j] for lin_cons->var connections [E_lcv, 1]

        Returns:
            Tensor: Updated variable node features [N, d]
        """
        # Constraints → Variables
        c5 = self.F5(h_c)
        msg_c = self.propagate(edge_index=edge_index_cv, x=(c5, h_u), w=w_c)

        # Norms → Variables
        v6 = self.F6(h_v)
        msg_n = self.propagate(edge_index=edge_index_nv, x=(v6, h_u), w=w_A)

        # Linear Constraints → Variables
        lc9 = self.F9(h_lc)
        msg_lc = self.propagate(edge_index=edge_index_lcv, x=(lc9, h_u), w=w_F)

        if self.mp_decouple_soc_lc:
            # Stage 1: coupled SOC signals (norm + cons share the same cone barrier)
            h_soc = h_u + self.Phi_soc(torch.cat([h_u, msg_n, msg_c], dim=1))
            # Stage 2: linear constraint correction (independent barrier)
            return h_soc + self.Phi_lc(torch.cat([h_soc, msg_lc], dim=1))

        # Combine all messages
        return h_u + self.Phi4(torch.cat([h_u, msg_c, msg_n, msg_lc], dim=1))

    def message(self, x_j: Tensor, w: Tensor) -> Tensor:
        return x_j * w.view(-1, 1)


class SOCPNet(nn.Module):
    """
    Multi-constraint SOCPNet (Second-Order Cone Programming Graph Neural Network)

    This is the main model class that implements the SOCPNet architecture as an
    iterative update operator. One forward call is one solver step, using the
    three-step message passing scheme.

    Architecture:
    - Each node's initial features: [obj_coeff, l_bound, r_bound, x0] where x0 is the first trajectory point
    - Forward pass performs ONE round of message passing and returns one prediction
    - Calling code handles iteration: features -> GNN -> x1 -> update features -> GNN -> x2 -> ...

    Graph Structure:
    - Variables: Represent optimization variables with features [obj_coeff_j, l_bound_j, r_bound_j, x_current_j]
    - Norms: Represent second-order cone constraints
    - Constraints: Represent conic constraints
    - Linear Constraints: Represent linear constraints

    Message Passing Schemes:
    - "three_step": Three-step scheme (Step 1: var→all, Step 2: norm↔cons, Step 3: all→var)

    Args:
        d (int): Hidden dimension size for all neural networks
        T (int): Training horizon. Normalizes the step index in the scale head.
        out_dim (int): Output dimension for final predictions
        F1 (nn.Module, optional): var → norm, step 1 (default: SimpleF, d → 128 → d)
        F2 (nn.Module, optional): var → cons, step 1 (default: SimpleF)
        F4 (nn.Module, optional): cons → norm, step 2 (default: SimpleF)
        F5 (nn.Module, optional): cons → var, step 3 (default: SimpleF)
        F6 (nn.Module, optional): norm → var, step 3 (default: SimpleF)
        F7 (nn.Module, optional): norm → cons, step 2 (default: SimpleF)
        F8 (nn.Module, optional): var → lin_cons, step 1 (default: SimpleF)
        F9 (nn.Module, optional): lin_cons → var, step 3 (default: SimpleF)
    """

    def __init__(
        self,
        d: int,
        T: int,
        out_dim: int,
        F1: Optional[nn.Module] = None,
        F2: Optional[nn.Module] = None,
        F4: Optional[nn.Module] = None,
        F5: Optional[nn.Module] = None,
        F6: Optional[nn.Module] = None,
        F7: Optional[nn.Module] = None,
        F8: Optional[nn.Module] = None,
        F9: Optional[nn.Module] = None,
        message_passing_scheme: str = "three_step",
        aggr: str = "mean",
        message_passing_depth: int = 1,
        num_attention_heads: int = 0,
        mp_decouple_soc_lc: bool = False,
        use_gru_update: bool = False,
        use_step_encoding: bool = False,
        use_iterate_features: bool = False,
        scale_direction: bool = False,
    ):
        """
        Initialize the SOCPNet model.

        Args:
            d (int): Hidden dimension size for all neural networks
            T (int): Training horizon. Normalizes the step index in the scale head.
            out_dim (int): Output dimension for final predictions
            F1 (nn.Module, optional): var → norm, step 1 (default: SimpleF, d → 128 → d)
            F2 (nn.Module, optional): var → cons, step 1 (default: SimpleF)
            F4 (nn.Module, optional): cons → norm, step 2 (default: SimpleF)
            F5 (nn.Module, optional): cons → var, step 3 (default: SimpleF)
            F6 (nn.Module, optional): norm → var, step 3 (default: SimpleF)
            F7 (nn.Module, optional): norm → cons, step 2 (default: SimpleF)
            F8 (nn.Module, optional): var → lin_cons, step 1 (default: SimpleF)
            F9 (nn.Module, optional): lin_cons → var, step 3 (default: SimpleF)
            message_passing_scheme (str): Message passing scheme to use. Options:
                - "three_step": Three-step message passing (Step 1: var→all, Step 2: norm↔cons, Step 3: all→var)
            aggr (str): Aggregation method for message passing (default: "mean")
            message_passing_depth (int): Number of sequential MP rounds before readout, each with independent weights (default: 1)
            num_attention_heads (int): If >0, apply multi-head self-attention on variable nodes after MP (default: 0)

        Note:
            - Individual F functions can be customized by passing them explicitly
            - If F functions are not provided, default SimpleF(d) will be used (d -> 128 -> d)
            - out_dim should always be 1 for SOCP problems (one scalar prediction per variable)
            - Call forward() once per solver step
            - Variable node features must be 4D: [obj_coeff, l_bound, r_bound, x_current] for each node
        """
        super().__init__()

        # Store parameters
        self.d = d
        self.T = T
        self.out_dim = out_dim
        self.message_passing_scheme = message_passing_scheme
        self.message_passing_depth = message_passing_depth
        self.aggr = aggr
        self.num_attention_heads = num_attention_heads
        self.mp_decouple_soc_lc = mp_decouple_soc_lc
        self.use_gru_update = use_gru_update
        self.use_step_encoding = use_step_encoding
        self.use_iterate_features = use_iterate_features
        # Scale/direction residual: split each update into a unit direction and a
        # log-step magnitude so the step size can span many orders of magnitude
        # (exponential decay toward the optimum, IPOPT-style). The per-graph scale
        # is returned explicitly from ``forward`` for the Lightning module.
        self.scale_direction = scale_direction
        if num_attention_heads > 0:
            assert self.d % num_attention_heads == 0, (
                f"embedding dim d={self.d} must be divisible by "
                f"num_attention_heads={num_attention_heads}"
            )
            self.var_attention = nn.MultiheadAttention(
                embed_dim=self.d, num_heads=num_attention_heads, batch_first=True
            )
            self.var_attn_norm = nn.LayerNorm(self.d)

        # Sinusoidal encoding of the iteration index, giving the model a
        # notion of "where along the central path" it is. A learned linear
        # projection adapts the fixed sinusoidal code before it is added to the
        # node embeddings.
        if use_step_encoding:
            self.step_proj = nn.Linear(self.d, self.d)

        # The local KKT info at the current iterate (constraint/bound slacks and
        # inverse-slack dual proxies) is injected into the embeddings at every
        # unroll step. The features are computed externally and passed via
        # ``iterate_feat_dict``.
        if use_iterate_features:
            self.iter_enc = nn.ModuleDict({
                nt: nn.Linear(k, self.d)
                for nt, k in ITERATE_FEAT_DIMS.items()
            })

        # Input feature transformation networks (simple linear layers per paper)
        # var_input_dim: 4 for trajectory mode (with x_current), 3 for end-to-end mode
        var_input_dim = getattr(self, "var_input_dim", 4)
        self.g_u = nn.Linear(var_input_dim, self.d)
        self.g_v = nn.Linear(1, self.d)
        self.g_c = nn.Linear(1, self.d)
        self.g_lc = nn.Linear(1, self.d)

        # Message passing layers: one independent set of weights per depth level
        if self.message_passing_scheme == "three_step":
            self.mp_layers = nn.ModuleList([
                nn.ModuleDict({
                    "step1": ThreeStepStep1(self.d, F1, F2, F8, aggr=self.aggr),
                    "step2": ThreeStepStep2(self.d, F7, F4, aggr=self.aggr),
                    "step3": ThreeStepStep3(self.d, F5, F6, F9, aggr=self.aggr,
                                            mp_decouple_soc_lc=self.mp_decouple_soc_lc),
                })
                for _ in range(self.message_passing_depth)
            ])
        else:
            raise ValueError(
                "Unsupported message_passing_scheme: "
                f"{self.message_passing_scheme}"
            )

        # Feature update: incorporates the prediction back into the variable
        # embeddings for the next iteration. With use_gru_update a gated
        # recurrent unit is used, otherwise a simple concat+Linear update.
        # Both expose the same (h_u, x_in) -> h_u interface, and only the selected
        # variant's parameters are created so none are left unused (avoids DDP
        # find_unused_parameters issues).
        if use_gru_update:
            self.feature_update = GRUFeatureUpdate(self.d)
        else:
            self.feature_update = LinearFeatureUpdate(self.d)

        # Layer normalization: one set of 4 norms per depth level, applied after each MP round
        self.mp_norms = nn.ModuleList([
            nn.ModuleDict({
                "var":      LayerNorm(self.d),
                "norm":     LayerNorm(self.d),
                "cons":     LayerNorm(self.d),
                "lin_cons": LayerNorm(self.d),
            })
            for _ in range(self.message_passing_depth)
        ])

        # Output layer: Single hidden layer with 128 neurons + ReLU (per paper)
        # Input: [h_u[i], I_1, I_2, I_3, I_4] where I_1, I_2, I_3, I_4 are aggregated embeddings
        self.f_out = nn.Sequential(
            nn.Linear(5 * self.d, 128), nn.ReLU(), nn.Linear(128, self.out_dim)
        )

        # Scale head: per-graph log-step magnitude. Input is the 4 per-graph
        # aggregates (I_1..I_4) plus the normalized iteration index.
        if self.scale_direction:
            self.scale_head = nn.Sequential(
                nn.Linear(4 * self.d + 1, 128), nn.ReLU(), nn.Linear(128, 1)
            )

    def initialize_embeddings(self, data: HeteroData) -> dict[str, Tensor]:
        # Initialize embeddings from input features
        x_dict = {
            "var": self.g_u(data["var"].x),  # [N_total, d]
            "norm": self.g_v(data["norm"].x),  # [K_tot_total, d]
            "cons": self.g_c(data["cons"].x),  # [C_total, d]
            "lin_cons": self.g_lc(data["lin_cons"].x),  # [b_total, d]
        }
        return x_dict

    def _step_encoding(self, step_idx: int, device, dtype) -> Tensor:
        """Sinusoidal positional encoding of the iteration index.

        Returns a [d] vector to be broadcast-added to every node embedding,
        after being adapted by a learned linear projection.
        """
        d = self.d
        pe = torch.zeros(d, device=device, dtype=dtype)
        idx = torch.arange(0, d, 2, device=device, dtype=dtype)
        div = torch.exp(idx * (-math.log(10000.0) / d))
        pos = torch.tensor(float(step_idx), device=device, dtype=dtype)
        pe[0::2] = torch.sin(pos * div[: pe[0::2].numel()])
        pe[1::2] = torch.cos(pos * div[: pe[1::2].numel()])
        return self.step_proj(pe)

    def forward(
        self,
        x_dict,
        edge_index_dict,
        edge_attr_dict,
        batch_dict,
        step_idx=None,
        iterate_feat_dict=None,
    ):
        """
        Forward pass of the SOCPNet model - performs ONE iteration only.

        This method performs one round of message passing and returns one
        prediction. The calling code is responsible for initializing embeddings
        and iterating.

        Args:
            x_dict (dict[str, Tensor]): Dictionary of node embeddings:
            - 'var': Variable node embeddings [N_total, d]
            - 'norm': Norm node embeddings [K_tot_total, d]
            - 'cons': Constraint node embeddings [C_total, d]
            - 'lin_cons': Linear constraint node embeddings [b_total, d]
            edge_index_dict (dict): Dictionary of edge indices for each edge type
            edge_attr_dict (dict): Dictionary of edge attributes for each edge type
            batch_dict (dict): Dictionary of batch assignment vectors for each node type
            step_idx (int, optional): Current unroll step index, used for the
                sinusoidal step encoding. Ignored unless ``use_step_encoding`` is set.
            iterate_feat_dict (dict, optional): Per-node-type "physics" features at
                the current iterate. Keys: var/norm/cons/lin_cons.
                Ignored unless ``use_iterate_features`` is set.

        Returns:
            Tuple[Tensor, dict[str, Tensor]]:
            - x_pred: Predicted variables [N_total, 1]
            - x_dict: Updated node embeddings dictionary
        """

        # Retrieve node embeddings
        h_u = x_dict["var"]
        h_v = x_dict["norm"]
        h_c = x_dict["cons"]
        h_lc = x_dict["lin_cons"]

        # Inject the iteration-index (central-path position) encoding,
        # broadcast-added to every node type.
        if self.use_step_encoding and step_idx is not None:
            se = self._step_encoding(step_idx, h_u.device, h_u.dtype)
            h_u = h_u + se
            h_v = h_v + se
            h_c = h_c + se
            h_lc = h_lc + se

        # Inject the local KKT information at this iterate
        if self.use_iterate_features and iterate_feat_dict is not None:
            h_u = h_u + self.iter_enc["var"](iterate_feat_dict["var"])
            h_v = h_v + self.iter_enc["norm"](iterate_feat_dict["norm"])
            h_c = h_c + self.iter_enc["cons"](iterate_feat_dict["cons"])
            h_lc = h_lc + self.iter_enc["lin_cons"](iterate_feat_dict["lin_cons"])

        # Run message_passing_depth rounds, each with independent weights
        for layer, norms in zip(self.mp_layers, self.mp_norms):
            h_u, h_v, h_c, h_lc = self._run_three_step_mp(
                h_u,
                h_v,
                h_c,
                h_lc,
                edge_index_dict,
                edge_attr_dict,
                layer["step1"],
                layer["step2"],
                layer["step3"],
            )
            # Apply LayerNorm after each complete MP round
            h_u   = norms["var"](h_u, batch_dict["var"])
            h_v   = norms["norm"](h_v, batch_dict["norm"])
            h_c   = norms["cons"](h_c, batch_dict["cons"])
            h_lc  = norms["lin_cons"](h_lc, batch_dict["lin_cons"])

        if self.num_attention_heads > 0:
            h_u = self._apply_var_attention(h_u, batch_dict["var"])

        readout, scale = self._compute_readout(
            h_u, h_v, h_c, h_lc, batch_dict, step_idx
        )

        x_dict["var"] = h_u
        x_dict["norm"] = h_v
        x_dict["cons"] = h_c
        x_dict["lin_cons"] = h_lc

        # In scale/direction mode, return the per-graph scale and the per-node
        # direction together with the embeddings.
        if self.scale_direction:
            return scale, readout, x_dict
        return readout, x_dict

    def _mean_per_graph(self, embeddings, node_batch, var_batch):
        """Mean of embeddings within each graph, repeated on each variable node of that graph."""
        num_graphs = node_batch.max().item() + 1
        d = embeddings.size(1)
        graph_means = torch.zeros(
            num_graphs, d, device=embeddings.device, dtype=embeddings.dtype
        )
        graph_means = torch.scatter_reduce(
            graph_means,
            0,
            node_batch.unsqueeze(1).expand(-1, d),
            embeddings,
            reduce="mean",
        )
        return graph_means[var_batch]

    def _apply_var_attention(self, h_u: Tensor, var_batch: Tensor) -> Tensor:
        """Multi-head self-attention on variable nodes within each graph."""
        mask = var_batch.unsqueeze(0) != var_batch.unsqueeze(1)
        h = h_u.unsqueeze(0)
        h_attn, _ = self.var_attention(h, h, h, attn_mask=mask)
        return self.var_attn_norm(h_u + h_attn.squeeze(0))

    def _compute_readout(self, h_u, h_v, h_c, h_lc, batch_dict, step_idx=None):
        """Compute readout: aggregate embeddings and apply f_out.

        Returns ``(x_pred, scale)``. In scale/direction mode ``x_pred`` is the
        (unnormalized) direction and ``scale`` is the per-graph log-step
        (computed from the aggregates plus the normalized iteration index,
        broadcast to every node of its graph). Otherwise ``scale`` is ``None``.
        """
        var_batch = batch_dict["var"]
        lin_cons_batch = batch_dict["lin_cons"]
        norm_batch = batch_dict["norm"]
        cons_batch = batch_dict["cons"]

        # Compute I_1, I_2, I_3, I_4: means per graph
        I_1 = self._mean_per_graph(
            h_u, var_batch, var_batch
        )  # Mean of variables
        I_2 = self._mean_per_graph(
            h_lc, lin_cons_batch, var_batch
        )  # Mean of linear constraints
        I_3 = self._mean_per_graph(h_v, norm_batch, var_batch)  # Mean of norms
        I_4 = self._mean_per_graph(
            h_c, cons_batch, var_batch
        )  # Mean of constraints

        # Concatenate: [h_u[i], I_1, I_2, I_3, I_4] for each variable node
        h_u_with_aggregates = torch.cat([h_u, I_1, I_2, I_3, I_4], dim=1)
        x_pred = self.f_out(h_u_with_aggregates)

        scale = None
        if self.scale_direction:
            step_norm = (float(step_idx) / self.T) if step_idx is not None else 0.0
            sn = torch.full(
                (h_u.size(0), 1), step_norm, device=h_u.device, dtype=h_u.dtype
            )
            scale_in = torch.cat([I_1, I_2, I_3, I_4, sn], dim=1)  # [N, 4d+1]
            scale_node = self.scale_head(scale_in)  # [N, 1]
            # Per-graph mean -> one scale per graph, broadcast back to each node.
            scale = self._mean_per_graph(scale_node, var_batch, var_batch)

        return x_pred, scale

    def _run_three_step_mp(
        self,
        h_u,
        h_v,
        h_c,
        h_lc,
        edge_index_dict,
        edge_attr_dict,
        step1_layer,
        step2_layer,
        step3_layer,
    ):
        """Run one three-step message passing iteration."""
        # Extract edge indices and weights
        VN_idx = edge_index_dict[("var", "V2N", "norm")]
        VN_w = edge_attr_dict[("var", "V2N", "norm")]
        VC_idx = edge_index_dict[("var", "V2C", "cons")]
        VC_w = edge_attr_dict[("var", "V2C", "cons")]
        NC_idx = edge_index_dict[("norm", "N2C", "cons")]
        CV_idx = edge_index_dict[("cons", "C2N", "norm")]
        C2V_idx = edge_index_dict[("cons", "C2V", "var")]
        C2V_w = edge_attr_dict[("cons", "C2V", "var")]
        NV_idx = edge_index_dict[("norm", "N2V", "var")]
        NV_w = edge_attr_dict[("norm", "N2V", "var")]
        V2LC_idx = edge_index_dict[("var", "V2LC", "lin_cons")]
        V2LC_w = edge_attr_dict[("var", "V2LC", "lin_cons")]
        LC2V_idx = edge_index_dict[("lin_cons", "LC2V", "var")]
        LC2V_w = edge_attr_dict[("lin_cons", "LC2V", "var")]

        # Step 1: Variables → All
        h_v, h_c, h_lc = step1_layer(
            h_u, h_v, h_c, h_lc, VN_idx, VN_w, VC_idx, VC_w, V2LC_idx, V2LC_w
        )
        # Step 2: Norm ↔ Constraint
        h_c, h_v = step2_layer(h_v, h_c, NC_idx, CV_idx)
        # Step 3: All → Variables
        h_u = step3_layer(
            h_u, h_v, h_c, h_lc, C2V_idx, C2V_w, NV_idx, NV_w, LC2V_idx, LC2V_w
        )

        return h_u, h_v, h_c, h_lc


class SOCPNetend2end(SOCPNet):
    """
    End-to-end SOCPNet model for direct optimal solution prediction.

    Inherits from SOCPNet and overrides behavior for end-to-end training:
    - Uses 3 features for variables (no x0): [obj_coeff, l_bound, r_bound]
    - Runs TWO message-passing layers (each using three_step scheme) in one forward pass
    - Applies readout only once at the end (no readout between layers)
    - Does NOT update features with predictions (no feature_update network)
    - Returns only prediction (not tuple with updated embeddings)

    Architecture:
    - Input features: [obj_coeff, l_bound, r_bound] (3D, no x0)
    - Layer 1: three_step message passing (Step1 → Step2 → Step3)
    - Layer 2: three_step message passing (Step1 → Step2 → Step3) using updated embeddings
    - Readout: node-level readout once at the end
    - Output: predictions for all variable nodes [N_total, 1]

    Args:
        d (int): Hidden dimension size for all neural networks
        out_dim (int): Output dimension for final predictions (should be 1)
        F1, F2, F4–F9 (nn.Module, optional): Transformation functions (same as parent)
    """

    def __init__(
        self,
        d: int,
        out_dim: int,
        F1: Optional[nn.Module] = None,
        F2: Optional[nn.Module] = None,
        F4: Optional[nn.Module] = None,
        F5: Optional[nn.Module] = None,
        F6: Optional[nn.Module] = None,
        F7: Optional[nn.Module] = None,
        F8: Optional[nn.Module] = None,
        F9: Optional[nn.Module] = None,
        aggr: str = "mean",
        num_attention_heads: int = 0,
        mp_decouple_soc_lc: bool = False,
    ):
        # Set var_input_dim before calling super().__init__ so parent uses it
        self.var_input_dim = 3  # End-to-end uses 3 features (no x0)

        # Initialize parent: three_step, two independent MP rounds (message_passing_depth=2)
        super().__init__(
            d=d,
            T=1,  # Dummy value, not used in end-to-end
            out_dim=out_dim,
            F1=F1,
            F2=F2,
            F4=F4,
            F5=F5,
            F6=F6,
            F7=F7,
            F8=F8,
            F9=F9,
            aggr=aggr,
            message_passing_scheme="three_step",
            message_passing_depth=2,
            num_attention_heads=num_attention_heads,
            mp_decouple_soc_lc=mp_decouple_soc_lc,
        )

        # Remove feature_update network (not used in end-to-end)
        delattr(self, "feature_update")

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_dict):
        """
        Forward pass: 2 MP layers + readout (no feature update).

        Returns:
            Tensor: Predicted variables [N_total, 1]
        """
        readout, _ = SOCPNet.forward(
            self, x_dict, edge_index_dict, edge_attr_dict, batch_dict
        )
        return readout


class FCNNEnd2End(nn.Module):
    """
    Plain fully-connected (MLP) baseline for direct optimal-solution prediction.

    Ignores all graph structure: flattens the raw problem data (variable
    features, norm/cons/lin_cons node features, and the sparse coupling
    coefficients materialized as dense matrices) into a single fixed-size
    vector per problem instance, and regresses the optimal variable values
    with a stack of Linear+ReLU layers. Ablation step 1 ("FCNN end-to-end").

    Problem instances in a given dataset share n_vars/n_cons/n_lin_cons, but
    the number of "norm" nodes (SOC block dimensions) varies per instance, so
    the norm axis is zero-padded up to ``n_norms_max`` (extra rows/cols are
    all-zero and carry no signal).
    """

    def __init__(
        self,
        n_vars: int,
        n_norms_max: int,
        n_cons: int,
        n_lin_cons: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
    ):
        super().__init__()
        self.n_vars = n_vars
        self.n_norms_max = n_norms_max
        self.n_cons = n_cons
        self.n_lin_cons = n_lin_cons

        in_dim = (
            n_vars * 3          # var.x flattened
            + n_norms_max       # norm.x flattened (zero-padded)
            + n_cons            # cons.x flattened
            + n_lin_cons        # lin_cons.x flattened
            + n_norms_max * n_vars   # dense V2N coefficients
            + n_cons * n_vars        # dense V2C coefficients
            + n_lin_cons * n_vars    # dense V2LC coefficients
            + n_cons * n_norms_max   # dense N2C grouping (binary)
        )
        self.in_dim = in_dim

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_vars))
        self.mlp = nn.Sequential(*layers)

    def _dense(self, edge_index: Tensor, edge_attr: Optional[Tensor],
               n_src: int, n_dst_max: int) -> Tensor:
        """Scatter a sparse (src->dst) edge list into a dense [n_dst_max, n_src] matrix."""
        device = edge_index.device
        mat = torch.zeros(n_dst_max, n_src, device=device, dtype=torch.float32)
        if edge_index.numel() == 0:
            return mat
        if edge_attr is None:
            vals = torch.ones(edge_index.shape[1], device=device, dtype=torch.float32)
        else:
            vals = edge_attr.reshape(-1).float()
        mat[edge_index[1], edge_index[0]] = vals
        return mat

    def _flatten_one(self, data: HeteroData) -> Tensor:
        device = data["var"].x.device
        n_norms = data["norm"].x.shape[0]

        var_x = data["var"].x.reshape(-1).float()

        norm_x = torch.zeros(self.n_norms_max, device=device, dtype=torch.float32)
        norm_x[:n_norms] = data["norm"].x.reshape(-1).float()

        cons_x = data["cons"].x.reshape(-1).float()
        lc_x = data["lin_cons"].x.reshape(-1).float()

        v2n = data[("var", "V2N", "norm")]
        a_vn = self._dense(v2n.edge_index, v2n.edge_attr, self.n_vars, self.n_norms_max).reshape(-1)

        v2c = data[("var", "V2C", "cons")]
        a_vc = self._dense(v2c.edge_index, v2c.edge_attr, self.n_vars, self.n_cons).reshape(-1)

        v2lc = data[("var", "V2LC", "lin_cons")]
        a_vlc = self._dense(v2lc.edge_index, v2lc.edge_attr, self.n_vars, self.n_lin_cons).reshape(-1)

        n2c = data[("norm", "N2C", "cons")]
        a_nc = self._dense(n2c.edge_index, None, self.n_norms_max, self.n_cons).reshape(-1)

        return torch.cat([var_x, norm_x, cons_x, lc_x, a_vn, a_vc, a_vlc, a_nc])

    def forward(self, data_list: list) -> Tensor:
        """
        Args:
            data_list: list of B un-batched HeteroData graphs (e.g. from
                ``Batch.to_data_list()``).

        Returns:
            Tensor: predicted variables, shape [B, n_vars].
        """
        feats = torch.stack([self._flatten_one(d) for d in data_list], dim=0)
        return self.mlp(feats)
