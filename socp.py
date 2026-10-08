"""
© 2026 Nokia
Licensed under the BSD 3-Clause License
SPDX-License-Identifier: BSD-3-Clause

SOCPNet Graph Builder and Constraint Builder

This module provides utilities for building SOCP problems and constructing
heterogeneous graphs. It includes both direct graph construction and a high-level
interface for building SOCP problems incrementally.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Union

import torch
from torch import Tensor
from torch_geometric.data import HeteroData


def build_direct_graph(
    f_vec: Tensor,  # [N] - Variable objective coefficients
    b_vec: Tensor,  # [K_tot] - Real norm constraint LHS values
    d_vec: Tensor,  # [C] - Constraint RHS values
    c_mat: Tensor,  # [C, N] - Constraint coefficient matrix
    A_indices: Tensor,
    A_values: Tensor,
    A_shape: Tuple[int, int],  # Sparse matrix A in COO format
    norm2cons: Tensor,  # [K_tot] - Mapping from norms to constraints
    l_bound: Tensor,  # [N] - Lower bounds on variables
    r_bound: Tensor,  # [N] - Upper bounds on variables
    F: Tensor,  # [b, N] - Linear constraint matrix
    g: Tensor,  # [b] - Linear constraint RHS vector
) -> HeteroData:
    """
    Create a HeteroData graph for the general multi-constraint SOCP case.

    This function constructs a heterogeneous graph from SOCP problem data where:
    - Variables represent optimization variables
    - Norms represent second-order cone constraints
    - Constraints (cons) represent SOCP constraints
    - Linear constraints (lin_cons) represent polyhedral/linear constraints F @ x <= g

    The graph structure includes:
    - Node features: f_j for variables, b_k for norms, d_m for constraints, g_k for lin_cons
    - Edge connections: var↔norm, var↔cons, norm↔cons, var↔lin_cons with appropriate weights

    Args:
        f_vec (Tensor): Variable objective coefficients [N]
        b_vec (Tensor): Norm constraint LHS offset values [K_tot]
        d_vec (Tensor): Constraint RHS values [C]
        c_mat (Tensor): Constraint coefficient matrix [C, N]
        A_indices (Tensor): Sparse matrix A indices in COO format [2, E]
        A_values (Tensor): Sparse matrix A values [E]
        A_shape (Tuple[int,int]): Shape of matrix A (K_tot, N)
        norm2cons (Tensor): Mapping from norm indices to constraint indices [K_tot]
        l_bound (Tensor): Lower bounds on variables [N]
        r_bound (Tensor): Upper bounds on variables [N]
        F (Tensor): Linear constraint matrix [b, N]
        g (Tensor): Linear constraint RHS vector [b]

    Returns:
        HeteroData: Constructed heterogeneous graph ready for SOCPNet

    Raises:
        AssertionError: If input dimensions are incompatible
    """
    K_tot, N = A_shape  # Total norms, number of variables
    C = d_vec.numel()  # Number of constraints

    assert f_vec.numel() == N, f"f_vec length {f_vec.numel()} != N {N}"
    assert b_vec.numel() == K_tot, (
        f"b_vec length {b_vec.numel()} != K_tot {K_tot}"
    )
    assert c_mat.shape == (C, N), (
        f"c_mat shape {c_mat.shape} != (C, N) ({C}, {N})"
    )
    assert norm2cons.numel() == K_tot, (
        f"norm2cons length {norm2cons.numel()} != K_tot {K_tot}"
    )

    # Validate that we have at least one edge
    if A_values.numel() == 0:
        raise ValueError(
            "A_values is empty - at least one non-zero element required"
        )

    data = HeteroData()

    # Add node features for each node type
    # Variables: f_j (objective coefficients), l_bound, r_bound
    # Stack features: [f_j, l_bound_j, r_bound_j] -> [N, 3]
    data["var"].x = torch.stack([f_vec, l_bound, r_bound], dim=1)

    # Norms: b_k (norm constraint LHS offset values)
    data["norm"].x = b_vec.view(K_tot, 1)

    # Constraints: d_m (constraint RHS values)
    data["cons"].x = d_vec.view(C, 1)

    # Linear constraints: g_k (linear constraint RHS values)
    b = g.numel()  # Number of linear constraints
    assert F.shape == (b, N), f"F shape {F.shape} != (b, N) ({b}, {N})"
    assert g.numel() == b, f"g length {g.numel()} != b {b}"
    data["lin_cons"].x = g.view(b, 1)

    # Build edge connections and attributes
    # Extract row and column indices from sparse matrix A
    rows_k, cols_j = (
        A_indices[0],
        A_indices[1],
    )  # Norm indices, variable indices

    # Variables → norms, weight A[k, j]
    data["var", "V2N", "norm"].edge_index = torch.stack([cols_j, rows_k], dim=0)
    data["var", "V2N", "norm"].edge_attr = A_values.unsqueeze(-1)  # [E, 1]

    # Norms → variables, same weights
    data["norm", "N2V", "var"].edge_index = torch.stack([rows_k, cols_j], dim=0)
    data["norm", "N2V", "var"].edge_attr = A_values.unsqueeze(-1)  # [E, 1]

    cons_idx = norm2cons.to(dtype=torch.long)

    # Norms → Constraints
    data["norm", "N2C", "cons"].edge_index = torch.stack(
        [torch.arange(K_tot, dtype=torch.long), cons_idx], dim=0
    )

    # Constraints → Norms
    data["cons", "C2N", "norm"].edge_index = torch.stack(
        [cons_idx, torch.arange(K_tot, dtype=torch.long)], dim=0
    )

    # Build fully-connected variable↔constraint edges using coefficient matrix c
    # This creates all possible var↔cons connections with weights c[m,j]
    m_idx = torch.arange(C, dtype=torch.long)  # Constraint indices 0..C-1
    j_idx = torch.arange(N, dtype=torch.long)  # Variable indices 0..N-1

    # Variables → Constraints
    # Create edges from each variable to each constraint
    vc_src = j_idx.repeat(C)  # Variable indices repeated C times
    vc_dst = m_idx.repeat_interleave(
        N
    )  # Constraint indices, each repeated N times
    data["var", "V2C", "cons"].edge_index = torch.stack([vc_src, vc_dst], dim=0)
    data["var", "V2C", "cons"].edge_attr = c_mat.reshape(
        -1, 1
    )  # [C*N, 1] - Flatten c matrix for edge weights

    # Constraints → Variables
    # Create edges from each constraint to each variable
    data["cons", "C2V", "var"].edge_index = torch.stack(
        [vc_dst, vc_src], dim=0
    )
    data["cons", "C2V", "var"].edge_attr = c_mat.reshape(
        -1, 1
    )  # [C*N, 1] - Same weights for reverse direction

    # Linear constraints: Connect lin_cons nodes to var nodes
    # Each linear constraint node k is connected to all variable nodes j with weight F[k,j]
    # Create fully-connected edges between lin_cons and var
    k_idx = torch.arange(b, dtype=torch.long)  # Linear constraint indices 0..b-1
    j_idx = torch.arange(N, dtype=torch.long)  # Variable indices 0..N-1

    # Linear constraints → Variables
    # Create edges from each linear constraint to each variable
    lc_src = k_idx.repeat_interleave(N)  # Linear constraint indices, each repeated N times
    lc_dst = j_idx.repeat(b)  # Variable indices repeated b times
    data["lin_cons", "LC2V", "var"].edge_index = torch.stack([lc_src, lc_dst], dim=0)
    data["lin_cons", "LC2V", "var"].edge_attr = F.reshape(-1, 1)  # [b*N, 1] - Flatten F matrix for edge weights

    # Variables → Linear constraints (bidirectional for message passing)
    data["var", "V2LC", "lin_cons"].edge_index = torch.stack([lc_dst, lc_src], dim=0)
    data["var", "V2LC", "lin_cons"].edge_attr = F.reshape(-1, 1)  # [b*N, 1] - Same weights

    return data


class SOCPConstraintBuilder:
    """
    High-level constraint builder for SOCP problems.

    This class provides an intuitive interface for building SOCP problems
    incrementally by adding constraints one at a time. It expects sparse
    matrix representations and handles edge creation and graph construction automatically.

    Features:
    - Add constraints with sparse matrix input (COO format)
    - Automatic edge creation from sparse matrices
    - Incremental graph building

    Example:
        f_vec = torch.tensor([1.0, 2.0, 3.0])
        l_bound = torch.zeros(3)
        r_bound = torch.ones(3)
        builder = SOCPConstraintBuilder(f_vec, l_bound, r_bound)
        builder.add_constraint(
            A_indices=torch.tensor([[0, 1, 0], [0, 1, 2]]),  # [2, E]
            A_values=torch.tensor([1.0, 1.0, 2.0]),          # [E]
            A_shape=(2, 3),                                  # (K_m, N)
            b_vec=torch.tensor([1.0, 2.0]),
            c_coeffs=torch.tensor([1.0, 1.0, 1.0]),
            d_rhs=5.0
        )
        # x0 + x1 <= 1,  x2 <= 0.5
        F = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # [b, N]
        g = torch.tensor([1.0, 0.5])                          # [b]
        builder.add_linear_constraints(F, g)
        data = builder.build()
    """

    def __init__(self, f_vec: Tensor, l_bound: Tensor, r_bound: Tensor):
        """
        Initialize the constraint builder with mandatory f_vec and bounds.

        Args:
            f_vec (Tensor): Variable objective coefficients [N]
            l_bound (Tensor): Lower bounds on variables [N]
            r_bound (Tensor): Upper bounds on variables [N]
        """
        self.f_vec = f_vec
        self.N = f_vec.numel()
        self.l_bound = l_bound
        self.r_bound = r_bound

        # Storage for constraint blocks
        self.blocks: List[Dict[str, Tensor]] = []
        self.constraint_count = 0

        # Storage for linear constraints (must be set before build())
        self.F: Optional[Tensor] = None
        self.g: Optional[Tensor] = None

    def add_constraint(
        self,
        A_indices: Tensor,
        A_values: Tensor,
        A_shape: Tuple[int, int],
        b_vec: Tensor,
        c_coeffs: Tensor,
        d_rhs: Union[float, Tensor],
    ) -> int:
        """Add a constraint with sparse matrix A."""
        K_m, N_m = A_shape

        # Validate that constraint matches builder's N
        assert N_m == self.N, f"Wrong number of variables: {N_m} != {self.N}"

        assert b_vec.numel() == K_m, (
            f"Wrong b_vec size: {b_vec.numel()} != {K_m}"
        )
        assert c_coeffs.numel() == self.N, (
            f"Wrong c_coeffs size: {c_coeffs.numel()} != {self.N}"
        )

        # Convert d_rhs to tensor
        if isinstance(d_rhs, (int, float)):
            d_rhs = torch.tensor([float(d_rhs)])

        # Store constraint data for later combination
        block = {
            "f_vec": self.f_vec,  # Use the builder's f_vec
            "b_vec": b_vec,  # Norm constraint LHS offset
            "d_vec": d_rhs,  # Constraint RHS
            "c_mat": c_coeffs.unsqueeze(
                0
            ),  # Constraint coefficient matrix [1, N]
            "A_indices": A_indices,
            "A_values": A_values,
            "A_shape": A_shape,
            "norm2cons": torch.zeros(
                K_m, dtype=torch.long
            ),  # All norms belong to this constraint
        }

        self.blocks.append(block)
        constraint_idx = self.constraint_count
        self.constraint_count += 1

        return constraint_idx

    def add_linear_constraints(self, F: Tensor, g: Tensor) -> None:
        """
        Add linear constraints F @ x <= g.

        Args:
            F (Tensor): Linear constraint matrix [b, N]
            g (Tensor): Linear constraint RHS vector [b]
        """
        assert F.shape[1] == self.N, f"F shape {F.shape} incompatible with N={self.N}"
        assert F.shape[0] == g.numel(), f"F rows {F.shape[0]} != g length {g.numel()}"
        
        # Store linear constraints (will be used in build())
        self.F = F
        self.g = g

    def build(self) -> HeteroData:
        """Build the graph from all constraints."""
        if len(self.blocks) == 0:
            raise ValueError("No constraints added")
        if self.F is None or self.g is None:
            raise ValueError("Linear constraints F and g must be set before build()")

        f_vec_combined = self.blocks[0]["f_vec"]

        b_vec_list = []
        d_vec_list = []
        c_mat_list = []
        A_indices_list = []
        A_values_list = []
        norm2cons_list = []

        norm_offset = 0
        cons_offset = 0

        for block in self.blocks:
            b_vec = block["b_vec"]
            d_vec = block["d_vec"]
            c_mat = block["c_mat"]
            A_indices = block["A_indices"]
            A_values = block["A_values"]
            A_shape = block["A_shape"]
            norm2cons = block["norm2cons"]

            b_vec_list.append(b_vec)
            d_vec_list.append(d_vec)
            c_mat_list.append(c_mat)
            norm2cons_list.append(
                norm2cons + cons_offset
            )  # Adjust constraint indices

            # Adjust A matrix indices (only norm indices, variable indices stay the same)
            adjusted_A_indices = A_indices.clone()
            adjusted_A_indices[0] += norm_offset  # Adjust norm indices
            # Variable indices stay the same since all blocks use same variables
            A_indices_list.append(adjusted_A_indices)
            A_values_list.append(A_values)

            norm_offset += A_shape[0]  # Number of norms in this block
            cons_offset += d_vec.numel()

        b_vec_combined = torch.cat(b_vec_list, dim=0)
        d_vec_combined = torch.cat(d_vec_list, dim=0)
        c_mat_combined = torch.cat(c_mat_list, dim=0)
        A_indices_combined = torch.cat(A_indices_list, dim=1)
        A_values_combined = torch.cat(A_values_list, dim=0)
        norm2cons_combined = torch.cat(norm2cons_list, dim=0)

        # Calculate combined shape
        total_vars = f_vec_combined.numel()
        total_norms = b_vec_combined.numel()
        A_shape_combined = (total_norms, total_vars)

        return build_direct_graph(
            f_vec_combined,
            b_vec_combined,
            d_vec_combined,
            c_mat_combined,
            A_indices_combined,
            A_values_combined,
            A_shape_combined,
            norm2cons_combined,
            l_bound=self.l_bound,
            r_bound=self.r_bound,
            F=self.F,
            g=self.g,
        )

    def clear(self) -> None:
        """Clear all constraints."""
        self.blocks.clear()
        self.constraint_count = 0
        self.F = None
        self.g = None

    def __len__(self) -> int:
        return len(self.blocks)

    def __repr__(self) -> str:
        return f"SOCPConstraintBuilder(N={self.N}, constraints={len(self.blocks)})"
