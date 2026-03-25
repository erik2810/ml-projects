"""
Physics-Informed Graph Neural Network architectures.

Two main models:

    PhysicsInformedGNN    Multi-layer network for node-level and graph-level
                          tasks, composing CotangentConv, DiffusionConv,
                          ReactionDiffusion, and CurvatureAttention layers
                          with physics-based regularisation.

    PhysicsInformedGraphGenerator
                          Generative model that produces spatial graphs
                          with physics-consistent structure, using the
                          reaction-diffusion mechanism as the core
                          generation process.

Architecture design philosophy:
    Every component has a clear differential-geometric interpretation.
    Edge weights come from cotangent geometry (not arbitrary learning).
    Diffusion provides the linear operator; reaction provides nonlinearity.
    Curvature features encode local shape as geometric invariants.
    Energy regularisation grounds predictions in physical consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional, Literal

from .layers import (
    CotangentConv,
    DiffusionConv,
    ReactionDiffusionLayer,
    CurvatureAttention,
    GeometricEdgeEncoder,
)
from .operators import (
    discrete_curvatures,
    geometric_edge_weights,
    weighted_laplacian,
    symmetric_normalised_laplacian,
    cotangent_laplacian,
)
from .energy import PhysicsRegulariser


# ---------------------------------------------------------------------------
# Curvature feature extractor (preprocessing)
# ---------------------------------------------------------------------------

class CurvatureEncoder(nn.Module):
    """Compute and encode discrete curvatures as node features.

    Extracts six geometric invariants per node (mean, Gaussian, two
    principal curvatures, shape index, curvedness) and encodes them
    via an MLP into a fixed-dimensional representation.

    These features are invariant to rigid motions and capture the
    local shape at each node — information that standard GNNs must
    learn from scratch but that DDG provides for free.
    """

    def __init__(self, out_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        positions: Tensor,
        adj: Tensor,
        faces: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Returns:
            features: (N, out_dim) curvature embeddings.
            raw: dict of raw curvature tensors.
        """
        cd = discrete_curvatures(positions, adj, faces)
        raw = torch.stack([
            cd['mean'], cd['gaussian'],
            cd['principal_1'], cd['principal_2'],
            cd['shape_index'], cd['curvedness'],
        ], dim=1)  # (N, 6)

        # Normalise for numerical stability
        mu = raw.mean(dim=0, keepdim=True)
        std = raw.std(dim=0, keepdim=True).clamp(min=1e-6)
        raw_norm = (raw - mu) / std

        return self.mlp(raw_norm), cd


# ---------------------------------------------------------------------------
# Physics-Informed GNN
# ---------------------------------------------------------------------------

class PhysicsInformedGNN(nn.Module):
    """Physics-informed graph neural network.

    Combines geometry-derived operators with learnable components in a
    principled architecture where each layer has a clear physical role:

        Layer 0:  CurvatureEncoder — geometric invariants as input features
        Layers 1..L-1: Interleaved CotangentConv and ReactionDiffusion
        Layer L:  CurvatureAttention — geometry-biased readout
        Output:   Task head (node classification, regression, or graph-level)

    The Laplacian eigendecomposition is computed once and shared across
    layers that need it (DiffusionConv, energy regularisation).

    Args:
        in_channels: input feature dimension (0 = use positions + curvatures only).
        hidden_channels: hidden dimension throughout the network.
        out_channels: output dimension.
        num_layers: number of message-passing blocks.
        task: 'node' for node-level, 'graph' for graph-level prediction.
        use_curvature: compute and use curvature features.
        use_diffusion: use multi-scale diffusion filters.
        use_reaction_diffusion: use reaction-diffusion layers.
        use_attention: use curvature-biased attention in the final layer.
        regularise: apply physics energy regularisation.
    """

    def __init__(
        self,
        in_channels: int = 0,
        hidden_channels: int = 64,
        out_channels: int = 1,
        num_layers: int = 4,
        task: Literal['node', 'graph'] = 'node',
        use_curvature: bool = True,
        use_diffusion: bool = True,
        use_reaction_diffusion: bool = True,
        use_attention: bool = True,
        regularise: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.task = task
        self.use_curvature = use_curvature
        self.use_diffusion = use_diffusion
        self.use_reaction_diffusion = use_reaction_diffusion
        self.use_attention = use_attention
        self.regularise_flag = regularise

        # Input projection
        curvature_dim = 16 if use_curvature else 0
        position_dim = 3
        input_dim = in_channels + position_dim + curvature_dim

        if use_curvature:
            self.curvature_encoder = CurvatureEncoder(out_dim=curvature_dim)

        self.input_proj = nn.Linear(input_dim, hidden_channels)

        # Message-passing blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block = nn.ModuleDict()

            # Primary convolution: cotangent-weighted
            block['conv'] = CotangentConv(
                hidden_channels, hidden_channels,
                learn_residual=True,
                residual_scale=0.1,
            )

            if use_diffusion and i % 2 == 0:
                block['diffusion'] = DiffusionConv(
                    hidden_channels, hidden_channels,
                    num_scales=3,
                )

            if use_reaction_diffusion:
                block['reaction_diffusion'] = ReactionDiffusionLayer(
                    hidden_channels,
                    reaction_hidden=hidden_channels,
                )

            block['norm'] = nn.LayerNorm(hidden_channels)
            block['dropout'] = nn.Dropout(dropout)

            self.blocks.append(block)

        # Final attention layer
        if use_attention:
            self.attention = CurvatureAttention(
                hidden_channels, hidden_channels,
                num_heads=4,
                dropout=dropout,
            )

        # Task head
        if task == 'graph':
            self.readout = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, out_channels),
            )
        else:
            self.readout = nn.Linear(hidden_channels, out_channels)

        # Physics regulariser
        if regularise:
            self.regulariser = PhysicsRegulariser(
                use_dirichlet=True,
                use_elastic=True,
                use_tv=False,
            )

    def forward(
        self,
        positions: Tensor,
        adj: Tensor,
        x: Optional[Tensor] = None,
        faces: Optional[Tensor] = None,
        return_energy: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Args:
            positions: (N, 3) node coordinates.
            adj: (N, N) adjacency matrix.
            x: (N, F) optional input features.
            faces: (F_tri, 3) optional triangle faces.
            return_energy: if True, also return physics regularisation loss.

        Returns:
            output: (N, out_channels) for node task, (out_channels,) for graph task.
            energy: scalar regularisation loss (only if return_energy=True).
        """
        N = positions.size(0)

        # Curvature features
        curv_dict = None
        input_parts = [positions]

        if self.use_curvature:
            curv_feats, curv_dict = self.curvature_encoder(positions, adj, faces)
            input_parts.append(curv_feats)

        if x is not None:
            input_parts.append(x)

        h = torch.cat(input_parts, dim=-1)
        h = self.input_proj(h)

        # Compute normalised Laplacian once
        if faces is not None:
            L = cotangent_laplacian(positions, faces)
        else:
            W = geometric_edge_weights(positions, adj)
            L = weighted_laplacian(W)

        # Normalise Laplacian
        D = (-L.diagonal()).clamp(min=1e-8)
        D_inv_sqrt = D.pow(-0.5)
        L_norm = D_inv_sqrt.unsqueeze(1) * L * D_inv_sqrt.unsqueeze(0)

        # Eigendecomposition (shared)
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        eigen = (eigenvalues, eigenvectors)

        # Message-passing blocks
        for block in self.blocks:
            residual = h

            # Cotangent convolution
            h = block['conv'](h, positions, adj, faces)
            h = F.silu(h)

            # Multi-scale diffusion (on alternate layers)
            if 'diffusion' in block:
                h_diff = block['diffusion'](h, positions, adj, faces, eigen)
                h = h + h_diff  # residual

            # Reaction-diffusion
            if 'reaction_diffusion' in block:
                h = block['reaction_diffusion'](h, L_norm)

            h = block['norm'](h + residual)
            h = block['dropout'](h)

        # Final attention
        if self.use_attention:
            h_attn = self.attention(h, positions, adj, curv_dict)
            h = h + h_attn

        # Readout
        if self.task == 'graph':
            h_graph = h.mean(dim=0)  # mean pooling
            output = self.readout(h_graph)
        else:
            output = self.readout(h)

        # Physics regularisation
        if return_energy:
            if self.regularise_flag:
                energy = self.regulariser(h, positions, adj, faces)
            else:
                energy = torch.tensor(0.0, device=output.device)
            return output, energy

        return output


# ---------------------------------------------------------------------------
# Physics-Informed Graph Generator
# ---------------------------------------------------------------------------

class PhysicsInformedGraphGenerator(nn.Module):
    """Generative model for spatial graphs using reaction-diffusion dynamics.

    The generation process mirrors morphogenesis:
        1. Start from a latent code z (the "genotype")
        2. Initialise node features on a fully connected seed
        3. Run reaction-diffusion dynamics to develop spatial structure
        4. Predict positions and adjacency from the evolved features

    The reaction-diffusion layers provide the core generation mechanism,
    directly implementing the Turing instability that drives biological
    pattern formation.  The model learns which patterns to generate
    through the latent code and learned reaction terms.

    Architecture:
        z → node_init(z) → [ReactionDiffusion × K] → pos_head, adj_head, mask_head
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_channels: int = 64,
        max_nodes: int = 64,
        num_rd_steps: int = 6,
        num_species: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.max_nodes = max_nodes
        self.num_species = num_species

        # Latent → initial node features
        self.node_init = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, max_nodes * hidden_channels * num_species),
        )

        # Reaction-diffusion blocks
        self.rd_layers = nn.ModuleList()
        for _ in range(num_rd_steps):
            self.rd_layers.append(
                ReactionDiffusionLayer(
                    channels=hidden_channels,
                    reaction_hidden=hidden_channels * 2,
                    num_species=num_species,
                    coupled=True,
                )
            )

        # Edge feature encoder for dynamic topology
        self.edge_encoder = GeometricEdgeEncoder(
            hidden_dim=hidden_channels,
            out_dim=hidden_channels,
        )

        # Output heads
        species_total = hidden_channels * num_species

        # Position prediction
        self.pos_head = nn.Sequential(
            nn.Linear(species_total, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3),
        )

        # Adjacency prediction (pairwise)
        self.adj_head = nn.Sequential(
            nn.Linear(species_total * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

        # Node existence mask
        self.mask_head = nn.Sequential(
            nn.Linear(species_total, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(
        self,
        z: Tensor,
        target_positions: Optional[Tensor] = None,
        target_adj: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            z: (latent_dim,) latent code.
            target_*: training targets (None during generation).

        Returns:
            dict with 'positions', 'adj_logits', 'mask_logits', 'loss' (if targets given).
        """
        N = self.max_nodes
        C = self.hidden_channels
        S = self.num_species

        # Initialise node features from latent code
        h_flat = self.node_init(z)  # (N * C * S,)
        h_all = h_flat.view(N, S, C)  # (N, S, C)
        species = [h_all[:, s, :] for s in range(S)]

        # Initial topology: K-nearest-neighbor from random positions
        # or from evolved features via inner product
        # Start with a soft fully-connected graph that the RD dynamics will refine
        init_adj = torch.ones(N, N, device=z.device) / N
        init_adj.fill_diagonal_(0.0)

        # Normalised Laplacian from initial adjacency
        D = init_adj.sum(dim=1).clamp(min=1e-8)
        D_inv = 1.0 / D
        L_norm = torch.diag(D_inv) @ init_adj - torch.eye(N, device=z.device)

        # Run reaction-diffusion dynamics
        for step, rd_layer in enumerate(self.rd_layers):
            species = rd_layer(species, L_norm)

            # Periodically update topology from current positions
            if step > 0 and step % 2 == 0:
                # Predict intermediate positions
                h_concat = torch.cat(species, dim=-1)
                pos_intermediate = self.pos_head(h_concat)
                # Update Laplacian based on geometry
                diff = pos_intermediate.unsqueeze(1) - pos_intermediate.unsqueeze(0)
                dist = diff.norm(dim=2).clamp(min=1e-8)
                sigma = dist.mean()
                new_adj = torch.exp(-dist.pow(2) / (2 * sigma.pow(2) + 1e-8))
                diag_mask = 1.0 - torch.eye(N, device=z.device)
                new_adj = new_adj * diag_mask
                D = new_adj.sum(dim=1).clamp(min=1e-8)
                D_inv = 1.0 / D
                L_norm = torch.diag(D_inv) @ new_adj - torch.eye(N, device=z.device)

        # Final predictions
        h_final = torch.cat(species, dim=-1)  # (N, S*C)

        positions_pred = self.pos_head(h_final)  # (N, 3)
        mask_logits = self.mask_head(h_final).squeeze(-1)  # (N,)

        # Pairwise adjacency
        h_i = h_final.unsqueeze(1).expand(-1, N, -1)  # (N, N, S*C)
        h_j = h_final.unsqueeze(0).expand(N, -1, -1)  # (N, N, S*C)
        adj_logits = self.adj_head(torch.cat([h_i, h_j], dim=-1)).squeeze(-1)  # (N, N)
        adj_logits = (adj_logits + adj_logits.T) / 2  # symmetrise
        diag_mask = torch.eye(N, device=z.device, dtype=torch.bool)
        adj_logits = adj_logits.masked_fill(diag_mask, float('-inf'))

        result = {
            'positions': positions_pred,
            'adj_logits': adj_logits,
            'mask_logits': mask_logits,
        }

        # Compute loss if targets provided
        if target_positions is not None and target_adj is not None:
            result['loss'] = self._compute_loss(
                positions_pred, adj_logits, mask_logits,
                target_positions, target_adj, target_mask,
            )

        return result

    def _compute_loss(
        self,
        pos_pred, adj_logits, mask_logits,
        pos_target, adj_target, mask_target,
    ) -> Tensor:
        """Multi-component loss with physics regularisation."""
        N = pos_pred.size(0)

        # Determine valid nodes
        if mask_target is not None:
            valid = mask_target > 0.5
        else:
            valid = torch.ones(N, dtype=torch.bool, device=pos_pred.device)

        n_valid = valid.sum().clamp(min=1)

        # Position loss (MSE on valid nodes)
        pos_loss = F.mse_loss(pos_pred[valid], pos_target[valid])

        # Adjacency loss (BCE on valid node pairs)
        valid_mask_2d = valid.unsqueeze(1) & valid.unsqueeze(0)
        # Upper triangle only (symmetric)
        upper = torch.triu(torch.ones(N, N, dtype=torch.bool, device=pos_pred.device), diagonal=1)
        pair_mask = valid_mask_2d & upper
        if pair_mask.sum() > 0:
            adj_loss = F.binary_cross_entropy_with_logits(
                adj_logits[pair_mask],
                adj_target[pair_mask].float(),
            )
        else:
            adj_loss = torch.tensor(0.0, device=pos_pred.device)

        # Mask loss
        if mask_target is not None:
            mask_loss = F.binary_cross_entropy_with_logits(mask_logits, mask_target.float())
        else:
            mask_loss = torch.tensor(0.0, device=pos_pred.device)

        # Elastic energy regularisation on predicted positions
        adj_binary = (torch.sigmoid(adj_logits) > 0.5).float() * valid_mask_2d.float()
        edges = adj_binary.triu().nonzero(as_tuple=False)
        if edges.size(0) > 0:
            edge_lens = (pos_pred[edges[:, 0]] - pos_pred[edges[:, 1]]).norm(dim=1)
            elastic = ((edge_lens - edge_lens.mean().detach()).pow(2)).mean()
        else:
            elastic = torch.tensor(0.0, device=pos_pred.device)

        total = pos_loss + adj_loss + 0.5 * mask_loss + 0.01 * elastic
        return total

    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 1,
        device: torch.device | None = None,
        threshold: float = 0.5,
    ) -> list[dict[str, Tensor]]:
        """Generate spatial graphs from random latent codes."""
        self.eval()
        results = []

        for _ in range(num_samples):
            z = torch.randn(self.latent_dim, device=device)
            out = self.forward(z)

            mask = torch.sigmoid(out['mask_logits']) > threshold
            adj = torch.sigmoid(out['adj_logits']) > threshold
            adj = adj.float() * mask.unsqueeze(1).float() * mask.unsqueeze(0).float()
            adj.fill_diagonal_(0.0)

            results.append({
                'positions': out['positions'][mask],
                'adjacency': adj[mask][:, mask],
                'num_nodes': mask.sum().item(),
            })

        return results
