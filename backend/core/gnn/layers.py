"""GNN layer implementations (GCN, GAT) using dense adjacency matrices."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.core.graph_utils import normalize_adj


class GCNLayer(nn.Module):
    """Spectral convolution from Kipf & Welling, ICLR 2017."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.dropout = dropout
        self._reset_parameters()

    def _reset_parameters(self):
        # Glorot uniform -- standard for GCNs, keeps variance stable across layers
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # pre-compute A_hat once per forward; caller can cache if needed
        a_hat = normalize_adj(adj, add_self_loops=True)

        x = F.dropout(x, p=self.dropout, training=self.training)
        # message passing: A_hat @ X @ W
        h = a_hat @ x @ self.weight
        if self.bias is not None:
            h = h + self.bias
        return h


class GATLayer(nn.Module):
    """Multi-head attention from Velickovic et al., ICLR 2018.

    Dense implementation -- works on (N, N) adjacency, not sparse edge lists.
    """

    def __init__(self, in_features: int, out_features: int, n_heads: int = 1,
                 concat: bool = True, dropout: float = 0.0,
                 leaky_relu_slope: float = 0.2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_features
        self.concat = concat
        self.dropout = dropout

        # shared linear projection across heads, then reshape
        self.W = nn.Parameter(torch.empty(in_features, n_heads * out_features))
        # attention coefficients: a_l and a_r for source and target
        # (split trick from the original paper -- avoids concatenation in inner loop)
        self.a_l = nn.Parameter(torch.empty(n_heads, out_features))
        self.a_r = nn.Parameter(torch.empty(n_heads, out_features))

        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_l.unsqueeze(0))  # treat as (1, H, F) for init
        nn.init.xavier_uniform_(self.a_r.unsqueeze(0))

    def forward(self, x, adj):
        N = x.size(0)

        x = F.dropout(x, p=self.dropout, training=self.training)
        # project: (N, F_in) -> (N, H, F_out)
        h = (x @ self.W).view(N, self.n_heads, self.head_dim)

        # attention logits via additive decomposition:
        #   e_ij = LeakyReLU( a_l^T h_i + a_r^T h_j )
        # this avoids the expensive (N, N, 2F) concat from the paper
        score_l = (h * self.a_l).sum(dim=-1)  # (N, H)
        score_r = (h * self.a_r).sum(dim=-1)  # (N, H)
        # broadcast: (N, 1, H) + (1, N, H) -> (N, N, H)
        attn_logits = score_l.unsqueeze(1) + score_r.unsqueeze(0)
        attn_logits = self.leaky_relu(attn_logits)  # (N, N, H)

        # mask out non-edges with -inf so softmax gives 0
        # adj is (N, N), add self-loops for attention (node attends to itself)
        mask = adj.unsqueeze(-1) + torch.eye(N, device=adj.device).unsqueeze(-1)
        attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=1)  # normalize over source nodes
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # weighted aggregation: for each head, sum neighbors
        # attn_weights: (N, N, H), h: (N, H, F_out)
        # -> einsum is cleaner than manual transposes here
        out = torch.einsum("ijh,jhf->ihf", attn_weights, h)

        if self.concat:
            # (N, H*F_out) -- used in hidden layers
            return out.reshape(N, self.n_heads * self.head_dim)
        else:
            # average heads -- typically for final layer
            return out.mean(dim=1)
