"""
Utility functions for graph operations using pure PyTorch tensors.

Graph representation used throughout:
    - adj:      (N, N) adjacency matrix, float tensor
    - features: (N, F) node feature matrix
    - edge_index: (2, E) COO-format edge list (used in message passing)
"""

import torch
from torch import Tensor


def adj_to_edge_index(adj: Tensor) -> Tensor:
    """Convert dense adjacency matrix to COO edge_index (2, E)."""
    row, col = torch.where(adj > 0)
    return torch.stack([row, col], dim=0)


def edge_index_to_adj(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Convert COO edge_index back to dense adjacency."""
    adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj


def compute_degree(adj: Tensor) -> Tensor:
    """Node degree vector from adjacency matrix."""
    return adj.sum(dim=1)


def normalize_adj(adj: Tensor, add_self_loops: bool = True) -> Tensor:
    """Symmetric normalization: D^{-1/2} A D^{-1/2}, following Kipf & Welling (2017)."""
    if add_self_loops:
        adj = adj + torch.eye(adj.size(0), device=adj.device)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
    # D^{-1/2} A D^{-1/2}
    return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)


def clustering_coefficient(adj: Tensor) -> Tensor:
    """Per-node clustering coefficient computed on the adjacency matrix.

    C_i = 2 * |triangles through i| / (deg_i * (deg_i - 1))
    """
    # A^3 diagonal gives 2x number of triangles through each node
    a3_diag = torch.diagonal(adj @ adj @ adj)
    deg = adj.sum(dim=1)
    denom = deg * (deg - 1)
    denom[denom == 0] = 1.0  # avoid division by zero for isolated/leaf nodes
    return a3_diag / denom


def graph_density(adj: Tensor) -> float:
    """Edge density of an undirected graph (ignoring self-loops)."""
    n = adj.size(0)
    if n < 2:
        return 0.0
    mask = 1.0 - torch.eye(n, device=adj.device)
    num_edges = (adj * mask).sum().item()
    return num_edges / (n * (n - 1))


def erdos_renyi(n: int, p: float, device: torch.device | None = None) -> Tensor:
    """Sample an Erdos-Renyi random graph G(n, p)."""
    rand = torch.rand(n, n, device=device)
    adj = (rand < p).float()
    # make symmetric and zero diagonal
    adj = torch.triu(adj, diagonal=1)
    adj = adj + adj.t()
    return adj


def barabasi_albert(n: int, m: int = 2, device: torch.device | None = None) -> Tensor:
    """Preferential attachment graph. Each new node attaches to m existing nodes."""
    adj = torch.zeros(n, n, device=device)
    # start with a small clique of size m+1
    for i in range(min(m + 1, n)):
        for j in range(i + 1, min(m + 1, n)):
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    for new_node in range(m + 1, n):
        deg = adj[:new_node].sum(dim=1)
        total_deg = deg.sum()
        if total_deg == 0:
            probs = torch.ones(new_node, device=device)
        else:
            probs = deg / total_deg
        targets = torch.multinomial(probs, min(m, new_node), replacement=False)
        adj[new_node, targets] = 1.0
        adj[targets, new_node] = 1.0
    return adj


def watts_strogatz(n: int, k: int = 4, p: float = 0.1,
                   device: torch.device | None = None) -> Tensor:
    """Small-world graph: start with ring lattice, rewire with probability p."""
    adj = torch.zeros(n, n, device=device)
    # ring lattice: each node connected to k//2 neighbors on each side
    for i in range(n):
        for j in range(1, k // 2 + 1):
            right = (i + j) % n
            adj[i, right] = 1.0
            adj[right, i] = 1.0

    # rewire
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if torch.rand(1).item() < p:
                old = (i + j) % n
                # pick a random target that isn't i and isn't already connected
                candidates = (adj[i] == 0).float()
                candidates[i] = 0.0
                if candidates.sum() == 0:
                    continue
                new = torch.multinomial(candidates, 1).item()
                adj[i, old] = 0.0
                adj[old, i] = 0.0
                adj[i, new] = 1.0
                adj[new, i] = 1.0
    return adj


def random_node_features(n: int, dim: int, device: torch.device | None = None) -> Tensor:
    """Random feature matrix, useful for testing."""
    return torch.randn(n, dim, device=device)


def one_hot_degree_features(adj: Tensor, max_degree: int = 10) -> Tensor:
    """Create node features from degree via one-hot encoding."""
    deg = adj.sum(dim=1).long().clamp(max=max_degree)
    return torch.nn.functional.one_hot(deg, num_classes=max_degree + 1).float()
