"""Framework-native Graph VAE and discrete graph diffusion.

Adapts :class:`backend.core.graphvae.model.GraphVAE` and ``DenoisingDiffusion``
to the ``core.BaseModel`` / ``core.BaseDataset`` protocol.

Registered components:

- ``graph_vae``          - Graph VAE (BaseModel).
- ``graph_diffusion``    - Discrete denoising diffusion on adjacency (BaseModel).
- ``social_skeletons``   - Synthetic Watts-Strogatz/BA mixture (BaseDataset).
"""

from __future__ import annotations

from algorithms.graphvae.dataset import SocialSkeletonsDataset
from algorithms.graphvae.model import GraphDiffusionModel, GraphVAEModel

__all__ = ["GraphVAEModel", "GraphDiffusionModel", "SocialSkeletonsDataset"]
