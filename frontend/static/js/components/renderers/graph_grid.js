// Mini-graph grid renderer shared by the Generator, VAE, and Diffusion
// panels. Each graph is drawn with renderMiniGraph plus a caption.

import { renderMiniGraph } from '../../graph_viz.js';

export function renderGraphGrid(containerId, graphs) {
  const grid = document.getElementById(containerId);
  if (!grid) return;
  grid.innerHTML = '';

  graphs.forEach(g => {
    const cell = document.createElement('div');
    cell.className = 'graph-cell';
    grid.appendChild(cell);

    const seen = new Set();
    const edges = [];
    for (const [s, t] of g.edges) {
      const key = Math.min(s, t) + ',' + Math.max(s, t);
      if (!seen.has(key)) {
        seen.add(key);
        edges.push([s, t]);
      }
    }

    renderMiniGraph(cell, {
      nodes: Array.from({ length: g.num_nodes }, (_, j) => ({ id: j })),
      edges,
    });

    const label = document.createElement('div');
    label.className = 'graph-cell-label';
    label.textContent = `n=${g.num_nodes} e=${g.num_edges} d=${g.density}`;
    cell.appendChild(label);
  });
}
