import { api } from './api.js';
import { renderGraph, renderMiniGraph, drawLossCurve } from './graph_viz.js';

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.remove('active');
      b.setAttribute('aria-selected', 'false');
    });
    btn.classList.add('active');
    btn.setAttribute('aria-selected', 'true');

    document.querySelectorAll('.panel').forEach(p => {
      p.classList.remove('active');
      p.hidden = true;
    });
    const target = document.getElementById(`panel-${btn.dataset.tab}`);
    if (target) {
      target.classList.add('active');
      target.hidden = false;
    }
  });
});

// ---------------------------------------------------------------------------
// Slider value display
// ---------------------------------------------------------------------------
document.querySelectorAll('.slider-field input[type="range"]').forEach(slider => {
  const output = slider.parentElement.querySelector('.slider-value');
  if (output) {
    slider.addEventListener('input', () => {
      output.textContent = parseFloat(slider.value).toFixed(
        slider.step.includes('.') ? slider.step.split('.')[1].length : 0
      );
    });
  }
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function setLoading(btn, loading) {
  btn.classList.toggle('loading', loading);
  btn.disabled = loading;
}

function log(outputId, msg) {
  const el = document.getElementById(outputId);
  if (!el) return;
  el.textContent += msg + '\n';
  el.scrollTop = el.scrollHeight;
}

let gnnGraphHandle = null;
let gnnGraphData = null;

// ---------------------------------------------------------------------------
// GNN Panel
// ---------------------------------------------------------------------------
document.getElementById('btn-load-graph')?.addEventListener('click', async function () {
  setLoading(this, true);
  try {
    const data = await api.loadKarateClub();
    gnnGraphData = data;
    document.getElementById('load-status').textContent = `${data.num_nodes} nodes, ${data.num_edges} edges`;
    document.getElementById('load-status').className = 'status-text success';
    document.getElementById('btn-train-gnn').disabled = false;

    const container = document.getElementById('gnn-graph-container');
    // convert edge pairs that include both directions — deduplicate
    const edgeSet = new Set();
    const edges = [];
    for (const [s, t] of data.edges) {
      const key = Math.min(s, t) + ',' + Math.max(s, t);
      if (!edgeSet.has(key)) {
        edgeSet.add(key);
        edges.push([s, t]);
      }
    }

    if (gnnGraphHandle) gnnGraphHandle.destroy();
    gnnGraphHandle = renderGraph(container, {
      nodes: Array.from({ length: data.num_nodes }, (_, i) => ({ id: i })),
      edges,
      labels: data.labels,
    });
  } catch (e) {
    document.getElementById('load-status').textContent = e.message;
    document.getElementById('load-status').className = 'status-text error';
  }
  setLoading(this, false);
});

document.getElementById('btn-train-gnn')?.addEventListener('click', async function () {
  setLoading(this, true);
  const statusLog = document.getElementById('gnn-status-log');
  statusLog.textContent = 'Training...\n';

  try {
    const params = {
      layer_type: document.getElementById('gnn-layer-type').value.toLowerCase(),
      hidden_dim: +document.getElementById('gnn-hidden-dim').value,
      num_layers: +document.getElementById('gnn-num-layers').value,
      lr: +document.getElementById('gnn-lr').value,
      epochs: +document.getElementById('gnn-epochs').value,
    };
    const result = await api.trainGNN(params);
    log('gnn-status-log', `Done — loss: ${result.final_loss}, acc: ${result.final_accuracy}`);

    // draw loss curve
    const container = document.getElementById('gnn-loss-container');
    container.innerHTML = '<canvas width="280" height="160"></canvas>';
    const canvas = container.querySelector('canvas');
    drawLossCurve(canvas, result.loss_curve);

    document.getElementById('predict-fieldset').disabled = false;
  } catch (e) {
    log('gnn-status-log', `Error: ${e.message}`);
  }
  setLoading(this, false);
});

document.getElementById('btn-predict-gnn')?.addEventListener('click', async function () {
  setLoading(this, true);
  try {
    const raw = document.getElementById('gnn-node-ids').value.trim();
    const nodeIds = raw ? raw.split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n)) : null;
    const result = await api.predictGNN(nodeIds);

    const predDiv = document.getElementById('gnn-predictions');
    predDiv.innerHTML = `<div style="margin-bottom:8px;color:var(--accent);font-weight:500">Accuracy: ${(result.accuracy * 100).toFixed(1)}%</div>`;
    for (const p of result.predictions) {
      predDiv.innerHTML += `<div class="pred-row"><span class="pred-label">Node ${p.node}</span><span class="pred-value">${p.predicted} (${(p.confidence * 100).toFixed(0)}%)</span></div>`;
    }

    if (gnnGraphHandle && gnnGraphData) {
      const preds = new Array(gnnGraphData.num_nodes).fill(-1);
      for (const p of result.predictions) preds[p.node] = p.predicted;
      gnnGraphHandle.updatePredictions(preds);
    }
  } catch (e) {
    log('gnn-status-log', `Predict error: ${e.message}`);
  }
  setLoading(this, false);
});

// ---------------------------------------------------------------------------
// Generator Panel
// ---------------------------------------------------------------------------
document.getElementById('btn-train-gen')?.addEventListener('click', async function () {
  setLoading(this, true);
  log('gen-status-log', 'Training generator...');
  try {
    const params = {
      hidden_dim: +document.getElementById('gen-hidden-dim').value,
      lr: +document.getElementById('gen-lr').value,
      epochs: +document.getElementById('gen-epochs').value,
    };
    const result = await api.trainGenerator(params);
    log('gen-status-log', `Done — final loss: ${result.final_loss}`);

    const container = document.getElementById('gen-loss-container');
    container.innerHTML = '<canvas width="240" height="120"></canvas>';
    drawLossCurve(container.querySelector('canvas'), result.loss_curve);
  } catch (e) {
    log('gen-status-log', `Error: ${e.message}`);
  }
  setLoading(this, false);
});

document.getElementById('btn-generate')?.addEventListener('click', async function () {
  setLoading(this, true);
  try {
    const params = {
      num_nodes: +document.getElementById('cond-nodes').value / 50, // normalize to [0, 1]
      density: +document.getElementById('cond-density').value,
      clustering: +document.getElementById('cond-clustering').value,
      num_samples: +document.getElementById('gen-num-samples').value,
    };
    const result = await api.generateGraphs(params);
    renderGraphGrid('gen-graph-grid', result.graphs);
  } catch (e) {
    log('gen-status-log', `Generate error: ${e.message}`);
  }
  setLoading(this, false);
});

// ---------------------------------------------------------------------------
// VAE / Diffusion Panel
// ---------------------------------------------------------------------------
document.getElementById('btn-train-vae')?.addEventListener('click', async function () {
  setLoading(this, true);
  log('vae-status-log', 'Training VAE...');
  try {
    const params = {
      latent_dim: +document.getElementById('vae-latent-dim').value,
      hidden_dim: +document.getElementById('vae-hidden-dim').value,
      lr: +document.getElementById('vae-lr').value,
      epochs: +document.getElementById('vae-epochs').value,
    };
    const result = await api.trainVAE(params);
    log('vae-status-log', `Done — final loss: ${result.final_loss}`);

    const container = document.getElementById('vae-loss-container');
    container.innerHTML = '<canvas width="240" height="120"></canvas>';
    drawLossCurve(container.querySelector('canvas'), result.loss_curve);
  } catch (e) {
    log('vae-status-log', `Error: ${e.message}`);
  }
  setLoading(this, false);
});

document.getElementById('btn-gen-vae')?.addEventListener('click', async function () {
  setLoading(this, true);
  try {
    const n = +document.getElementById('vae-gen-samples').value;
    const result = await api.generateVAE(n);
    renderGraphGrid('vae-graph-grid', result.graphs);
  } catch (e) {
    log('vae-status-log', `Generate error: ${e.message}`);
  }
  setLoading(this, false);
});

document.getElementById('btn-train-diff')?.addEventListener('click', async function () {
  setLoading(this, true);
  log('diff-status-log', 'Training diffusion model...');
  try {
    const params = {
      hidden_dim: +document.getElementById('diff-hidden-dim').value,
      timesteps: +document.getElementById('diff-timesteps').value,
      lr: +document.getElementById('diff-lr').value,
      epochs: +document.getElementById('diff-epochs').value,
    };
    const result = await api.trainDiffusion(params);
    log('diff-status-log', `Done — final loss: ${result.final_loss}`);

    const container = document.getElementById('diff-loss-container');
    container.innerHTML = '<canvas width="240" height="120"></canvas>';
    drawLossCurve(container.querySelector('canvas'), result.loss_curve);
  } catch (e) {
    log('diff-status-log', `Error: ${e.message}`);
  }
  setLoading(this, false);
});

document.getElementById('btn-gen-diff')?.addEventListener('click', async function () {
  setLoading(this, true);
  try {
    const n = +document.getElementById('diff-gen-samples').value;
    const result = await api.generateDiffusion(n);
    renderGraphGrid('vae-graph-grid', result.graphs);
  } catch (e) {
    log('diff-status-log', `Generate error: ${e.message}`);
  }
  setLoading(this, false);
});

document.getElementById('btn-interpolate')?.addEventListener('click', async function () {
  setLoading(this, true);
  try {
    const params = {
      graph_idx_a: +document.getElementById('interp-idx-a').value,
      graph_idx_b: +document.getElementById('interp-idx-b').value,
      steps: +document.getElementById('interp-steps').value,
    };
    const result = await api.interpolateVAE(params);
    // show interpolation steps in the graph grid
    const graphs = [result.source, ...result.steps, result.target];
    renderGraphGrid('vae-graph-grid', graphs);
  } catch (e) {
    log('vae-status-log', `Interpolate error: ${e.message}`);
  }
  setLoading(this, false);
});

// ---------------------------------------------------------------------------
// Spatial 3D Panel
// ---------------------------------------------------------------------------

let spatialModelType = 'vae';

document.getElementById('sp-model-type')?.addEventListener('change', function () {
  spatialModelType = this.value;
});

document.getElementById('btn-train-spatial')?.addEventListener('click', async function () {
  setLoading(this, true);
  const statusLog = document.getElementById('sp-status-log');
  statusLog.textContent = '';
  const modelType = document.getElementById('sp-model-type').value;
  log('sp-status-log', `Training spatial ${modelType}...`);

  try {
    const baseParams = {
      data_type: document.getElementById('sp-data-type').value,
      num_train: +document.getElementById('sp-num-train').value,
      num_nodes: +document.getElementById('sp-num-nodes').value,
      hidden_dim: +document.getElementById('sp-hidden-dim').value,
      epochs: +document.getElementById('sp-epochs').value,
      lr: +document.getElementById('sp-lr').value,
    };

    let result;
    if (modelType === 'vae') {
      result = await api.trainSpatialVAE({ ...baseParams, latent_dim: 32 });
    } else {
      result = await api.trainSpatialDiffusion({ ...baseParams, timesteps: 50 });
    }

    log('sp-status-log', `Done — loss: ${result.final_loss} | params: ${result.num_params.toLocaleString()}`);

    const container = document.getElementById('sp-loss-container');
    container.innerHTML = '<canvas width="240" height="120"></canvas>';
    drawLossCurve(container.querySelector('canvas'), result.loss_curve);

    spatialModelType = modelType;
  } catch (e) {
    log('sp-status-log', `Error: ${e.message}`);
  }
  setLoading(this, false);
});

document.getElementById('btn-gen-spatial')?.addEventListener('click', async function () {
  setLoading(this, true);
  try {
    const n = +document.getElementById('sp-gen-samples').value;
    const numNodes = +document.getElementById('sp-num-nodes').value;
    let result;

    if (spatialModelType === 'vae') {
      result = await api.generateSpatialVAE({ num_samples: n });
    } else {
      result = await api.generateSpatialDiffusion({ num_samples: n, num_nodes: numNodes });
    }

    renderSpatialGrid('sp-3d-grid', result.graphs);
    log('sp-status-log', `Generated ${result.graphs.length} trees`);
  } catch (e) {
    log('sp-status-log', `Generate error: ${e.message}`);
  }
  setLoading(this, false);
});

document.getElementById('btn-gen-synthetic')?.addEventListener('click', async function () {
  setLoading(this, true);
  try {
    const params = {
      type: document.getElementById('sp-data-type').value,
      num_graphs: 4,
      num_nodes: +document.getElementById('sp-num-nodes').value,
    };
    const result = await api.generateSynthetic(params);
    renderSpatialGrid('sp-3d-grid', result.graphs);
    log('sp-status-log', `Showing ${result.graphs.length} synthetic samples`);
  } catch (e) {
    log('sp-status-log', `Error: ${e.message}`);
  }
  setLoading(this, false);
});

document.getElementById('btn-analyze-spatial')?.addEventListener('click', async function () {
  setLoading(this, true);
  try {
    const result = await api.analyzeSpatial({
      model: spatialModelType,
      num_samples: 20,
    });

    const metricsDiv = document.getElementById('sp-metrics');
    metricsDiv.innerHTML = '';
    for (const [key, val] of Object.entries(result)) {
      metricsDiv.innerHTML += `<div class="metric-row"><span class="metric-name">${key}</span><span class="metric-value">${typeof val === 'number' ? val.toFixed(6) : val}</span></div>`;
    }
    log('sp-status-log', 'Evaluation complete');
  } catch (e) {
    log('sp-status-log', `Analysis error: ${e.message}`);
  }
  setLoading(this, false);
});

// ---------------------------------------------------------------------------
// Graph grid rendering (2D — for GNN / Generator / VAE panels)
// ---------------------------------------------------------------------------
function renderGraphGrid(containerId, graphs) {
  const grid = document.getElementById(containerId);
  if (!grid) return;
  grid.innerHTML = '';

  graphs.forEach((g, i) => {
    const cell = document.createElement('div');
    cell.className = 'graph-cell';
    grid.appendChild(cell);

    // deduplicate edges (API returns both directions)
    const edgeSet = new Set();
    const edges = [];
    for (const [s, t] of g.edges) {
      const key = Math.min(s, t) + ',' + Math.max(s, t);
      if (!edgeSet.has(key)) {
        edgeSet.add(key);
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


// ---------------------------------------------------------------------------
// Spatial 3D grid rendering (isometric projection)
// ---------------------------------------------------------------------------
function renderSpatialGrid(containerId, graphs) {
  const grid = document.getElementById(containerId);
  if (!grid) return;
  grid.innerHTML = '';

  graphs.forEach((g, idx) => {
    const cell = document.createElement('div');
    cell.className = 'spatial-cell';
    grid.appendChild(cell);

    renderSpatialTree(cell, g);

    const feats = g.features || {};
    const label = document.createElement('div');
    label.className = 'spatial-cell-label';
    const parts = [`n=${g.num_nodes}`];
    if (feats.num_branch_points !== undefined) parts.push(`br=${feats.num_branch_points}`);
    if (feats.num_tips !== undefined) parts.push(`tips=${feats.num_tips}`);
    if (feats.strahler_order !== undefined) parts.push(`S=${feats.strahler_order}`);
    label.textContent = parts.join('  ');
    cell.appendChild(label);
  });
}


/**
 * Render a spatial tree as an isometric 2D projection via SVG.
 *
 * Projects 3D positions (x, y, z) to 2D using a simple isometric transform:
 *   screenX = (x - z) * cos(30)
 *   screenY = -y + (x + z) * sin(30) * 0.5
 */
function renderSpatialTree(container, graphData) {
  const size = container.clientWidth || 260;
  const pad = 30;

  const positions = graphData.positions;
  const edges = graphData.edges;
  const parent = graphData.parent;
  const nodeTypes = graphData.node_types || [];
  const n = positions.length;

  if (n === 0) return;

  // isometric projection
  const cos30 = Math.cos(Math.PI / 6);
  const sin30 = Math.sin(Math.PI / 6);

  const projected = positions.map(([x, y, z]) => ({
    px: (x - z) * cos30,
    py: -y + (x + z) * sin30 * 0.5,
  }));

  // fit to viewport
  const xs = projected.map(p => p.px);
  const ys = projected.map(p => p.py);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const scale = (size - 2 * pad) / Math.max(rangeX, rangeY);

  const cx = size / 2;
  const cy = size / 2;
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;

  function toScreen(p) {
    return {
      x: cx + (p.px - midX) * scale,
      y: cy + (p.py - midY) * scale,
    };
  }

  const svg = d3.select(container).append('svg')
    .attr('viewBox', `0 0 ${size} ${size}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  // subtle axis indicators
  const axisLen = 25;
  const origin = { x: pad + 5, y: size - pad - 5 };
  const axes = [
    { dx: axisLen * cos30, dy: -axisLen * sin30 * 0.5, label: 'x' },
    { dx: 0, dy: -axisLen, label: 'y' },
    { dx: -axisLen * cos30, dy: -axisLen * sin30 * 0.5, label: 'z' },
  ];
  axes.forEach(a => {
    svg.append('line')
      .attr('class', 'axis-line')
      .attr('x1', origin.x).attr('y1', origin.y)
      .attr('x2', origin.x + a.dx).attr('y2', origin.y + a.dy);
    svg.append('text')
      .attr('class', 'axis-label')
      .attr('x', origin.x + a.dx * 1.3)
      .attr('y', origin.y + a.dy * 1.3 + 3)
      .text(a.label);
  });

  // draw edges using parent array (preferred) or edge list
  const edgeGroup = svg.append('g');
  if (parent && parent.length === n) {
    for (let i = 0; i < n; i++) {
      const p = parent[i];
      if (p >= 0 && p < n) {
        const s = toScreen(projected[p]);
        const e = toScreen(projected[i]);
        edgeGroup.append('line')
          .attr('class', 'tree-edge')
          .attr('x1', s.x).attr('y1', s.y)
          .attr('x2', e.x).attr('y2', e.y);
      }
    }
  } else if (edges) {
    edges.forEach(([s, t]) => {
      const ps = toScreen(projected[s]);
      const pt = toScreen(projected[t]);
      edgeGroup.append('line')
        .attr('class', 'tree-edge')
        .attr('x1', ps.x).attr('y1', ps.y)
        .attr('x2', pt.x).attr('y2', pt.y);
    });
  }

  // draw nodes
  const nodeGroup = svg.append('g');
  // compute child counts for coloring
  const childCounts = new Array(n).fill(0);
  if (parent) {
    for (let i = 0; i < n; i++) {
      const p = parent[i];
      if (p >= 0 && p < n) childCounts[p]++;
    }
  }

  for (let i = 0; i < n; i++) {
    const sc = toScreen(projected[i]);
    let nodeClass = 'node-default';

    if (i === 0) {
      nodeClass = 'node-soma';
    } else if (childCounts[i] === 0) {
      nodeClass = 'node-tip';
    } else if (childCounts[i] >= 2) {
      nodeClass = 'node-branch';
    }

    // depth-based sizing
    const r = i === 0 ? 4.5 : (childCounts[i] >= 2 ? 3 : 2);

    nodeGroup.append('circle')
      .attr('class', `tree-node ${nodeClass}`)
      .attr('cx', sc.x)
      .attr('cy', sc.y)
      .attr('r', r);
  }
}
