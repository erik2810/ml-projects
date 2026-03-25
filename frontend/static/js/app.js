import { api } from './api.js';
import { renderGraph, renderMiniGraph, renderFixedGraph, renderWLGraph, drawLossCurve } from './graph_viz.js';

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
      num_nodes: +document.getElementById('cond-nodes').value,
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
    const result = await api.interpolateVAEGeometric(params);
    // Render as interpolation strip with fixed positions
    const strip = document.getElementById('vae-interp-strip');
    if (strip) {
      strip.innerHTML = '';
      strip.hidden = false;
      result.steps.forEach((step, i) => {
        const stepDiv = document.createElement('div');
        stepDiv.className = 'interp-step';
        if (i === 0 || i === result.steps.length - 1) {
          stepDiv.classList.add('is-endpoint');
        }
        const graphDiv = document.createElement('div');
        graphDiv.className = 'interp-step-graph';
        graphDiv.style.width = '140px';
        graphDiv.style.height = '140px';
        stepDiv.appendChild(graphDiv);
        const label = document.createElement('span');
        label.className = 'interp-step-label';
        const t = i / Math.max(result.steps.length - 1, 1);
        label.textContent = i === 0 ? 'Source' : i === result.steps.length - 1 ? 'Target' : `t=${t.toFixed(2)}`;
        stepDiv.appendChild(label);
        strip.appendChild(stepDiv);
        renderFixedGraph(graphDiv, { positions: step.positions, edges: step.edges });
      });
    } else {
      // Fallback: render in grid
      renderGraphGrid('vae-graph-grid', result.steps);
    }
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

    if (g.type === 'mesh') {
      renderSpatialMesh(cell, g);
    } else {
      renderSpatialTree(cell, g);
    }

    const feats = g.features || {};
    // Show mesh name if available
    if (g.name) {
      const nameEl = document.createElement('div');
      nameEl.className = 'spatial-cell-name';
      nameEl.textContent = g.name;
      cell.appendChild(nameEl);
    }
    const label = document.createElement('div');
    label.className = 'spatial-cell-label';
    const parts = [`n=${g.num_nodes}  e=${g.num_edges}`];
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


/**
 * Render a spatial mesh (general graph with cycles) as pink wireframe.
 * Uses the same isometric projection as renderSpatialTree but with
 * edge-list rendering and uniform node color.
 */
function renderSpatialMesh(container, graphData) {
  const size = container.clientWidth || 260;
  const pad = 30;

  const positions = graphData.positions;
  const edges = graphData.edges;
  const n = positions.length;

  if (n === 0) return;

  const cos30 = Math.cos(Math.PI / 6);
  const sin30 = Math.sin(Math.PI / 6);

  const projected = positions.map(([x, y, z]) => ({
    px: (x - z) * cos30,
    py: -y + (x + z) * sin30 * 0.5,
  }));

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

  // axis indicators
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

  // draw edges (pink wireframe)
  const edgeGroup = svg.append('g');
  if (edges) {
    edges.forEach(([s, t]) => {
      const ps = toScreen(projected[s]);
      const pt = toScreen(projected[t]);
      edgeGroup.append('line')
        .attr('class', 'mesh-edge')
        .attr('x1', ps.x).attr('y1', ps.y)
        .attr('x2', pt.x).attr('y2', pt.y);
    });
  }

  // draw nodes (uniform pink)
  const nodeGroup = svg.append('g');
  for (let i = 0; i < n; i++) {
    const sc = toScreen(projected[i]);
    nodeGroup.append('circle')
      .attr('class', 'tree-node node-mesh')
      .attr('cx', sc.x)
      .attr('cy', sc.y)
      .attr('r', 2.5);
  }
}


// ---------------------------------------------------------------------------
// Physics GNN Panel
// ---------------------------------------------------------------------------

let physicsDataLoaded = false;
let physicsGraphData = null;

function renderPhysicsGraph(containerId, graphData, colorField = 'labels') {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = '';

  const positions = graphData.positions;
  const edges = graphData.edges;
  const n = positions.length;
  if (n === 0) return;

  const size = Math.min(el.clientWidth || 500, el.clientHeight || 400, 500);
  const pad = 35;

  const cos30 = Math.cos(Math.PI / 6);
  const sin30 = Math.sin(Math.PI / 6);
  const projected = positions.map(([x, y, z]) => ({
    px: (x - z) * cos30,
    py: -y + (x + z) * sin30 * 0.5,
  }));

  const xs = projected.map(p => p.px);
  const ys = projected.map(p => p.py);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const scale = (size - 2 * pad) / Math.max(rangeX, rangeY);
  const cx = size / 2, cy = size / 2;
  const midX = (minX + maxX) / 2, midY = (minY + maxY) / 2;

  function toScreen(p) {
    return { x: cx + (p.px - midX) * scale, y: cy + (p.py - midY) * scale };
  }

  const svg = d3.select(el).append('svg')
    .attr('viewBox', `0 0 ${size} ${size}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  // Edges
  const edgeGroup = svg.append('g');
  if (edges) {
    edges.forEach(([s, t]) => {
      const ps = toScreen(projected[s]), pt = toScreen(projected[t]);
      edgeGroup.append('line')
        .attr('class', 'physics-edge')
        .attr('x1', ps.x).attr('y1', ps.y)
        .attr('x2', pt.x).attr('y2', pt.y);
    });
  }

  // Color palette
  const classColors = ['#3b82f6', '#f59e0b', '#ef4444'];
  const heatmap = (v) => {
    const r = Math.round(255 * Math.min(1, v * 2));
    const b = Math.round(255 * Math.min(1, (1 - v) * 2));
    const g = Math.round(255 * Math.max(0, 1 - Math.abs(v - 0.5) * 4));
    return `rgb(${r},${g},${b})`;
  };

  // Nodes
  const nodeGroup = svg.append('g');
  const colorValues = graphData[colorField] || graphData.labels || [];
  const isDiscrete = colorField === 'labels' || colorField === 'predictions';

  for (let i = 0; i < n; i++) {
    const sc = toScreen(projected[i]);
    let fill;
    if (isDiscrete && colorValues[i] !== undefined) {
      fill = classColors[colorValues[i] % classColors.length];
    } else if (colorValues[i] !== undefined) {
      const vals = colorValues;
      const mn = Math.min(...vals), mx = Math.max(...vals);
      const norm = mx > mn ? (vals[i] - mn) / (mx - mn) : 0.5;
      fill = heatmap(norm);
    } else {
      fill = '#6366f1';
    }

    nodeGroup.append('circle')
      .attr('cx', sc.x).attr('cy', sc.y)
      .attr('r', 3.5)
      .attr('fill', fill)
      .attr('stroke', '#1e293b')
      .attr('stroke-width', 0.8);
  }

  // Legend
  if (isDiscrete && colorValues.length > 0) {
    const legendLabels = colorField === 'labels'
      ? ['Cold / Flat / Low', 'Warm / Curved / Mid', 'Hot / Sharp / High']
      : ['Predicted 0', 'Predicted 1', 'Predicted 2'];
    const legendG = svg.append('g').attr('transform', `translate(${pad}, ${size - 20})`);
    legendLabels.forEach((lbl, i) => {
      legendG.append('circle').attr('cx', i * 100).attr('cy', 0).attr('r', 4).attr('fill', classColors[i]);
      legendG.append('text').attr('x', i * 100 + 8).attr('y', 4)
        .attr('fill', '#94a3b8').attr('font-size', '9px').text(lbl);
    });
  }
}

function renderHeatmapGraph(container, graphData, values, title) {
  const el = typeof container === 'string' ? document.getElementById(container) : container;
  if (!el) return;
  el.innerHTML = '';

  const positions = graphData.positions;
  const edges = graphData.edges;
  const n = positions.length;
  if (n === 0) return;

  const size = 220;
  const pad = 25;

  const cos30 = Math.cos(Math.PI / 6);
  const sin30 = Math.sin(Math.PI / 6);
  const projected = positions.map(([x, y, z]) => ({
    px: (x - z) * cos30,
    py: -y + (x + z) * sin30 * 0.5,
  }));

  const xs = projected.map(p => p.px);
  const ys = projected.map(p => p.py);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const scl = (size - 2 * pad) / Math.max(maxX - minX || 1, maxY - minY || 1);
  const cx = size / 2, cy = size / 2;
  const midX = (minX + maxX) / 2, midY = (minY + maxY) / 2;
  const toS = p => ({ x: cx + (p.px - midX) * scl, y: cy + (p.py - midY) * scl });

  const svg = d3.select(el).append('svg')
    .attr('viewBox', `0 0 ${size} ${size + 20}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  // Title
  svg.append('text').attr('x', size / 2).attr('y', 14).attr('text-anchor', 'middle')
    .attr('fill', '#94a3b8').attr('font-size', '10px').text(title || '');

  const g = svg.append('g').attr('transform', 'translate(0, 16)');

  if (edges) {
    edges.forEach(([s, t]) => {
      const ps = toS(projected[s]), pt = toS(projected[t]);
      g.append('line').attr('x1', ps.x).attr('y1', ps.y).attr('x2', pt.x).attr('y2', pt.y)
        .attr('stroke', '#334155').attr('stroke-width', 0.6).attr('stroke-opacity', 0.4);
    });
  }

  const mn = Math.min(...values), mx = Math.max(...values);
  for (let i = 0; i < n; i++) {
    const sc = toS(projected[i]);
    const norm = mx > mn ? (values[i] - mn) / (mx - mn) : 0.5;
    const r = Math.round(255 * Math.min(1, norm * 2));
    const b = Math.round(255 * Math.min(1, (1 - norm) * 2));
    const green = Math.round(200 * Math.max(0, 1 - Math.abs(norm - 0.5) * 3));
    g.append('circle').attr('cx', sc.x).attr('cy', sc.y).attr('r', 3)
      .attr('fill', `rgb(${r},${green},${b})`).attr('stroke', '#1e293b').attr('stroke-width', 0.6);
  }
}

// Dataset generation
document.getElementById('btn-phys-dataset')?.addEventListener('click', async function () {
  setLoading(this, true);
  try {
    const params = {
      scenario: document.getElementById('phys-scenario').value,
      num_nodes: +document.getElementById('phys-num-nodes').value,
      grid_size: +document.getElementById('phys-grid-size').value,
    };
    const result = await api.physicsDataset(params);
    physicsGraphData = result.graph;
    physicsDataLoaded = true;

    renderPhysicsGraph('phys-main-viz', result.graph, 'labels');
    document.getElementById('phys-main-viz').scrollIntoView({ behavior: 'smooth', block: 'center' });
    document.getElementById('phys-viz-title').textContent = result.graph.name || 'Physical System';
    document.getElementById('phys-dataset-status').textContent = `${result.graph.num_nodes} nodes, ${result.graph.num_edges} edges`;
    document.getElementById('phys-dataset-status').className = 'status-text success';

    // Enable buttons
    document.getElementById('btn-phys-train').disabled = false;
    document.getElementById('btn-phys-ablation').disabled = false;
    document.getElementById('btn-phys-energy').disabled = false;
    document.getElementById('btn-phys-curvature').disabled = false;
    document.getElementById('btn-phys-heat').disabled = false;

    // Hide prediction panels
    document.getElementById('phys-predictions-viz').hidden = true;
    document.getElementById('phys-heat-strip').hidden = true;
    document.getElementById('phys-rd-strip').hidden = true;
    document.getElementById('phys-accuracy-display').hidden = true;

    log('phys-status-log', `Loaded ${params.scenario}: ${result.graph.num_nodes} nodes`);
  } catch (e) {
    log('phys-status-log', `Error: ${e.message}`);
  }
  setLoading(this, false);
});

// Training
document.getElementById('btn-phys-train')?.addEventListener('click', async function () {
  setLoading(this, true);
  document.getElementById('phys-status-log').textContent = '';
  log('phys-status-log', 'Training Physics-Informed GNN...');

  try {
    const params = {
      hidden_dim: +document.getElementById('phys-hidden-dim').value,
      num_layers: +document.getElementById('phys-num-layers').value,
      epochs: +document.getElementById('phys-epochs').value,
      lr: +document.getElementById('phys-lr').value,
      use_curvature: document.getElementById('phys-use-curvature').checked,
      use_diffusion: document.getElementById('phys-use-diffusion').checked,
      use_reaction_diffusion: document.getElementById('phys-use-rd').checked,
      use_attention: document.getElementById('phys-use-attention').checked,
      regularise: document.getElementById('phys-use-reg').checked,
    };

    const result = await api.physicsTrain(params);

    log('phys-status-log', `Done — train acc: ${result.train_accuracy} | val acc: ${result.val_accuracy} | params: ${result.num_params.toLocaleString()} | ${result.train_time}s`);

    // Loss curve
    const container = document.getElementById('phys-loss-container');
    container.innerHTML = '<canvas width="240" height="120"></canvas>';
    drawLossCurve(container.querySelector('canvas'), result.train_losses);

    // Show predictions
    if (physicsGraphData && result.predictions) {
      const predData = { ...physicsGraphData, predictions: result.predictions };
      document.getElementById('phys-predictions-viz').hidden = false;
      renderPhysicsGraph('phys-pred-container', predData, 'predictions');

      const accDisplay = document.getElementById('phys-accuracy-display');
      accDisplay.hidden = false;
      accDisplay.innerHTML = `
        <div class="physics-acc-row"><span>Train accuracy</span><strong>${(result.train_accuracy * 100).toFixed(1)}%</strong></div>
        <div class="physics-acc-row"><span>Val accuracy</span><strong>${(result.val_accuracy * 100).toFixed(1)}%</strong></div>
        <div class="physics-acc-row"><span>Parameters</span><strong>${result.num_params.toLocaleString()}</strong></div>
        <div class="physics-acc-row"><span>Best epoch</span><strong>${result.best_epoch}</strong></div>
      `;
    }
  } catch (e) {
    log('phys-status-log', `Error: ${e.message}`);
  }
  setLoading(this, false);
});

// Ablation
document.getElementById('btn-phys-ablation')?.addEventListener('click', async function () {
  setLoading(this, true);
  log('phys-status-log', 'Running ablation study (7 configurations)...');

  try {
    const result = await api.physicsAblation({
      epochs: +document.getElementById('phys-epochs').value,
    });

    const table = document.getElementById('phys-ablation-table');
    table.innerHTML = '';

    // Header
    const header = document.createElement('div');
    header.className = 'metric-row metric-header';
    header.innerHTML = '<span class="metric-name">Configuration</span><span class="metric-value">Accuracy</span><span class="metric-value">Params</span>';
    table.appendChild(header);

    // Find best accuracy for highlighting
    const bestAcc = Math.max(...result.results.map(r => r.accuracy));

    result.results.forEach(r => {
      const row = document.createElement('div');
      row.className = 'metric-row' + (r.accuracy === bestAcc ? ' metric-best' : '');
      row.innerHTML = `<span class="metric-name">${r.name}</span><span class="metric-value">${(r.accuracy * 100).toFixed(1)}%</span><span class="metric-value">${r.num_params}</span>`;
      table.appendChild(row);
    });

    log('phys-status-log', 'Ablation complete');
  } catch (e) {
    log('phys-status-log', `Ablation error: ${e.message}`);
  }
  setLoading(this, false);
});

// Energy analysis
document.getElementById('btn-phys-energy')?.addEventListener('click', async function () {
  setLoading(this, true);
  try {
    const result = await api.physicsEnergy();
    const display = document.getElementById('phys-energy-display');
    display.innerHTML = '';
    for (const [key, val] of Object.entries(result)) {
      display.innerHTML += `<div class="metric-row"><span class="metric-name">${key.replace(/_/g, ' ')}</span><span class="metric-value">${val}</span></div>`;
    }
    log('phys-status-log', `Energy: Dirichlet=${result.dirichlet} TV=${result.total_variation} Elastic=${result.elastic}`);
  } catch (e) {
    log('phys-status-log', `Energy error: ${e.message}`);
  }
  setLoading(this, false);
});

// Curvature analysis
document.getElementById('btn-phys-curvature')?.addEventListener('click', async function () {
  setLoading(this, true);
  try {
    const result = await api.physicsCurvatures();
    const display = document.getElementById('phys-curvature-display');
    display.innerHTML = '';
    for (const [key, data] of Object.entries(result)) {
      display.innerHTML += `<div class="metric-row"><span class="metric-name">${key.replace(/_/g, ' ')}</span><span class="metric-value">[${data.min}, ${data.max}] &mu;=${data.mean}</span></div>`;
    }

    // Visualise curvedness on the graph
    if (physicsGraphData && result.curvedness) {
      const curvData = { ...physicsGraphData, values: result.curvedness.values };
      renderPhysicsGraph('phys-main-viz', curvData, 'values');
      document.getElementById('phys-viz-title').textContent = 'Curvedness Heatmap';
    }

    log('phys-status-log', 'Curvature analysis complete');
  } catch (e) {
    log('phys-status-log', `Curvature error: ${e.message}`);
  }
  setLoading(this, false);
});

// Heat diffusion visualization
document.getElementById('btn-phys-heat')?.addEventListener('click', async function () {
  setLoading(this, true);
  try {
    const source = +document.getElementById('phys-heat-source').value;
    const result = await api.physicsHeatDiffusion({ source_nodes: [source], num_steps: 6, t_max: 5.0 });

    const strip = document.getElementById('phys-heat-strip');
    strip.hidden = false;
    strip.innerHTML = '';

    result.steps.forEach((step, i) => {
      const cell = document.createElement('div');
      cell.className = 'interp-step spatial-cell';
      cell.style.minWidth = '160px';
      cell.style.minHeight = '180px';
      strip.appendChild(cell);
      renderHeatmapGraph(cell, physicsGraphData, step.values, `t = ${step.t}`);
    });

    log('phys-status-log', `Heat diffusion from node ${source}: ${result.steps.length} timesteps`);
  } catch (e) {
    log('phys-status-log', `Heat diffusion error: ${e.message}`);
  }
  setLoading(this, false);
});

// Reaction-diffusion patterns
document.getElementById('btn-phys-rd')?.addEventListener('click', async function () {
  setLoading(this, true);
  log('phys-status-log', 'Running Gray-Scott reaction-diffusion...');
  try {
    const params = {
      grid_size: 8,
      feed_rate: +document.getElementById('phys-rd-feed').value,
      kill_rate: +document.getElementById('phys-rd-kill').value,
      num_steps: 6,
      steps_per_frame: 80,
    };
    const result = await api.physicsRDPattern(params);

    const strip = document.getElementById('phys-rd-strip');
    strip.hidden = false;
    strip.innerHTML = '';

    const rdGraph = {
      positions: result.positions,
      edges: result.edges,
      num_nodes: result.num_nodes,
    };

    result.frames.forEach((frame, i) => {
      const cell = document.createElement('div');
      cell.className = 'interp-step spatial-cell';
      cell.style.minWidth = '160px';
      cell.style.minHeight = '180px';
      strip.appendChild(cell);
      renderHeatmapGraph(cell, rdGraph, frame.B, `step ${frame.step}`);
    });

    log('phys-status-log', `Gray-Scott: f=${params.feed_rate} k=${params.kill_rate}, ${result.frames.length} frames`);
  } catch (e) {
    log('phys-status-log', `RD error: ${e.message}`);
  }
  setLoading(this, false);
});


// ---------------------------------------------------------------------------
// Mesh Interpolation Panel
// ---------------------------------------------------------------------------

// Helper: load showcase meshes into grid and update dropdowns
async function loadShowcaseMeshes() {
  const result = await api.generateMeshes({ mesh_type: 'showcase', num_meshes: 6 });
  renderSpatialGrid('mesh-3d-grid', result.graphs);
  document.getElementById('mesh-output-title').textContent = 'Showcase Shapes';

  // Hide interpolation section until interpolation is run
  const interpSection = document.getElementById('mesh-interp-section');
  if (interpSection) interpSection.hidden = true;

  // Populate source/target dropdowns with shape names
  const selA = document.getElementById('mesh-interp-a');
  const selB = document.getElementById('mesh-interp-b');
  const prevA = selA.value;
  const prevB = selB.value;
  selA.innerHTML = '';
  selB.innerHTML = '';
  result.graphs.forEach((g, i) => {
    const name = g.name || `Mesh ${i}`;
    selA.innerHTML += `<option value="${i}">${name}</option>`;
    selB.innerHTML += `<option value="${i}">${name}</option>`;
  });
  // Restore previous selection or use defaults
  selA.value = prevA < result.graphs.length ? prevA : '0';
  selB.value = prevB < result.graphs.length ? prevB : String(Math.min(3, result.graphs.length - 1));

  return result;
}

let meshShowcaseLoaded = false;

document.getElementById('btn-view-meshes')?.addEventListener('click', async function () {
  setLoading(this, true);
  try {
    await loadShowcaseMeshes();
    meshShowcaseLoaded = true;
    log('mesh-status-log', 'Loaded 6 showcase shapes');
  } catch (e) {
    log('mesh-status-log', `Error: ${e.message}`);
  }
  setLoading(this, false);
});

document.getElementById('btn-train-mesh-vae')?.addEventListener('click', async function () {
  setLoading(this, true);
  const statusLog = document.getElementById('mesh-status-log');
  statusLog.textContent = '';

  try {
    // Auto-load showcase meshes if not done yet
    if (!meshShowcaseLoaded) {
      log('mesh-status-log', 'Loading showcase shapes...');
      await loadShowcaseMeshes();
      meshShowcaseLoaded = true;
    }

    log('mesh-status-log', 'Training mesh VAE...');
    const params = {
      num_train: +document.getElementById('mesh-num-train').value,
      hidden_dim: +document.getElementById('mesh-hidden-dim').value,
      latent_dim: +document.getElementById('mesh-latent-dim').value,
      epochs: +document.getElementById('mesh-epochs').value,
      lr: +document.getElementById('mesh-lr').value,
    };

    const result = await api.trainMeshVAE(params);
    log('mesh-status-log', `Done — loss: ${result.final_loss} | params: ${result.num_params.toLocaleString()}`);

    const container = document.getElementById('mesh-loss-container');
    container.innerHTML = '<canvas width="240" height="120"></canvas>';
    drawLossCurve(container.querySelector('canvas'), result.loss_curve);
  } catch (e) {
    log('mesh-status-log', `Error: ${e.message}`);
  }
  setLoading(this, false);
});

document.getElementById('btn-gen-mesh-vae')?.addEventListener('click', async function () {
  setLoading(this, true);
  try {
    const n = +document.getElementById('mesh-gen-samples').value;
    const result = await api.generateMeshVAE({ num_samples: n });
    renderSpatialGrid('mesh-3d-grid', result.graphs);
    document.getElementById('mesh-output-title').textContent = 'Generated Meshes (VAE Prior)';
    const interpSection = document.getElementById('mesh-interp-section');
    if (interpSection) interpSection.hidden = true;
    log('mesh-status-log', `Generated ${result.graphs.length} meshes from VAE`);
  } catch (e) {
    log('mesh-status-log', `Generate error: ${e.message}`);
  }
  setLoading(this, false);
});

document.getElementById('btn-interp-mesh')?.addEventListener('click', async function () {
  setLoading(this, true);
  try {
    const idxA = +document.getElementById('mesh-interp-a').value;
    const idxB = +document.getElementById('mesh-interp-b').value;
    const params = {
      graph_idx_a: idxA,
      graph_idx_b: idxB,
      steps: +document.getElementById('mesh-interp-steps').value,
    };
    const result = await api.interpolateMeshVAE(params);
    const nameA = result.source_name || `#${idxA}`;
    const nameB = result.target_name || `#${idxB}`;

    // Show source + target in the main grid
    const srcData = result.source;
    srcData.name = nameA;
    const tgtData = result.target;
    tgtData.name = nameB;
    renderSpatialGrid('mesh-3d-grid', [srcData, tgtData]);
    document.getElementById('mesh-output-title').textContent = `Source & Target`;

    // Show interpolation strip
    const interpSection = document.getElementById('mesh-interp-section');
    if (interpSection) interpSection.hidden = false;
    document.getElementById('mesh-interp-title').textContent =
      `Interpolation: ${nameA} → ${nameB}`;

    const strip = document.getElementById('mesh-interp-strip');
    strip.innerHTML = '';

    const nSteps = result.graphs.length;
    result.graphs.forEach((g, i) => {
      const step = document.createElement('div');
      step.className = 'interp-step spatial-cell';
      strip.appendChild(step);

      renderSpatialMesh(step, g);

      const label = document.createElement('div');
      label.className = 'interp-step-label';
      if (i === 0) label.textContent = nameA;
      else if (i === nSteps - 1) label.textContent = nameB;
      else label.textContent = `t=${(i / (nSteps - 1)).toFixed(2)}`;
      step.appendChild(label);
    });

    log('mesh-status-log', `Interpolation: ${nameA} → ${nameB}, ${nSteps} steps`);
  } catch (e) {
    log('mesh-status-log', `Interpolation error: ${e.message}`);
  }
  setLoading(this, false);
});


// ---------------------------------------------------------------------------
// WL Test Panel — Interactive Stepper
// ---------------------------------------------------------------------------
const WL_PALETTE = ['#6366f1', '#f59e0b', '#10b981', '#f43f5e', '#06b6d4', '#8b5cf6', '#ec4899', '#14b8a6'];

function buildWLHistogram(nodeColors) {
  const counts = {};
  if (!nodeColors) return [];
  for (const c of nodeColors) counts[c] = (counts[c] || 0) + 1;
  return Object.entries(counts)
    .map(([ci, count]) => ({ colorIdx: +ci, count }))
    .sort((a, b) => a.colorIdx - b.colorIdx);
}

function renderWLHistBar(container, histA, histB, numNodes) {
  container.innerHTML = '';
  function makeRow(hist, label) {
    const row = document.createElement('div');
    row.className = 'wl-hist-row';
    const lbl = document.createElement('span');
    lbl.className = 'wl-hist-label';
    lbl.textContent = label;
    row.appendChild(lbl);
    const bar = document.createElement('div');
    bar.className = 'wl-hist-bar';
    for (const h of hist) {
      const seg = document.createElement('div');
      seg.className = 'wl-hist-seg';
      seg.style.width = ((h.count / numNodes) * 100) + '%';
      seg.style.background = WL_PALETTE[h.colorIdx % WL_PALETTE.length];
      seg.title = `Color ${h.colorIdx}: ${h.count}`;
      bar.appendChild(seg);
    }
    row.appendChild(bar);
    return row;
  }
  container.appendChild(makeRow(histA, 'G₁'));
  container.appendChild(makeRow(histB, 'G₂'));
}

document.getElementById('btn-run-wl')?.addEventListener('click', async function () {
  setLoading(this, true);
  try {
    const pairIdx = +document.getElementById('wl-pair').value;
    const iterations = +document.getElementById('wl-iterations').value;
    const result = await api.runWLTest({ pair_index: pairIdx, iterations });
    const examples = await api.loadWLExamples();
    const pair = examples.pairs[pairIdx];

    // Verdict badge
    const verdictEl = document.getElementById('wl-verdict');
    verdictEl.hidden = false;
    if (result.distinguished) {
      verdictEl.className = 'wl-verdict wl-verdict--success';
      verdictEl.innerHTML = `<span class="wl-verdict-icon">&#x2713;</span> Distinguished at iteration ${result.distinguishing_iteration}`;
    } else {
      verdictEl.className = 'wl-verdict wl-verdict--fail';
      verdictEl.innerHTML = `<span class="wl-verdict-icon">&#x2717;</span> Indistinguishable &mdash; WL fails`;
    }

    if (result.description) {
      document.getElementById('wl-result-title').textContent = result.name;
    }

    // Build interactive stepper
    const strip = document.getElementById('wl-iteration-strip');
    strip.innerHTML = '';

    let currentStep = 0;
    const maxStep = result.iterations.length - 1;

    // Navigation controls
    const nav = document.createElement('div');
    nav.className = 'wl-stepper-nav';

    const btnPrev = document.createElement('button');
    btnPrev.className = 'wl-nav-btn';
    btnPrev.innerHTML = '&#8592;';
    const stepText = document.createElement('span');
    stepText.className = 'wl-step-text';
    const btnNext = document.createElement('button');
    btnNext.className = 'wl-nav-btn';
    btnNext.innerHTML = '&#8594;';
    const btnPlay = document.createElement('button');
    btnPlay.className = 'wl-nav-btn wl-play-btn';
    btnPlay.innerHTML = '&#9654;';

    nav.appendChild(btnPrev);
    nav.appendChild(stepText);
    nav.appendChild(btnNext);
    nav.appendChild(btnPlay);
    strip.appendChild(nav);

    // Slider + dots
    const sliderWrap = document.createElement('div');
    sliderWrap.className = 'wl-slider-wrap';
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '0';
    slider.max = String(maxStep);
    slider.value = '0';
    slider.className = 'wl-slider';
    sliderWrap.appendChild(slider);

    const dots = document.createElement('div');
    dots.className = 'wl-step-dots';
    for (let i = 0; i <= maxStep; i++) {
      const dot = document.createElement('span');
      dot.className = 'wl-dot' + (i === 0 ? ' active' : '');
      dot.dataset.step = i;
      dots.appendChild(dot);
    }
    sliderWrap.appendChild(dots);
    strip.appendChild(sliderWrap);

    // Graph panels
    const graphsRow = document.createElement('div');
    graphsRow.className = 'wl-graphs-row';

    const wrapA = document.createElement('div');
    wrapA.className = 'wl-graph-panel';
    const labelA = document.createElement('div');
    labelA.className = 'wl-graph-label';
    labelA.textContent = pair.graphA.name;
    wrapA.appendChild(labelA);
    const vizA = document.createElement('div');
    vizA.className = 'wl-graph-viz';
    vizA.style.width = '180px';
    vizA.style.height = '180px';
    wrapA.appendChild(vizA);
    graphsRow.appendChild(wrapA);

    const vsDiv = document.createElement('div');
    vsDiv.className = 'wl-vs';
    vsDiv.textContent = 'vs';
    graphsRow.appendChild(vsDiv);

    const wrapB = document.createElement('div');
    wrapB.className = 'wl-graph-panel';
    const labelB = document.createElement('div');
    labelB.className = 'wl-graph-label';
    labelB.textContent = pair.graphB.name;
    wrapB.appendChild(labelB);
    const vizB = document.createElement('div');
    vizB.className = 'wl-graph-viz';
    vizB.style.width = '180px';
    vizB.style.height = '180px';
    wrapB.appendChild(vizB);
    graphsRow.appendChild(wrapB);

    strip.appendChild(graphsRow);

    // Histogram area
    const histArea = document.createElement('div');
    histArea.className = 'wl-histogram-area';
    strip.appendChild(histArea);

    // Match badge
    const matchBadge = document.createElement('div');
    matchBadge.className = 'wl-step-match';
    strip.appendChild(matchBadge);

    // Color count
    const colorInfo = document.createElement('div');
    colorInfo.className = 'wl-color-info';
    strip.appendChild(colorInfo);

    function renderStep(step) {
      currentStep = step;
      const iter = result.iterations[step];

      stepText.textContent = `Iteration ${iter.step}`;
      slider.value = String(step);
      btnPrev.disabled = step === 0;
      btnNext.disabled = step === maxStep;

      dots.querySelectorAll('.wl-dot').forEach((d, i) => {
        d.classList.toggle('active', i === step);
        d.classList.toggle('past', i < step);
      });

      renderWLGraph(vizA, { positions: pair.graphA.positions, edges: pair.graphA.edges, nodeColors: iter.colors_a });
      renderWLGraph(vizB, { positions: pair.graphB.positions, edges: pair.graphB.edges, nodeColors: iter.colors_b });

      const histA = buildWLHistogram(iter.colors_a);
      const histB = buildWLHistogram(iter.colors_b);
      renderWLHistBar(histArea, histA, histB, iter.colors_a.length);

      if (iter.histograms_match) {
        matchBadge.className = 'wl-step-match wl-step-match--same';
        matchBadge.textContent = 'Histograms match';
      } else {
        matchBadge.className = 'wl-step-match wl-step-match--diff';
        matchBadge.textContent = 'Histograms differ — distinguished!';
      }

      const nA = iter.num_colors_a || new Set(iter.colors_a).size;
      const nB = iter.num_colors_b || new Set(iter.colors_b).size;
      colorInfo.innerHTML = `<span>${pair.graphA.name}: <strong>${nA}</strong> colors</span><span>${pair.graphB.name}: <strong>${nB}</strong> colors</span>`;
    }

    btnPrev.addEventListener('click', () => { if (currentStep > 0) renderStep(currentStep - 1); });
    btnNext.addEventListener('click', () => { if (currentStep < maxStep) renderStep(currentStep + 1); });
    slider.addEventListener('input', () => renderStep(+slider.value));
    dots.querySelectorAll('.wl-dot').forEach(dot => {
      dot.addEventListener('click', () => renderStep(+dot.dataset.step));
    });

    let playing = false, playTimer = null;
    btnPlay.addEventListener('click', () => {
      if (playing) {
        clearInterval(playTimer); playing = false;
        btnPlay.innerHTML = '&#9654;'; btnPlay.classList.remove('playing');
      } else {
        playing = true; btnPlay.innerHTML = '&#9646;&#9646;'; btnPlay.classList.add('playing');
        if (currentStep >= maxStep) renderStep(0);
        playTimer = setInterval(() => {
          if (currentStep < maxStep) renderStep(currentStep + 1);
          else { clearInterval(playTimer); playing = false; btnPlay.innerHTML = '&#9654;'; btnPlay.classList.remove('playing'); }
        }, 1200);
      }
    });

    renderStep(0);
    log('wl-status-log', `WL test: ${result.distinguished ? 'Distinguished' : 'Indistinguishable'}`);
  } catch (e) {
    log('wl-status-log', `Error: ${e.message}`);
  }
  setLoading(this, false);
});


// ---------------------------------------------------------------------------
// Hyperbolic GNN
// ---------------------------------------------------------------------------

const HYP_COLORS = ['#3b82f6', '#f59e0b', '#ef4444', '#10b981', '#8b5cf6', '#ec4899'];

function renderPoincareDisk(containerId, positions, edges, geodesicArcs, options = {}) {
  const el = typeof containerId === 'string' ? document.getElementById(containerId) : containerId;
  if (!el) return;
  el.innerHTML = '';

  const labels = options.labels || [];
  const size = el.clientWidth || 400;

  const svg = d3.select(el).append('svg')
    .attr('viewBox', '-1.1 -1.1 2.2 2.2')
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .attr('width', size)
    .attr('height', size);

  // Unit circle boundary
  svg.append('circle')
    .attr('cx', 0).attr('cy', 0).attr('r', 1)
    .attr('class', 'hyp-boundary');

  // Concentric distance circles
  [0.5, 0.75, 0.9].forEach(r => {
    svg.append('circle')
      .attr('cx', 0).attr('cy', 0).attr('r', r)
      .attr('class', 'hyp-distance-circle');
  });

  // Draw edges or geodesic arcs
  if (geodesicArcs && geodesicArcs.length > 0) {
    geodesicArcs.forEach(arc => {
      const lineGen = d3.line().x(d => d[0]).y(d => d[1]).curve(d3.curveBasis);
      svg.append('path')
        .attr('d', lineGen(arc))
        .attr('class', 'hyp-geodesic');
    });
  } else if (edges) {
    edges.forEach(([s, t]) => {
      if (positions[s] && positions[t]) {
        svg.append('line')
          .attr('x1', positions[s][0]).attr('y1', positions[s][1])
          .attr('x2', positions[t][0]).attr('y2', positions[t][1])
          .attr('class', 'hyp-edge');
      }
    });
  }

  // Draw nodes
  const tooltip = d3.select(el).append('div')
    .style('position', 'absolute').style('pointer-events', 'none')
    .style('background', 'var(--bg-raised)').style('color', 'var(--text-primary)')
    .style('padding', '2px 6px').style('border-radius', '4px')
    .style('font-size', '11px').style('display', 'none');

  positions.forEach((pos, i) => {
    const label = labels[i] !== undefined ? labels[i] : 0;
    const color = HYP_COLORS[label % HYP_COLORS.length];
    svg.append('circle')
      .attr('cx', pos[0]).attr('cy', pos[1]).attr('r', 0.025)
      .attr('fill', color)
      .attr('class', 'hyp-node')
      .on('mouseenter', function (event) {
        tooltip.style('display', 'block').text(`Node ${i}`);
        const rect = el.getBoundingClientRect();
        tooltip.style('left', (event.clientX - rect.left + 8) + 'px')
          .style('top', (event.clientY - rect.top - 20) + 'px');
      })
      .on('mouseleave', () => tooltip.style('display', 'none'));
  });
}

function renderEuclideanEmbedding(containerId, positions, edges, labels) {
  const el = typeof containerId === 'string' ? document.getElementById(containerId) : containerId;
  if (!el || !positions || positions.length === 0) return;
  el.innerHTML = '';

  const size = el.clientWidth || 300;

  // Compute bounds
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  positions.forEach(([x, y]) => {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  });
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const pad = 30;
  const scale = (size - 2 * pad) / Math.max(rangeX, rangeY);
  const cx = size / 2, cy = size / 2;
  const midX = (minX + maxX) / 2, midY = (minY + maxY) / 2;

  const toScreen = ([x, y]) => ({
    sx: cx + (x - midX) * scale,
    sy: cy + (y - midY) * scale,
  });

  const svg = d3.select(el).append('svg')
    .attr('viewBox', `0 0 ${size} ${size}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  // Edges
  if (edges) {
    edges.forEach(([s, t]) => {
      if (positions[s] && positions[t]) {
        const ps = toScreen(positions[s]), pt = toScreen(positions[t]);
        svg.append('line')
          .attr('x1', ps.sx).attr('y1', ps.sy).attr('x2', pt.sx).attr('y2', pt.sy)
          .attr('stroke', '#334155').attr('stroke-width', 0.8).attr('stroke-opacity', 0.4);
      }
    });
  }

  // Nodes
  positions.forEach((pos, i) => {
    const sc = toScreen(pos);
    const label = labels && labels[i] !== undefined ? labels[i] : 0;
    const color = HYP_COLORS[label % HYP_COLORS.length];
    svg.append('circle')
      .attr('cx', sc.sx).attr('cy', sc.sy).attr('r', 4)
      .attr('fill', color).attr('stroke', '#1e293b').attr('stroke-width', 0.8);
  });
}

// Generate graph
document.getElementById('btn-hyp-graph')?.addEventListener('click', async function () {
  setLoading(this, true);
  try {
    const type = document.getElementById('hyp-graph-type').value;
    const depth = +document.getElementById('hyp-depth').value;
    const data = await api.hyperbolicGraph({ graph_type: type, num_nodes: depth, branching_factor: 2 });

    // Store globally for later use
    window._hypState = {
      positions: data.nodes.map(n => [n.x, n.y]),
      edges: data.edges,
      labels: data.nodes.map(n => n.label),
    };

    // Render initial positions
    renderPoincareDisk('hyp-disk-viz', window._hypState.positions, data.edges, null, { labels: window._hypState.labels });
    document.getElementById('hyp-disk-viz').scrollIntoView({ behavior: 'smooth', block: 'center' });

    // Enable simulation and training buttons
    document.getElementById('btn-hyp-sim-start').disabled = false;
    document.getElementById('btn-hyp-sim-reset').disabled = false;
    document.getElementById('btn-hyp-train').disabled = false;
    document.getElementById('btn-hyp-compare').disabled = false;

    // Update status
    document.getElementById('hyp-graph-status').textContent = `${data.num_nodes} nodes, ${data.num_edges} edges`;
    document.getElementById('hyp-graph-status').className = 'status-text success';

    // Update title
    document.getElementById('hyp-viz-title').textContent =
      type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) + ' in Poincar\u00e9 Disk';

    // Reset viz sections
    document.getElementById('hyp-predictions-viz').hidden = true;
    document.getElementById('hyp-compare-viz').hidden = true;
    document.getElementById('hyp-accuracy-display').hidden = true;

    log('hyp-status-log', `Loaded ${type}: ${data.num_nodes} nodes, ${data.num_edges} edges`);
  } catch (e) {
    log('hyp-status-log', `Error: ${e.message}`);
  }
  setLoading(this, false);
});

// Simulation loop
async function hyperbolicSimLoop() {
  if (!window._hypSimRunning) return;
  try {
    const result = await api.hyperbolicSimStep({ n_steps: 5 });
    window._hypState.positions = result.positions;
    document.getElementById('hyp-energy-val').textContent = result.energy.toFixed(4);
    document.getElementById('hyp-step-val').textContent =
      (+document.getElementById('hyp-step-val').textContent) + result.step;

    // Fetch geodesic arcs for curved edges
    const geoData = await api.hyperbolicGeodesics({ n_samples: 20 });
    renderPoincareDisk('hyp-disk-viz', result.positions, window._hypState.edges, geoData.arcs, { labels: window._hypState.labels });

    if (window._hypSimRunning) setTimeout(hyperbolicSimLoop, 100);
  } catch (e) {
    window._hypSimRunning = false;
    log('hyp-status-log', `Simulation error: ${e.message}`);
    document.getElementById('btn-hyp-sim-start').disabled = false;
    document.getElementById('btn-hyp-sim-stop').disabled = true;
  }
}

// Start simulation
document.getElementById('btn-hyp-sim-start')?.addEventListener('click', async function () {
  setLoading(this, true);
  try {
    const params = {
      curvature: +document.getElementById('hyp-curvature').value,
      spring_k: +document.getElementById('hyp-spring-k').value,
      target_length: +document.getElementById('hyp-target-len').value,
      charge_c: +document.getElementById('hyp-charge').value,
      lr: 0.01,
    };
    await api.hyperbolicSimInit(params);

    document.getElementById('btn-hyp-sim-stop').disabled = false;
    document.getElementById('btn-hyp-sim-start').disabled = true;
    document.getElementById('hyp-step-val').textContent = '0';
    window._hypSimRunning = true;
    hyperbolicSimLoop();
  } catch (e) {
    log('hyp-status-log', `Init error: ${e.message}`);
  }
  setLoading(this, false);
});

// Stop simulation
document.getElementById('btn-hyp-sim-stop')?.addEventListener('click', function () {
  window._hypSimRunning = false;
  document.getElementById('btn-hyp-sim-start').disabled = false;
  document.getElementById('btn-hyp-sim-stop').disabled = true;
});

// Reset simulation
document.getElementById('btn-hyp-sim-reset')?.addEventListener('click', async function () {
  window._hypSimRunning = false;
  setLoading(this, true);
  try {
    await api.hyperbolicSimReset();
    renderPoincareDisk('hyp-disk-viz', window._hypState.positions, window._hypState.edges, null, { labels: window._hypState.labels });
    document.getElementById('hyp-energy-val').textContent = '\u2014';
    document.getElementById('hyp-step-val').textContent = '0';
    document.getElementById('btn-hyp-sim-start').disabled = false;
    document.getElementById('btn-hyp-sim-stop').disabled = true;
  } catch (e) {
    log('hyp-status-log', `Reset error: ${e.message}`);
  }
  setLoading(this, false);
});

// Train hyperbolic GNN
document.getElementById('btn-hyp-train')?.addEventListener('click', async function () {
  setLoading(this, true);
  document.getElementById('hyp-status-log').textContent = '';
  log('hyp-status-log', 'Training Hyperbolic GNN...');

  try {
    const config = {
      hidden_dim: +document.getElementById('hyp-hidden-dim').value,
      num_layers: +document.getElementById('hyp-num-layers').value,
      epochs: +document.getElementById('hyp-epochs').value,
      lr: +document.getElementById('hyp-lr').value,
      curvature: +document.getElementById('hyp-curvature').value,
      layer_type: document.getElementById('hyp-layer-type').value,
    };
    const result = await api.hyperbolicTrain(config);

    // Draw loss curve
    const container = document.getElementById('hyp-loss-container');
    container.innerHTML = '<canvas width="240" height="120"></canvas>';
    drawLossCurve(container.querySelector('canvas'), result.losses);

    // Show accuracy
    const accDisplay = document.getElementById('hyp-accuracy-display');
    accDisplay.hidden = false;
    accDisplay.innerHTML = `
      <div class="physics-acc-row"><span>Train accuracy</span><strong>${(result.train_acc * 100).toFixed(1)}%</strong></div>
      <div class="physics-acc-row"><span>Val accuracy</span><strong>${(result.val_acc * 100).toFixed(1)}%</strong></div>
    `;

    // Show predictions on Poincare disk
    const predViz = document.getElementById('hyp-predictions-viz');
    predViz.hidden = false;
    renderPoincareDisk('hyp-pred-disk', result.embeddings, window._hypState.edges, null, { labels: result.predictions });

    log('hyp-status-log', `Done \u2014 ${(result.train_acc * 100).toFixed(1)}% train / ${(result.val_acc * 100).toFixed(1)}% val`);
  } catch (e) {
    log('hyp-status-log', `Error: ${e.message}`);
  }
  setLoading(this, false);
});

// Compare Euclidean vs Hyperbolic
document.getElementById('btn-hyp-compare')?.addEventListener('click', async function () {
  setLoading(this, true);
  log('hyp-status-log', 'Training embeddings...');

  try {
    const params = {
      embed_dim: +document.getElementById('hyp-embed-dim').value,
      epochs: +document.getElementById('hyp-comp-epochs').value,
      lr: 0.01,
    };
    const result = await api.hyperbolicCompare(params);

    // Show comparison visualizations
    const compViz = document.getElementById('hyp-compare-viz');
    compViz.hidden = false;

    // Euclidean embedding
    renderEuclideanEmbedding('hyp-euclidean-viz', result.euclidean.positions, window._hypState.edges, window._hypState.labels);

    // Poincare embedding
    renderPoincareDisk('hyp-poincare-viz', result.hyperbolic.positions, window._hypState.edges, null, { labels: window._hypState.labels });

    // Show distortion metrics
    const display = document.getElementById('hyp-compare-display');
    display.innerHTML = `<table>
      <tr><th></th><th>Euclidean</th><th>Poincar\u00e9</th></tr>
      <tr><td>Distortion</td><td>${result.euclidean.distortion.toFixed(4)}</td><td>${result.hyperbolic.distortion.toFixed(4)}</td></tr>
      <tr><td>Mean Edge Dist Error</td><td>${result.euclidean.avg_dist_error.toFixed(4)}</td><td>${result.hyperbolic.avg_dist_error.toFixed(4)}</td></tr>
    </table>`;

    log('hyp-status-log', 'Comparison complete');
  } catch (e) {
    log('hyp-status-log', `Error: ${e.message}`);
  }
  setLoading(this, false);
});
