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
// WL Test Panel
// ---------------------------------------------------------------------------
document.getElementById('btn-run-wl')?.addEventListener('click', async function () {
  setLoading(this, true);
  try {
    const pairIdx = +document.getElementById('wl-pair').value;
    const iterations = +document.getElementById('wl-iterations').value;
    const result = await api.runWLTest({ pair_index: pairIdx, iterations });

    // Also fetch example data for positions/edges
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

    document.getElementById('wl-result-title').textContent = result.name;

    // Build iteration strip
    const strip = document.getElementById('wl-iteration-strip');
    strip.innerHTML = '';

    for (const step of result.iterations) {
      const stepDiv = document.createElement('div');
      stepDiv.className = 'wl-step';

      const stepLabel = document.createElement('div');
      stepLabel.className = 'wl-step-label';
      stepLabel.textContent = `Iteration ${step.step}`;
      stepDiv.appendChild(stepLabel);

      const graphsRow = document.createElement('div');
      graphsRow.className = 'wl-step-graphs';

      // Graph A
      const graphAWrap = document.createElement('div');
      graphAWrap.className = 'wl-graph-wrap';
      const graphALabel = document.createElement('div');
      graphALabel.className = 'wl-graph-label';
      graphALabel.textContent = pair.graphA.name;
      graphAWrap.appendChild(graphALabel);
      const graphAViz = document.createElement('div');
      graphAViz.className = 'wl-graph-viz';
      graphAViz.style.width = '150px';
      graphAViz.style.height = '150px';
      graphAWrap.appendChild(graphAViz);
      graphsRow.appendChild(graphAWrap);

      // Graph B
      const graphBWrap = document.createElement('div');
      graphBWrap.className = 'wl-graph-wrap';
      const graphBLabel = document.createElement('div');
      graphBLabel.className = 'wl-graph-label';
      graphBLabel.textContent = pair.graphB.name;
      graphBWrap.appendChild(graphBLabel);
      const graphBViz = document.createElement('div');
      graphBViz.className = 'wl-graph-viz';
      graphBViz.style.width = '150px';
      graphBViz.style.height = '150px';
      graphBWrap.appendChild(graphBViz);
      graphsRow.appendChild(graphBWrap);

      stepDiv.appendChild(graphsRow);

      // Match indicator
      const matchDiv = document.createElement('div');
      matchDiv.className = step.histograms_match ? 'wl-match wl-match--same' : 'wl-match wl-match--diff';
      matchDiv.textContent = step.histograms_match ? 'Histograms match' : 'Histograms differ';
      stepDiv.appendChild(matchDiv);

      strip.appendChild(stepDiv);

      renderWLGraph(graphAViz, { positions: pair.graphA.positions, edges: pair.graphA.edges, nodeColors: step.colors_a });
      renderWLGraph(graphBViz, { positions: pair.graphB.positions, edges: pair.graphB.edges, nodeColors: step.colors_b });
    }

    log('wl-status-log', `WL test: ${result.distinguished ? 'Distinguished' : 'Indistinguishable'}`);
  } catch (e) {
    log('wl-status-log', `Error: ${e.message}`);
  }
  setLoading(this, false);
});
