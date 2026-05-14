// GNN panel: load Karate Club, train a GCN/GAT, predict on held-out nodes.
//
// Each panel module follows the same shape:
//   1. A per-panel state object (scoped to the module, not a global).
//   2. An `init()` function that wires up DOM event listeners once.
//   3. Private helpers below init.
// app.js is responsible for calling init() during boot.

import { api } from '../../api.js';
import { renderGraph, drawLossCurve } from '../../graph_viz.js';
import { log, setLoading } from '../shared.js';

const state = {
  handle: null,
  data: null,
};

export function initGnnPanel() {
  document.getElementById('btn-load-graph')?.addEventListener('click', onLoadGraph);
  document.getElementById('btn-train-gnn')?.addEventListener('click', onTrain);
  document.getElementById('btn-predict-gnn')?.addEventListener('click', onPredict);
}

async function onLoadGraph() {
  setLoading(this, true);
  try {
    const data = await api.loadKarateClub();
    state.data = data;
    document.getElementById('load-status').textContent = `${data.num_nodes} nodes, ${data.num_edges} edges`;
    document.getElementById('load-status').className = 'status-text success';
    document.getElementById('btn-train-gnn').disabled = false;

    const container = document.getElementById('gnn-graph-container');
    const edges = dedupeEdges(data.edges);

    if (state.handle) state.handle.destroy();
    state.handle = renderGraph(container, {
      nodes: Array.from({ length: data.num_nodes }, (_, i) => ({ id: i })),
      edges,
      labels: data.labels,
    });
  } catch (e) {
    document.getElementById('load-status').textContent = e.message;
    document.getElementById('load-status').className = 'status-text error';
  }
  setLoading(this, false);
}

async function onTrain() {
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

    const container = document.getElementById('gnn-loss-container');
    container.innerHTML = '<canvas width="280" height="160"></canvas>';
    drawLossCurve(container.querySelector('canvas'), result.loss_curve);

    document.getElementById('predict-fieldset').disabled = false;
  } catch (e) {
    log('gnn-status-log', `Error: ${e.message}`);
  }
  setLoading(this, false);
}

async function onPredict() {
  setLoading(this, true);
  try {
    const raw = document.getElementById('gnn-node-ids').value.trim();
    const nodeIds = raw
      ? raw.split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n))
      : null;
    const result = await api.predictGNN(nodeIds);

    const predDiv = document.getElementById('gnn-predictions');
    predDiv.innerHTML = `<div style="margin-bottom:8px;color:var(--accent);font-weight:500">Accuracy: ${(result.accuracy * 100).toFixed(1)}%</div>`;
    for (const p of result.predictions) {
      predDiv.innerHTML += `<div class="pred-row"><span class="pred-label">Node ${p.node}</span><span class="pred-value">${p.predicted} (${(p.confidence * 100).toFixed(0)}%)</span></div>`;
    }

    if (state.handle && state.data) {
      const preds = new Array(state.data.num_nodes).fill(-1);
      for (const p of result.predictions) preds[p.node] = p.predicted;
      state.handle.updatePredictions(preds);
    }
  } catch (e) {
    log('gnn-status-log', `Predict error: ${e.message}`);
  }
  setLoading(this, false);
}

function dedupeEdges(edges) {
  const seen = new Set();
  const out = [];
  for (const [s, t] of edges) {
    const key = Math.min(s, t) + ',' + Math.max(s, t);
    if (!seen.has(key)) {
      seen.add(key);
      out.push([s, t]);
    }
  }
  return out;
}
