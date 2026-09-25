// Generator panel: train a conditional graph VAE, generate samples by
// specifying target density / clustering.

import { api } from '../../api.js';
import { drawLossCurve } from '../../graph_viz.js';
import { log, setLoading } from '../shared.js';
import { renderGraphGrid } from '../renderers/graph_grid.js';

export function initGeneratorPanel() {
  document.getElementById('btn-train-gen')?.addEventListener('click', onTrain);
  document.getElementById('btn-generate')?.addEventListener('click', onGenerate);
}

async function onTrain() {
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
}

async function onGenerate() {
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
}
