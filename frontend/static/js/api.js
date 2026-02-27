const API_BASE = '/api';

async function request(path, { method = 'GET', body = null } = {}) {
  const opts = { method, headers: {} };
  if (body !== null) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(`${API_BASE}${path}`, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const api = {
  // GNN
  loadKarateClub: () => request('/gnn/karate-club'),
  trainGNN: (params) => request('/gnn/train', { method: 'POST', body: params }),
  predictGNN: (nodeIds) => request('/gnn/predict', { method: 'POST', body: { node_ids: nodeIds } }),

  // Generator
  trainGenerator: (params) => request('/generator/train', { method: 'POST', body: params }),
  generateGraphs: (params) => request('/generator/generate', { method: 'POST', body: params }),

  // VAE
  trainVAE: (params) => request('/vae/train', { method: 'POST', body: params }),
  generateVAE: (n) => request(`/vae/generate?num_samples=${n}`, { method: 'POST' }),
  interpolateVAE: (params) => request('/vae/interpolate', { method: 'POST', body: params }),

  // Diffusion
  trainDiffusion: (params) => request('/vae/diffusion/train', { method: 'POST', body: params }),
  generateDiffusion: (n) => request(`/vae/diffusion/generate?num_samples=${n}`, { method: 'POST' }),

  // Spatial 3D
  generateSynthetic: (params) => request('/spatial/synthetic', { method: 'POST', body: params }),
  trainSpatialVAE: (params) => request('/spatial/vae/train', { method: 'POST', body: params }),
  trainSpatialDiffusion: (params) => request('/spatial/diffusion/train', { method: 'POST', body: params }),
  generateSpatialVAE: (params) => request('/spatial/vae/generate', { method: 'POST', body: params }),
  generateSpatialDiffusion: (params) => request('/spatial/diffusion/generate', { method: 'POST', body: params }),
  analyzeSpatial: (params) => request('/spatial/analyze', { method: 'POST', body: params }),
  shollProfile: (idx) => request(`/spatial/sholl/${idx}`, { method: 'POST' }),
};
