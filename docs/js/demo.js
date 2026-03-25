/* ===========================================================================
   Graph ML Lab — Demo (static, no backend)
   All data is pre-computed or generated client-side with D3.
   =========================================================================== */

(async function () {
  'use strict';

  const EXAMPLES = await fetch('data/examples.json').then(r => r.json());

  // ── Karate Club (hardcoded from Zachary 1977) ───────────────────────────
  const KARATE_EDGES = [
    [0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,10],[0,11],
    [0,12],[0,13],[0,17],[0,19],[0,21],[0,31],
    [1,2],[1,3],[1,7],[1,13],[1,17],[1,19],[1,21],[1,30],
    [2,3],[2,7],[2,8],[2,9],[2,13],[2,27],[2,28],[2,32],
    [3,7],[3,12],[3,13],
    [4,6],[4,10],[5,6],[5,10],[5,16],[6,16],
    [8,30],[8,32],[8,33],[9,33],[13,33],
    [14,32],[14,33],[15,32],[15,33],[18,32],[18,33],[19,33],
    [20,32],[20,33],[22,32],[22,33],
    [23,25],[23,27],[23,29],[23,32],[23,33],
    [24,25],[24,27],[24,31],[25,31],
    [26,29],[26,33],[27,33],[28,31],[28,33],
    [29,32],[29,33],[30,32],[30,33],[31,32],[31,33],[32,33],
  ];

  const KARATE_LABELS = [
    0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,
    1,0,1,1,1,1,1,1,1,1,1,1,1,1,
  ];

  const COLOR_A = '#6366f1';
  const COLOR_B = '#f59e0b';

  // ── Scroll animations ───────────────────────────────────────────────────
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
  }, { threshold: 0.15 });
  document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));

  // nav scroll shadow
  const nav = document.querySelector('nav');
  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 20);
  });

  // active nav link
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-links a');
  window.addEventListener('scroll', () => {
    let current = '';
    for (const s of sections) {
      if (window.scrollY >= s.offsetTop - 100) current = s.id;
    }
    navLinks.forEach(a => {
      a.classList.toggle('active', a.getAttribute('href') === '#' + current);
    });
  });

  // ── GNN: Karate Club visualization ──────────────────────────────────────
  function renderKarateClub(containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML = '';

    const width = el.clientWidth || 540;
    const height = el.clientHeight || 440;

    const svg = d3.select(el).append('svg')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    const nodes = Array.from({length: 34}, (_, i) => ({
      id: i,
      label: KARATE_LABELS[i],
      messageStep: -1,
    }));

    const links = KARATE_EDGES.map(([s, t]) => ({source: s, target: t}));

    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(35))
      .force('charge', d3.forceManyBody().strength(-100))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide(10));

    const linkSel = svg.append('g').selectAll('line')
      .data(links).join('line')
      .attr('class', 'link');

    const nodeSel = svg.append('g').selectAll('g')
      .data(nodes).join('g')
      .attr('class', 'node')
      .call(d3.drag()
        .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    nodeSel.append('circle')
      .attr('r', d => (d.id === 0 || d.id === 33) ? 10 : 6)
      .attr('fill', d => d.label === 0 ? COLOR_A : COLOR_B);

    nodeSel.append('text')
      .text(d => d.id)
      .attr('fill', '#fff')
      .attr('font-size', d => (d.id === 0 || d.id === 33) ? '10px' : '7px');

    sim.on('tick', () => {
      linkSel.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      nodeSel.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    return { nodes, links, linkSel, nodeSel, sim };
  }

  let karateHandle = null;
  karateHandle = renderKarateClub('gnn-viz');

  // message passing animation
  document.getElementById('btn-message-pass')?.addEventListener('click', () => {
    if (!karateHandle) return;
    const {nodes, links, linkSel, nodeSel} = karateHandle;

    const adj = {};
    for (const {source, target} of links) {
      const s = typeof source === 'object' ? source.id : source;
      const t = typeof target === 'object' ? target.id : target;
      (adj[s] = adj[s] || []).push(t);
      (adj[t] = adj[t] || []).push(s);
    }

    let visited = new Set([0]);
    let frontier = [0];
    let step = 0;
    nodes[0].messageStep = 0;

    function wave() {
      if (frontier.length === 0) return;
      step++;
      const nextFrontier = [];

      linkSel.each(function (d) {
        const s = typeof d.source === 'object' ? d.source.id : d.source;
        const t = typeof d.target === 'object' ? d.target.id : d.target;
        if (frontier.includes(s) || frontier.includes(t)) {
          d3.select(this).classed('message-active', true);
        }
      });

      for (const nid of frontier) {
        for (const neighbor of (adj[nid] || [])) {
          if (!visited.has(neighbor)) {
            visited.add(neighbor);
            nodes[neighbor].messageStep = step;
            nextFrontier.push(neighbor);
          }
        }
      }

      nodeSel.filter(d => d.messageStep === step)
        .classed('pulse', true)
        .select('circle')
        .transition().duration(300)
        .attr('fill', '#818cf8')
        .transition().duration(400)
        .attr('fill', d => d.label === 0 ? COLOR_A : COLOR_B);

      frontier = nextFrontier;
      if (frontier.length > 0) setTimeout(wave, 600);
      else {
        setTimeout(() => {
          linkSel.classed('message-active', false);
          nodeSel.classed('pulse', false);
        }, 1000);
      }
    }

    linkSel.classed('message-active', false).classed('highlighted', false);
    nodeSel.classed('pulse', false);
    nodes.forEach(n => n.messageStep = -1);
    wave();
  });

  document.getElementById('btn-reset-gnn')?.addEventListener('click', () => {
    if (karateHandle) {
      karateHandle.linkSel.classed('message-active', false);
      karateHandle.nodeSel.classed('pulse', false)
        .select('circle').attr('fill', d => d.label === 0 ? COLOR_A : COLOR_B);
    }
  });

  // ── GNN training chart ──────────────────────────────────────────────────
  function drawChart(containerId, data, yKey, { color = COLOR_A, yLabel = '' } = {}) {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML = '';

    const margin = {top: 16, right: 16, bottom: 28, left: 40};
    const width = el.clientWidth || 400;
    const height = el.clientHeight || 200;
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select(el).append('svg')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear()
      .domain([data.epochs[0], data.epochs[data.epochs.length - 1]])
      .range([0, innerW]);

    const vals = data[yKey];
    const y = d3.scaleLinear()
      .domain([d3.min(vals) * 0.9, d3.max(vals) * 1.1])
      .range([innerH, 0]);

    g.append('g').attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(6).tickSize(-innerH).tickFormat(d => d))
      .call(g => g.select('.domain').remove())
      .call(g => g.selectAll('.tick line').attr('stroke', '#e2e2ea').attr('stroke-opacity', 0.3))
      .call(g => g.selectAll('.tick text').attr('fill', '#5a5a72').attr('font-size', '10px'));

    g.append('g')
      .call(d3.axisLeft(y).ticks(4).tickSize(-innerW).tickFormat(d3.format('.2f')))
      .call(g => g.select('.domain').remove())
      .call(g => g.selectAll('.tick line').attr('stroke', '#e2e2ea').attr('stroke-opacity', 0.3))
      .call(g => g.selectAll('.tick text').attr('fill', '#5a5a72').attr('font-size', '10px'));

    const line = d3.line()
      .x((_, i) => x(data.epochs[i]))
      .y(d => y(d))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(vals)
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', 2)
      .attr('d', line);

    if (yLabel) {
      g.append('text')
        .attr('x', 4).attr('y', -4)
        .attr('fill', color)
        .attr('font-size', '11px')
        .attr('font-family', 'var(--font-body)')
        .text(yLabel);
    }
  }

  drawChart('gnn-chart', EXAMPLES.trainingCurves.gnn, 'trainLoss', { color: COLOR_A, yLabel: 'Train loss' });

  // ── Generator section ───────────────────────────────────────────────────
  function generateErdosRenyi(n, p) {
    const edges = [];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.random() < p) edges.push([i, j]);
      }
    }
    return edges;
  }

  function generateBarabasiAlbert(n, m) {
    const seed = Math.min(m + 1, n);
    const edges = [];
    const degree = new Array(n).fill(0);

    for (let i = 0; i < seed; i++) {
      for (let j = i + 1; j < seed; j++) {
        edges.push([i, j]);
        degree[i]++;
        degree[j]++;
      }
    }

    for (let v = seed; v < n; v++) {
      const totalDeg = degree.reduce((s, d) => s + d, 0) || 1;
      const targets = new Set();
      let attempts = 0;
      while (targets.size < Math.min(m, v) && attempts < 200) {
        let r = Math.random() * totalDeg;
        for (let u = 0; u < v; u++) {
          r -= degree[u];
          if (r <= 0 && !targets.has(u)) {
            targets.add(u);
            break;
          }
        }
        attempts++;
      }
      for (const u of targets) {
        edges.push([u, v]);
        degree[u]++;
        degree[v]++;
      }
    }
    return edges;
  }

  function generateWattsStrogatz(n, k, p) {
    const edges = [];
    const edgeSet = new Set();
    const addEdge = (a, b) => {
      const lo = Math.min(a, b), hi = Math.max(a, b);
      const key = lo + ',' + hi;
      if (!edgeSet.has(key)) { edgeSet.add(key); edges.push([lo, hi]); }
    };

    for (let i = 0; i < n; i++) {
      for (let j = 1; j <= Math.floor(k / 2); j++) {
        addEdge(i, (i + j) % n);
      }
    }

    for (let i = 0; i < n; i++) {
      for (let j = 1; j <= Math.floor(k / 2); j++) {
        if (Math.random() < p) {
          const oldTarget = (i + j) % n;
          const lo = Math.min(i, oldTarget), hi = Math.max(i, oldTarget);
          const oldKey = lo + ',' + hi;
          edgeSet.delete(oldKey);
          const idx = edges.findIndex(e => e[0] === lo && e[1] === hi);
          if (idx >= 0) edges.splice(idx, 1);

          let newTarget;
          let tries = 0;
          do {
            newTarget = Math.floor(Math.random() * n);
            tries++;
          } while ((newTarget === i || edgeSet.has(Math.min(i, newTarget) + ',' + Math.max(i, newTarget))) && tries < 50);
          if (tries < 50) addEdge(i, newTarget);
          else { edgeSet.add(oldKey); edges.push([lo, hi]); }
        }
      }
    }
    return edges;
  }

  function computeGraphProps(n, edges) {
    const maxEdges = n * (n - 1) / 2;
    const density = maxEdges > 0 ? edges.length / maxEdges : 0;

    const adj = {};
    for (let i = 0; i < n; i++) adj[i] = new Set();
    for (const [a, b] of edges) { adj[a].add(b); adj[b].add(a); }

    let ccSum = 0;
    for (let i = 0; i < n; i++) {
      const neighbors = [...adj[i]];
      const k = neighbors.length;
      if (k < 2) continue;
      let triangles = 0;
      for (let a = 0; a < k; a++) {
        for (let b = a + 1; b < k; b++) {
          if (adj[neighbors[a]].has(neighbors[b])) triangles++;
        }
      }
      ccSum += (2 * triangles) / (k * (k - 1));
    }
    const avgClustering = n > 0 ? ccSum / n : 0;

    return { numNodes: n, numEdges: edges.length, density, avgClustering };
  }

  function generateGraphsFromSliders() {
    const n = parseInt(document.getElementById('ctrl-nodes').value, 10);
    const targetDensity = parseFloat(document.getElementById('ctrl-density').value);
    const typeFilter = document.getElementById('ctrl-type').value;

    const generators = [];
    if (typeFilter === 'all' || typeFilter === 'er') {
      generators.push({ type: 'er', label: 'Erdos-Renyi', fn: () => generateErdosRenyi(n, targetDensity) });
      generators.push({ type: 'er', label: 'Erdos-Renyi', fn: () => generateErdosRenyi(n, targetDensity) });
    }
    if (typeFilter === 'all' || typeFilter === 'ba') {
      const m = Math.max(1, Math.round(targetDensity * (n - 1) / 2));
      generators.push({ type: 'ba', label: 'Barabasi-Albert', fn: () => generateBarabasiAlbert(n, m) });
      generators.push({ type: 'ba', label: 'Barabasi-Albert', fn: () => generateBarabasiAlbert(n, Math.max(1, m - 1)) });
    }
    if (typeFilter === 'all' || typeFilter === 'ws') {
      const k = Math.max(2, Math.round(targetDensity * (n - 1)));
      generators.push({ type: 'ws', label: 'Watts-Strogatz', fn: () => generateWattsStrogatz(n, k, 0.2) });
      generators.push({ type: 'ws', label: 'Watts-Strogatz', fn: () => generateWattsStrogatz(n, k, 0.5) });
    }

    const grid = document.getElementById('gen-grid');
    if (!grid) return;
    grid.innerHTML = '';

    for (const gen of generators) {
      const edges = gen.fn();
      const props = computeGraphProps(n, edges);

      const card = document.createElement('div');
      card.className = 'graph-card';

      const viz = document.createElement('div');
      viz.className = 'graph-card-viz';
      card.appendChild(viz);

      const info = document.createElement('div');
      info.className = 'graph-card-info';
      info.innerHTML = `
        <h4>${gen.label}</h4>
        <div class="graph-props">
          <span>Nodes:</span> <span>${props.numNodes}</span>
          <span>Edges:</span> <span>${props.numEdges}</span>
          <span>Density:</span> <span>${props.density.toFixed(3)}</span>
          <span>Clustering:</span> <span>${props.avgClustering.toFixed(3)}</span>
        </div>`;
      card.appendChild(info);
      grid.appendChild(card);

      renderSmallGraph(viz, n, edges);
    }
  }

  function renderSmallGraph(container, numNodes, edges) {
    const size = container.clientWidth || 200;
    const h = container.clientHeight || 200;

    const svg = d3.select(container).append('svg')
      .attr('viewBox', `0 0 ${size} ${h}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    const nodes = Array.from({length: numNodes}, (_, i) => ({id: i}));
    const links = edges.map(([s, t]) => ({source: s, target: t}));

    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(22))
      .force('charge', d3.forceManyBody().strength(-50))
      .force('center', d3.forceCenter(size / 2, h / 2))
      .force('collision', d3.forceCollide(6));

    const linkSel = svg.append('g').selectAll('line')
      .data(links).join('line')
      .attr('stroke', '#9ca3af').attr('stroke-width', 1).attr('stroke-opacity', 0.5);

    const nodeSel = svg.append('g').selectAll('circle')
      .data(nodes).join('circle')
      .attr('r', 4).attr('fill', COLOR_A).attr('stroke', '#fff').attr('stroke-width', 1);

    sim.on('tick', () => {
      linkSel.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      nodeSel.attr('cx', d => d.x).attr('cy', d => d.y);
    });

    setTimeout(() => sim.stop(), 3000);
  }

  generateGraphsFromSliders();

  document.querySelectorAll('.generator-controls input[type="range"]').forEach(input => {
    const valSpan = document.getElementById(input.id + '-val');
    if (valSpan) {
      input.addEventListener('input', () => {
        valSpan.textContent = parseFloat(input.value).toFixed(
          input.step.includes('.') ? input.step.split('.')[1].length : 0
        );
      });
    }
  });

  document.getElementById('btn-gen-new')?.addEventListener('click', generateGraphsFromSliders);
  document.getElementById('ctrl-type')?.addEventListener('change', generateGraphsFromSliders);

  // ── VAE: geometric graph interpolation strips ─────────────────────────

  function renderGraphWireframe(container, graphData) {
    const w = container.clientWidth || 100;
    const h = container.clientHeight || 100;
    const pad = 12;
    const positions = graphData.positions;
    const edges = graphData.edges;

    if (!positions || positions.length === 0) return;

    // Compute bounding box
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const [x, y] of positions) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const scale = Math.min((w - 2 * pad) / rangeX, (h - 2 * pad) / rangeY);
    const cx = w / 2, cy = h / 2;
    const midX = (minX + maxX) / 2, midY = (minY + maxY) / 2;

    const screenPos = positions.map(([x, y]) => [
      cx + (x - midX) * scale,
      cy + (y - midY) * scale,
    ]);

    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    svg.style.width = '100%';
    svg.style.height = '100%';

    // Draw edges
    for (const [i, j] of edges) {
      if (i < screenPos.length && j < screenPos.length) {
        const line = document.createElementNS(ns, 'line');
        line.setAttribute('x1', screenPos[i][0]);
        line.setAttribute('y1', screenPos[i][1]);
        line.setAttribute('x2', screenPos[j][0]);
        line.setAttribute('y2', screenPos[j][1]);
        line.setAttribute('stroke', '#6366f1');
        line.setAttribute('stroke-width', '1');
        line.setAttribute('stroke-opacity', '0.45');
        svg.appendChild(line);
      }
    }

    // Draw nodes
    for (const [x, y] of screenPos) {
      const circle = document.createElementNS(ns, 'circle');
      circle.setAttribute('cx', x);
      circle.setAttribute('cy', y);
      circle.setAttribute('r', '3');
      circle.setAttribute('fill', '#6366f1');
      circle.setAttribute('stroke', '#fff');
      circle.setAttribute('stroke-width', '0.8');
      svg.appendChild(circle);
    }

    container.innerHTML = '';
    container.appendChild(svg);
  }

  function renderGraphInterpolationStrips() {
    const container = document.getElementById('graph-interp-strips');
    if (!container || !EXAMPLES.graphInterpolation) return;
    container.innerHTML = '';

    for (const pair of EXAMPLES.graphInterpolation) {
      const pairDiv = document.createElement('div');
      pairDiv.className = 'graph-interp-pair';

      const title = document.createElement('h4');
      title.textContent = `${pair.source} → ${pair.target}`;
      pairDiv.appendChild(title);

      const strip = document.createElement('div');
      strip.className = 'graph-interp-strip';

      for (let i = 0; i < pair.steps.length; i++) {
        const step = pair.steps[i];
        const stepDiv = document.createElement('div');
        stepDiv.className = 'graph-interp-step';
        if (i === 0 || i === pair.steps.length - 1) {
          stepDiv.classList.add('is-endpoint');
        }

        const vizDiv = document.createElement('div');
        vizDiv.className = 'graph-interp-step-viz';
        stepDiv.appendChild(vizDiv);

        const label = document.createElement('span');
        label.className = 'graph-interp-step-label';
        label.textContent = step.label;
        stepDiv.appendChild(label);

        strip.appendChild(stepDiv);
        renderGraphWireframe(vizDiv, step);
      }

      pairDiv.appendChild(strip);
      container.appendChild(pairDiv);
    }
  }

  renderGraphInterpolationStrips();

  drawChart('vae-chart', EXAMPLES.trainingCurves.vae, 'totalLoss', { color: '#c084fc', yLabel: 'ELBO loss' });

  // ── Spatial 3D: Isometric tree rendering ──────────────────────────────
  const COS30 = Math.cos(Math.PI / 6);
  const SIN30 = Math.sin(Math.PI / 6);

  function isoProject(pos3d) {
    const [x, y, z] = pos3d;
    const sx = (x - z) * COS30;
    const sy = -y + (x + z) * SIN30 * 0.5;
    return [sx, sy];
  }

  function renderSpatialTree(container, treeData) {
    const w = container.clientWidth || 200;
    const h = container.clientHeight || 200;

    const projected = treeData.positions.map(p => isoProject(p));

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const [sx, sy] of projected) {
      if (sx < minX) minX = sx;
      if (sx > maxX) maxX = sx;
      if (sy < minY) minY = sy;
      if (sy > maxY) maxY = sy;
    }

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const pad = 20;
    const scaleX = (w - 2 * pad) / rangeX;
    const scaleY = (h - 2 * pad) / rangeY;
    const scale = Math.min(scaleX, scaleY);
    const cx = w / 2;
    const cy = h / 2;
    const midX = (minX + maxX) / 2;
    const midY = (minY + maxY) / 2;

    const svg = d3.select(container).append('svg')
      .attr('viewBox', `0 0 ${w} ${h}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    for (const [a, b] of treeData.edges) {
      const [x1, y1] = projected[a];
      const [x2, y2] = projected[b];
      svg.append('line')
        .attr('class', 'tree-edge')
        .attr('x1', cx + (x1 - midX) * scale)
        .attr('y1', cy + (y1 - midY) * scale)
        .attr('x2', cx + (x2 - midX) * scale)
        .attr('y2', cy + (y2 - midY) * scale);
    }

    for (const [sx, sy] of projected) {
      svg.append('circle')
        .attr('class', 'tree-node')
        .attr('cx', cx + (sx - midX) * scale)
        .attr('cy', cy + (sy - midY) * scale)
        .attr('r', 3);
    }
  }

  function renderSpatialTrees() {
    const grid = document.getElementById('spatial-tree-grid');
    if (!grid || !EXAMPLES.spatialTrees) return;
    grid.innerHTML = '';

    for (const tree of EXAMPLES.spatialTrees) {
      const card = document.createElement('div');
      card.className = 'spatial-card';

      const viz = document.createElement('div');
      viz.className = 'spatial-card-viz';
      card.appendChild(viz);

      const info = document.createElement('div');
      info.className = 'spatial-card-info';
      const feats = tree.features || {};
      info.innerHTML = `
        <h4>${tree.name}</h4>
        <div class="spatial-props">
          <span>${tree.num_nodes} nodes</span>
          <span>${tree.num_edges} edges</span>
          ${feats.branch_points !== undefined ? `<span>${feats.branch_points} branch pts</span>` : ''}
        </div>`;
      card.appendChild(info);
      grid.appendChild(card);

      renderSpatialTree(viz, tree);
    }
  }

  renderSpatialTrees();

  // ── Mesh: Isometric wireframe rendering ───────────────────────────────
  function renderMeshWireframe(container, meshData, opts = {}) {
    const w = opts.size || container.clientWidth || 200;
    const h = opts.size || container.clientHeight || w;

    const projected = meshData.positions.map(p => isoProject(p));

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const [sx, sy] of projected) {
      if (sx < minX) minX = sx;
      if (sx > maxX) maxX = sx;
      if (sy < minY) minY = sy;
      if (sy > maxY) maxY = sy;
    }

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const pad = opts.pad || 16;
    const scaleX = (w - 2 * pad) / rangeX;
    const scaleY = (h - 2 * pad) / rangeY;
    const scale = Math.min(scaleX, scaleY);
    const cx = w / 2;
    const cy = h / 2;
    const midX = (minX + maxX) / 2;
    const midY = (minY + maxY) / 2;

    const svg = d3.select(container).append('svg')
      .attr('viewBox', `0 0 ${w} ${h}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    for (const [a, b] of meshData.edges) {
      if (a >= meshData.positions.length || b >= meshData.positions.length) continue;
      const [x1, y1] = projected[a];
      const [x2, y2] = projected[b];
      svg.append('line')
        .attr('class', 'mesh-edge')
        .attr('x1', cx + (x1 - midX) * scale)
        .attr('y1', cy + (y1 - midY) * scale)
        .attr('x2', cx + (x2 - midX) * scale)
        .attr('y2', cy + (y2 - midY) * scale);
    }

    const nodeR = opts.nodeRadius || 3;
    for (const [sx, sy] of projected) {
      svg.append('circle')
        .attr('class', 'mesh-node')
        .attr('cx', cx + (sx - midX) * scale)
        .attr('cy', cy + (sy - midY) * scale)
        .attr('r', nodeR);
    }
  }

  function renderMeshGallery() {
    const grid = document.getElementById('mesh-gallery-grid');
    if (!grid || !EXAMPLES.meshes) return;
    grid.innerHTML = '';

    for (const mesh of EXAMPLES.meshes) {
      const card = document.createElement('div');
      card.className = 'mesh-card';

      const viz = document.createElement('div');
      viz.className = 'mesh-card-viz';
      card.appendChild(viz);

      const info = document.createElement('div');
      info.className = 'mesh-card-info';
      info.innerHTML = `
        <h4>${mesh.name}</h4>
        <div class="mesh-props">
          <span>${mesh.num_nodes} vertices</span>
          <span>${mesh.num_edges} edges</span>
        </div>`;
      card.appendChild(info);
      grid.appendChild(card);

      renderMeshWireframe(viz, mesh);
    }
  }

  renderMeshGallery();

  // ── Mesh interpolation: multiple pairs ─────────────────────────────────
  function renderMeshInterpolationStrips() {
    const container = document.getElementById('mesh-interp-strips');
    const interps = EXAMPLES.meshInterpolations;
    if (!container || !interps) return;
    container.innerHTML = '';

    for (const pair of interps) {
      const pairEl = document.createElement('div');
      pairEl.className = 'interp-pair';

      // Title
      const title = document.createElement('div');
      title.className = 'interp-pair-title';
      title.innerHTML = `${pair.source} <span class="pair-arrow">&rarr;</span> ${pair.target}`;
      pairEl.appendChild(title);

      // Meta info
      const meta = document.createElement('div');
      meta.className = 'interp-pair-meta';
      const srcStep = pair.steps[0];
      const tgtStep = pair.steps[pair.steps.length - 1];
      meta.textContent = `${srcStep.num_nodes}v/${srcStep.num_edges}e \u2192 ${tgtStep.num_nodes}v/${tgtStep.num_edges}e \u00b7 ${pair.steps.length} steps`;
      pairEl.appendChild(meta);

      // Strip
      const strip = document.createElement('div');
      strip.className = 'mesh-interp-strip';

      for (let i = 0; i < pair.steps.length; i++) {
        const step = pair.steps[i];
        const isEndpoint = (i === 0 || i === pair.steps.length - 1);

        const stepEl = document.createElement('div');
        stepEl.className = 'mesh-interp-step' + (isEndpoint ? ' is-endpoint' : '');

        const viz = document.createElement('div');
        viz.className = 'mesh-interp-step-viz';
        stepEl.appendChild(viz);

        const label = document.createElement('div');
        label.className = 'mesh-interp-step-label';
        label.textContent = step.label;
        stepEl.appendChild(label);

        strip.appendChild(stepEl);

        renderMeshWireframe(viz, step, { pad: 12, nodeRadius: 2.5, size: 130 });
      }

      pairEl.appendChild(strip);
      container.appendChild(pairEl);
    }
  }

  renderMeshInterpolationStrips();

  // Mesh VAE training chart
  if (EXAMPLES.trainingCurvesSpatial && EXAMPLES.trainingCurvesSpatial.meshVae) {
    drawChart('mesh-vae-chart', EXAMPLES.trainingCurvesSpatial.meshVae, 'loss', { color: '#ec4899', yLabel: 'Mesh VAE loss' });
  }

  // ── WL Test: Interactive color refinement visualization ─────────────
  const WL_PALETTE = ['#6366f1', '#f59e0b', '#10b981', '#f43f5e', '#06b6d4', '#8b5cf6', '#ec4899', '#14b8a6'];

  function renderWLGraphD3(container, graphData, nodeColors, opts = {}) {
    const w = opts.size || container.clientWidth || 180;
    const h = opts.size || container.clientHeight || 180;
    const pad = 20;
    const positions = graphData.positions;
    const edges = graphData.edges;

    if (!positions || positions.length === 0) return;

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const [x, y] of positions) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const scale = Math.min((w - 2 * pad) / rangeX, (h - 2 * pad) / rangeY);
    const cx = w / 2, cy = h / 2;
    const midX = (minX + maxX) / 2, midY = (minY + maxY) / 2;

    const screenPos = positions.map(([x, y]) => [
      cx + (x - midX) * scale,
      cy + (y - midY) * scale,
    ]);

    // Use D3 for animated transitions
    const svg = d3.select(container).selectAll('svg').data([0]);
    const svgEnter = svg.enter().append('svg')
      .attr('viewBox', `0 0 ${w} ${h}`)
      .attr('preserveAspectRatio', 'xMidYMid meet')
      .style('width', '100%')
      .style('height', '100%');
    const svgMerged = svgEnter.merge(svg);

    // Edges
    let edgeG = svgMerged.selectAll('g.wl-edges').data([0]);
    edgeG = edgeG.enter().append('g').attr('class', 'wl-edges').merge(edgeG);

    const edgeData = edges.filter(([i, j]) => i < screenPos.length && j < screenPos.length);
    const lines = edgeG.selectAll('line').data(edgeData, (d, i) => i);
    lines.enter().append('line')
      .attr('stroke', '#9ca3af')
      .attr('stroke-width', 1.5)
      .attr('stroke-opacity', 0.4)
      .attr('x1', d => screenPos[d[0]][0]).attr('y1', d => screenPos[d[0]][1])
      .attr('x2', d => screenPos[d[1]][0]).attr('y2', d => screenPos[d[1]][1]);

    // Nodes
    let nodeG = svgMerged.selectAll('g.wl-nodes').data([0]);
    nodeG = nodeG.enter().append('g').attr('class', 'wl-nodes').merge(nodeG);

    const nodeData = screenPos.map(([x, y], i) => ({
      x, y, i,
      color: nodeColors && nodeColors.length > i
        ? WL_PALETTE[nodeColors[i] % WL_PALETTE.length]
        : '#6366f1'
    }));

    const circles = nodeG.selectAll('circle').data(nodeData, d => d.i);
    circles.enter().append('circle')
      .attr('cx', d => d.x).attr('cy', d => d.y)
      .attr('r', opts.nodeRadius || 7)
      .attr('fill', d => d.color)
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
    .merge(circles)
      .transition().duration(400).ease(d3.easeCubicOut)
      .attr('fill', d => d.color);

    // Node labels (color index)
    if (opts.showLabels !== false) {
      const labels = nodeG.selectAll('text').data(nodeData, d => d.i);
      labels.enter().append('text')
        .attr('x', d => d.x).attr('y', d => d.y)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'central')
        .attr('font-size', '8px')
        .attr('font-family', 'var(--font-mono, monospace)')
        .attr('font-weight', '700')
        .attr('fill', '#fff')
        .attr('pointer-events', 'none')
        .text(d => nodeColors ? nodeColors[d.i] : '')
      .merge(labels)
        .transition().duration(400)
        .text(d => nodeColors ? nodeColors[d.i] : '');
    }
  }

  function buildHistogramData(nodeColors) {
    const counts = {};
    if (!nodeColors) return [];
    for (const c of nodeColors) {
      counts[c] = (counts[c] || 0) + 1;
    }
    return Object.entries(counts)
      .map(([color, count]) => ({ colorIdx: +color, count }))
      .sort((a, b) => a.colorIdx - b.colorIdx);
  }

  function renderHistogramBar(container, histA, histB, numNodes) {
    container.innerHTML = '';
    const wrap = document.createElement('div');
    wrap.className = 'wl-histogram-compare';

    function makeBar(hist, label) {
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
        seg.title = `Color ${h.colorIdx}: ${h.count} node${h.count > 1 ? 's' : ''}`;
        bar.appendChild(seg);
      }
      row.appendChild(bar);
      return row;
    }

    wrap.appendChild(makeBar(histA, 'G₁'));
    wrap.appendChild(makeBar(histB, 'G₂'));
    container.appendChild(wrap);
  }

  function renderWLSection() {
    const container = document.getElementById('wl-pairs-container');
    if (!container || !EXAMPLES.wlTest) return;
    container.innerHTML = '';

    for (const pair of EXAMPLES.wlTest) {
      const pairEl = document.createElement('div');
      pairEl.className = 'wl-pair fade-in';

      // Header with title + verdict
      const header = document.createElement('div');
      header.className = 'wl-pair-header';

      const titleRow = document.createElement('div');
      titleRow.className = 'wl-pair-title-row';

      const title = document.createElement('h3');
      title.className = 'wl-pair-title';
      title.textContent = pair.name;
      titleRow.appendChild(title);

      const verdict = document.createElement('span');
      if (pair.distinguished) {
        verdict.className = 'wl-verdict-badge wl-verdict--success';
        verdict.innerHTML = '&#x2713; Distinguished at iteration ' + pair.distinguishing_iteration;
      } else {
        verdict.className = 'wl-verdict-badge wl-verdict--fail';
        verdict.innerHTML = '&#x2717; Indistinguishable';
      }
      titleRow.appendChild(verdict);
      header.appendChild(titleRow);

      // Description
      if (pair.description) {
        const desc = document.createElement('p');
        desc.className = 'wl-pair-desc';
        desc.textContent = pair.description;
        header.appendChild(desc);
      }

      pairEl.appendChild(header);

      // Interactive viewer
      const viewer = document.createElement('div');
      viewer.className = 'wl-viewer';

      // Current step state
      let currentStep = 0;
      const maxStep = pair.iterations.length - 1;

      // Step indicator
      const stepIndicator = document.createElement('div');
      stepIndicator.className = 'wl-step-indicator';

      const btnPrev = document.createElement('button');
      btnPrev.className = 'wl-nav-btn';
      btnPrev.innerHTML = '&#8592;';
      btnPrev.title = 'Previous iteration';

      const stepText = document.createElement('span');
      stepText.className = 'wl-step-text';

      const btnNext = document.createElement('button');
      btnNext.className = 'wl-nav-btn';
      btnNext.innerHTML = '&#8594;';
      btnNext.title = 'Next iteration';

      const btnPlay = document.createElement('button');
      btnPlay.className = 'wl-nav-btn wl-play-btn';
      btnPlay.innerHTML = '&#9654;';
      btnPlay.title = 'Auto-play';

      stepIndicator.appendChild(btnPrev);
      stepIndicator.appendChild(stepText);
      stepIndicator.appendChild(btnNext);
      stepIndicator.appendChild(btnPlay);
      viewer.appendChild(stepIndicator);

      // Slider
      const sliderWrap = document.createElement('div');
      sliderWrap.className = 'wl-slider-wrap';
      const slider = document.createElement('input');
      slider.type = 'range';
      slider.min = '0';
      slider.max = String(maxStep);
      slider.value = '0';
      slider.className = 'wl-slider';
      sliderWrap.appendChild(slider);

      // Step dots
      const dots = document.createElement('div');
      dots.className = 'wl-step-dots';
      for (let i = 0; i <= maxStep; i++) {
        const dot = document.createElement('span');
        dot.className = 'wl-dot' + (i === 0 ? ' active' : '');
        dot.dataset.step = i;
        dots.appendChild(dot);
      }
      sliderWrap.appendChild(dots);
      viewer.appendChild(sliderWrap);

      // Graphs side by side
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
      wrapA.appendChild(vizA);
      graphsRow.appendChild(wrapA);

      // VS divider
      const vs = document.createElement('div');
      vs.className = 'wl-vs';
      vs.textContent = 'vs';
      graphsRow.appendChild(vs);

      const wrapB = document.createElement('div');
      wrapB.className = 'wl-graph-panel';
      const labelB = document.createElement('div');
      labelB.className = 'wl-graph-label';
      labelB.textContent = pair.graphB.name;
      wrapB.appendChild(labelB);
      const vizB = document.createElement('div');
      vizB.className = 'wl-graph-viz';
      wrapB.appendChild(vizB);
      graphsRow.appendChild(wrapB);

      viewer.appendChild(graphsRow);

      // Histogram comparison
      const histArea = document.createElement('div');
      histArea.className = 'wl-histogram-area';
      viewer.appendChild(histArea);

      // Match verdict for current step
      const matchBadge = document.createElement('div');
      matchBadge.className = 'wl-step-match';
      viewer.appendChild(matchBadge);

      // Color count info
      const colorInfo = document.createElement('div');
      colorInfo.className = 'wl-color-info';
      viewer.appendChild(colorInfo);

      pairEl.appendChild(viewer);
      container.appendChild(pairEl);
      observer.observe(pairEl);

      // Render function
      function renderStep(step) {
        currentStep = step;
        const iter = pair.iterations[step];

        stepText.textContent = `Iteration ${iter.step}`;
        slider.value = String(step);

        // Update dots
        dots.querySelectorAll('.wl-dot').forEach((d, i) => {
          d.classList.toggle('active', i === step);
          d.classList.toggle('past', i < step);
        });

        // Update nav buttons
        btnPrev.disabled = step === 0;
        btnNext.disabled = step === maxStep;

        // Render graphs with animated color transitions
        renderWLGraphD3(vizA, pair.graphA, iter.colors_a, { nodeRadius: 8 });
        renderWLGraphD3(vizB, pair.graphB, iter.colors_b, { nodeRadius: 8 });

        // Histograms
        const histA = buildHistogramData(iter.colors_a);
        const histB = buildHistogramData(iter.colors_b);
        const numNodes = iter.colors_a.length;
        renderHistogramBar(histArea, histA, histB, numNodes);

        // Match badge
        if (iter.histograms_match) {
          matchBadge.className = 'wl-step-match wl-step-match--same';
          matchBadge.textContent = 'Histograms match — cannot distinguish';
        } else {
          matchBadge.className = 'wl-step-match wl-step-match--diff';
          matchBadge.textContent = 'Histograms differ — graphs distinguished!';
        }

        // Color count
        const nA = iter.num_colors_a || new Set(iter.colors_a).size;
        const nB = iter.num_colors_b || new Set(iter.colors_b).size;
        colorInfo.innerHTML = `<span>${pair.graphA.name}: <strong>${nA}</strong> color${nA > 1 ? 's' : ''}</span><span>${pair.graphB.name}: <strong>${nB}</strong> color${nB > 1 ? 's' : ''}</span>`;
      }

      // Events
      btnPrev.addEventListener('click', () => { if (currentStep > 0) renderStep(currentStep - 1); });
      btnNext.addEventListener('click', () => { if (currentStep < maxStep) renderStep(currentStep + 1); });

      slider.addEventListener('input', () => renderStep(+slider.value));

      dots.querySelectorAll('.wl-dot').forEach(dot => {
        dot.addEventListener('click', () => renderStep(+dot.dataset.step));
      });

      let playing = false;
      let playTimer = null;
      btnPlay.addEventListener('click', () => {
        if (playing) {
          clearInterval(playTimer);
          playing = false;
          btnPlay.innerHTML = '&#9654;';
          btnPlay.classList.remove('playing');
        } else {
          playing = true;
          btnPlay.innerHTML = '&#9646;&#9646;';
          btnPlay.classList.add('playing');
          if (currentStep >= maxStep) renderStep(0);
          playTimer = setInterval(() => {
            if (currentStep < maxStep) {
              renderStep(currentStep + 1);
            } else {
              clearInterval(playTimer);
              playing = false;
              btnPlay.innerHTML = '&#9654;';
              btnPlay.classList.remove('playing');
            }
          }, 1200);
        }
      });

      // Initial render
      renderStep(0);
    }
  }

  renderWLSection();

  // ── Physics GNN: Heat diffusion on grid graph ─────────────────────────
  function renderPhysicsSection() {
    const container = document.getElementById('physics-demo-graphs');
    if (!container) return;

    const SIZE = 7;

    function makeGrid(size) {
      const nodes = [];
      const edges = [];
      for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
          const i = r * size + c;
          nodes.push({ id: i, x: c, y: r });
          if (c < size - 1) edges.push([i, i + 1]);
          if (r < size - 1) edges.push([i, i + size]);
        }
      }
      return { nodes, edges };
    }

    const { nodes, edges } = makeGrid(SIZE);

    // Build adjacency list
    const adj = new Map();
    nodes.forEach(n => adj.set(n.id, []));
    edges.forEach(([a, b]) => { adj.get(a).push(b); adj.get(b).push(a); });

    // Initial heat: source at center
    const sourceId = Math.floor(SIZE * SIZE / 2); // node 24
    const heatInitial = new Float64Array(nodes.length);
    heatInitial[sourceId] = 1.0;

    // Diffuse heat
    const ITERS = 20;
    const ALPHA = 0.2;
    let heat = Float64Array.from(heatInitial);
    for (let it = 0; it < ITERS; it++) {
      const next = Float64Array.from(heat);
      for (let i = 0; i < nodes.length; i++) {
        let laplacian = 0;
        for (const j of adj.get(i)) {
          laplacian += heat[j] - heat[i];
        }
        next[i] = heat[i] + ALPHA * laplacian;
      }
      heat = next;
    }
    const heatDiffused = heat;

    // Color scale: blue -> orange -> red
    const colorScale = d3.scaleSequential()
      .domain([0, 1])
      .interpolator(t => {
        if (t < 0.5) {
          return d3.interpolateRgb('#4a9eff', '#f59e0b')(t * 2);
        }
        return d3.interpolateRgb('#f59e0b', '#ef4444')((t - 0.5) * 2);
      });

    // Normalize heat for color mapping
    const maxHeatDiffused = Math.max(...heatDiffused);

    function renderGraph(parentEl, title, heatValues) {
      const panel = document.createElement('div');
      panel.className = 'physics-graph-panel';

      const heading = document.createElement('h4');
      heading.textContent = title;
      panel.appendChild(heading);

      const vizDiv = document.createElement('div');
      vizDiv.className = 'physics-graph-viz';
      panel.appendChild(vizDiv);
      parentEl.appendChild(panel);

      const w = 240;
      const h = 240;
      const pad = 30;
      const cellW = (w - 2 * pad) / (SIZE - 1);
      const cellH = (h - 2 * pad) / (SIZE - 1);

      const svg = d3.select(vizDiv).append('svg')
        .attr('viewBox', `0 0 ${w} ${h}`)
        .attr('preserveAspectRatio', 'xMidYMid meet');

      // Edges
      svg.append('g').selectAll('line')
        .data(edges)
        .join('line')
        .attr('x1', d => pad + nodes[d[0]].x * cellW)
        .attr('y1', d => pad + nodes[d[0]].y * cellH)
        .attr('x2', d => pad + nodes[d[1]].x * cellW)
        .attr('y2', d => pad + nodes[d[1]].y * cellH)
        .attr('class', 'physics-edge');

      // Determine max for this specific view
      const maxVal = Math.max(...heatValues, 0.001);

      // Nodes
      svg.append('g').selectAll('circle')
        .data(nodes)
        .join('circle')
        .attr('cx', d => pad + d.x * cellW)
        .attr('cy', d => pad + d.y * cellH)
        .attr('r', 8)
        .attr('class', 'physics-node')
        .attr('fill', d => colorScale(heatValues[d.id] / maxVal))
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5);
    }

    renderGraph(container, 'Initial State', Array.from(heatInitial));
    renderGraph(container, 'After Diffusion', Array.from(heatDiffused));
  }

  renderPhysicsSection();

  // ── Hyperbolic GNN Section ──────────────────────────────────────────
  function renderHyperbolicSection() {
    const container = document.getElementById('hyperbolic-demo-disk');
    if (!container) return;

    const DEPTH = 4;

    // Generate binary tree
    function makeBinaryTree(depth) {
      const nodes = [];
      const edges = [];
      let id = 0;
      function addNode(level, parentId) {
        const nodeId = id++;
        nodes.push({ id: nodeId, level });
        if (parentId >= 0) edges.push([parentId, nodeId]);
        if (level < depth) {
          addNode(level + 1, nodeId);
          addNode(level + 1, nodeId);
        }
      }
      addNode(0, -1);
      return { nodes, edges };
    }

    const { nodes, edges } = makeBinaryTree(DEPTH);

    // Embed nodes in Poincaré disk using radial layout
    // Build children map
    const children = new Map();
    nodes.forEach(n => children.set(n.id, []));
    edges.forEach(([parent, child]) => {
      children.get(parent).push(child);
    });

    const positions = new Map();

    function layoutNode(nodeId, level, angleStart, angleEnd) {
      if (level === 0) {
        positions.set(nodeId, [0, 0]);
      } else {
        const r = level / (DEPTH + 1) * 0.85;
        const angle = (angleStart + angleEnd) / 2;
        positions.set(nodeId, [r * Math.cos(angle), r * Math.sin(angle)]);
      }
      const kids = children.get(nodeId);
      if (kids.length === 0) return;
      const span = angleEnd - angleStart;
      const step = span / kids.length;
      kids.forEach((kid, i) => {
        layoutNode(kid, level + 1, angleStart + i * step, angleStart + (i + 1) * step);
      });
    }

    layoutNode(0, 0, 0, 2 * Math.PI);

    // Geodesic arc approximation (linear interpolation for short edges)
    function geodesicArc(p1, p2, nSamples) {
      const points = [];
      for (let i = 0; i <= nSamples; i++) {
        const t = i / nSamples;
        points.push([p1[0] * (1 - t) + p2[0] * t, p1[1] * (1 - t) + p2[1] * t]);
      }
      return points;
    }

    // Color by depth level
    const levelColors = ['#ef4444', '#f59e0b', '#6366f1', '#818cf8', '#a5b4fc'];

    // Create SVG
    const svg = d3.select(container).append('svg')
      .attr('viewBox', '-1.1 -1.1 2.2 2.2')
      .attr('preserveAspectRatio', 'xMidYMid meet');

    // Boundary circle (dashed)
    svg.append('circle')
      .attr('cx', 0).attr('cy', 0).attr('r', 1)
      .attr('class', 'hyperbolic-boundary');

    // Concentric distance circles
    [0.3, 0.6, 0.85].forEach(r => {
      svg.append('circle')
        .attr('cx', 0).attr('cy', 0).attr('r', r)
        .attr('class', 'hyperbolic-grid-circle');
    });

    // Draw edges as paths
    const lineGen = d3.line().x(d => d[0]).y(d => d[1]).curve(d3.curveBasis);
    svg.append('g').selectAll('path')
      .data(edges)
      .join('path')
      .attr('d', d => {
        const p1 = positions.get(d[0]);
        const p2 = positions.get(d[1]);
        const pts = geodesicArc(p1, p2, 12);
        return lineGen(pts);
      })
      .attr('class', 'hyperbolic-edge');

    // Draw nodes
    svg.append('g').selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('cx', d => positions.get(d.id)[0])
      .attr('cy', d => positions.get(d.id)[1])
      .attr('r', d => d.level === 0 ? 0.06 : (d.level === 1 ? 0.045 : 0.035))
      .attr('class', 'hyperbolic-node')
      .attr('fill', d => levelColors[Math.min(d.level, levelColors.length - 1)]);
  }

  renderHyperbolicSection();

  // ── Guide Section (injected via JS for parser compatibility) ──────────
  (function renderGuideSection() {
    const guideEl = document.getElementById('guide');
    if (!guideEl) {
      // If the HTML parser dropped the section, re-insert from source
      const footer = document.querySelector('footer');
      if (!footer) return;
      fetch('index.html')
        .then(r => r.ok ? r.text() : Promise.reject())
        .then(html => {
          const start = html.indexOf('<section id="guide">');
          const end = html.indexOf('</section>', start);
          if (start < 0 || end < 0) return;
          const sectionHTML = html.substring(start, end + '</section>'.length);
          const parser = new DOMParser();
          const doc = parser.parseFromString(sectionHTML, 'text/html');
          const section = doc.querySelector('section');
          if (section) {
            document.body.insertBefore(section, footer);
            section.querySelectorAll('.fade-in').forEach(el => observer.observe(el));
          }
        })
        .catch(() => {});
    }
  })();

  // Re-observe new fade-in elements added by JS
  document.querySelectorAll('.fade-in:not(.visible)').forEach(el => observer.observe(el));

})();
