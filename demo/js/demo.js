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
  const COLOR_EDGE = '#c7c7d4';

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

    // animate in waves from node 0
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

      // highlight current edges
      linkSel.each(function (d) {
        const s = typeof d.source === 'object' ? d.source.id : d.source;
        const t = typeof d.target === 'object' ? d.target.id : d.target;
        if (frontier.includes(s) || frontier.includes(t)) {
          d3.select(this).classed('message-active', true);
        }
      });

      // activate new nodes
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
        // cleanup
        setTimeout(() => {
          linkSel.classed('message-active', false);
          nodeSel.classed('pulse', false);
        }, 1000);
      }
    }
    // reset first
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
  function renderExampleGraphs() {
    const grid = document.getElementById('gen-grid');
    if (!grid) return;
    grid.innerHTML = '';

    let graphsToShow = EXAMPLES.generatedGraphs;
    const typeFilter = document.getElementById('ctrl-type').value;
    if (typeFilter !== 'all') {
      const prefix = {er: 'erdos', ba: 'barabasi', ws: 'watts'}[typeFilter];
      graphsToShow = graphsToShow.filter(g => g.type.startsWith(prefix));
    }

    for (const g of graphsToShow) {
      const card = document.createElement('div');
      card.className = 'graph-card';

      const viz = document.createElement('div');
      viz.className = 'graph-card-viz';
      card.appendChild(viz);

      const info = document.createElement('div');
      info.className = 'graph-card-info';
      info.innerHTML = `
        <h4>${g.name}</h4>
        <div class="graph-props">
          <span>Nodes:</span> <span>${g.properties.numNodes}</span>
          <span>Edges:</span> <span>${g.properties.numEdges}</span>
          <span>Density:</span> <span>${g.properties.density.toFixed(3)}</span>
          <span>Clustering:</span> <span>${g.properties.avgClustering.toFixed(3)}</span>
        </div>`;
      card.appendChild(info);
      grid.appendChild(card);

      // render small force graph
      renderSmallGraph(viz, g.nodes.length, g.edges);
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

  renderExampleGraphs();

  // slider updates
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

  document.getElementById('btn-gen-new')?.addEventListener('click', renderExampleGraphs);
  document.getElementById('ctrl-type')?.addEventListener('change', renderExampleGraphs);

  // ── VAE: interpolation ──────────────────────────────────────────────────
  const interpSteps = EXAMPLES.interpolation.steps;

  function renderInterpolation(stepIndex) {
    const el = document.getElementById('interp-viz');
    if (!el || !interpSteps[stepIndex]) return;
    el.innerHTML = '';

    const step = interpSteps[stepIndex];
    document.getElementById('interp-alpha').textContent = step.alpha.toFixed(1);

    renderSmallGraph(el, step.nodes.length, step.edges);
  }

  renderInterpolation(0);

  document.getElementById('interp-slider')?.addEventListener('input', function () {
    renderInterpolation(+this.value);
  });

  // VAE training chart
  drawChart('vae-chart', EXAMPLES.trainingCurves.vae, 'totalLoss', { color: '#c084fc', yLabel: 'ELBO loss' });

})();
