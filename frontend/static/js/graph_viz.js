// d3 is loaded globally from CDN

const COLORS = {
  community0: '#6366f1',
  community1: '#f59e0b',
  unlabeled: '#4b5563',
  edge: '#374151',
  edgeHighlight: '#818cf8',
  bg: '#0a0e1a',
};

/**
 * Render a force-directed graph inside a container element.
 * Returns a handle with update/destroy methods.
 */
export function renderGraph(container, { nodes, edges, labels = null, predictions = null } = {}) {
  const el = typeof container === 'string' ? document.querySelector(container) : container;
  el.innerHTML = '';

  const width = el.clientWidth || 500;
  const height = el.clientHeight || 400;

  const svg = d3.select(el).append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  const graphNodes = nodes.map((n, i) => ({
    id: typeof n === 'object' ? n.id : i,
    index: i,
    label: labels ? labels[i] : -1,
    predicted: predictions ? predictions[i] : -1,
  }));

  const graphEdges = edges.map(([s, t]) => ({ source: s, target: t }));

  const sim = d3.forceSimulation(graphNodes)
    .force('link', d3.forceLink(graphEdges).id(d => d.id).distance(40))
    .force('charge', d3.forceManyBody().strength(-120))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide(12));

  const linkGroup = svg.append('g');
  const nodeGroup = svg.append('g');

  const link = linkGroup.selectAll('line')
    .data(graphEdges)
    .join('line')
    .attr('stroke', COLORS.edge)
    .attr('stroke-width', 1.2)
    .attr('stroke-opacity', 0.5);

  const node = nodeGroup.selectAll('g')
    .data(graphNodes)
    .join('g')
    .call(drag(sim));

  node.append('circle')
    .attr('r', d => 5 + (labels ? 2 : 0))
    .attr('fill', d => nodeColor(d))
    .attr('stroke', '#1a1a2e')
    .attr('stroke-width', 1.5);

  node.append('text')
    .text(d => d.id)
    .attr('text-anchor', 'middle')
    .attr('dy', '0.35em')
    .attr('font-size', '8px')
    .attr('fill', '#c8ccd4')
    .attr('pointer-events', 'none');

  // tooltip
  const tooltip = d3.select(el).append('div')
    .attr('class', 'graph-tooltip');

  node.on('mouseenter', (event, d) => {
    let text = `Node ${d.id}`;
    if (d.label >= 0) text += ` | true: ${d.label}`;
    if (d.predicted >= 0) text += ` | pred: ${d.predicted}`;
    tooltip.text(text).classed('visible', true);
  })
  .on('mousemove', (event) => {
    const rect = el.getBoundingClientRect();
    tooltip
      .style('left', (event.clientX - rect.left + 12) + 'px')
      .style('top', (event.clientY - rect.top - 8) + 'px');
  })
  .on('mouseleave', () => {
    tooltip.classed('visible', false);
  });

  sim.on('tick', () => {
    link
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);
    node.attr('transform', d => `translate(${d.x},${d.y})`);
  });

  return {
    updatePredictions(preds) {
      graphNodes.forEach((n, i) => { n.predicted = preds[i] ?? -1; });
      node.select('circle').attr('fill', d => nodeColor(d));
    },
    destroy() {
      sim.stop();
      el.innerHTML = '';
    }
  };
}

/**
 * Render a mini graph (for grid cells).
 */
export function renderMiniGraph(container, { nodes, edges, label = '' } = {}) {
  const el = typeof container === 'string' ? document.querySelector(container) : container;
  el.innerHTML = '';
  const size = el.clientWidth || 180;

  const svg = d3.select(el).append('svg')
    .attr('viewBox', `0 0 ${size} ${size}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  const gNodes = nodes.map((_, i) => ({ id: i }));
  const gEdges = edges.map(([s, t]) => ({ source: s, target: t }));

  const sim = d3.forceSimulation(gNodes)
    .force('link', d3.forceLink(gEdges).id(d => d.id).distance(25))
    .force('charge', d3.forceManyBody().strength(-60))
    .force('center', d3.forceCenter(size / 2, size / 2))
    .force('collision', d3.forceCollide(8));

  const links = svg.append('g').selectAll('line')
    .data(gEdges).join('line')
    .attr('stroke', '#374151').attr('stroke-width', 1).attr('stroke-opacity', 0.4);

  const circles = svg.append('g').selectAll('circle')
    .data(gNodes).join('circle')
    .attr('r', 4).attr('fill', '#6366f1').attr('stroke', '#1a1a2e').attr('stroke-width', 1);

  sim.on('tick', () => {
    links.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    circles.attr('cx', d => d.x).attr('cy', d => d.y);
  });

  // stop sim after settling
  setTimeout(() => sim.stop(), 2500);
}

/**
 * Draw a simple loss curve on a canvas element.
 */
export function drawLossCurve(canvas, losses, { color = '#f59e0b', label = 'Loss' } = {}) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const pad = { top: 20, right: 16, bottom: 24, left: 44 };

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0a0e1a';
  ctx.fillRect(0, 0, w, h);

  if (!losses || losses.length < 2) return;

  const maxVal = Math.max(...losses) * 1.1;
  const minVal = Math.min(...losses) * 0.9;
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  // axes
  ctx.strokeStyle = '#252b40';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, h - pad.bottom);
  ctx.lineTo(w - pad.right, h - pad.bottom);
  ctx.stroke();

  // y-axis labels
  ctx.fillStyle = '#5c6478';
  ctx.font = '9px monospace';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const val = minVal + (maxVal - minVal) * (1 - i / 4);
    const y = pad.top + plotH * (i / 4);
    ctx.fillText(val.toFixed(2), pad.left - 4, y + 3);
  }

  // curve
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  losses.forEach((v, i) => {
    const x = pad.left + (i / (losses.length - 1)) * plotW;
    const y = pad.top + plotH * (1 - (v - minVal) / (maxVal - minVal));
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // label
  ctx.fillStyle = color;
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(label, pad.left + 4, pad.top + 12);
}

function nodeColor(d) {
  if (d.predicted >= 0) {
    return d.predicted === 0 ? COLORS.community0 : COLORS.community1;
  }
  if (d.label >= 0) {
    return d.label === 0 ? COLORS.community0 : COLORS.community1;
  }
  return COLORS.unlabeled;
}

function drag(simulation) {
  return d3.drag()
    .on('start', (event, d) => {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x; d.fy = d.y;
    })
    .on('drag', (event, d) => {
      d.fx = event.x; d.fy = event.y;
    })
    .on('end', (event, d) => {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null; d.fy = null;
    });
}
