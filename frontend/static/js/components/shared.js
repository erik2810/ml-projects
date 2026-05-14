// Shared frontend helpers shared across panel modules.
//
// Panel modules import these utilities directly — no globals. The init*
// functions are idempotent-safe: wiring the same listener twice is fine
// because `addEventListener` is used with a plain function reference.

export function setLoading(btn, loading) {
  btn.classList.toggle('loading', loading);
  btn.disabled = loading;
}

export function log(outputId, msg) {
  const el = document.getElementById(outputId);
  if (!el) return;
  el.textContent += msg + '\n';
  el.scrollTop = el.scrollHeight;
}

// ---------------------------------------------------------------------------
// Tab switching — tabs are <button class="tab-btn" data-tab="..."> and the
// matching content panels are <section id="panel-...">.
// ---------------------------------------------------------------------------
export function initTabs() {
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
}

// ---------------------------------------------------------------------------
// Slider value display — every .slider-field range input gets its textual
// companion updated as the user drags.
// ---------------------------------------------------------------------------
export function initSliders() {
  document.querySelectorAll('.slider-field input[type="range"]').forEach(slider => {
    const output = slider.parentElement.querySelector('.slider-value');
    if (!output) return;
    slider.addEventListener('input', () => {
      const decimals = slider.step.includes('.') ? slider.step.split('.')[1].length : 0;
      output.textContent = parseFloat(slider.value).toFixed(decimals);
    });
  });
}
