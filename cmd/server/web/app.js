async function j(url, opts) { const r = await fetch(url, opts); return r.json(); }

async function refreshStats() {
  const s = await j('/api/stats');
  document.getElementById('c-mem').textContent = s.memories.toLocaleString();
  document.getElementById('c-vec').textContent = s.indexed_vectors.toLocaleString();
  document.getElementById('c-imp').textContent = s.avg_importance;
  document.getElementById('c-acc').textContent = s.total_access.toLocaleString();
  document.getElementById('c-lat').textContent = s.last_query_ms;
}

async function refreshFeed() {
  const mems = await j('/api/memories');
  const el = document.getElementById('feed');
  el.innerHTML = '';
  for (const m of mems) {
    const d = document.createElement('div');
    d.className = 'mem';
    d.innerHTML = `<span>#${m.id} ${m.text}</span>
      <span class="meta">imp ${m.importance.toFixed(2)} · ×${m.access_count}
      <button onclick="forget(${m.id})">forget</button></span>`;
    el.appendChild(d);
  }
}

async function forget(id) {
  await j('/api/memory/' + id, {method: 'DELETE'});
  refreshStats(); refreshFeed();
}

async function recall() {
  const q = document.getElementById('q-text').value;
  if (!q.trim()) return;
  const r = await j('/api/recall?q=' + encodeURIComponent(q) + '&k=5');
  const el = document.getElementById('hits');
  el.innerHTML = '';
  for (const h of r.hits) {
    const d = document.createElement('div');
    d.className = 'hit';
    d.innerHTML = `<div class="txt"><span class="score">${h.score.toFixed(3)}</span> · #${h.id} ${h.text}</div>
      <div class="bars">
        <span class="bar">vector <b>${h.vec_sim.toFixed(3)}</b></span>
        <span class="bar">bm25 <b>${h.bm25.toFixed(3)}</b></span>
        <span class="bar">decay <b>${h.decay.toFixed(3)}</b></span>
        <span class="bar">importance <b>${h.importance.toFixed(2)}</b></span>
        <span class="bar">recalled ×${h.access_count}</span>
      </div>`;
    el.appendChild(d);
  }
  if (!r.hits.length) el.innerHTML = '<div class="hit">no memories match</div>';
  refreshStats(); refreshFeed();
}

document.getElementById('btn-recall').onclick = recall;
document.getElementById('q-text').addEventListener('keydown', e => { if (e.key === 'Enter') recall(); });
document.getElementById('btn-remember').onclick = async () => {
  const t = document.getElementById('r-text').value;
  const imp = parseFloat(document.getElementById('r-imp').value);
  if (!t.trim()) return;
  await j('/api/remember', {method:'POST', body: JSON.stringify({text: t, importance: imp})});
  document.getElementById('r-text').value = '';
  refreshStats(); refreshFeed();
};
document.getElementById('btn-seed').onclick = async () => {
  await j('/api/seed', {method: 'POST'});
  setTimeout(() => { refreshStats(); refreshFeed(); }, 1500);
};

refreshStats(); refreshFeed();
setInterval(refreshStats, 3000);
