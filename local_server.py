#!/usr/bin/env python3
"""
local_server.py — Hero Title Brain Analyzer: local web UI with full TRIBE v2 pipeline.

Run:
    python local_server.py
    # then open http://localhost:5000 in your browser

Requirements:
    pip install flask openai python-dotenv numpy scipy matplotlib tqdm
    # plus TRIBE v2: git clone https://github.com/facebookresearch/tribev2 && pip install -e tribev2
    # optional: pip install nilearn  (for precise Destrieux atlas mapping)

Environment:
    OPENAI_API_KEY must be set in .env or shell environment
"""

import json
import os
import sys
import tempfile
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request

load_dotenv()

app = Flask(__name__)

# ──────────────────────────────────────────────
# In-memory job store  {job_id: {status, step, progress, result, error}}
# ──────────────────────────────────────────────
_jobs: dict = {}
_jobs_lock = threading.Lock()


def _set_job(job_id: str, **kwargs):
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


# ──────────────────────────────────────────────
# Pipeline runner (background thread)
# ──────────────────────────────────────────────
def _run_pipeline(job_id: str, title: str):
    tmp_dir = tempfile.mkdtemp(prefix="tribe_")
    try:
        # ── Step 1: Generate variants ────────────────────────────
        _set_job(job_id, step="Step 1/4 — Generating 60 variants via GPT…", progress=5)
        from generate import generate_variants

        variants = generate_variants(title)
        variants_path = os.path.join(tmp_dir, "variants.json")
        with open(variants_path, "w", encoding="utf-8") as f:
            json.dump(variants, f, ensure_ascii=False)

        n = sum(len(v) for k, v in variants.items() if isinstance(v, list))
        _set_job(job_id, step=f"Step 1/4 — Generated {n} variants ✓", progress=20)

        # ── Step 2: TRIBE v2 analysis ────────────────────────────
        _set_job(
            job_id,
            step="Step 2/4 — Running TRIBE v2 brain analysis (this takes a few minutes)…",
            progress=22,
        )
        from analyze import analyze_all, save_activations

        cache_dir = os.environ.get("TRIBE_CACHE_DIR", "./cache")
        activations = analyze_all(variants_path, cache_folder=cache_dir)

        _set_job(job_id, step="Step 2/4 — TRIBE v2 analysis complete ✓", progress=70)

        # ── Step 3: Aggregate ROI scores ─────────────────────────
        _set_job(job_id, step="Step 3/4 — Aggregating cortical vertices to ROI scores…", progress=72)
        from aggregate import aggregate_all

        scores = aggregate_all(activations)
        _set_job(job_id, step="Step 3/4 — ROI aggregation complete ✓", progress=85)

        # ── Step 4: CAS ranking ──────────────────────────────────
        _set_job(job_id, step="Step 4/4 — Computing CAS rankings…", progress=88)
        from compare import compute_rankings

        rankings = compute_rankings(scores)
        _set_job(
            job_id,
            status="done",
            step="Analysis complete ✓",
            progress=100,
            result=rankings,
        )

    except ImportError as e:
        _set_job(
            job_id,
            status="error",
            error=(
                f"Missing dependency: {e}. "
                "Make sure TRIBE v2 is installed: "
                "git clone https://github.com/facebookresearch/tribev2 && pip install -e tribev2"
            ),
        )
    except Exception as e:
        _set_job(job_id, status="error", error=str(e))
    finally:
        # Clean up temp dir
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ──────────────────────────────────────────────
# HTML UI
# ──────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Hero Title Brain Analyzer — Local</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#0d1117;--surface:#161b22;--surface2:#21262d;--border:#30363d;
    --purple:#a371f7;--cyan:#39d0d8;--green:#3fb950;--orange:#f0883e;--red:#f85149;
    --text:#e6edf3;--muted:#8b949e;--font:'Segoe UI',system-ui,sans-serif;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:var(--font);min-height:100vh}
  .wrap{max-width:1100px;margin:0 auto;padding:0 20px}

  header{border-bottom:1px solid var(--border);padding:18px 0}
  header .wrap{display:flex;align-items:center;gap:14px}
  .logo{width:40px;height:40px;background:linear-gradient(135deg,var(--purple),var(--cyan));
        border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0}
  .brand h1{font-size:1.15rem;font-weight:700;letter-spacing:-.01em}
  .brand p{font-size:.78rem;color:var(--muted);margin-top:2px}
  .badge{margin-left:auto;background:#1a2a15;border:1px solid #3fb95066;border-radius:20px;
         padding:4px 12px;font-size:.72rem;color:var(--green)}

  main{padding:40px 0 80px}
  .section-title{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;
                 color:var(--muted);margin-bottom:12px}

  .info-bar{background:#0d2510;border:1px solid #3fb95044;border-radius:8px;
            padding:12px 16px;font-size:.8rem;color:#5fdd7a;display:flex;gap:8px;
            align-items:flex-start;margin-bottom:28px}
  .info-bar svg{flex-shrink:0;margin-top:1px}

  .input-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
              padding:28px;margin-bottom:28px}
  .input-card h2{font-size:1.05rem;margin-bottom:6px}
  .input-card p{font-size:.85rem;color:var(--muted);margin-bottom:20px;line-height:1.5}
  .input-row{display:flex;gap:10px;flex-wrap:wrap}
  .input-row input{flex:1;min-width:260px;background:var(--surface2);border:1px solid var(--border);
                   border-radius:8px;padding:12px 16px;color:var(--text);font-size:.95rem;
                   outline:none;transition:border-color .2s}
  .input-row input:focus{border-color:var(--purple)}
  .input-row input::placeholder{color:var(--muted)}
  .btn{padding:12px 24px;border-radius:8px;border:none;cursor:pointer;font-size:.9rem;font-weight:600;transition:opacity .2s}
  .btn-primary{background:linear-gradient(135deg,var(--purple),#7c3aed);color:#fff}
  .btn-primary:hover{opacity:.88}
  .btn-primary:disabled{opacity:.4;cursor:not-allowed}

  #loading{display:none;background:var(--surface);border:1px solid var(--border);
           border-radius:12px;padding:36px;text-align:center;margin-bottom:28px}
  .spinner{width:40px;height:40px;border:3px solid var(--border);border-top-color:var(--purple);
           border-radius:50%;animation:spin .8s linear infinite;margin:0 auto 20px}
  @keyframes spin{to{transform:rotate(360deg)}}
  #loading-step{color:var(--muted);font-size:.88rem;margin-top:6px}
  .progress-bar{background:var(--surface2);border-radius:4px;height:6px;margin-top:16px;overflow:hidden}
  .progress-fill{height:100%;background:linear-gradient(90deg,var(--purple),var(--cyan));
                 border-radius:4px;transition:width .6s ease;width:0}

  #error-box{display:none;background:#1c1014;border:1px solid #5a1d1d;border-radius:8px;
             padding:16px 20px;color:var(--red);font-size:.88rem;margin-bottom:20px;white-space:pre-wrap}

  #results{display:none}
  .score-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-bottom:28px}
  .score-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px 20px}
  .score-card .label{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
  .score-card .value{font-size:1.9rem;font-weight:700;line-height:1}
  .score-card .sub{font-size:.78rem;color:var(--muted);margin-top:4px}
  .val-purple{color:var(--purple)} .val-cyan{color:var(--cyan)} .val-green{color:var(--green)}

  .chart-row{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px}
  @media(max-width:680px){.chart-row{grid-template-columns:1fr}}
  .chart-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:20px}
  .chart-card h3{font-size:.85rem;font-weight:600;margin-bottom:16px;color:var(--muted);
                 text-transform:uppercase;letter-spacing:.04em}
  .chart-card canvas{max-height:300px}

  .cat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:28px}
  @media(max-width:600px){.cat-grid{grid-template-columns:1fr}}
  .cat-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px 18px}
  .cat-card .cat-name{font-size:.8rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px}
  .cat-card .stat-row{display:flex;justify-content:space-between;font-size:.82rem;
                      padding:3px 0;border-bottom:1px solid var(--border)}
  .cat-card .stat-row:last-child{border-bottom:none}
  .stat-val{color:var(--text);font-weight:600}
  .similar-color{color:#a371f7} .alternative-color{color:#39d0d8} .exaggerated-color{color:#f0883e}

  .table-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden;margin-bottom:28px}
  .table-card h3{font-size:.85rem;font-weight:600;padding:16px 20px;border-bottom:1px solid var(--border);
                 text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
  .tbl-wrap{overflow-x:auto}
  table{width:100%;border-collapse:collapse;font-size:.84rem}
  th{padding:10px 14px;text-align:left;color:var(--muted);font-weight:600;font-size:.75rem;
     text-transform:uppercase;letter-spacing:.04em;border-bottom:1px solid var(--border);background:var(--surface2)}
  td{padding:10px 14px;border-bottom:1px solid var(--border);vertical-align:middle}
  tr:last-child td{border-bottom:none}
  tr:hover td{background:var(--surface2)}
  .rank-num{color:var(--muted);font-weight:700;width:40px}
  .cat-pill{padding:2px 10px;border-radius:20px;font-size:.72rem;font-weight:600;display:inline-block}
  .pill-similar{background:#2d1f5a;color:#a371f7}
  .pill-alternative{background:#0d2d30;color:#39d0d8}
  .pill-exaggerated{background:#3d2005;color:#f0883e}
  .pill-original{background:#1a2a15;color:#3fb950}
  .cas-bar{display:flex;align-items:center;gap:8px}
  .cas-track{flex:1;background:var(--surface2);border-radius:3px;height:6px;min-width:60px}
  .cas-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--purple),var(--cyan))}
  .cas-num{font-weight:600;font-size:.82rem;width:44px;text-align:right;color:var(--text)}
  .copy-btn{background:none;border:1px solid var(--border);color:var(--muted);border-radius:4px;
            padding:2px 8px;font-size:.72rem;cursor:pointer;transition:all .15s}
  .copy-btn:hover{border-color:var(--purple);color:var(--purple)}
</style>
</head>
<body>
<header>
  <div class="wrap">
    <div class="logo">🧠</div>
    <div class="brand">
      <h1>Hero Title Brain Analyzer</h1>
      <p>TRIBE v2 · Composite Attention Score · Copywriting Optimization</p>
    </div>
    <div class="badge">● Local · TRIBE v2</div>
  </div>
</header>

<main>
  <div class="wrap">

    <div class="info-bar">
      <svg width="16" height="16" fill="none" stroke="#5fdd7a" stroke-width="2" viewBox="0 0 24 24">
        <polyline points="20 6 9 17 4 12"/>
      </svg>
      <span><b>Full TRIBE v2 mode:</b> Real fMRI brain activation predictions on fsaverage5 mesh (~20,484 vertices). ROI scores are genuine neuroscience predictions, not heuristics.</span>
    </div>

    <div class="input-card">
      <h2>Analyze a Hero Title</h2>
      <p>Enter your hero section title. The analyzer generates 60 variants via GPT, runs each through Meta's TRIBE v2 model to predict cortical brain activation, aggregates into 6 ROI scores, and ranks by Composite Attention Score (CAS).</p>
      <div class="input-row">
        <input id="title-input" type="text" placeholder='e.g. "Trasforma il tuo business con l&apos;AI"'/>
        <button class="btn btn-primary" id="analyze-btn" onclick="startAnalysis()">Analyze →</button>
      </div>
    </div>

    <div id="error-box"></div>

    <div id="loading">
      <div class="spinner"></div>
      <div style="font-size:.95rem;font-weight:600">Running full TRIBE v2 pipeline…</div>
      <div id="loading-step">Starting…</div>
      <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
      <div style="font-size:.75rem;color:var(--muted);margin-top:12px">TRIBE v2 analysis typically takes 2–10 minutes depending on hardware.</div>
    </div>

    <div id="results">
      <div class="section-title">Results — Real TRIBE v2 Brain Scores</div>
      <div class="score-grid" id="score-grid"></div>
      <div class="chart-row">
        <div class="chart-card"><h3>Top 10 Titles by CAS</h3><canvas id="chart-top10"></canvas></div>
        <div class="chart-card"><h3>ROI Radar — Best vs Original</h3><canvas id="chart-radar"></canvas></div>
      </div>
      <div class="section-title">Category Breakdown</div>
      <div class="cat-grid" id="cat-grid"></div>
      <div class="table-card">
        <h3>Full Rankings — Top 30</h3>
        <div class="tbl-wrap">
          <table>
            <thead><tr><th>#</th><th>Title</th><th>Category</th><th style="min-width:160px">CAS Score</th></tr></thead>
            <tbody id="rankings-tbody"></tbody>
          </table>
        </div>
      </div>
    </div>

  </div>
</main>

<script>
let top10Chart = null, radarChart = null, pollTimer = null;

document.getElementById('title-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') startAnalysis();
});

function startAnalysis() {
  const title = document.getElementById('title-input').value.trim();
  if (!title) { showError('Please enter a hero title.'); return; }
  hideError(); hideResults(); showLoading(true);
  setProgress(0, 'Connecting to pipeline…');

  fetch('/analyze', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({title})
  })
  .then(r => r.json())
  .then(d => {
    if (d.error) { showLoading(false); showError(d.error); return; }
    pollJob(d.job_id);
  })
  .catch(err => { showLoading(false); showError('Network error: ' + err.message); });
}

function pollJob(jobId) {
  fetch(`/job/${jobId}`)
  .then(r => r.json())
  .then(job => {
    setProgress(job.progress || 0, job.step || '…');
    if (job.status === 'running') {
      pollTimer = setTimeout(() => pollJob(jobId), 2500);
    } else if (job.status === 'done') {
      showLoading(false);
      renderResults(job.result);
    } else {
      showLoading(false);
      showError(job.error || 'Analysis failed.');
    }
  })
  .catch(err => { showLoading(false); showError('Polling error: ' + err.message); });
}

function setProgress(pct, msg) {
  document.getElementById('loading-step').textContent = msg;
  document.getElementById('progress-fill').style.width = pct + '%';
}
function showLoading(v) {
  document.getElementById('loading').style.display = v ? 'block' : 'none';
  document.getElementById('analyze-btn').disabled = v;
  if (!v && pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
}
function showError(msg) {
  const el = document.getElementById('error-box');
  el.textContent = '⚠ ' + msg;
  el.style.display = 'block';
}
function hideError() { document.getElementById('error-box').style.display = 'none'; }
function hideResults() { document.getElementById('results').style.display = 'none'; }
function catColor(cat) {
  return {similar:'#a371f7',alternative:'#39d0d8',exaggerated:'#f0883e',original:'#3fb950'}[cat]||'#8b949e';
}
function catPill(cat) { return `<span class="cat-pill pill-${cat}">${cat}</span>`; }

function renderResults(data) {
  const {ranking_overall, ranking_by_category, category_stats, original, original_percentile} = data;
  const best = ranking_overall[0];
  const pctColor = original_percentile >= 75 ? 'val-green' : original_percentile >= 50 ? 'val-cyan' : 'val-purple';

  document.getElementById('score-grid').innerHTML = `
    <div class="score-card">
      <div class="label">Your Title CAS</div>
      <div class="value val-purple">${original.cas.toFixed(3)}</div>
      <div class="sub">${original.title}</div>
    </div>
    <div class="score-card">
      <div class="label">Percentile vs 60 Variants</div>
      <div class="value ${pctColor}">${original_percentile}%</div>
      <div class="sub">beats ${original_percentile}% of variants</div>
    </div>
    <div class="score-card">
      <div class="label">Best Variant CAS</div>
      <div class="value val-cyan">${best.cas.toFixed(3)}</div>
      <div class="sub">[${best.category}] ${best.title}</div>
    </div>
    <div class="score-card">
      <div class="label">CAS Improvement</div>
      <div class="value val-green">+${((best.cas - original.cas) * 100).toFixed(1)}%</div>
      <div class="sub">best variant vs original</div>
    </div>`;

  const top10 = ranking_overall.slice(0, 10);
  if (top10Chart) top10Chart.destroy();
  top10Chart = new Chart(document.getElementById('chart-top10'), {
    type: 'bar',
    data: {
      labels: top10.map(e => `#${e.rank} ${e.title.length > 32 ? e.title.slice(0,30)+'…' : e.title}`),
      datasets: [{
        data: top10.map(e => e.cas),
        backgroundColor: top10.map(e => catColor(e.category) + 'cc'),
        borderColor: top10.map(e => catColor(e.category)),
        borderWidth: 1, borderRadius: 4,
      }]
    },
    options: {
      indexAxis: 'y', responsive: true,
      plugins: { legend: {display:false}, tooltip: {callbacks:{label: ctx => ` CAS: ${ctx.raw.toFixed(4)}`}} },
      scales: {
        x: {grid:{color:'#21262d'}, ticks:{color:'#8b949e'}},
        y: {grid:{display:false}, ticks:{color:'#e6edf3', font:{size:11}}}
      }
    }
  });

  const roiLabels = ['Language\nNetwork','Attn Dorsal','Attn Ventral','Prefrontal','Emotional','Default Mode'];
  const roiKeys = ['language_network','attention_dorsal','attention_ventral','prefrontal','emotional','default_mode'];
  if (radarChart) radarChart.destroy();
  radarChart = new Chart(document.getElementById('chart-radar'), {
    type: 'radar',
    data: {
      labels: roiLabels,
      datasets: [
        { label: `Best: ${best.title.slice(0,25)}…`, data: roiKeys.map(k => best.roi_scores[k]),
          borderColor:'#39d0d8', backgroundColor:'#39d0d822', pointBackgroundColor:'#39d0d8', borderWidth:2 },
        { label: 'Original', data: roiKeys.map(k => original.roi_scores[k]),
          borderColor:'#a371f7', backgroundColor:'#a371f722', pointBackgroundColor:'#a371f7',
          borderWidth:2, borderDash:[5,4] }
      ]
    },
    options: {
      responsive: true,
      plugins: {legend:{labels:{color:'#8b949e', font:{size:11}}}},
      scales: {r:{min:0,max:1, grid:{color:'#30363d'}, ticks:{display:false},
                  pointLabels:{color:'#8b949e', font:{size:11}}, angleLines:{color:'#30363d'}}}
    }
  });

  document.getElementById('cat-grid').innerHTML = ['similar','alternative','exaggerated'].map(cat => {
    const s = category_stats[cat];
    const bc = ranking_by_category[cat][0];
    return `<div class="cat-card">
      <div class="cat-name ${cat}-color">${cat}</div>
      <div class="stat-row"><span style="color:var(--muted)">Mean CAS</span><span class="stat-val">${s.mean_cas.toFixed(3)}</span></div>
      <div class="stat-row"><span style="color:var(--muted)">Max CAS</span><span class="stat-val">${s.max_cas.toFixed(3)}</span></div>
      <div class="stat-row"><span style="color:var(--muted)">Min CAS</span><span class="stat-val">${s.min_cas.toFixed(3)}</span></div>
      <div class="stat-row"><span style="color:var(--muted)">Std Dev</span><span class="stat-val">${s.std.toFixed(3)}</span></div>
      <div class="stat-row" style="border:none;padding-top:8px;flex-direction:column;gap:4px;align-items:flex-start">
        <span style="color:var(--muted);font-size:.75rem">Best:</span>
        <span style="font-size:.78rem;color:var(--text)">${bc.title}</span>
      </div>
    </div>`;
  }).join('');

  const maxCas = ranking_overall[0].cas;
  const rows = ranking_overall.slice(0, 30).map(e => {
    const pct = ((e.cas / maxCas) * 100).toFixed(0);
    return `<tr>
      <td class="rank-num">${e.rank}</td>
      <td><span style="margin-right:6px">${e.title}</span>
          <button class="copy-btn" onclick="navigator.clipboard.writeText(${JSON.stringify(e.title)})">copy</button></td>
      <td>${catPill(e.category)}</td>
      <td><div class="cas-bar">
        <div class="cas-track"><div class="cas-fill" style="width:${pct}%"></div></div>
        <span class="cas-num">${e.cas.toFixed(3)}</span>
      </div></td>
    </tr>`;
  }).join('');

  const origPct = ((original.cas / maxCas) * 100).toFixed(0);
  document.getElementById('rankings-tbody').innerHTML = rows + `<tr style="background:#1a2a1522">
    <td class="rank-num" style="color:var(--green)">orig</td>
    <td><span style="color:var(--green)">${original.title}</span>
        <button class="copy-btn" onclick="navigator.clipboard.writeText(${JSON.stringify(original.title)})">copy</button></td>
    <td>${catPill('original')}</td>
    <td><div class="cas-bar">
      <div class="cas-track"><div class="cas-fill" style="width:${origPct}%;background:var(--green)"></div></div>
      <span class="cas-num">${original.cas.toFixed(3)}</span>
    </div></td>
  </tr>`;

  document.getElementById('results').style.display = 'block';
  document.getElementById('results').scrollIntoView({behavior:'smooth', block:'start'});
}
</script>
</body>
</html>
"""


# ──────────────────────────────────────────────
# Flask routes
# ──────────────────────────────────────────────

@app.get("/")
def root():
    return Response(HTML, mimetype="text/html")


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/analyze")
def analyze():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "title is required"}), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY not set — add it to .env or export it in your shell"}), 500

    job_id = uuid.uuid4().hex[:10]
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "step": "Starting…", "progress": 0}

    thread = threading.Thread(target=_run_pipeline, args=(job_id, title), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.get("/job/<job_id>")
def job_status(job_id):
    with _jobs_lock:
        job = dict(_jobs.get(job_id, {}))
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print("🧠 Hero Title Brain Analyzer — Local Server")
    print("=" * 60)
    print(f"   URL:  http://localhost:{port}")
    print(f"   Mode: Full TRIBE v2 pipeline")
    print()

    # Warn if OPENAI_API_KEY missing
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠  WARNING: OPENAI_API_KEY not set. Add it to .env or export it.")
        print()

    # Warn if TRIBE v2 not importable
    try:
        import tribev2  # noqa
        print("   TRIBE v2: ✓ installed")
    except ImportError:
        print("   TRIBE v2: ✗ not found")
        print("   Install:  git clone https://github.com/facebookresearch/tribev2")
        print("             pip install -e tribev2")
    print()

    app.run(host="0.0.0.0", port=port, debug=False)
