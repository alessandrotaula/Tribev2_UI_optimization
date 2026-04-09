"""
Hero Title Brain Analyzer — Vercel Flask entrypoint.

GET  /         → Full HTML SPA (UI)
POST /analyze  → Generate 60 variants via OpenAI + heuristic ROI scoring + CAS ranking
GET  /health   → {"ok": true}
"""

import json
import math
import os
import re
import statistics
import time
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request

load_dotenv()

app = Flask(__name__)

# ──────────────────────────────────────────────
# CAS weights (same as compare.py)
# ──────────────────────────────────────────────
CAS_WEIGHTS = {
    "language_network": 0.25,
    "attention_dorsal": 0.20,
    "attention_ventral": 0.20,
    "prefrontal": -0.10,
    "emotional": 0.25,
    "default_mode": 0.10,
}

# ──────────────────────────────────────────────
# Heuristic scorer (replaces TRIBE v2 on Vercel)
# ──────────────────────────────────────────────
_ACTION_VERBS = {
    "trasforma", "scopri", "ottieni", "raggiungi", "migliora", "accelera",
    "domina", "crea", "costruisci", "lancia", "aumenta", "raddoppia",
    "transform", "discover", "get", "achieve", "improve", "accelerate",
    "dominate", "create", "build", "launch", "boost", "double", "unlock",
    "master", "start", "grow", "scale", "win", "crush", "maximize",
}
_EMOTIONAL_WORDS = {
    "incredibile", "rivoluzionario", "straordinario", "potente", "garantito",
    "segreto", "esclusivo", "gratis", "libero", "facile", "veloce",
    "incredible", "revolutionary", "extraordinary", "powerful", "guaranteed",
    "secret", "exclusive", "free", "easy", "fast", "proven", "amazing",
    "unstoppable", "explosive", "massive", "ultimate", "epic",
}
_SURPRISE_WORDS = {
    "perché", "come", "cosa", "mai", "davvero", "sorprendente",
    "why", "how", "what", "never", "really", "surprising", "finally",
    "without", "senza", "invece", "actually", "hidden", "nascoste",
}
_NARRATIVE_WORDS = {
    "tuo", "tua", "tuoi", "tue", "tu", "te", "vita", "storia",
    "your", "you", "life", "story", "journey", "future", "dream",
    "futuro", "sogno", "percorso", "mondo",
}
_ABSTRACT_WORDS = {
    "strategia", "ottimizzazione", "implementazione", "paradigma", "framework",
    "strategy", "optimization", "implementation", "paradigm", "framework",
    "synergy", "leverage", "scalability", "methodology", "infrastructure",
}


def _raw_roi_scores(title: str) -> dict:
    """Compute raw (unnormalized) heuristic ROI scores for a title."""
    t = title.lower()
    words = re.findall(r"\w+", t)
    n = max(len(words), 1)

    action_count = sum(1 for w in words if w in _ACTION_VERBS)
    emotion_count = sum(1 for w in words if w in _EMOTIONAL_WORDS)
    surprise_count = sum(1 for w in words if w in _SURPRISE_WORDS)
    narrative_count = sum(1 for w in words if w in _NARRATIVE_WORDS)
    abstract_count = sum(1 for w in words if w in _ABSTRACT_WORDS)

    unique_ratio = len(set(words)) / n
    has_question = 1.0 if "?" in title else 0.0
    has_exclamation = 1.0 if "!" in title else 0.0

    # Avg word length as proxy for complexity
    avg_word_len = sum(len(w) for w in words) / n

    language_network = unique_ratio * 0.6 + (1.0 / (1 + avg_word_len / 8)) * 0.4
    attention_dorsal = min(1.0, (action_count / n) * 3 + 0.1)
    attention_ventral = min(1.0, has_question * 0.4 + (surprise_count / n) * 3 + 0.1)
    prefrontal = min(1.0, (n / 15) * 0.5 + (abstract_count / n) * 3 + avg_word_len / 12)
    emotional = min(1.0, (emotion_count / n) * 4 + has_exclamation * 0.3 + 0.05)
    default_mode = min(1.0, (narrative_count / n) * 4 + 0.1)

    return {
        "language_network": language_network,
        "attention_dorsal": attention_dorsal,
        "attention_ventral": attention_ventral,
        "prefrontal": prefrontal,
        "emotional": emotional,
        "default_mode": default_mode,
    }


def _normalize_scores(all_entries: list) -> None:
    """Normalize each ROI to [0,1] across all titles (in-place)."""
    rois = list(CAS_WEIGHTS.keys())
    for roi in rois:
        vals = [e["roi_scores"][roi] for e in all_entries]
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx != mn else 1.0
        for e in all_entries:
            e["roi_scores"][roi] = (e["roi_scores"][roi] - mn) / rng


def _cas(roi_scores: dict) -> float:
    return sum(roi_scores.get(roi, 0.0) * w for roi, w in CAS_WEIGHTS.items())


# ──────────────────────────────────────────────
# OpenAI variant generation (inline, no file I/O)
# ──────────────────────────────────────────────
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
MAX_RETRIES = 2

_SINGLE_PROMPT = """\
You are a conversion copywriter. Given a hero section title, generate variants in 3 categories.

Original title: "{title}"

Return ONLY a JSON object with exactly this structure:
{{
  "similar": ["title1", "title2", ..., "title10"],
  "alternative": ["title1", "title2", ..., "title10"],
  "exaggerated": ["title1", "title2", ..., "title10"]
}}

Rules:
- similar: same tone/message, small A/B-test variations
- alternative: different angles (benefit-led, problem-led, question, curiosity gap)
- exaggerated: hyperbolic, bold, direct-response style
- exactly 10 strings per category
- no markdown, no extra keys, no explanations
"""


def _extract_list(parsed: dict, key: str) -> list:
    """Robustly extract a list from parsed JSON, trying multiple key variants."""
    # Direct key
    v = parsed.get(key)
    if isinstance(v, list) and v:
        return v
    # Search all values for the first non-empty list if key missing
    for val in parsed.values():
        if isinstance(val, list) and val and key in str(parsed)[:200]:
            return val
    # Last resort: any list
    for val in parsed.values():
        if isinstance(val, list) and len(val) >= 5:
            return val
    return []


def _generate_variants(title: str, api_key: str) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    prompt = _SINGLE_PROMPT.format(title=title)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
                timeout=50,
            )
            raw = resp.choices[0].message.content
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected dict, got {type(parsed).__name__}")

            result = {"input_title": title}
            for cat in ["similar", "alternative", "exaggerated"]:
                lst = _extract_list(parsed, cat)
                if not lst:
                    raise ValueError(f"Missing '{cat}' list in response")
                # Pad/trim to exactly 10
                while len(lst) < 10:
                    lst.append(lst[-1] + " (alt)")
                result[cat] = [str(t) for t in lst[:10]]
            return result

        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"OpenAI generation failed: {e}") from e
            time.sleep(1)


# ──────────────────────────────────────────────
# Full analysis pipeline
# ──────────────────────────────────────────────
def _run_analysis(title: str, api_key: str) -> dict:
    # Step 1: Generate variants
    variants = _generate_variants(title, api_key)

    # Step 2: Build all_entries list for scoring
    all_entries = [{"title": title, "category": "original", "roi_scores": _raw_roi_scores(title)}]
    for cat in ["similar", "alternative", "exaggerated"]:
        for t in variants[cat]:
            all_entries.append({"title": t, "category": cat, "roi_scores": _raw_roi_scores(t)})

    # Step 3: Normalize ROI scores across all 61 titles
    _normalize_scores(all_entries)

    # Step 4: Compute CAS
    for e in all_entries:
        e["cas"] = round(_cas(e["roi_scores"]), 4)
        e["roi_scores"] = {k: round(v, 4) for k, v in e["roi_scores"].items()}

    original_entry = all_entries[0]
    variant_entries = all_entries[1:]

    # Step 5: Rankings
    sorted_variants = sorted(variant_entries, key=lambda x: x["cas"], reverse=True)
    for i, e in enumerate(sorted_variants):
        e["rank"] = i + 1

    ranking_by_category = {}
    for cat in ["similar", "alternative", "exaggerated"]:
        cat_entries = sorted([e for e in variant_entries if e["category"] == cat],
                             key=lambda x: x["cas"], reverse=True)
        for i, e in enumerate(cat_entries):
            e["category_rank"] = i + 1
        ranking_by_category[cat] = cat_entries

    category_stats = {}
    for cat in ["similar", "alternative", "exaggerated"]:
        vals = [e["cas"] for e in ranking_by_category[cat]]
        category_stats[cat] = {
            "mean_cas": round(statistics.mean(vals), 4),
            "max_cas": round(max(vals), 4),
            "min_cas": round(min(vals), 4),
            "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "median_cas": round(statistics.median(vals), 4) if vals else 0.0,
        }

    all_cas = [e["cas"] for e in variant_entries]
    n_below = sum(1 for c in all_cas if c < original_entry["cas"])
    original_percentile = round((n_below / len(all_cas)) * 100)

    return {
        "ranking_overall": sorted_variants,
        "ranking_by_category": ranking_by_category,
        "category_stats": category_stats,
        "original": {
            "title": original_entry["title"],
            "cas": original_entry["cas"],
            "roi_scores": original_entry["roi_scores"],
        },
        "original_percentile": original_percentile,
        "weights_used": CAS_WEIGHTS,
    }


# ──────────────────────────────────────────────
# HTML UI
# ──────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Hero Title Brain Analyzer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#0d1117;--surface:#161b22;--surface2:#21262d;--border:#30363d;
    --purple:#a371f7;--cyan:#39d0d8;--green:#3fb950;--orange:#f0883e;--red:#f85149;
    --text:#e6edf3;--muted:#8b949e;--font:'Segoe UI',system-ui,sans-serif;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:var(--font);min-height:100vh}

  /* ── Layout ── */
  .wrap{max-width:1100px;margin:0 auto;padding:0 20px}

  /* ── Header ── */
  header{border-bottom:1px solid var(--border);padding:18px 0}
  header .wrap{display:flex;align-items:center;gap:14px}
  .logo{width:40px;height:40px;background:linear-gradient(135deg,var(--purple),var(--cyan));
        border-radius:10px;display:flex;align-items:center;justify-content:center;
        font-size:20px;flex-shrink:0}
  .brand h1{font-size:1.15rem;font-weight:700;letter-spacing:-.01em}
  .brand p{font-size:.78rem;color:var(--muted);margin-top:2px}
  .badge{margin-left:auto;background:var(--surface2);border:1px solid var(--border);
         border-radius:20px;padding:4px 12px;font-size:.72rem;color:var(--muted)}

  /* ── Main ── */
  main{padding:40px 0 80px}
  .section-title{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;
                 color:var(--muted);margin-bottom:12px}

  /* ── Input card ── */
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
  .btn{padding:12px 24px;border-radius:8px;border:none;cursor:pointer;font-size:.9rem;
       font-weight:600;transition:opacity .2s}
  .btn-primary{background:linear-gradient(135deg,var(--purple),#7c3aed);color:#fff}
  .btn-primary:hover{opacity:.88}
  .btn-primary:disabled{opacity:.4;cursor:not-allowed}

  /* ── Disclaimer ── */
  .disclaimer{background:#1c1c1e;border:1px solid #3d3220;border-radius:8px;
              padding:12px 16px;font-size:.8rem;color:#9a7e4a;display:flex;gap:8px;
              align-items:flex-start;margin-bottom:28px}
  .disclaimer svg{flex-shrink:0;margin-top:1px}

  /* ── Loading ── */
  #loading{display:none;background:var(--surface);border:1px solid var(--border);
           border-radius:12px;padding:36px;text-align:center;margin-bottom:28px}
  .spinner{width:40px;height:40px;border:3px solid var(--border);
           border-top-color:var(--purple);border-radius:50%;
           animation:spin .8s linear infinite;margin:0 auto 20px}
  @keyframes spin{to{transform:rotate(360deg)}}
  #loading-step{color:var(--muted);font-size:.88rem;margin-top:6px}
  .progress-bar{background:var(--surface2);border-radius:4px;height:4px;margin-top:16px;overflow:hidden}
  .progress-fill{height:100%;background:linear-gradient(90deg,var(--purple),var(--cyan));
                 border-radius:4px;transition:width 1.2s ease;width:0}

  /* ── Error ── */
  #error-box{display:none;background:#1c1014;border:1px solid #5a1d1d;border-radius:8px;
             padding:16px 20px;color:var(--red);font-size:.88rem;margin-bottom:20px}

  /* ── Results ── */
  #results{display:none}

  /* ── Score cards row ── */
  .score-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;
              margin-bottom:28px}
  .score-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
              padding:18px 20px}
  .score-card .label{font-size:.72rem;color:var(--muted);text-transform:uppercase;
                     letter-spacing:.05em;margin-bottom:6px}
  .score-card .value{font-size:1.9rem;font-weight:700;line-height:1}
  .score-card .sub{font-size:.78rem;color:var(--muted);margin-top:4px}
  .val-purple{color:var(--purple)}
  .val-cyan{color:var(--cyan)}
  .val-green{color:var(--green)}

  /* ── 2-col chart row ── */
  .chart-row{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px}
  @media(max-width:680px){.chart-row{grid-template-columns:1fr}}
  .chart-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
              padding:20px}
  .chart-card h3{font-size:.85rem;font-weight:600;margin-bottom:16px;color:var(--muted);
                 text-transform:uppercase;letter-spacing:.04em}
  .chart-card canvas{max-height:300px}

  /* ── Category stats ── */
  .cat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:28px}
  @media(max-width:600px){.cat-grid{grid-template-columns:1fr}}
  .cat-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
            padding:16px 18px}
  .cat-card .cat-name{font-size:.8rem;font-weight:700;text-transform:uppercase;
                      letter-spacing:.05em;margin-bottom:10px}
  .cat-card .stat-row{display:flex;justify-content:space-between;font-size:.82rem;
                      padding:3px 0;border-bottom:1px solid var(--border)}
  .cat-card .stat-row:last-child{border-bottom:none}
  .stat-val{color:var(--text);font-weight:600}
  .similar-color{color:#a371f7}
  .alternative-color{color:#39d0d8}
  .exaggerated-color{color:#f0883e}

  /* ── Table ── */
  .table-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
              overflow:hidden;margin-bottom:28px}
  .table-card h3{font-size:.85rem;font-weight:600;padding:16px 20px;border-bottom:1px solid var(--border);
                 text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
  .tbl-wrap{overflow-x:auto}
  table{width:100%;border-collapse:collapse;font-size:.84rem}
  th{padding:10px 14px;text-align:left;color:var(--muted);font-weight:600;
     font-size:.75rem;text-transform:uppercase;letter-spacing:.04em;
     border-bottom:1px solid var(--border);background:var(--surface2)}
  td{padding:10px 14px;border-bottom:1px solid var(--border);vertical-align:middle}
  tr:last-child td{border-bottom:none}
  tr:hover td{background:var(--surface2)}
  .rank-num{color:var(--muted);font-weight:700;width:40px}
  .cat-pill{padding:2px 10px;border-radius:20px;font-size:.72rem;font-weight:600;
            display:inline-block}
  .pill-similar{background:#2d1f5a;color:#a371f7}
  .pill-alternative{background:#0d2d30;color:#39d0d8}
  .pill-exaggerated{background:#3d2005;color:#f0883e}
  .pill-original{background:#1a2a15;color:#3fb950}
  .cas-bar{display:flex;align-items:center;gap:8px}
  .cas-track{flex:1;background:var(--surface2);border-radius:3px;height:6px;min-width:60px}
  .cas-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--purple),var(--cyan))}
  .cas-num{font-weight:600;font-size:.82rem;width:44px;text-align:right;color:var(--text)}

  /* ── Copy button ── */
  .copy-btn{background:none;border:1px solid var(--border);color:var(--muted);
            border-radius:4px;padding:2px 8px;font-size:.72rem;cursor:pointer;
            transition:all .15s}
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
    <div class="badge">v2 · Vercel</div>
  </div>
</header>

<main>
  <div class="wrap">

    <div class="disclaimer">
      <svg width="16" height="16" fill="none" stroke="#9a7e4a" stroke-width="2" viewBox="0 0 24 24">
        <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/>
        <line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
      <span><b>Simplified mode on Vercel:</b> ROI scores use text-feature heuristics (action verbs, emotional words, complexity). For full TRIBE v2 brain activation prediction, run <code>python main.py --title "..."</code> locally.</span>
    </div>

    <div class="input-card">
      <h2>Analyze a Hero Title</h2>
      <p>Enter your hero section title. The analyzer generates 60 variants (similar, alternative, exaggerated) via GPT and ranks them by Composite Attention Score (CAS) — a weighted formula reflecting linguistic fluency, attention capture, emotional response, and cognitive load.</p>
      <div class="input-row">
        <input id="title-input" type="text"
               placeholder='e.g. "Trasforma il tuo business con l&apos;AI"'
               value=""/>
        <button class="btn btn-primary" id="analyze-btn" onclick="startAnalysis()">Analyze →</button>
      </div>
    </div>

    <div id="error-box"></div>

    <div id="loading">
      <div class="spinner"></div>
      <div style="font-size:.95rem;font-weight:600">Analyzing your title…</div>
      <div id="loading-step">Connecting to OpenAI…</div>
      <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
    </div>

    <div id="results">

      <div class="section-title">Results</div>

      <!-- Score cards -->
      <div class="score-grid" id="score-grid"></div>

      <!-- Charts row -->
      <div class="chart-row">
        <div class="chart-card">
          <h3>Top 10 Titles by CAS</h3>
          <canvas id="chart-top10"></canvas>
        </div>
        <div class="chart-card">
          <h3>ROI Radar — Best vs Original</h3>
          <canvas id="chart-radar"></canvas>
        </div>
      </div>

      <!-- Category stats -->
      <div class="section-title">Category Breakdown</div>
      <div class="cat-grid" id="cat-grid"></div>

      <!-- Full rankings table -->
      <div class="table-card">
        <h3>Full Rankings — Top 30</h3>
        <div class="tbl-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Title</th>
                <th>Category</th>
                <th style="min-width:160px">CAS Score</th>
              </tr>
            </thead>
            <tbody id="rankings-tbody"></tbody>
          </table>
        </div>
      </div>

    </div><!-- /results -->

  </div>
</main>

<script>
let top10Chart = null;
let radarChart = null;

// Enter key support
document.getElementById('title-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') startAnalysis();
});

function startAnalysis() {
  const title = document.getElementById('title-input').value.trim();
  if (!title) {
    showError('Please enter a hero title.');
    return;
  }

  hideError();
  hideResults();
  showLoading(true);

  const steps = [
    [0,   'Sending to OpenAI — generating 60 variants…'],
    [30,  'Generating similar variants…'],
    [55,  'Generating alternative variants…'],
    [75,  'Generating exaggerated variants…'],
    [88,  'Scoring with heuristic ROI model…'],
    [95,  'Computing CAS rankings…'],
  ];
  let si = 0;
  const stepInterval = setInterval(() => {
    if (si < steps.length) {
      const [pct, msg] = steps[si++];
      document.getElementById('loading-step').textContent = msg;
      document.getElementById('progress-fill').style.width = pct + '%';
    }
  }, 1800);

  fetch('/analyze', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({title})
  })
  .then(r => r.json().then(d => ({ok: r.ok, data: d})))
  .then(({ok, data}) => {
    clearInterval(stepInterval);
    showLoading(false);
    if (!ok) {
      showError(data.error || 'Analysis failed. Check your OPENAI_API_KEY is set in Vercel env vars.');
    } else {
      document.getElementById('progress-fill').style.width = '100%';
      renderResults(data);
    }
  })
  .catch(err => {
    clearInterval(stepInterval);
    showLoading(false);
    showError('Network error: ' + err.message);
  });
}

function showLoading(v) {
  document.getElementById('loading').style.display = v ? 'block' : 'none';
  document.getElementById('analyze-btn').disabled = v;
  if (v) document.getElementById('progress-fill').style.width = '0';
}
function showError(msg) {
  const el = document.getElementById('error-box');
  el.textContent = '⚠ ' + msg;
  el.style.display = 'block';
}
function hideError() { document.getElementById('error-box').style.display = 'none'; }
function hideResults() { document.getElementById('results').style.display = 'none'; }

function catColor(cat) {
  return {similar:'#a371f7', alternative:'#39d0d8', exaggerated:'#f0883e', original:'#3fb950'}[cat] || '#8b949e';
}
function catPill(cat) {
  return `<span class="cat-pill pill-${cat}">${cat}</span>`;
}

function renderResults(data) {
  const {ranking_overall, ranking_by_category, category_stats, original, original_percentile} = data;

  // ── Score cards ──
  const pctColor = original_percentile >= 75 ? 'val-green' : original_percentile >= 50 ? 'val-cyan' : 'val-purple';
  const best = ranking_overall[0];
  document.getElementById('score-grid').innerHTML = `
    <div class="score-card">
      <div class="label">Your Title CAS</div>
      <div class="value val-purple">${original.cas.toFixed(3)}</div>
      <div class="sub">${original.title}</div>
    </div>
    <div class="score-card">
      <div class="label">Percentile vs 60 Variants</div>
      <div class="value ${pctColor}">${original_percentile}%</div>
      <div class="sub">beats ${original_percentile}% of generated variants</div>
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
    </div>
  `;

  // ── Top 10 bar chart ──
  const top10 = ranking_overall.slice(0, 10);
  if (top10Chart) top10Chart.destroy();
  const labels10 = top10.map((e, i) => `#${e.rank} ${e.title.length > 32 ? e.title.slice(0,30)+'…' : e.title}`);
  top10Chart = new Chart(document.getElementById('chart-top10'), {
    type: 'bar',
    data: {
      labels: labels10,
      datasets: [{
        data: top10.map(e => e.cas),
        backgroundColor: top10.map(e => catColor(e.category) + 'cc'),
        borderColor: top10.map(e => catColor(e.category)),
        borderWidth: 1,
        borderRadius: 4,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: { legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` CAS: ${ctx.raw.toFixed(4)}` } }
      },
      scales: {
        x: { grid: { color: '#21262d' }, ticks: { color: '#8b949e' } },
        y: { grid: { display: false }, ticks: { color: '#e6edf3', font: { size: 11 } } }
      }
    }
  });

  // ── ROI radar chart ──
  const roiLabels = ['Language\nNetwork','Attn Dorsal','Attn Ventral','Prefrontal','Emotional','Default Mode'];
  const roiKeys = ['language_network','attention_dorsal','attention_ventral','prefrontal','emotional','default_mode'];
  if (radarChart) radarChart.destroy();
  radarChart = new Chart(document.getElementById('chart-radar'), {
    type: 'radar',
    data: {
      labels: roiLabels,
      datasets: [
        {
          label: `Best: ${best.title.slice(0,25)}…`,
          data: roiKeys.map(k => best.roi_scores[k]),
          borderColor: '#39d0d8',
          backgroundColor: '#39d0d822',
          pointBackgroundColor: '#39d0d8',
          borderWidth: 2,
        },
        {
          label: 'Original',
          data: roiKeys.map(k => original.roi_scores[k]),
          borderColor: '#a371f7',
          backgroundColor: '#a371f722',
          pointBackgroundColor: '#a371f7',
          borderWidth: 2,
          borderDash: [5, 4],
        }
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#8b949e', font: { size: 11 } } } },
      scales: {
        r: {
          min: 0, max: 1,
          grid: { color: '#30363d' },
          ticks: { display: false },
          pointLabels: { color: '#8b949e', font: { size: 11 } },
          angleLines: { color: '#30363d' },
        }
      }
    }
  });

  // ── Category stats ──
  const catNames = ['similar','alternative','exaggerated'];
  document.getElementById('cat-grid').innerHTML = catNames.map(cat => {
    const s = category_stats[cat];
    const best_cat = ranking_by_category[cat][0];
    return `
      <div class="cat-card">
        <div class="cat-name ${cat}-color">${cat}</div>
        <div class="stat-row"><span style="color:var(--muted)">Mean CAS</span><span class="stat-val">${s.mean_cas.toFixed(3)}</span></div>
        <div class="stat-row"><span style="color:var(--muted)">Max CAS</span><span class="stat-val">${s.max_cas.toFixed(3)}</span></div>
        <div class="stat-row"><span style="color:var(--muted)">Min CAS</span><span class="stat-val">${s.min_cas.toFixed(3)}</span></div>
        <div class="stat-row"><span style="color:var(--muted)">Std Dev</span><span class="stat-val">${s.std.toFixed(3)}</span></div>
        <div class="stat-row" style="border:none;padding-top:8px;flex-direction:column;gap:4px;align-items:flex-start">
          <span style="color:var(--muted);font-size:.75rem">Best title:</span>
          <span style="font-size:.78rem;color:var(--text)">${best_cat.title}</span>
        </div>
      </div>`;
  }).join('');

  // ── Rankings table ──
  const maxCas = ranking_overall[0].cas;
  const tbody = document.getElementById('rankings-tbody');
  const rows = ranking_overall.slice(0, 30).map(e => {
    const pct = ((e.cas / maxCas) * 100).toFixed(0);
    return `<tr>
      <td class="rank-num">${e.rank}</td>
      <td>
        <span style="margin-right:6px">${e.title}</span>
        <button class="copy-btn" onclick="navigator.clipboard.writeText(${JSON.stringify(e.title)})">copy</button>
      </td>
      <td>${catPill(e.category)}</td>
      <td>
        <div class="cas-bar">
          <div class="cas-track"><div class="cas-fill" style="width:${pct}%"></div></div>
          <span class="cas-num">${e.cas.toFixed(3)}</span>
        </div>
      </td>
    </tr>`;
  }).join('');

  // Add original at end
  const origPct = ((original.cas / maxCas) * 100).toFixed(0);
  tbody.innerHTML = rows + `<tr style="background:#1a2a1522">
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
        return jsonify({"error": "OPENAI_API_KEY is not set on this deployment. Add it in Vercel → Settings → Environment Variables."}), 500

    try:
        result = _run_analysis(title, api_key)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
