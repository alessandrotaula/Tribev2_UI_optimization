"""
report.py — Generazione report Markdown + grafici PNG (matplotlib).

Produce:
1. chart_category_boxplot.png — boxplot CAS per le 3 categorie + originale
2. chart_top10_bar.png — bar chart top 10 titoli per CAS
3. chart_roi_radar_best.png — radar chart ROI scores per best di ogni categoria vs originale
4. chart_roi_heatmap.png — heatmap 61×6 (titoli × ROI), ordinata per CAS
5. report.md — report finale con interpretazione LLM
"""

import json
import os
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Backend non-interattivo
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Colori per categoria
CATEGORY_COLORS = {
    "similar": "#4A90D9",
    "alternative": "#50C878",
    "exaggerated": "#FF6B6B",
    "original": "#FFD700",
}

CATEGORY_LABELS = {
    "similar": "Similar",
    "alternative": "Alternative",
    "exaggerated": "Exaggerated",
}

ROI_LABELS = {
    "language_network": "Language\nNetwork",
    "attention_dorsal": "Dorsal\nAttention",
    "attention_ventral": "Ventral\nAttention",
    "prefrontal": "Prefrontal",
    "emotional": "Emotional",
    "default_mode": "Default\nMode",
}


def _setup_style():
    """Configura lo stile globale matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "text.color": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "grid.color": "#2a2a4a",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 10,
    })


def generate_boxplot(rankings: dict, output_dir: str) -> str:
    """Genera boxplot CAS per le 3 categorie + originale come linea tratteggiata."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["similar", "alternative", "exaggerated"]
    data = []
    colors = []

    for cat in categories:
        cas_values = [e["cas"] for e in rankings["ranking_by_category"][cat]]
        data.append(cas_values)
        colors.append(CATEGORY_COLORS[cat])

    bp = ax.boxplot(
        data,
        labels=[CATEGORY_LABELS[c] for c in categories],
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"color": "#e0e0e0"},
        capprops={"color": "#e0e0e0"},
        flierprops={"markerfacecolor": "#e0e0e0", "markersize": 4},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Linea tratteggiata per l'originale
    original_cas = rankings["original"]["cas"]
    ax.axhline(
        y=original_cas,
        color=CATEGORY_COLORS["original"],
        linestyle="--",
        linewidth=2,
        label=f"Original (CAS={original_cas:.3f})",
    )

    ax.set_ylabel("Composite Attention Score (CAS)")
    ax.set_title("CAS Distribution by Category", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    path = os.path.join(output_dir, "chart_category_boxplot.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_top10_bar(rankings: dict, output_dir: str) -> str:
    """Genera bar chart top 10 titoli per CAS, colorati per categoria."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 8))

    top10 = rankings["ranking_overall"][:10]
    titles = [f"#{e['rank']} {e['title'][:50]}..." if len(e["title"]) > 50
              else f"#{e['rank']} {e['title']}" for e in top10]
    cas_values = [e["cas"] for e in top10]
    colors = [CATEGORY_COLORS[e["category"]] for e in top10]

    bars = ax.barh(range(len(top10) - 1, -1, -1), cas_values, color=colors, alpha=0.8)

    ax.set_yticks(range(len(top10) - 1, -1, -1))
    ax.set_yticklabels(titles, fontsize=8)
    ax.set_xlabel("Composite Attention Score (CAS)")
    ax.set_title("Top 10 Titles by CAS", fontsize=14, fontweight="bold")

    # Originale come linea verticale
    ax.axvline(
        x=rankings["original"]["cas"],
        color=CATEGORY_COLORS["original"],
        linestyle="--",
        linewidth=2,
        label=f"Original (CAS={rankings['original']['cas']:.3f})",
    )

    # Legenda per categorie
    handles = [
        mpatches.Patch(color=CATEGORY_COLORS[c], label=CATEGORY_LABELS[c], alpha=0.8)
        for c in ["similar", "alternative", "exaggerated"]
    ]
    handles.append(plt.Line2D([0], [0], color=CATEGORY_COLORS["original"],
                              linestyle="--", linewidth=2, label="Original"))
    ax.legend(handles=handles, loc="lower right")

    # Valori sulle barre
    for bar, cas in zip(bars, cas_values):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{cas:.3f}",
            va="center",
            fontsize=8,
            color="#e0e0e0",
        )

    ax.grid(True, axis="x", alpha=0.3)

    path = os.path.join(output_dir, "chart_top10_bar.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_radar_chart(rankings: dict, scores: dict, output_dir: str) -> str:
    """Genera radar chart ROI scores per il best di ogni categoria vs originale."""
    _setup_style()

    roi_names = list(ROI_LABELS.keys())
    roi_labels = list(ROI_LABELS.values())
    n_rois = len(roi_names)

    # Angoli per il radar
    angles = np.linspace(0, 2 * np.pi, n_rois, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#16213e")

    # Originale
    orig_values = [rankings["original"]["roi_scores"].get(r, 0) for r in roi_names]
    orig_values += orig_values[:1]
    ax.plot(angles, orig_values, "o-", color=CATEGORY_COLORS["original"],
            linewidth=2, label="Original", markersize=6)
    ax.fill(angles, orig_values, color=CATEGORY_COLORS["original"], alpha=0.1)

    # Best per categoria
    for cat in ["similar", "alternative", "exaggerated"]:
        best = rankings["ranking_by_category"][cat][0]
        values = [best["roi_scores"].get(r, 0) for r in roi_names]
        values += values[:1]
        ax.plot(angles, values, "o-", color=CATEGORY_COLORS[cat],
                linewidth=2, label=f"Best {CATEGORY_LABELS[cat]}", markersize=6)
        ax.fill(angles, values, color=CATEGORY_COLORS[cat], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(roi_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("ROI Scores — Best per Category vs Original",
                 fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    path = os.path.join(output_dir, "chart_roi_radar_best.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_heatmap(rankings: dict, output_dir: str) -> str:
    """Genera heatmap 61×6 (titoli × ROI), ordinata per CAS."""
    _setup_style()

    roi_names = list(ROI_LABELS.keys())

    # Raccogli tutti i titoli ordinati per CAS (include originale)
    all_entries = []

    # Inserisci l'originale nella posizione corretta
    orig_entry = {
        "title": rankings["original"]["title"],
        "category": "original",
        "cas": rankings["original"]["cas"],
        "roi_scores": rankings["original"]["roi_scores"],
    }
    all_entries.append(orig_entry)
    all_entries.extend(rankings["ranking_overall"])

    # Ordina per CAS
    all_entries.sort(key=lambda x: x["cas"], reverse=True)

    # Costruisci matrice
    n_titles = len(all_entries)
    matrix = np.zeros((n_titles, len(roi_names)))
    labels = []
    row_colors = []

    for i, entry in enumerate(all_entries):
        for j, roi in enumerate(roi_names):
            matrix[i, j] = entry["roi_scores"].get(roi, 0)
        short_title = entry["title"][:40] + "..." if len(entry["title"]) > 40 else entry["title"]
        labels.append(f"{short_title}")
        row_colors.append(CATEGORY_COLORS[entry["category"]])

    fig, ax = plt.subplots(figsize=(12, max(16, n_titles * 0.3)))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(roi_names)))
    ax.set_xticklabels([ROI_LABELS[r] for r in roi_names], fontsize=8, rotation=45, ha="right")

    ax.set_yticks(range(n_titles))
    ax.set_yticklabels(labels, fontsize=6)

    # Colora le label per categoria
    for i, color in enumerate(row_colors):
        ax.get_yticklabels()[i].set_color(color)

    ax.set_title("Brain Activation Heatmap (sorted by CAS)",
                 fontsize=14, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.5, label="Normalized ROI Score")

    # Legenda
    handles = [
        mpatches.Patch(color=CATEGORY_COLORS[c], label=l)
        for c, l in {**CATEGORY_LABELS, "original": "Original"}.items()
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.35, 1.0))

    path = os.path.join(output_dir, "chart_roi_heatmap.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _generate_llm_interpretation(
    input_title: str,
    rankings: dict,
    api_key: Optional[str] = None,
) -> str:
    """Genera interpretazione e raccomandazioni via LLM."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return (
            "*Interpretazione LLM non disponibile: OPENAI_API_KEY non configurata.*\n\n"
            "Per generare l'interpretazione, configura la chiave API nel file .env."
        )

    client = OpenAI(api_key=key)

    system_prompt = (
        "You are a neuroscience-informed UX copywriter. "
        "You will receive brain activation data (predicted fMRI ROI scores) "
        "for 61 hero section titles. Your job is to interpret the patterns "
        "and give actionable copywriting recommendations. "
        "Be specific, cite the data, avoid generic advice. "
        "Write in the same language as the input titles."
    )

    top5 = rankings["ranking_overall"][:5]
    bottom3 = rankings["ranking_overall"][-3:]

    user_prompt = (
        f'Original title: "{input_title}"\n'
        f'Original CAS: {rankings["original"]["cas"]}\n'
        f'Original percentile: {rankings["original_percentile"]}\n\n'
        f'Category stats:\n{json.dumps(rankings["category_stats"], indent=2)}\n\n'
        f'Top 5 overall titles with ROI scores:\n{json.dumps(top5, indent=2)}\n\n'
        f'Bottom 3 titles (worst performing):\n{json.dumps(bottom3, indent=2)}\n\n'
        "Tasks:\n"
        "1. INTERPRETATION (3-5 paragraphs): What do these brain activation patterns "
        "tell us about how people process these titles? Which ROI patterns "
        "correlate with high/low CAS? What makes the winning titles work neurologically?\n\n"
        "2. RECOMMENDATIONS (bullet list): Based on this data, what specific "
        "copywriting principles should be applied to optimize the hero title? "
        "Give at least 5 actionable recommendations with data backing."
    )

    try:
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
            max_tokens=3000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"*Errore nella generazione dell'interpretazione LLM: {e}*"


def generate_report(
    input_title: str,
    scores: dict,
    rankings: dict,
    output_dir: str = "output",
    api_key: Optional[str] = None,
) -> str:
    """
    Genera il report completo: grafici PNG + report Markdown.

    Returns:
        Percorso del report.md generato.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Genera grafici ---
    print("  → Generating charts...")
    chart_paths = []
    chart_paths.append(generate_boxplot(rankings, output_dir))
    chart_paths.append(generate_top10_bar(rankings, output_dir))
    chart_paths.append(generate_radar_chart(rankings, scores, output_dir))
    chart_paths.append(generate_heatmap(rankings, output_dir))
    print(f"    ✓ {len(chart_paths)} charts generated")

    # --- Interpretazione LLM ---
    print("  → Generating LLM interpretation...")
    interpretation = _generate_llm_interpretation(input_title, rankings, api_key)
    print("    ✓ LLM interpretation generated")

    # --- Componi report Markdown ---
    print("  → Composing report...")

    original = rankings["original"]
    cat_stats = rankings["category_stats"]
    top3 = rankings["ranking_overall"][:3]
    rank_medals = ["🥇", "🥈", "🥉"]

    # Trova la categoria migliore
    best_cat = max(cat_stats, key=lambda c: cat_stats[c]["mean_cas"])

    lines = []
    lines.append("# Hero Title Brain Analysis Report\n")
    lines.append("## Input\n")
    lines.append(f'- **Titolo originale**: "{input_title}"')
    lines.append(f"- **Data analisi**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **Totale varianti analizzate**: 61 (1 originale + 60 varianti)")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary\n")
    lines.append(
        f'La categoria **{best_cat}** ha ottenuto il CAS medio più alto '
        f'({cat_stats[best_cat]["mean_cas"]:.3f}). '
        f'Il titolo vincitore è della categoria **{top3[0]["category"]}** con CAS '
        f'{top3[0]["cas"]:.3f}. '
        f'Il titolo originale si posiziona al **percentile {rankings["original_percentile"]}** '
        f'(CAS {original["cas"]:.3f}), '
        f'{"superando" if rankings["original_percentile"] > 50 else "sotto"} la mediana '
        f"delle varianti generate."
    )
    lines.append("")

    # Titolo originale
    lines.append("## Titolo Originale\n")
    lines.append(f'> "{input_title}"\n')
    lines.append(f"- **CAS**: {original['cas']:.4f} (percentile {rankings['original_percentile']} su 61)")
    lines.append(f"- **ROI breakdown**:")
    for roi, score in original["roi_scores"].items():
        label = ROI_LABELS.get(roi, roi).replace("\n", " ")
        lines.append(f"  - {label}: {score:.3f}")
    lines.append("")

    # Top 3
    lines.append("## Top 3 Overall\n")
    for i, entry in enumerate(top3):
        lines.append(f"### {rank_medals[i]} Rank {i+1} — {CATEGORY_LABELS.get(entry['category'], entry['category'])}\n")
        lines.append(f'**"{entry["title"]}"**\n')
        lines.append(f"- CAS: {entry['cas']:.4f}")
        # Insight sintetico basato sui ROI
        top_roi = max(entry["roi_scores"], key=lambda r: entry["roi_scores"][r])
        top_roi_label = ROI_LABELS.get(top_roi, top_roi).replace("\n", " ")
        lines.append(
            f"- Insight: Picco in **{top_roi_label}** ({entry['roi_scores'][top_roi]:.3f}). "
        )
        if entry["roi_scores"].get("emotional", 0) > 0.7:
            lines.append("  Forte attivazione emotiva rilevata.")
        if entry["roi_scores"].get("prefrontal", 0) < 0.3:
            lines.append("  Basso carico cognitivo (positivo per conversione).")
        lines.append("")

    # Grafici
    lines.append("## Grafici\n")
    lines.append("![CAS Boxplot](chart_category_boxplot.png)\n")
    lines.append("![Top 10 Bar](chart_top10_bar.png)\n")
    lines.append("![ROI Radar Best](chart_roi_radar_best.png)\n")
    lines.append("![ROI Heatmap](chart_roi_heatmap.png)\n")

    # Analisi per categoria
    lines.append("## Analisi per Categoria\n")
    for cat in ["similar", "alternative", "exaggerated"]:
        stats = cat_stats[cat]
        lines.append(f"### {CATEGORY_LABELS[cat]} (20 varianti)\n")
        lines.append(
            f"- **Media CAS**: {stats['mean_cas']:.4f} | "
            f"**Best**: {stats['max_cas']:.4f} | "
            f"**Worst**: {stats['min_cas']:.4f} | "
            f"**Std**: {stats['std']:.4f}"
        )

        # Top 3 per categoria
        cat_top3 = rankings["ranking_by_category"][cat][:3]
        lines.append(f"\n**Top 3 {CATEGORY_LABELS[cat]}:**\n")
        for j, e in enumerate(cat_top3):
            lines.append(f'{j+1}. "{e["title"]}" — CAS {e["cas"]:.4f}')
        lines.append("")

    # Tabella ROI completa
    lines.append("## ROI Analysis\n")
    lines.append("| Rank | Category | CAS | " + " | ".join(
        ROI_LABELS[r].replace("\n", " ") for r in ROI_LABELS
    ) + " | Title |")
    lines.append("| --- | --- | --- | " + " | ".join("---" for _ in ROI_LABELS) + " | --- |")

    # Aggiungi originale
    orig_roi_str = " | ".join(
        f"{original['roi_scores'].get(r, 0):.3f}" for r in ROI_LABELS
    )
    lines.append(
        f"| ★ | original | {original['cas']:.4f} | {orig_roi_str} | "
        f"{original['title'][:60]} |"
    )

    # Aggiungi tutte le varianti
    for entry in rankings["ranking_overall"]:
        roi_str = " | ".join(
            f"{entry['roi_scores'].get(r, 0):.3f}" for r in ROI_LABELS
        )
        title_short = entry["title"][:60]
        lines.append(
            f"| {entry['rank']} | {entry['category']} | {entry['cas']:.4f} | "
            f"{roi_str} | {title_short} |"
        )
    lines.append("")

    # Interpretazione e raccomandazioni LLM
    lines.append("## Interpretazione LLM\n")
    lines.append(interpretation)
    lines.append("")

    # Disclaimer
    lines.append("---\n")
    lines.append("*Report generato da Hero Title Brain Analyzer. "
                 "Le attivazioni cerebrali sono predette da TRIBE v2 (Meta FAIR) "
                 "e rappresentano stime statistiche, non misurazioni reali. "
                 "TRIBE v2 è rilasciato sotto licenza CC BY-NC (solo uso non commerciale).*")

    # Scrivi report
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"    ✓ Report saved to {report_path}")
    return report_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--scores", default="output/scores.json")
    parser.add_argument("--rankings", default="output/rankings.json")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    with open(args.scores, "r") as f:
        scores = json.load(f)
    with open(args.rankings, "r") as f:
        rankings = json.load(f)

    generate_report(args.title, scores, rankings, args.output_dir)
