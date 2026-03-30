"""
compare.py — Comparazione e ranking dei titoli per Composite Attention Score (CAS).

Il CAS è uno score composito pesato basato sui ROI scores.
I pesi riflettono l'ipotesi neuroscientifica che:
- Comprensione fluida + risposta emotiva + cattura attentiva = alta conversione
- Alto carico cognitivo (prefrontale) = bassa conversione
"""

import json
import statistics
from typing import Optional


# Pesi CAS (configurabili)
DEFAULT_WEIGHTS = {
    "language_network": 0.25,    # comprensione fluida
    "attention_dorsal": 0.20,    # attenzione diretta
    "attention_ventral": 0.20,   # cattura attenzione involontaria
    "prefrontal": -0.10,         # penalizza alto carico cognitivo
    "emotional": 0.25,           # risposta emotiva
    "default_mode": 0.10,        # narrativa e self-reference
}


def composite_attention_score(
    roi_scores: dict[str, float],
    weights: Optional[dict[str, float]] = None,
) -> float:
    """
    Calcola il Composite Attention Score (CAS) da ROI scores normalizzati.

    Il segno negativo su 'prefrontal' riflette l'ipotesi che un alto carico
    cognitivo riduca la conversione (il messaggio è troppo complesso).

    Args:
        roi_scores: dict ROI → score normalizzato [0, 1]
        weights: dict ROI → peso (default: DEFAULT_WEIGHTS)

    Returns:
        CAS score (float)
    """
    w = weights or DEFAULT_WEIGHTS
    return sum(roi_scores.get(roi, 0.0) * weight for roi, weight in w.items())


def compute_rankings(
    scores: dict,
    weights: Optional[dict[str, float]] = None,
) -> dict:
    """
    Calcola ranking e statistiche per tutti i titoli.

    Args:
        scores: output di aggregate_all() — ROI scores per ogni titolo

    Returns:
        dict con ranking_overall, ranking_by_category, category_stats,
        original_cas, original_percentile
    """
    w = weights or DEFAULT_WEIGHTS

    # Calcola CAS per l'originale
    original_cas = composite_attention_score(scores["original"]["roi_scores"], w)

    # Calcola CAS per tutte le varianti
    all_entries = []
    category_entries = {"similar": [], "alternative": [], "exaggerated": []}

    for cat in ["similar", "alternative", "exaggerated"]:
        for item in scores[cat]:
            cas = composite_attention_score(item["roi_scores"], w)
            entry = {
                "title": item["title"],
                "category": cat,
                "cas": round(cas, 4),
                "roi_scores": {k: round(v, 4) for k, v in item["roi_scores"].items()},
            }
            all_entries.append(entry)
            category_entries[cat].append(entry)

    # Ranking overall (ordinato per CAS decrescente)
    all_entries.sort(key=lambda x: x["cas"], reverse=True)
    for i, entry in enumerate(all_entries):
        entry["rank"] = i + 1

    # Ranking per categoria
    ranking_by_category = {}
    for cat in ["similar", "alternative", "exaggerated"]:
        cat_sorted = sorted(category_entries[cat], key=lambda x: x["cas"], reverse=True)
        for i, entry in enumerate(cat_sorted):
            entry["category_rank"] = i + 1
        ranking_by_category[cat] = cat_sorted

    # Statistiche per categoria
    category_stats = {}
    for cat in ["similar", "alternative", "exaggerated"]:
        cas_values = [e["cas"] for e in category_entries[cat]]
        category_stats[cat] = {
            "mean_cas": round(statistics.mean(cas_values), 4),
            "max_cas": round(max(cas_values), 4),
            "min_cas": round(min(cas_values), 4),
            "std": round(statistics.stdev(cas_values), 4) if len(cas_values) > 1 else 0.0,
            "median_cas": round(statistics.median(cas_values), 4),
        }

    # Percentile dell'originale rispetto alle 60 varianti
    all_cas = [e["cas"] for e in all_entries]
    n_below = sum(1 for c in all_cas if c < original_cas)
    original_percentile = round((n_below / len(all_cas)) * 100)

    return {
        "ranking_overall": all_entries,
        "ranking_by_category": ranking_by_category,
        "category_stats": category_stats,
        "original": {
            "title": scores["original"]["title"],
            "cas": round(original_cas, 4),
            "roi_scores": {
                k: round(v, 4) for k, v in scores["original"]["roi_scores"].items()
            },
        },
        "original_percentile": original_percentile,
        "weights_used": w,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="output/scores.json")
    parser.add_argument("--output", default="output/rankings.json")
    args = parser.parse_args()

    with open(args.scores, "r") as f:
        scores = json.load(f)

    rankings = compute_rankings(scores)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=2, ensure_ascii=False)
    print(f"Saved rankings to {args.output}")
    print(f"\nOriginal CAS: {rankings['original']['cas']} "
          f"(percentile {rankings['original_percentile']})")
    print(f"\nTop 3:")
    for entry in rankings["ranking_overall"][:3]:
        print(f"  #{entry['rank']} [{entry['category']}] CAS={entry['cas']:.4f}: {entry['title']}")
