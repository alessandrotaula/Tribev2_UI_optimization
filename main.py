#!/usr/bin/env python3
"""
main.py — Hero Title Brain Analyzer: entry point CLI.

Pipeline:
1. Genera 60 varianti del titolo hero via LLM (3 categorie × 20)
2. Analizza ogni variante con TRIBE v2 per attivazioni cerebrali predette
3. Aggrega i vertici corticali in ROI scores
4. Calcola Composite Attention Score (CAS) e ranking
5. Produce report Markdown + grafici PNG

Usage:
    python main.py --title "Il tuo titolo hero"
    python main.py --title "Your hero title" --mock
    python main.py --title "..." --skip-generate --skip-analyze
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Hero Title Brain Analyzer — Analizza titoli hero con TRIBE v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python main.py --title "Trasforma il tuo business con l\'AI"\n'
            '  python main.py --title "Your hero title" --mock\n'
            '  python main.py --title "..." --skip-generate  # usa variants.json esistente\n'
        ),
    )
    parser.add_argument(
        "--title", required=True,
        help="Input hero title da analizzare"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Usa attivazioni simulate (non richiede TRIBE v2 installato)"
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Salta la generazione, usa output/variants.json esistente"
    )
    parser.add_argument(
        "--skip-analyze", action="store_true",
        help="Salta l'analisi TRIBE v2, usa output/activations esistente"
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory per gli output (default: output)"
    )
    parser.add_argument(
        "--cache-dir", default="./cache",
        help="Directory cache per il modello TRIBE v2 (default: ./cache)"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("🧠 Hero Title Brain Analyzer")
    print("=" * 60)
    print(f'📝 Input: "{args.title}"')
    if args.mock:
        print("⚠  Modalità MOCK: attivazioni simulate")
    print()

    # ─── Step 1: Generazione varianti ────────────────────────────
    variants_path = os.path.join(output_dir, "variants.json")

    if args.skip_generate:
        if not os.path.exists(variants_path):
            print(f"❌ Errore: --skip-generate richiede {variants_path}")
            sys.exit(1)
        print(f"Step 1/5 — Skipping generation (using {variants_path})")
        with open(variants_path, "r", encoding="utf-8") as f:
            variants = json.load(f)
    else:
        print("Step 1/5 — Generating 60 variants via LLM...")
        from generate import generate_variants

        variants = generate_variants(args.title)
        with open(variants_path, "w", encoding="utf-8") as f:
            json.dump(variants, f, indent=2, ensure_ascii=False)

        n_variants = sum(len(v) for k, v in variants.items() if isinstance(v, list))
        print(f"  ✓ Generated {n_variants} variants → {variants_path}")
    print()

    # ─── Step 2: Analisi TRIBE v2 ───────────────────────────────
    activations_path = os.path.join(output_dir, "activations")

    if args.skip_analyze:
        npz_path = activations_path + ".npz"
        json_path = os.path.join(output_dir, "activations_summary.json")
        if not (os.path.exists(npz_path) or os.path.exists(json_path)):
            print(f"❌ Errore: --skip-analyze richiede file attivazioni in {output_dir}/")
            sys.exit(1)
        print(f"Step 2/5 — Skipping TRIBE v2 analysis (using cached activations)")
        # Ricarica le attivazioni dal JSON completo (se disponibile)
        # oppure dall'NPZ
        activations_json_path = os.path.join(output_dir, "activations_full.json")
        if os.path.exists(activations_json_path):
            with open(activations_json_path, "r") as f:
                activations = json.load(f)
        else:
            print("  ⚠ Full activations JSON not found, re-running analysis...")
            args.skip_analyze = False

    if not args.skip_analyze:
        print("Step 2/5 — Running TRIBE v2 analysis on 61 titles...")
        from analyze import analyze_all, save_activations

        activations = analyze_all(
            variants_path,
            mock=args.mock,
            cache_folder=args.cache_dir,
        )
        save_activations(activations, activations_path)

        # Salva anche il JSON completo per --skip-analyze
        full_json_path = os.path.join(output_dir, "activations_full.json")
        with open(full_json_path, "w", encoding="utf-8") as f:
            json.dump(activations, f, ensure_ascii=False)
        print(f"  ✓ Activations saved")
    print()

    # ─── Step 3: Aggregazione ROI ────────────────────────────────
    print("Step 3/5 — Aggregating cortical vertices to ROI scores...")
    from aggregate import aggregate_all

    scores = aggregate_all(activations)
    scores_path = os.path.join(output_dir, "scores.json")
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"  ✓ ROI scores → {scores_path}")
    print()

    # ─── Step 4: Ranking e CAS ──────────────────────────────────
    print("Step 4/5 — Computing Composite Attention Score (CAS) and rankings...")
    from compare import compute_rankings

    rankings = compute_rankings(scores)
    # Propaga flag mock nei rankings
    rankings["mock"] = activations.get("mock", args.mock)

    rankings_path = os.path.join(output_dir, "rankings.json")
    with open(rankings_path, "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Rankings → {rankings_path}")
    print(f"  📊 Original CAS: {rankings['original']['cas']:.4f} "
          f"(percentile {rankings['original_percentile']})")
    print(f"  🏆 Best: \"{rankings['ranking_overall'][0]['title'][:60]}\" "
          f"(CAS {rankings['ranking_overall'][0]['cas']:.4f})")
    print()

    # ─── Step 5: Report ─────────────────────────────────────────
    print("Step 5/5 — Generating report and charts...")
    from report import generate_report

    report_path = generate_report(args.title, scores, rankings, output_dir)
    print()

    # ─── Riepilogo ───────────────────────────────────────────────
    print("=" * 60)
    print("✅ Analisi completata!")
    print("=" * 60)
    print(f"  📊 {report_path}")
    print(f"  📈 4 charts (PNG) in {output_dir}/")
    print(f"  🗃️  variants.json, scores.json, rankings.json")
    print()
    print(f'  🏆 Winner: "{rankings["ranking_overall"][0]["title"][:70]}"')
    print(f"     Category: {rankings['ranking_overall'][0]['category']}")
    print(f"     CAS: {rankings['ranking_overall'][0]['cas']:.4f}")
    print()


if __name__ == "__main__":
    main()
