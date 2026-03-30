# Hero Title Brain Analyzer

Pipeline CLI che analizza titoli hero section usando TRIBE v2 (Meta FAIR) per predire le attivazioni cerebrali e ottimizzare il copy per la conversione.

## Come funziona

1. **Genera** 60 varianti del titolo (3 categorie × 20) via Claude
2. **Analizza** ogni variante con TRIBE v2 per ottenere attivazioni fMRI predette
3. **Aggrega** i vertici corticali in ROI scores (language, attention, emotion, etc.)
4. **Calcola** un Composite Attention Score (CAS) pesato e ranking
5. **Produce** report Markdown + 4 grafici PNG + interpretazione LLM

## Setup

```bash
# 1. Clona e installa TRIBE v2
git clone https://github.com/facebookresearch/tribev2
cd tribev2 && pip install -e .
cd ..

# 2. Installa dipendenze
pip install -r requirements.txt

# 3. Configura API key
cp .env.example .env
# Modifica .env con la tua ANTHROPIC_API_KEY
```

## Uso

```bash
# Analisi completa con TRIBE v2
python main.py --title "Trasforma il tuo business con l'AI"

# Modalità mock (senza TRIBE v2, attivazioni simulate)
python main.py --title "Trasforma il tuo business con l'AI" --mock

# Salta generazione varianti (usa output/variants.json esistente)
python main.py --title "..." --skip-generate

# Salta anche l'analisi TRIBE v2 (usa attivazioni cached)
python main.py --title "..." --skip-generate --skip-analyze
```

## Output

```
output/
├── variants.json              # 60 varianti generate
├── activations.npz            # attivazioni TRIBE v2 (numpy compressed)
├── activations_summary.json   # sommario attivazioni
├── scores.json                # ROI scores normalizzati
├── rankings.json              # ranking e CAS
├── report.md                  # report finale
├── chart_category_boxplot.png # boxplot CAS per categoria
├── chart_top10_bar.png        # top 10 titoli per CAS
├── chart_roi_radar_best.png   # radar ROI best per categoria
└── chart_roi_heatmap.png      # heatmap 61×6 titoli × ROI
```

## Note su TRIBE v2

- **Repo**: [facebookresearch/tribev2](https://github.com/facebookresearch/tribev2)
- **HuggingFace**: [facebook/tribev2](https://huggingface.co/facebook/tribev2)
- **Output**: predizioni su mesh corticale fsaverage5 (~20k vertici)
- **Input testo**: il testo viene convertito in speech con timing word-level

Il modello predice la risposta cerebrale media (soggetto "average") su fsaverage5.
Per titoli brevi, le predizioni vengono mediate sui timestep.

## Disclaimer

TRIBE v2 è rilasciato sotto licenza **CC BY-NC** (solo uso non commerciale).
Questo tool è a scopo di ricerca e sperimentazione.
Le attivazioni cerebrali sono stime statistiche, non misurazioni reali.
