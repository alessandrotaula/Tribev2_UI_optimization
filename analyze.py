"""
analyze.py — Wrapper TRIBE v2 per predizione attivazioni cerebrali da testo.

TRIBE v2 (facebookresearch/tribev2) predice risposte fMRI su mesh corticale
fsaverage5 (~20k vertici) da stimoli multimodali (video, audio, testo).

Per input testuale: il testo viene scritto in un file temporaneo, passato come
text_path a get_events_dataframe(), che lo converte internamente in speech con
timing word-level. Il modello predice attivazioni con shape (n_timesteps, n_vertices).
Per titoli brevi (pochi secondi), si media sui timestep per ottenere un vettore
di attivazione per titolo.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

# Flag per gestire l'assenza di TRIBE v2 (fallback a mock)
TRIBE_AVAILABLE = False
TribeModel = None

try:
    from tribev2 import TribeModel as _TribeModel

    TribeModel = _TribeModel
    TRIBE_AVAILABLE = True
except ImportError:
    pass


def _load_model(cache_folder: str = "./cache") -> object:
    """Carica il modello TRIBE v2 pretrained."""
    if not TRIBE_AVAILABLE:
        raise ImportError(
            "tribev2 non è installato. Installa con:\n"
            "  git clone https://github.com/facebookresearch/tribev2\n"
            "  cd tribev2 && pip install -e .\n"
            "Oppure usa --mock per generare dati simulati."
        )
    print("  → Loading TRIBE v2 model (questo può richiedere qualche minuto)...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache_folder)
    print("    ✓ Model loaded")
    return model


def _predict_single_text(model, text: str) -> np.ndarray:
    """
    Predice l'attivazione cerebrale per un singolo testo.

    TRIBE v2 accetta input testuale via text_path. Il testo viene convertito
    internamente in speech con timing word-level.

    Per titoli brevi (< 5 sec di speech), la predizione avrà pochi timestep.
    Mediamo sui timestep per ottenere un singolo vettore di attivazione.

    Returns:
        np.ndarray: vettore di attivazione, shape (n_vertices,)
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(text)
        text_path = f.name

    try:
        # Costruisci l'events dataframe dal testo
        df = model.get_events_dataframe(text_path=text_path)

        # Predici le attivazioni cerebrali
        preds, segments = model.predict(events=df)
        # preds shape: (n_timesteps, n_vertices)

        # Media sui timestep per ottenere un singolo vettore
        activation = preds.mean(axis=0)  # shape: (n_vertices,)

        return activation.astype(np.float32)
    finally:
        os.unlink(text_path)


def _generate_mock_activation(text: str, n_vertices: int = 20484) -> np.ndarray:
    """
    Genera attivazioni mock deterministiche basate sul testo.

    Usa un hash del testo come seed per generare pattern realistici:
    - Baseline gaussiana per tutti i vertici
    - Boost nelle aree linguistiche per testi più lunghi
    - Boost nelle aree emotive per testi con parole forti
    - Varianza proporzionale alla complessità del testo

    n_vertices=20484 corrisponde a fsaverage5 (10242 per emisfero).
    """
    seed = hash(text) % (2**32)
    rng = np.random.RandomState(seed)

    # Baseline: attivazione gaussiana
    activation = rng.randn(n_vertices).astype(np.float32) * 0.3

    # Fattori basati sul testo
    word_count = len(text.split())
    char_count = len(text)
    has_question = "?" in text
    has_exclamation = "!" in text

    # Parole "emotive" comuni nel copywriting
    emotional_words = {
        "free", "now", "urgent", "limited", "exclusive", "guaranteed",
        "proven", "secret", "discover", "amazing", "incredible", "transform",
        "revolutionary", "ultimate", "powerful", "instant", "massive",
        "libera", "gratis", "ora", "urgente", "limitato", "esclusivo",
        "garantito", "provato", "segreto", "scopri", "incredibile",
        "trasforma", "rivoluzionario", "ultimo", "potente", "istantaneo",
    }
    text_lower = text.lower()
    emotion_score = sum(1 for w in emotional_words if w in text_lower)

    # Simula regioni cerebrali su fsaverage5
    # (approssimazione: dividiamo i vertici in blocchi funzionali)
    n_hemi = n_vertices // 2

    # Aree linguistiche (STG, MTG, IFG) — ~vertici 2000-5000 per emisfero sinistro
    lang_start, lang_end = 2000, 5000
    activation[lang_start:lang_end] += 0.2 + 0.02 * word_count

    # Aree attentive dorsali (IPS, FEF) — ~vertici 5000-6500
    att_d_start, att_d_end = 5000, 6500
    activation[att_d_start:att_d_end] += 0.15 + (0.1 if has_question else 0)

    # Aree attentive ventrali (TPJ) — ~vertici 6500-7500
    att_v_start, att_v_end = 6500, 7500
    activation[att_v_start:att_v_end] += 0.1 + (0.15 if has_exclamation else 0)

    # Prefrontale (dlPFC, vmPFC) — ~vertici 0-2000
    pfc_start, pfc_end = 0, 2000
    activation[pfc_start:pfc_end] += 0.05 + 0.015 * word_count

    # Emotive (amygdala proxy, insula) — ~vertici 7500-8500
    emo_start, emo_end = 7500, 8500
    activation[emo_start:emo_end] += 0.1 + 0.08 * emotion_score

    # Default mode network (mPFC, PCC, angular) — ~vertici 8500-10000
    dmn_start, dmn_end = 8500, 10000
    activation[dmn_start:dmn_end] += 0.05 + (0.1 if char_count > 50 else 0)

    # Anche l'emisfero destro (spostato di n_hemi)
    activation[n_hemi + lang_start : n_hemi + lang_end] += 0.1 + 0.01 * word_count
    activation[n_hemi + emo_start : n_hemi + emo_end] += 0.08 + 0.06 * emotion_score

    return activation


def analyze_all(
    variants_path: str,
    mock: bool = False,
    cache_folder: str = "./cache",
) -> dict:
    """
    Analizza tutti i titoli (originale + 60 varianti) con TRIBE v2.

    Args:
        variants_path: percorso al file variants.json
        mock: se True, usa attivazioni simulate invece del modello reale
        cache_folder: percorso cache per il modello TRIBE v2

    Returns:
        dict con attivazioni per ogni titolo, struttura:
        {
            "n_vertices": int,
            "original": {"title": str, "activation": list[float]},
            "similar": [{"title": str, "activation": list[float]}, ...],
            "alternative": [...],
            "exaggerated": [...]
        }
    """
    with open(variants_path, "r", encoding="utf-8") as f:
        variants = json.load(f)

    input_title = variants["input_title"]
    categories = ["similar", "alternative", "exaggerated"]

    # Raccogli tutti i titoli
    all_titles = [("original", input_title)]
    for cat in categories:
        for title in variants[cat]:
            all_titles.append((cat, title))

    print(f"  → Analyzing {len(all_titles)} titles...")

    # Carica modello o usa mock
    model = None
    if not mock:
        try:
            model = _load_model(cache_folder)
        except ImportError as e:
            print(f"  ⚠ {e}")
            print("  → Falling back to mock activations")
            mock = True

    # Predici attivazioni
    predict_fn = (
        _generate_mock_activation if mock else lambda text: _predict_single_text(model, text)
    )

    results = {"n_vertices": 20484, "mock": mock}

    # Originale
    print(f"  → Analyzing original title...")
    act = predict_fn(input_title)
    results["n_vertices"] = len(act)
    results["original"] = {
        "title": input_title,
        "activation": act.tolist(),
    }

    # Varianti per categoria
    for cat in categories:
        results[cat] = []
        print(f"  → Analyzing {cat} ({len(variants[cat])} titles)...")
        for title in tqdm(variants[cat], desc=f"    {cat}", leave=False):
            act = predict_fn(title)
            results[cat].append({
                "title": title,
                "activation": act.tolist(),
            })

    return results


def save_activations(activations: dict, output_path: str) -> None:
    """Salva le attivazioni in formato JSON (per portabilità) e NPZ (per efficienza)."""
    # JSON (senza i vettori completi per dimensione)
    summary = {
        "n_vertices": activations["n_vertices"],
        "mock": activations.get("mock", False),
        "original": {
            "title": activations["original"]["title"],
            "activation_shape": len(activations["original"]["activation"]),
        },
    }
    for cat in ["similar", "alternative", "exaggerated"]:
        summary[cat] = [
            {"title": item["title"], "activation_shape": len(item["activation"])}
            for item in activations[cat]
        ]

    json_path = output_path.replace(".npz", "_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # NPZ (dati completi)
    arrays = {
        "original": np.array(activations["original"]["activation"], dtype=np.float32),
    }
    titles = {"original_title": activations["original"]["title"]}

    for cat in ["similar", "alternative", "exaggerated"]:
        for i, item in enumerate(activations[cat]):
            key = f"{cat}_{i:02d}"
            arrays[key] = np.array(item["activation"], dtype=np.float32)
            titles[f"{key}_title"] = item["title"]

    npz_path = output_path if output_path.endswith(".npz") else output_path + ".npz"
    np.savez_compressed(npz_path, **arrays)

    # Salva anche i titoli
    titles_path = npz_path.replace(".npz", "_titles.json")
    with open(titles_path, "w", encoding="utf-8") as f:
        json.dump(titles, f, indent=2, ensure_ascii=False)

    print(f"    ✓ Activations saved to {npz_path} and {json_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="output/variants.json")
    parser.add_argument("--mock", action="store_true", help="Use mock activations")
    parser.add_argument("--output", default="output/activations")
    args = parser.parse_args()

    activations = analyze_all(args.variants, mock=args.mock)
    save_activations(activations, args.output)
