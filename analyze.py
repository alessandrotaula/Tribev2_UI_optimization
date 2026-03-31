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
import numpy as np
from tqdm import tqdm

# Flag disponibilità TRIBE v2
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
            "  cd tribev2 && pip install -e ."
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


def analyze_all(
    variants_path: str,
    cache_folder: str = "./cache",
) -> dict:
    """
    Analizza tutti i titoli (originale + 60 varianti) con TRIBE v2.

    Args:
        variants_path: percorso al file variants.json
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

    model = _load_model(cache_folder)
    predict_fn = lambda text: _predict_single_text(model, text)

    results = {"n_vertices": 20484}

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
    parser.add_argument("--output", default="output/activations")
    args = parser.parse_args()

    activations = analyze_all(args.variants)
    save_activations(activations, args.output)
