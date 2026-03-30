"""
aggregate.py — Aggregazione vertici corticali → ROI scores.

TRIBE v2 produce attivazioni su mesh fsaverage5 (~20484 vertici).
Questo modulo aggrega i vertici in Regions of Interest (ROI) rilevanti
per l'analisi di stimoli linguistici/copywriting.

Supporta due modalità:
1. Atlas reale: Glasser 360 o Destrieux via nilearn (se disponibile)
2. Atlas approssimato: mapping funzionale basato su coordinate fsaverage5
"""

import json
from typing import Optional

import numpy as np

# ROI rilevanti per copywriting (come da spec)
ROI_DEFINITIONS = {
    "language_network": {
        "description": "IFG, STG, MTG — elaborazione linguistica, comprensione",
        "brodmann": [22, 41, 42, 44, 45, 21, 38],
    },
    "attention_dorsal": {
        "description": "IPS, FEF — attenzione spaziale e salienza",
        "brodmann": [7, 8, 6],
    },
    "attention_ventral": {
        "description": "TPJ, IFG — attenzione involontaria, sorpresa",
        "brodmann": [39, 40, 47],
    },
    "prefrontal": {
        "description": "dlPFC, vmPFC — carico cognitivo, decision making",
        "brodmann": [9, 10, 11, 46],
    },
    "emotional": {
        "description": "Amygdala proxy, insula — risposta emotiva",
        "brodmann": [13, 14, 25, 34],
    },
    "default_mode": {
        "description": "mPFC, PCC, angular — elaborazione narrativa, self-reference",
        "brodmann": [10, 23, 29, 30, 31, 39],
    },
}


def _build_approximate_atlas(n_vertices: int = 20484) -> dict[str, np.ndarray]:
    """
    Costruisce un atlas approssimato per fsaverage5 mappando blocchi di vertici
    a ROI funzionali. Questo è un fallback quando nilearn/atlas reali non sono
    disponibili.

    La mappatura è basata sull'organizzazione topografica di fsaverage5:
    - Emisfero sinistro: vertici 0 a n_vertices//2 - 1
    - Emisfero destro: vertici n_vertices//2 a n_vertices - 1

    I vertici sono ordinati approssimativamente per posizione anatomica.
    Questa è un'approssimazione — per risultati precisi usare l'atlas Glasser.
    """
    n_hemi = n_vertices // 2
    atlas = {}

    # Definizioni approssimative per emisfero sinistro
    # (le aree linguistiche sono lateralizzate a sinistra)
    roi_vertex_ranges = {
        "language_network": [
            (2000, 5000),  # LH: STG, MTG, IFG
            (n_hemi + 2000, n_hemi + 4000),  # RH (minor)
        ],
        "attention_dorsal": [
            (5000, 6500),  # LH: IPS, FEF
            (n_hemi + 5000, n_hemi + 6500),  # RH
        ],
        "attention_ventral": [
            (6500, 7500),  # LH: TPJ
            (n_hemi + 6500, n_hemi + 7500),  # RH
        ],
        "prefrontal": [
            (0, 2000),  # LH: dlPFC, vmPFC
            (n_hemi, n_hemi + 2000),  # RH
        ],
        "emotional": [
            (7500, 8500),  # LH: insula, amygdala proxy
            (n_hemi + 7500, n_hemi + 8500),  # RH
        ],
        "default_mode": [
            (8500, 10000),  # LH: mPFC, PCC, angular
            (n_hemi + 8500, n_hemi + 10000),  # RH
        ],
    }

    for roi_name, ranges in roi_vertex_ranges.items():
        indices = []
        for start, end in ranges:
            end = min(end, n_vertices)
            indices.extend(range(start, end))
        atlas[roi_name] = np.array(indices, dtype=np.int64)

    return atlas


def _try_load_nilearn_atlas(n_vertices: int = 20484) -> Optional[dict[str, np.ndarray]]:
    """
    Tenta di caricare un atlas reale via nilearn per fsaverage5.

    Prova in ordine:
    1. Destrieux atlas (incluso in nilearn)
    2. Schaefer parcellation (400 ROI → aggregato in 7 network Yeo)

    Returns None se nilearn non è disponibile.
    """
    try:
        from nilearn import datasets, surface

        # Carica fsaverage5
        fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

        # Prova Destrieux atlas
        destrieux = datasets.fetch_atlas_surf_destrieux()

        # Mappa le label Destrieux alle nostre ROI
        # Destrieux ha ~75 regioni per emisfero
        label_to_roi = {}

        # Language network: gyri temporali e frontali inferiori
        lang_labels = [
            "G_temp_sup-G_T_transv",
            "G_temp_sup-Lateral",
            "G_temp_sup-Plan_polar",
            "G_temp_sup-Plan_tempo",
            "G_temporal_middle",
            "G_front_inf-Opercular",
            "G_front_inf-Orbital",
            "G_front_inf-Triangul",
            "S_temporal_sup",
        ]
        for label in lang_labels:
            label_to_roi[label] = "language_network"

        # Attention dorsal: solco intraparietale, FEF
        att_d_labels = [
            "S_intrapariet_and_P_trans",
            "G_parietal_sup",
            "G_front_sup",
            "S_precentral-sup-part",
        ]
        for label in att_d_labels:
            label_to_roi[label] = "attention_dorsal"

        # Attention ventral: TPJ, giro angolare
        att_v_labels = [
            "G_pariet_inf-Supramar",
            "G_pariet_inf-Angular",
            "S_temporal_inf",
        ]
        for label in att_v_labels:
            label_to_roi[label] = "attention_ventral"

        # Prefrontal
        pfc_labels = [
            "G_front_middle",
            "G_front_sup",
            "G_orbital",
            "S_front_middle",
            "S_front_sup",
            "G_rectus",
            "S_orbital-H_Shaped",
        ]
        for label in pfc_labels:
            label_to_roi[label] = "prefrontal"

        # Emotional
        emo_labels = [
            "G_insular_short",
            "G_insular_long",
            "S_circular_insula_ant",
            "S_circular_insula_inf",
            "S_circular_insula_sup",
            "G_subcallosal",
        ]
        for label in emo_labels:
            label_to_roi[label] = "emotional"

        # Default mode network
        dmn_labels = [
            "G_cingul-Post-dorsal",
            "G_cingul-Post-ventral",
            "G_precuneus",
            "G_front_medial",
            "Pole_temporal",
        ]
        for label in dmn_labels:
            label_to_roi[label] = "default_mode"

        # Costruisci l'atlas
        labels_lh = destrieux.labels
        map_lh = destrieux.map_left
        map_rh = destrieux.map_right

        atlas = {roi: [] for roi in ROI_DEFINITIONS}

        for i, label_name in enumerate(labels_lh):
            label_str = label_name if isinstance(label_name, str) else label_name.decode()
            if label_str in label_to_roi:
                roi = label_to_roi[label_str]
                # Vertici dell'emisfero sinistro
                vertices_lh = np.where(map_lh == i)[0]
                atlas[roi].extend(vertices_lh.tolist())
                # Vertici dell'emisfero destro (offset)
                vertices_rh = np.where(map_rh == i)[0]
                atlas[roi].extend((vertices_rh + n_vertices // 2).tolist())

        # Converti in array numpy
        for roi in atlas:
            indices = np.array(atlas[roi], dtype=np.int64)
            # Filtra indici fuori range
            atlas[roi] = indices[indices < n_vertices]

        # Verifica che ogni ROI abbia vertici
        if all(len(v) > 0 for v in atlas.values()):
            return atlas

    except Exception as e:
        print(f"  ⚠ Could not load nilearn atlas: {e}")

    return None


def build_atlas(n_vertices: int = 20484) -> dict[str, np.ndarray]:
    """
    Costruisce l'atlas per aggregazione ROI.

    Tenta prima un atlas reale via nilearn, poi fallback all'approssimazione.
    """
    atlas = _try_load_nilearn_atlas(n_vertices)
    if atlas is not None:
        print("  → Using nilearn Destrieux atlas for ROI mapping")
        return atlas

    print("  → Using approximate atlas for ROI mapping (install nilearn for precise mapping)")
    return _build_approximate_atlas(n_vertices)


def aggregate_to_roi(activation: np.ndarray, atlas: dict[str, np.ndarray]) -> dict[str, float]:
    """
    Aggrega un vettore di attivazione (n_vertices,) in ROI scores.

    Returns:
        dict con score medio per ogni ROI
    """
    scores = {}
    for roi_name, indices in atlas.items():
        valid_indices = indices[indices < len(activation)]
        if len(valid_indices) > 0:
            scores[roi_name] = float(activation[valid_indices].mean())
        else:
            scores[roi_name] = 0.0
    return scores


def normalize_scores(all_scores: list[dict[str, float]]) -> list[dict[str, float]]:
    """
    Normalizza i ROI scores in range [0, 1] rispetto al batch corrente (min-max).

    Ogni ROI è normalizzata indipendentemente.
    """
    if not all_scores:
        return all_scores

    roi_names = list(all_scores[0].keys())

    # Calcola min e max per ogni ROI
    roi_min = {}
    roi_max = {}
    for roi in roi_names:
        values = [s[roi] for s in all_scores]
        roi_min[roi] = min(values)
        roi_max[roi] = max(values)

    # Normalizza
    normalized = []
    for scores in all_scores:
        norm = {}
        for roi in roi_names:
            range_val = roi_max[roi] - roi_min[roi]
            if range_val > 1e-10:
                norm[roi] = (scores[roi] - roi_min[roi]) / range_val
            else:
                norm[roi] = 0.5  # Se tutti i valori sono uguali
        normalized.append(norm)

    return normalized


def aggregate_all(activations: dict) -> dict:
    """
    Aggrega tutte le attivazioni in ROI scores normalizzati.

    Args:
        activations: output di analyze_all()

    Returns:
        dict con ROI scores per ogni titolo, struttura come da spec
    """
    n_vertices = activations.get("n_vertices", 20484)
    atlas = build_atlas(n_vertices)

    # Raccogli tutti gli score grezzi
    all_raw_scores = []
    titles_metadata = []  # (category, title)

    # Originale
    act = np.array(activations["original"]["activation"], dtype=np.float32)
    raw_scores = aggregate_to_roi(act, atlas)
    all_raw_scores.append(raw_scores)
    titles_metadata.append(("original", activations["original"]["title"]))

    # Varianti
    for cat in ["similar", "alternative", "exaggerated"]:
        for item in activations[cat]:
            act = np.array(item["activation"], dtype=np.float32)
            raw_scores = aggregate_to_roi(act, atlas)
            all_raw_scores.append(raw_scores)
            titles_metadata.append((cat, item["title"]))

    # Normalizza tutto il batch insieme
    normalized = normalize_scores(all_raw_scores)

    # Ricomponi la struttura output
    result = {}
    idx = 0

    # Originale
    result["original"] = {
        "title": titles_metadata[idx][1],
        "roi_scores": normalized[idx],
    }
    idx += 1

    # Varianti per categoria
    for cat in ["similar", "alternative", "exaggerated"]:
        result[cat] = []
        for item in activations[cat]:
            result[cat].append({
                "title": titles_metadata[idx][1],
                "roi_scores": normalized[idx],
            })
            idx += 1

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", default="output/activations.json")
    parser.add_argument("--output", default="output/scores.json")
    args = parser.parse_args()

    with open(args.activations, "r") as f:
        activations = json.load(f)

    scores = aggregate_all(activations)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"Saved ROI scores to {args.output}")
