# Phase 2 — Pipeline de données

**Date :** 8 novembre 2026  
**Statut :** implémenté (pas d’entraînement)  
**Conformité :** décisions Phase 1 conservées (split 42/9/9, entrée 512×288, masques 0/1, pas de letterbox)

## Objectif

Fournir un pipeline unique, déterministe et testé pour charger les paires image/masque **depuis les originaux** (`frames/`), les préparer pour Keras, et appliquer une augmentation **train-only**.

Les artefacts letterbox 512×512 sous `data/processed/` restent des archives d’exploration ; ils ne sont **pas** l’entrée d’entraînement par défaut.

## Modules

| Module | Rôle |
| --- | --- |
| `src/config.py` | Charge `configs/default.json` |
| `src/preprocessing.py` | Alignement, remap 0/2→0/1, resize 512×288, normalisation |
| `src/augmentation.py` | Flip H, rotation/zoom/translation limités, photométrie ; masque nearest |
| `src/dataset.py` | Manifeste verrouillé, `RoadAnomalyDataset`, batches |
| `src/metrics.py` | Precision/Recall/F1/IoU/Dice/Accuracy + conventions §7.3 |

## Vérifications

- Hash du split contrôlé à chaque `load_split_manifest()`.
- Aucun `scene_group` ne traverse deux splits.
- Aucun `shutil.rmtree` dans le pipeline Phase 2.
- Masques préparés ∈ `{0,1}` uniquement.
- Toute anomalie survit au resize (assert).

## Figures

Générées dans `experiments/figures/` :

- `phase2_01_512x288_example.png`
- `phase2_02_augmentation_sync.png`
- `phase2_03_split_examples.png`
- `phase2_04_pipeline_summary.png`

Commande :

```bash
python scripts/run_phase2_visualizations.py
python -m unittest discover -s tests -v
```

## Hors scope (Phases suivantes)

- Entraînement E0/E1/E2
- Checkpointing
- Streamlit
