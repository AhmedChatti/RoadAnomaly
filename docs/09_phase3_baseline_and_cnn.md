# Phase 3 — Baseline et CNN from scratch

**Date :** 8 novembre 2026  
**Matériel :** CPU local (pas de GPU TensorFlow Windows)  
**Test set :** non utilisé pour la sélection

## Expériences

### E0 — tout fond

- Aucun entraînement.
- Sur **validation** : IoU = 0, Dice = 0, Recall = 0, Accuracy ≈ **86,2 %**.
- Figure : `experiments/figures/phase3_E0_metric_trap.png`
- Métadonnées : `experiments/metadata/E0_all_background.json`

### E1 — FCN U-Net-lite from scratch

- Architecture : encodeur–décodeur, `base_filters=8`, **sans BatchNorm** (BN + `batch_size=2` faisait collapser vers le fond).
- Perte : BCE pondérée (poids positif 15) + soft Dice.
- Adam lr=1e-3, batch=2, seed=42, early stopping patience=12 sur `val_mean_dice`.
- Meilleur checkpoint : epoch **66**, `val_mean_dice` (thr=0.5 pendant le train) ≈ **0,563**.
- Seuil choisi **sur validation** : **0,35** → mean Dice/image ≈ **0,569**, mean IoU ≈ **0,445**, mean Recall ≈ **0,850**.
- Durée ≈ **28 min** (78 epochs avant early stop).
- Checkpoint : `checkpoints/E1_cnn_scratch_20261108_1510/best.keras`
- Métadonnées : `experiments/metadata/E1_cnn_scratch_20261108_1510.json`

## Comparaison validation (réelle)

| Expérience | mean Dice | mean IoU | mean Recall | Accuracy globale |
| --- | ---: | ---: | ---: | ---: |
| E0 | 0,000 | 0,000 | 0,000 | 0,862 |
| E1 | 0,569 | 0,445 | 0,850 | 0,867 |

Source : `experiments/results/phase3_E0_E1_val_comparison.csv`

## Figures

- `phase3_E0_metric_trap.png`
- `phase3_E1_training_curves.png`
- `phase3_E1_val_qualitative.png`

## Décisions techniques notées

1. Batch size abaissé à 2 pour la mémoire CPU.
2. BatchNorm retiré après diagnostic (collapse fond).
3. Biais final initialisé au logit(0,1).
4. Le test reste bloqué jusqu’à la Phase 6.

## Hors scope (Phase 4+)

U-Net + MobileNetV2, fine-tuning, évaluation test finale.

## Gel

Phase 3 **figée** le 8 novembre 2026. Audit et plan Phase 4 : `docs/10_phase3_freeze_and_phase4_plan.md`.  
Ne pas réentraîner E0/E1 ni modifier les métriques ci-dessus.
