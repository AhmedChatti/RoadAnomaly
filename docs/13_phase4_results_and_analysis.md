# Phase 4 results and analysis (validation)

**Status:** COMPLETED (validation comparison)  
**Date:** 2026-11-08  
**Git SHA used for E2 training:** `9e734a1d0af48342f92c084216b1dd1d4c254741`  
**Split lock:** `9a01c05c…04585c78` (42/9/9)  
**Test set:** **not used** in this document.

## Experiments included

| ID | Description | Status |
| --- | --- | --- |
| E0 | All-background baseline | Executed (Phase 3) |
| E1 | CNN from scratch (`E1_…_1510`) | Executed / frozen |
| E2a | U-Net + MobileNetV2, frozen encoder | Executed (Colab T4) |
| E2b | Selective fine-tune from `block_13_expand` | Executed (Colab T4) |

Source table: `experiments/results/phase4/phase4_E0_E1_E2_val_comparison.csv`  
Metadata: `experiments/metadata/phase4/E2a_…1714.json`, `E2b_…1721.json`

## Validation metrics (mean per image)

Operational thresholds are **validation-selected** (except E0 fixed 0.5).

| Exp | Train thr | Op. thr | Best epoch | Dice | IoU | Precision | Recall | F1 | Accuracy | Duration (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | — | 0.50 | — | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.862 | 0 |
| E1 | 0.50 | 0.35 | 66 | 0.569 | 0.445 | 0.522 | 0.850 | 0.569 | 0.867 | (Phase 3) |
| E2a | 0.50 | 0.15 | 10 | 0.705 | 0.584 | 0.678 | 0.835 | 0.705 | 0.935 | ~395 |
| E2b | 0.50 | 0.55 | 34 | **0.778** | **0.686** | **0.846** | 0.793 | **0.778** | **0.954** | ~437 |

Checkpoint Dice monitored during training uses fixed thr **0.5** (`best_val_metric_value`): E2a 0.688; E2b 0.778. Operational metrics above use the selected thresholds.

## Checkpoint hashes (from metadata)

| Exp | Drive checkpoint | SHA-256 |
| --- | --- | --- |
| E1 | local/Drive Phase 3 path | `567bde32…0fb234` |
| E2a | `…/E2a_…1714/best.keras` | `428845e976a8f4773ce929ec66520227b49928142d4c34e38974af671f38b080` |
| E2b | `…/E2b_…1721/best.keras` | `280170dc64347b0b4cc8ce3af9be40dec4ac3733b27d1abeb620cc0c8e88e432` |

Drive verification of E2 weights is performed in Stage 5 before test access.

## Interpretation (evidence-based)

1. **E0 accuracy trap.** Accuracy ≈ 0.86 with Dice/IoU/Recall = 0. Accuracy alone is invalid for this imbalance (~10% anomaly pixels).
2. **E1 from scratch.** Non-trivial localization (Dice 0.569) with high recall (0.85) and moderate precision.
3. **E2a transfer learning.** Frozen ImageNet encoder + trained decoder improves val Dice to 0.705 vs E1. Selected threshold is low (0.15), indicating calibrated probabilities differ from E1.
4. **E2b fine-tuning.** Best validation Dice (0.778) and IoU (0.686). Higher precision than E2a, slightly lower recall. Initialized from E2a best checkpoint as required.
5. **Overfitting risk.** Small train set (n=42). E2a stopped early (best @10 / 22 epochs). E2b best @34 / 46 epochs. Curves live on Drive (`*_training_curves.png`); not reproduced locally here.
6. **No test claims in this section.** Generalization to unseen test scenes is Stage 5 only.

## Limitations of the comparison

- n_val = 9 images → high uncertainty; no confidence intervals invented.
- Near-duplicate scenes and category imbalance remain.
- Test split has **no lost-tire** examples (frozen protocol note).
- Thresholds differ across models; comparing at a single shared threshold would change rankings and is **not** the protocol used for operational selection.
- Parameter counts were not logged in metadata; omitted rather than guessed.
- Figures on Drive are not copied into Git (`.gitignore`).

## Provisional model selection for Stage 5 (validation only)

**Primary selected model:** E2b (`E2b_unet_mobilenetv2_finetune_20261108_1721`)  
**Reason:** highest validation mean Dice / IoU / F1 among completed experiments.  
**Operational threshold:** 0.55 (validation-selected).  
**Also evaluate once on test (frozen protocol):** E0, E1, E2a, E2b with their frozen operational thresholds — no further threshold tuning.

Selection is frozen **before** test access.
