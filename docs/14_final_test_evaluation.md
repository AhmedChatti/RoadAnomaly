# Évaluation finale sur le jeu de test (Stage 5)

**Statut :** TERMINÉ (one-shot Colab, 2026-11-08)  
**Notebook :** `notebooks/03_final_test_evaluation_colab.ipynb`  
**Sélection figée avant accès test :** `experiments/metadata/phase4/FINAL_MODEL_SELECTION_FREEZE.json`  
**Garde-fou :** `experiments/metadata/phase4/STAGE5_TEST_EVAL_DONE.json`  
**Tableau résumé :** `experiments/results/phase4/phase4_final_test_comparison.csv`

## Règles respectées

- Aucun retuning de seuil sur le test.
- Aucune re-sélection de modèle après observation du test.
- Une seule passe d’évaluation.
- SHA-256 Drive vérifiés pour E1 / E2a / E2b (`match True`).
- Chargement Keras-safe (rebuild architecture + poids) pour éviter l’incompatibilité `GlorotUniform.input_axes`.

## Modèle primaire (sélection validation, inchangée)

| Champ | Valeur |
| --- | --- |
| Expérience | `E2b_unet_mobilenetv2_finetune_20261108_1721` |
| Seuil opérationnel | 0.55 |
| Dice val (mean/image) | ≈ 0.778 |
| Dice test (mean/image) | ≈ **0.818** |
| Dice test (global) | ≈ **0.871** |
| Checkpoint SHA-256 | `280170dc64347b0b4cc8ce3af9be40dec4ac3733b27d1abeb620cc0c8e88e432` |

## Résultats test (mean / image, seuils figés)

| Exp | Thr | Dice | IoU | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | 0.50 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.921 |
| E1 | 0.35 | 0.398 | 0.287 | 0.297 | 0.951 | 0.398 | 0.767 |
| E2a | 0.15 | 0.613 | 0.454 | 0.465 | 0.960 | 0.613 | 0.935 |
| **E2b** | **0.55** | **0.818** | **0.705** | **0.778** | **0.897** | **0.818** | **0.979** |

## Résultats test (globaux pixels)

| Exp | Dice | IoU | Accuracy |
| --- | ---: | ---: | ---: |
| E0 | 0.000 | 0.000 | 0.921 |
| E1 | 0.394 | 0.245 | 0.767 |
| E2a | 0.695 | 0.533 | 0.935 |
| **E2b** | **0.871** | **0.771** | **0.979** |

## Lecture val → test (Dice mean/image)

| Exp | Val | Test | Δ |
| --- | ---: | ---: | ---: |
| E1 | 0.569 | 0.398 | −0.171 |
| E2a | 0.705 | 0.613 | −0.092 |
| E2b | 0.778 | 0.818 | +0.040 |

Avec **n_test = 9**, un écart positif sur E2b n’est **pas** une preuve de généralisation supérieure ; il reflète surtout la variance d’un petit jeu. E1 et E2a baissent comme attendu. L’ordre relatif **E2b > E2a > E1 > E0** est conservé. L’accuracy d’E0 (~0,92) reste trompeuse (Dice = 0).

## Artefacts Drive

- `…/results/phase4/phase4_final_test_comparison.csv`
- `…/results/phase4/*_test_per_image.csv`
- `…/metadata/phase4/STAGE5_TEST_EVAL_DONE.json`
