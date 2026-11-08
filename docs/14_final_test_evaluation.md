# Évaluation finale sur le jeu de test (Stage 5)

**Statut :** PROTOCOLE FIGÉ — exécution one-shot via Colab (`notebooks/03_final_test_evaluation_colab.ipynb`)  
**Sélection figée avant accès test :** `experiments/metadata/phase4/FINAL_MODEL_SELECTION_FREEZE.json`

## Règles

- Aucun retuning de seuil sur le test.
- Aucune re-sélection de modèle après observation du test.
- Une seule passe d’évaluation (garde-fou `STAGE5_TEST_EVAL_DONE.json` sur Drive).
- Vérification SHA-256 des checkpoints Drive contre les métadonnées avant prédiction.

## Modèle primaire (sélection validation)

| Champ | Valeur |
| --- | --- |
| Expérience | `E2b_unet_mobilenetv2_finetune_20261108_1721` |
| Seuil opérationnel | 0.55 |
| Dice val (mean/image) | ≈ 0.778 |
| Checkpoint SHA-256 | `280170dc64347b0b4cc8ce3af9be40dec4ac3733b27d1abeb620cc0c8e88e432` |

## Modèles évalués une fois sur test

E0 (thr 0.5), E1 (thr 0.35), E2a (thr 0.15), E2b (thr 0.55).

## Résultats test

**À compléter uniquement après exécution Colab réussie.**  
Fichiers attendus sur Drive :

- `experiments/results/phase4/phase4_final_test_comparison.csv`
- `experiments/results/phase4/*_test_per_image.csv`
- `experiments/metadata/phase4/STAGE5_TEST_EVAL_DONE.json`

Tant que ces fichiers n’existent pas, **aucune métrique test n’est inventée** dans le rapport ou les slides.
