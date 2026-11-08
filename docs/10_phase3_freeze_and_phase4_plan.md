# Phase 3 freeze audit + Phase 4 implementation plan

**Date :** 8 novembre 2026  
**Statut Phase 3 :** **FROZEN** (résultats figés ; pas de réentraînement)  
**Statut Phase 4 :** **PLAN ONLY** (pas d’entraînement démarré)

Ce document (1) archive l’audit de gel de la Phase 3 et (2) prépare l’exécution Colab de la Phase 4.  
Il **ne modifie pas** le protocole Phase 1, le pipeline Phase 2, ni les métriques scientifiques Phase 3.

---

## A. Gel Phase 3

### A.1 Expériences canoniques

| ID | Rôle | Artefacts |
| --- | --- | --- |
| **E0_all_background** | Baseline triviale | `experiments/metadata/E0_all_background.json`, `experiments/results/E0_all_background_val_per_image.csv`, `experiments/figures/phase3_E0_metric_trap.png` |
| **E1_cnn_scratch_20261108_1510** | CNN from scratch (officiel) | metadata + history JSON, threshold grid, per-image CSV, curves + qualitative figures, `checkpoints/E1_cnn_scratch_20261108_1510/best.keras` |

**Non-canonique (conservé, non utilisé pour la comparaison officielle) :**  
`E1_cnn_scratch_20261108_1501` — run intermédiaire (BN / poids positif 9 / arrêt tôt). Métadonnées et checkpoint **préservés** pour traçabilité uniquement.

### A.2 Résultats validation figés (inchangés)

| Expérience | mean Dice | mean IoU | mean Recall | Accuracy globale |
| --- | ---: | ---: | ---: | ---: |
| E0 | 0.000 | 0.000 | 0.000 | 0.862 |
| E1 | **0.569** | **0.445** | **0.850** | 0.867 |

Source : `experiments/results/phase3_E0_E1_val_comparison.csv`  
Split lock SHA-256 : `9a01c05cfc7d33783c4f64b91b0ae423a4d3ad57439fd71ffc029a8f04585c78`

### A.3 Seuils (séparation explicite)

| Usage | Seuil | Où c’est enregistré |
| --- | ---: | --- |
| Sélection de checkpoint pendant l’entraînement (`MeanDiceCallback` → `val_mean_dice`) | **0.5** | `threshold_selection.default_during_training`, grille CSV ligne `0.5`, `best_val_metric_value` ≈ 0.563 @ epoch 66 |
| Seuil opérationnel choisi **après** entraînement, **sur validation seulement** | **0.35** | `threshold_selection.value`, `validation_metrics.threshold`, grille CSV ligne `0.35` |

Ces deux valeurs sont **intentionnellement distinctes**. Le test n’entre pas dans la sélection.

### A.4 Audit exécuté

Script : `scripts/audit_phase3_freeze.py`  
Rapport machine : `experiments/metadata/phase3_freeze_audit.json`

Vérifications **PASS** :

1. Métadonnées, historiques (78 epochs), figures, CSV présents et non vides.  
2. Checkpoint E1 rechargeable ; prédictions val shape `(N,288,512,1)`, probs ∈ [0,1], non constantes ; mean Dice rechargé = Dice figé.  
3. `0.35` et `0.5` enregistrés séparément.  
4. `evaluate_predictor_on_split(..., "test", allow_test=False)` lève `RuntimeError` ; le script Phase 3 n’appelle que `train`/`val`.  
5. Code Phase 3 présent (`src/models/*`, `losses`, `training`, `evaluation`, `scripts/run_phase3_experiments.py`) ; tests pipeline Phase 2 : **22/22 OK**.  
6. Checkpoint SHA-256 : `567bde326ee5b776bc34132b23012543b35f195ec8d60229d01196eb1a0fb234` (~5.8 Mo).

### A.5 Correctif de traçabilité (sans changer les métriques)

Lors du gel, le champ `default_during_training: 0.5` avait été écrasé par la réécriture post-grille de `threshold_selection` (le script ne le recopait pas).  
**Correction :** réinjection du champ + note dans `E1_cnn_scratch_20261108_1510.json` et dans `scripts/run_phase3_experiments.py` pour les runs futurs.  
**Aucun chiffre de performance n’a été recalculé ni modifié.**

### A.6 Limites Phase 3 (à reporter telles quelles)

1. Entraînement **CPU local**, `batch_size=2`, modèle réduit (`base_filters=8`, **sans BatchNorm**).  
2. BN retiré après collapse fond (documenté) — décision d’ingénierie, pas un changement de protocole.  
3. Jeu très petit (42/9/9) → variance élevée, risque de surapprentissage.  
4. Seuil 0.35 optimisé sur 9 images de val → estimation bruyante.  
5. Pas de tests unitaires dédiés `models`/`losses`/`training` (couverture = tests Phase 2 + audit de gel + smoke d’entraînement).  
6. Les poids `*.keras` sont dans `.gitignore` : **préservation locale + sauvegarde manuelle obligatoire** avant Phase 4 (Drive/Colab).

### A.7 Règles de conservation

- Ne pas réentraîner E0/E1 pour « améliorer » les chiffres Phase 3.  
- Ne pas supprimer `checkpoints/E1_cnn_scratch_20261108_1510/`.  
- Conserver `E1_1501` comme archive non-canonique.  
- Test set : **toujours bloqué** jusqu’à la Phase 6.

---

## B. Plan d’implémentation Phase 4 (U-Net + MobileNetV2)

**Objectif :** répondre à RQ2 / H2 — un U-Net avec encodeur ImageNet améliore-t-il le Dice/IoU val vs E1 ?

**Hors scope de ce document :** lancer l’entraînement. Attendre l’approbation explicite.

### B.1 Expériences prévues

| ID | Encodeur | Décodeur | Normalisation | Notes |
| --- | --- | --- | --- | --- |
| **E2a** | MobileNetV2 ImageNet **gelé** | U-Net decoder entraîné | `imagenet` | Transfer learning minimal |
| **E2b** | MobileNetV2 ImageNet **fine-tuné** (lr bas) | U-Net decoder | `imagenet` | Comparer sur val à E2a ; choisir le meilleur pour la suite |

Sélection inter-modèles : **meilleur `val_mean_dice` après sélection de seuil sur val**.  
Test : **non utilisé**.

### B.2 Contraintes figées (héritées Phases 1–3)

- Split 42/9/9 + SHA `9a01c05c…` inchangé.  
- Entrée **512×288**, pas de letterbox.  
- Masques modèle `{0,1}` dans le loader.  
- Augmentation **train-only**, sync image/masque (config Phase 2).  
- Checkpoint : max `val_mean_dice` (seuil train-time **0.5**).  
- Seuil opérationnel : grille sur **val** uniquement (comme E1).  
- Métriques primaires : Precision / Recall / F1 / IoU / Dice ; accuracy secondaire.  
- Stack : TensorFlow / Keras uniquement.

### B.3 Environnement cible : Google Colab GPU + Drive

1. Monter Google Drive (`/content/drive/MyDrive/RoadAnomaly/...`).  
2. Cloner / synchroniser le dépôt (code + `data/processed/split_manifest.csv` + frames ou archives images).  
3. Racine de travail Drive pour artefacts Phase 4, ex. :
   - `Drive/RoadAnomaly/checkpoints/E2a_.../`
   - `Drive/RoadAnomaly/checkpoints/E2b_.../`
   - `Drive/RoadAnomaly/experiments/{metadata,results,figures}/`
4. Notebook Colab : `notebooks/colab_phase4_unet_mobilenetv2.ipynb` (à créer à l’implémentation).  
5. Script reproductible local/Colab : `scripts/run_phase4_experiments.py`.  
6. Après chaque run : écrire metadata JSON + SHA du checkpoint + versions TF/CUDA.

### B.4 Architecture proposée

- Backbone : `keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(288,512,3))`.  
- Skip connections depuis niveaux MobileNetV2 standard (blocs intermédiaires) vers un décodeur U-Net (upsample + concat + conv).  
- Tête : `Conv2D(1, 1, activation="sigmoid")`, biais initial ≈ `logit(0.1)` (comme E1).  
- Entrée preprocess : pipeline existant `normalize="imagenet"` (déjà prévu dans `configs/default.json`).  
- Perte : reprendre `weighted_bce + soft_dice` (positif weight à calibrer légèrement si besoin, documenter tout écart vs 15).  
- Optimiseur Adam :
  - E2a : lr décodeur `1e-3` (config `learning_rate_decoder`)  
  - E2b stage 1 : comme E2a puis stage 2 : dégeler encodeur partiel / total avec `learning_rate_finetune_encoder=1e-4`  
- Batch size Colab : viser 8 (descendre à 4 si OOM) ; seed 42 ; early stop patience 12.

### B.5 Fichiers à créer (lors de l’approbation Phase 4)

| Fichier | Rôle |
| --- | --- |
| `src/models/unet_mobilenetv2.py` | Construction E2a/E2b |
| `src/training.py` (extension) | `train_unet_mobilenetv2(...)` sans toucher à la logique E1 figée |
| `scripts/run_phase4_experiments.py` | Orchestration E2a→E2b, figures, metadata |
| `notebooks/colab_phase4_unet_mobilenetv2.ipynb` | Exécution GPU + montage Drive |
| `docs/11_phase4_unet_mobilenetv2.md` | Compte rendu post-run (après résultats réels) |

**Ne pas modifier :** `docs/02_...`, split lock, preprocess Phase 2, artefacts E0/E1, seuils E1.

### B.6 Critères d’arrêt Phase 4

1. E2a et E2b entraînés sur Colab GPU, checkpoints + metadata sur Drive **et** copiés dans le dépôt local.  
2. Courbes + panel qualitatif val + grille de seuils val.  
3. Tableau val E0 / E1 / E2a / E2b (chiffres réels uniquement).  
4. Meilleur modèle Phase 4 désigné sur val (Dice après seuil val).  
5. **Stop** et demander approbation Phase 5 (ablations / AE write-up).

### B.7 Risques Phase 4

| Risque | Mitigation |
| --- | --- |
| OOM GPU | Réduire batch ; geler plus de couches ; mixed precision si stable |
| Surapprentissage fort (n=42) | Early stop strict ; comparer E2a vs E2b sur val ; ne pas toucher au test |
| Mismatch normalisation | Forcer `imagenet` pour MobileNetV2 ; interdire `scale_0_1` pour E2 |
| Perte de checkpoints Colab | Sauvegarde Drive à chaque best epoch + hash SHA-256 dans metadata |
| Fuite test | Garder `allow_test=False` ; audit script réutilisable |

---

## C. Décision requise

**Phase 3 est gelée.**  
**Phase 4 n’est pas démarrée.**  

Approuver explicitement la **Phase 4** pour autoriser l’implémentation du code + notebook Colab et, ensuite, l’entraînement GPU.
