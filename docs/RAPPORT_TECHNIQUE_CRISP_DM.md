# Rapport technique — Détection et localisation des anomalies routières par segmentation sémantique supervisée

**Module :** Deep Learning  
**Dataset :** RoadAnomaly (Lis et al., ICCV 2019)  
**Langue :** français  
**Date :** 8 novembre 2026  
**SHA Git (entraînement E2) :** `9e734a1d0af48342f92c084216b1dd1d4c254741`  
**Verrou de split :** `9a01c05cfc7d33783c4f64b91b0ae423a4d3ad57439fd71ffc029a8f04585c78`

> **Avertissement méthodologique.** Les métriques de **validation** ci-dessous sont mesurées. Les métriques de **test** ne sont reportées que si Stage 5 a produit `phase4_final_test_comparison.csv`. Aucun chiffre manquant n’est inventé.

---

## Résumé

Ce projet construit un système de **segmentation sémantique binaire supervisée** pour localiser des obstacles routiers inhabituels (animaux, cônes, débris, etc.). Sur un jeu de 60 images (split 42/9/9, résolution d’entrée 512×288), nous comparons une baseline triviale (E0), un CNN entraîné de zéro (E1) et un U-Net avec encodeur MobileNetV2 pré-entraîné (E2a gelé, E2b fine-tuning partiel). En **validation**, E2b obtient le meilleur Dice moyen par image (**0,778** au seuil 0,55), devant E2a (**0,705**) et E1 (**0,569**). L’accuracy d’E0 (~0,86) illustre le piège du déséquilibre de classes. Ce travail **ne reproduit pas** Lis et al. (2019) : leur méthode est une détection d’anomalies par resynthèse d’image ; la nôtre est une segmentation supervisée sur RoadAnomaly.

---

## 1. Introduction

Les segmenteurs routiers classiques (Cityscapes, etc.) échouent sur des objets hors vocabulaire. RoadAnomaly fournit 60 scènes annotées au pixel pour ce cas. Notre contribution est un pipeline reproductible TensorFlow/Keras, des expériences E0–E2 documentées, et une analyse critique des limites d’un très petit jeu.

---

## 2. Travaux connexes et distinction ICCV 2019

Lis, Nakka, Fua et Salzmann (ICCV 2019) détectent l’inattendu via **resynthèse d’image** et carte de discordance, typiquement après entraînement sur des scènes « normales » (ex. Cityscapes).  
**Différences :**

| | Lis et al. 2019 | Ce projet |
| --- | --- | --- |
| Supervision | Anomalie non supervisée / resynthèse | Masques binaires supervisés |
| Données d’entraînement | Scènes normales externes | Split de RoadAnomaly (42 images) |
| Sortie | Carte de discordance | Masque 0/1 |
| Comparaison de scores | Non applicable directement | Interne au protocole 42/9/9 |

---

## 3. CRISP-DM — Business Understanding

### 3.1 Problème
Localiser des obstacles inattendus dans une image de route pour un prototype académique de perception.

### 3.2 Objectifs métier (BO)
1. Détecter l’obstacle inattendu.  
2. Le localiser (région pixel).  
3. Couvrir tailles et types variés.  
4. Produire une sortie inspectable par un humain.

### 3.3 Objectifs data science (DSO)
Segmentation binaire ; métriques Precision, Recall, F1, IoU, Dice ; sélection sur validation ; test unique.

### 3.4 Questions de recherche
- **RQ1 :** un CNN from scratch bat-il E0 ?  
- **RQ2 :** un encodeur ImageNet gelé améliore-t-il E1 ?  
- **RQ3 :** le fine-tuning partiel apporte-t-il un gain vs encodeur gelé ?  
- **RQ4 :** un AE « normal-only » est-il défendable ? → **non** (60/60 images contiennent des anomalies).

### 3.5 Critères de succès
Amélioration claire du Dice/IoU val vs E0 ; protocole reproductible ; analyse honnête des limites. Pas de certification ADAS.

---

## 4. CRISP-DM — Data Understanding

- 60 images WebP 1280×720 ; 60 masques sémantiques `{0,2}`.  
- ~9,85 % pixels anomalie / ~90,15 % fond.  
- Catégories (préfixes) : animals 31, obstacles 15, cones 8, lost tires 3, vehicles 3.  
- Scènes répétées / quasi-doublons ; pas de sous-ensemble normal-only.  
- Licences : images tierces (`credits.txt`) ; annotations CC BY-SA 4.0.  

Figures : `reports/figures/01`–`06`, `10`, `11`.

---

## 5. CRISP-DM — Data Preparation

Décisions **figées** (`docs/02`, `configs/default.json`) :

- Split **42 / 9 / 9**, anti-fuite par groupes de scènes.  
- Hash d’affectation `9a01c05c…04585c78`.  
- Redimensionnement **512×288** (pas le letterbox historique 512×512 du rapport exploratoire).  
- Conversion loader `{0,2}→{0,1}` ; originaux intacts.  
- Normalisation E1 : `scale_0_1` ; E2 : `mobilenet_v2` (`uint8/127.5−1`).  
- Augmentation **train only**.  

Pipeline : `src/preprocessing.py`, `augmentation.py`, `dataset.py` ; doc `docs/08_phase2_data_pipeline.md`.

---

## 6. CRISP-DM — Modeling

### 6.1 E0 — Baseline tout fond
Prédiction constante 0. Sert à démontrer l’échec de l’accuracy.

### 6.2 E1 — CNN from scratch
U-Net-lite (`base_filters=8`, **sans BatchNorm** après collapse observé à batch 2). Loss BCE pondérée (pos=15) + soft Dice ; Adam 1e-3 ; batch 2 ; seed 42. Checkpoint : val mean Dice @ 0,5. Run canonique `E1_cnn_scratch_20261108_1510` (meilleure époque 66).

### 6.3 E2a — U-Net + MobileNetV2 (encodeur gelé)
ImageNet ; encodeur gelé ; décodeur entraîné ; lr 1e-3 ; batch 4 ; BN gelées.  
ID : `E2a_unet_mobilenetv2_frozen_20261108_1714`.

### 6.4 E2b — Fine-tuning sélectif
Initialisé depuis le meilleur E2a ; dégel exact depuis `block_13_expand` ; BN gelées ; lr 1e-4.  
ID : `E2b_unet_mobilenetv2_finetune_20261108_1721`.

Config : `configs/phase4.json`. Entraînement Colab Tesla T4.

---

## 7. CRISP-DM — Evaluation

### 7.1 Méthode
- Checkpointing : Dice moyen/image validation au seuil **0,5**.  
- Seuil opérationnel : grille validation 0,1–0,9.  
- Test : **une fois**, après gel (`FINAL_MODEL_SELECTION_FREEZE.json`).

### 7.2 Résultats validation (mean per image)

| Exp | Seuil op. | Dice | IoU | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | 0,50 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,862 |
| E1 | 0,35 | 0,569 | 0,445 | 0,522 | 0,850 | 0,569 | 0,867 |
| E2a | 0,15 | 0,705 | 0,584 | 0,678 | 0,835 | 0,705 | 0,935 |
| E2b | 0,55 | **0,778** | **0,686** | **0,846** | 0,793 | **0,778** | **0,954** |

Source : `experiments/results/phase4/phase4_E0_E1_E2_val_comparison.csv` et métadonnées Phase 3/4.

### 7.3 Réponses provisoires aux RQ (validation)
- **RQ1 :** oui — E1 ≫ E0 en Dice/IoU.  
- **RQ2 :** oui — E2a > E1 en Dice val.  
- **RQ3 :** oui sur val — E2b > E2a ; risque de surapprentissage non exclu (n=42).  
- **RQ4 :** AE normal-only non justifié.

### 7.4 Résultats test
Voir `docs/14_final_test_evaluation.md`. **Ne pas confondre avec la validation.**

### 7.5 Limites
n_val=n_test=9 ; pas de lost-tire en test ; scènes web biaisées ; seuils différents entre modèles ; pas d’IC inventés.

---

## 8. CRISP-DM — Deployment

**Implémenté :** pipeline Colab/Drive pour entraînement et artefacts.  
**Non implémenté :** application Streamlit grand public.  
**Proposition :** démo locale upload → même prétraitement → checkpoint E2b → masque + overlay, avec disclaimer « prototype académique ».

---

## 9. Conclusion

Le transfer learning MobileNetV2 améliore nettement les scores de **validation** par rapport au CNN from scratch, et le fine-tuning partiel (E2b) donne le meilleur Dice val observé. Ces conclusions restent conditionnées à un jeu minuscule et doivent être confirmées par l’évaluation test one-shot. Le projet documente un protocole reproductible et refuse les comparaisons fallacieuses avec ICCV 2019.

---

## Références

1. Lis, K., Nakka, K., Fua, P., & Salzmann, M. (2019). Detecting the Unexpected via Image Resynthesis. ICCV.  
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net. MICCAI.  
3. Sandler, M., et al. (2018). MobileNetV2. CVPR.  
4. Dataset credits : `credits.txt` ; `readme.md`.

## Annexes (fichiers projet)

- Protocole : `docs/02_protocole_scientifique_phase1.md`  
- Analyse Phase 4 : `docs/13_phase4_results_and_analysis.md`  
- Index présentation : `docs/PRESENTATION_MATERIALS_INDEX.md`, `docs/15_presentation_slide_content.md`
