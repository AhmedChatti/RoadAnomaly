# Structure du rapport technique (méthodologie CRISP-DM)

**Langue :** français  
**Volume cible :** environ 8–15 pages (hors annexes), sauf consigne contraire de l’université  
**Cohérence :** aligné sur `docs/02_protocole_scientifique_phase1.md`  
**Résultats expérimentaux :** sections marquées placeholder tant qu’aucun run n’existe

Ce document est un **plan de rédaction**, pas le rapport final. Le rapport final sera produit en Phase 9 à partir d’évidences réelles.

---

## Page de garde (hors CRISP-DM)

- Titre FR + titre EN  
- Noms des membres du groupe  
- Module Deep Learning / établissement / année  
- Encadrant  
- Date  
- Mention : prototype académique ; dataset RoadAnomaly (Lis et al., 2019)

---

## Résumé / Abstract (hors CRISP-DM, ~½ page)

**À écrire après les expériences.**  
Contenu prévu : problème, approche (segmentation supervisée), données (60 images, split 42/9/9), modèles (baseline, CNN, U-Net+MobileNetV2), principaux scores `[Résultats à compléter après les expériences]`, limite majeure (petit jeu, pas de routes « normales » pour un AE).

---

## Introduction (hors CRISP-DM, ~1 page)

1. Motivation : objets hors vocabulaire des segmenteurs classiques.  
2. Positionnement : détection d’anomalies routières ≠ classification ; ≠ détection par boîtes.  
3. Contributions du projet (préparation, modèles, évaluation, démo).  
4. Différence explicite avec Lis et al. (2019).  
5. Annonce de la structure CRISP-DM du rapport.

**Figures suggérées :** aucune obligatoire ; éventuellement Fig. 11 (correspondance masque) en teaser.

---

# Phase CRISP-DM 1 — Business Understanding

### 1.1 Contexte et problème

- Scènes routières contenant des obstacles inattendus (animaux, rochers, cônes, pneus, véhicules atypiques).  
- Besoin de **localiser** la région, pas seulement de dire « anomalie / pas d’anomalie ».

**Source déjà rédigée :** `reports/01_business_and_data_understanding.md` §1.

### 1.2 Objectifs métier (Business Objectives — BO)

Reprendre les 4 BO du protocole :

1. Détecter l’obstacle inattendu.  
2. Le localiser dans l’image.  
3. Couvrir tailles et types variés.  
4. Produire une sortie vérifiable par un humain ; périmètre prototype.

### 1.3 Objectifs data science (Data Science Objectives — DSO)

- Segmentation binaire supervisée.  
- Métriques : Precision, Recall, F1, IoU, Dice (+ accuracy secondaire).  
- Split fixe ; sélection sur val ; test unique.

### 1.4 Résultats attendus

- Notebook reproductible, modèles comparés, application de démonstration, analyse critique.  
- **Non attendu :** déploiement production, freinage automatique, certification.

### 1.5 Pertinence pratique

Aide à la perception pour ADAS / recherche ; jeu public hétérogène → pertinence **illustrative**, pas flotte instrumentée.

**Tableau BO ↔ DSO ↔ métrique** (obligatoire dans le rapport).

---

# Phase CRISP-DM 2 — Data Understanding

### 2.1 Description du jeu RoadAnomaly

- 60 images WebP RGB 1280×720.  
- Masques sémantiques : `0` fond, `2` anomalie.  
- Instance masks pour 46/60.  
- Origine : Lis et al., ICCV 2019 ; images tierces (`credits.txt`).

### 2.2 Structure des dossiers

Décrire `frames/`, `*.labels/labels_semantic.png`, `frame_list.json`.

### 2.3 Analyse exploratoire

**Figures à inclure (déjà générées) :**

| Figure | Message |
| --- | --- |
| `01_category_counts.png` | Déséquilibre des préfixes |
| `02_resolution_and_obstacle_size.png` | Résolution fixe ; taille d’obstacle variable |
| `03_anomaly_area_by_category.png` | Petits objets (cones, tires) |
| `04_pixel_class_balance.png` | 9,85 % anomalies → piège de l’accuracy |
| `05_...` / `11_...` | Correspondance image–masque |
| `06_...` | Extrêmes 0,37 % vs 42,3 % |
| `10_...` | Scènes répétées |

### 2.4 Limites et biais

- Pas d’image sans anomalie.  
- Mélange Wikimedia / dashcam / photos atypiques.  
- Filigranes (pneus).  
- Catégories déséquilibrées.  
- Annotations parfois plus grossières sur parties fines (jambes).

**Tableaux :** inventaire par catégorie ; stats d’aire d’anomalie (depuis `frame_inventory.csv`).

---

# Phase CRISP-DM 3 — Data Preparation

### 3.1 Conversion des masques

Officiel `0/2` → modèle `0/1` dans le loader. Ne pas altérer `frames/`.

### 3.2 Redimensionnement

**Décision Phase 1 :** 512×288 (16:9, divisible par 32), **sans** padding letterbox pour l’entraînement.  
Documenter aussi l’artefact historique 512×512 letterbox (`data/processed/`) et pourquoi le padding (43,75 % de noir) a été écarté comme entrée par défaut.

### 3.3 Normalisation

- CNN from scratch : `[0, 1]`.  
- MobileNetV2 : statistiques ImageNet.

### 3.4 Augmentation

Train uniquement ; flip horizontal ; rotations/zoom/luminosité limités ; masque en nearest-neighbour synchronisé.

### 3.5 Split train / val / test

42 / 9 / 9 ; groupes répétés en train ; hash `assignment_sha256` ; figure `07_split_composition.png`.

### 3.6 Prévention des fuites

- Pas de split aléatoire cassant une séquence.  
- Test jamais utilisé pour le seuil ou early stopping.  
- Même prétraitement train/val/test/app.

**Figure :** schéma du pipeline (à dessiner en Phase 2/9).  
**Figure existante utile :** `08_preprocessing_letterbox.png` comme *contre-exemple pédagogique* du padding, puis expliquer le choix 512×288.

---

# Phase CRISP-DM 4 — Modeling

### 4.1 Stratégie expérimentale progressive

| Expérience | Rôle |
| --- | --- |
| E0 Baseline tout-fond | Piège métrique |
| E1 CNN / FCN from scratch | Chapitre CNN sans transfer |
| E2a U-Net + MobileNetV2 gelé | Modèle principal candidat |
| E2b Fine-tuning encodeur | Ablation |
| E3 Ablations limitées | Loss / résolution si temps |
| E4 Autoencodeur | **Non entraîné** ; § faisabilité |

### 4.2 Architecture E1

Décrire encodeur–décodeur léger, sigmoïde, justification CPU/Colab et n=42.

### 4.3 Architecture E2 (U-Net + MobileNetV2)

- Skip connections.  
- Transfer learning ImageNet.  
- Gel vs fine-tune (LR plus faible sur l’encodeur).

### 4.4 Fonction de perte et hyperparamètres

Référencer `configs/default.json` : Adam, batch, epochs, early stopping, seed 42, critère = Dice moyen val.

### 4.5 Autoencodeurs — justification d’exclusion

Section dédiée (exigence professeur + Phase 1) : pas de scènes normales ; risque de reconstruire les anomalies ; papier ICCV critique les AE naïfs ; pas de corpus externe dans le scope.

**Figures :** diagrammes d’architecture (à produire avec le code, Phase 3–4).  
**Tableaux :** hyperparamètres par expérience.  
`[Résultats d’entraînement à compléter après les expériences]`

---

# Phase CRISP-DM 5 — Evaluation

### 5.1 Métriques

Définir Precision, Recall, F1, IoU, Dice, Accuracy ; conventions dénominateur nul (protocole §7.3).

### 5.2 Protocole

Val pour sélection ; seuil sur val ; test unique.

### 5.3 Résultats quantitatifs

| Tableau | Contenu |
| --- | --- |
| T1 | E0 / E1 / E2a / E2b sur **validation** |
| T2 | Modèle final sur **test** (une ligne + par image) |
| T3 | Ablations si réalisées |

`[Résultats à compléter après les expériences]`

### 5.4 Résultats qualitatifs

Grilles : image / GT / prédiction / overlay / erreurs.  
Inclure succès **et** échecs (petits objets, faux positifs).

### 5.5 Analyse critique

Forces, faiblesses, surapprentissage, variance du test n=9, absence de pneus en test, non-généralisabilité « tout objet inattendu du monde réel ».

### 5.6 Améliorations possibles

Plus de données, cross-validation / bootstrap, calibration, autres encodeurs, évaluation hors distribution — sans promettre l’impossible.

---

# Phase CRISP-DM 6 — Deployment (si applicable)

### 6.1 Démonstration

Application Streamlit : upload → prétraitement partagé → checkpoint réel → masque + overlay.

### 6.2 Architecture logicielle de la démo

Notebook / `src` / checkpoint / app ; une seule fonction d’inférence.

### 6.3 Usage pratique envisageable

Prototype de laboratoire / pédagogie / pré-filtre humain — **pas** freinage.

### 6.4 Limites et disclaimer

Petit jeu, biais web, pas de mesure de fausses alarmes sur routes vides, pas de certification.

`[Captures d’écran Streamlit à compléter après la Phase 8]`

---

## Conclusion (hors CRISP-DM)

Synthèse BO/DSO, réponse aux RQ1–RQ4, ouverture.

`[À écrire après interprétation des résultats réels]`

---

## Références

1. Lis et al., ICCV 2019.  
2. Ronneberger et al., U-Net, 2015.  
3. Sandler et al., MobileNetV2, 2018 (si cite l’encodeur).  
4. Notes dataset `readme.md` / `credits.txt`.  
5. Supports de cours du module (CNN, segmentation, hyperparamètres) — à citer selon les consignes locales.

---

## Annexes suggérées

- A. Liste complète train/val/test.  
- B. Hash du split.  
- C. Model card.  
- D. Versions logicielles Colab.  
- E. Section AE (si trop longue pour le corps).

---

## Checklist de rédaction (auteur)

- [ ] Les six phases CRISP-DM apparaissent comme titres principaux.  
- [ ] Aucun chiffre d’expérience inventé.  
- [ ] Distinction claire papier ICCV vs notre protocole.  
- [ ] Termes alignés sur le protocole (512×288, 0/1, Dice val, etc.).  
- [ ] Figures numérotées et référencées dans le texte.
