# Protocole scientifique et fondations du projet

**Projet :** Détection et localisation des anomalies routières par segmentation sémantique supervisée  
**Titre anglais :** Road Anomaly Detection Through Supervised Semantic Segmentation  
**Phase :** 1 — protocole et fondations (pas d’entraînement)  
**Date :** 8 novembre 2026  
**Décisions verrouillées :** approbation explicite des recommandations de la Phase 0

Ce document fixe le protocole expérimental. Toute modification ultérieure du split, du prétraitement, des métriques ou du critère de sélection de modèle doit être écrite ici **avant** d’être appliquée dans le code.

---

## 1. Formulation du problème

### 1.1 Trois niveaux à ne pas confondre

| Niveau | Définition dans ce projet |
| --- | --- |
| **Problème de vision** | Détection d’anomalies routières : objets inattendus et potentiellement dangereux dans une scène de route (animaux, rochers, cônes, pneus, véhicules atypiques). |
| **Tâche d’apprentissage** | Segmentation sémantique binaire au niveau du pixel. |
| **Paradigme** | Apprentissage **supervisé**, en utilisant les masques pixel du jeu RoadAnomaly. |

Ce n’est **pas** une classification d’image.  
Ce n’est **pas** une détection d’objets par boîtes englobantes.  
Ce n’est **pas** une reproduction de la méthode ICCV 2019 (resynthèse d’image).

### 1.2 Entrée / sortie

| | Spécification |
| --- | --- |
| Entrée | Une photographie RGB de scène routière |
| Sortie du modèle | Une carte de probabilités \(p \in [0,1]\) par pixel (activation sigmoïde) |
| Sortie rapportée | Un masque binaire après seuillage : fond = 0, anomalie = 1 |
| Supervision | Masque sémantique officiel : valeur `0` = fond, valeur `2` = anomalie |

Les préfixes de nom de fichier (`animals`, `cones`, …) servent à **stratifier** et à **analyser** les erreurs. Ils ne sont **pas** des classes prédites.

### 1.3 Objectifs métier (business)

1. Localiser l’obstacle inattendu dans l’image.
2. Produire une région qu’un humain peut vérifier.
3. Couvrir des tailles très différentes (du pneu ~0,37 % au troupeau ~42 %).
4. Rester honnête sur le périmètre : prototype académique, pas système de conduite autonome certifié.

### 1.4 Objectifs data science

1. Entraîner et comparer des modèles de segmentation binaire sur le split fixe 42 / 9 / 9.
2. Évaluer avec Precision, Recall, F1, IoU, Dice (accuracy uniquement comme métrique secondaire / contre-exemple).
3. Sélectionner le modèle et le seuil sur la **validation** uniquement.
4. Évaluer **une seule fois** le modèle final sur le test.
5. Documenter les limites et les cas d’échec.

---

## 2. Questions de recherche et hypothèses

### 2.1 Questions de recherche

**RQ1.** Un réseau convolutif entraîné de zéro sur 42 images peut-il localiser des anomalies mieux qu’une baseline triviale « tout fond » ?

**RQ2.** Un U-Net avec encodeur MobileNetV2 pré-entraîné (ImageNet) améliore-t-il l’IoU / le Dice de validation par rapport au CNN de zéro ?

**RQ3.** Sur ce petit jeu, le fine-tuning de l’encodeur apporte-t-il un gain durable face à un encodeur gelé, ou augmente-t-il surtout le surapprentissage ?

**RQ4.** Un autoencodeur de détection d’anomalies par reconstruction est-il scientifiquement défendable avec les seules 60 images RoadAnomaly ?

### 2.2 Hypothèses

| ID | Hypothèse | Critère de lecture |
| --- | --- | --- |
| H0 | La baseline tout-fond atteint ~90 % d’accuracy pixel et un IoU anomalie ≈ 0. | Vérification mécanique sur le test / val |
| H1 | Le CNN de zéro bat H0 en IoU / Dice validation, mais échoue souvent sur les petits objets. | Comparaison E0 vs E1 sur val |
| H2 | Le U-Net + transfer learning bat E1 en IoU / Dice validation. | Comparaison E1 vs E2 sur val |
| H3 | Le fine-tuning complet n’est pas forcément meilleur qu’un encodeur gelé ; on choisit la config selon la **validation**. | Ablation E2-frozen vs E2-finetune |
| H4 | Sans corpus « route normale », un autoencodeur classique n’est pas une baseline valide ici. | Analyse de faisabilité (pas d’expérience forcée) |

Les hypothèses seront acceptées ou rejetées **après** les expériences réelles. Aucun résultat chiffré n’est inventé dans ce document.

---

## 3. Relation avec CRISP-DM et le cahier des charges

| Étape CRISP-DM | État | Livrable projet |
| --- | --- | --- |
| Compréhension métier | Fait (Phase 0 + rapport existant) | §1 de ce protocole ; rapport technique |
| Compréhension des données | Fait | Audit, figures, notebook historique |
| Préparation des données | Split fait ; prétraitement d’entraînement à finaliser en Phase 2 | §5–6 |
| Modélisation | À venir (Phases 3–5) | Expériences E0–E3 |
| Évaluation | À venir (Phase 6) | §7 |
| Déploiement / démo | À venir (Phase 8) | Application Streamlit |

---

## 4. Référence ICCV 2019 — comparaison protocolaire

**Référence :** Lis, K., Nakka, K., Fua, P., and Salzmann, M. (2019). *Detecting the Unexpected via Image Resynthesis.* ICCV.  
Accès ouvert : https://openaccess.thecvf.com/content_ICCV_2019/html/Lis_Detecting_the_Unexpected_via_Image_Resynthesis_ICCV_2019_paper.html

### 4.1 Méthode du papier (vérifiée)

1. Un segmenteur sémantique (classes connues, type Cityscapes / BDD) produit une carte de labels.
2. Un générateur (pix2pixHD) **resynthétise** une image à partir de cette carte.
3. Les objets inattendus sont mal étiquetés → mal resynthétisés.
4. Un **réseau de discrépance** compare image originale, image resynthétisée et labels, et produit un score d’anomalie par pixel.
5. Entraînement **sans** voir les classes inattendues : simulation par échange aléatoire de labels d’instances sur Cityscapes.
6. RoadAnomaly (60 images) sert surtout à l’**évaluation** ; métrique rapportée : courbes ROC / **AUROC**.

### 4.2 Tableau de comparaison (à reprendre dans le rapport)

| Aspect | Référence ICCV 2019 | Notre projet |
| --- | --- | --- |
| Objectif principal | Détecter des classes jamais vues à l’entraînement | Détecter / localiser des obstacles routiers inattendus |
| Paradigme | Détection d’inattendu sans observer ces objets en train | Supervisé |
| Méthodologie | Resynthèse d’image + réseau de discrépance | Segmentation sémantique binaire |
| Supervision | Cityscapes (+ BDD pour les segmenteurs) ; anomalies simulées | Masques pixel RoadAnomaly |
| Prédiction | Score d’anomalie par pixel | Masque binaire anomalie / fond |
| Évaluation | AUROC sur Lost and Found et RoadAnomaly (sans entraîner sur ces anomalies) | Precision, Recall, F1, IoU, Dice sur split 42/9/9 |
| Jeu RoadAnomaly | Benchmark de test | Données d’entraînement, validation et test |

**Règle scientifique :** aucun tableau de résultats ne doit présenter nos scores comme une reproduction ou une comparaison directe aux AUROC du papier, sauf protocole identique (hors scope).

---

## 5. Protocole de partition des données

### 5.1 Décision verrouillée

**Conserver le split existant 42 / 9 / 9** sans redistribution.

Source de vérité : `data/processed/split_manifest.csv` et `reports/tables/split_assignment.csv`.

| Split | n | Usage autorisé |
| --- | ---: | --- |
| `train` | 42 | Optimisation des poids ; augmentation |
| `val` | 9 | Early stopping, sélection de modèle, sélection de seuil, ablations |
| `test` | 9 | **Une** évaluation finale du modèle choisi ; jamais pour le tuning |

### 5.2 Règles anti-fuite (déjà appliquées)

Les groupes suivants restent entièrement dans `train` :

- `lost_tires03`, `lost_tires08`, `lost_tires09`
- `obstacles03`–`obstacles06` (sculpture cardboard czech hedgehog)
- `obstacles12_rocks2`, `rocks4`, `rocks5`, `rocks6`

Aucun `scene_group` ne traverse deux splits (vérifié : 0).

### 5.3 Limites assumées du split

- Test de **9 images** → forte variance ; publier les scores **par image**.
- Aucun `lost_tires` en validation/test → pas de score test « pneus ».
- Médiane d’aire d’anomalie : val ≈ 10,0 % vs test ≈ 4,3 % → la validation est un peu plus « facile ».
- Les préfixes de fichier sont déséquilibrés (31 animals / 60).

### 5.4 Hash de reproductibilité

Avant chaque entraînement, enregistrer un empreinte du manifeste (ex. SHA-256 du CSV trié par `frame_id`). Si le hash change, le run n’est pas comparable aux précédents.

---

## 6. Protocole de prétraitement

### 6.1 Décisions verrouillées

| Paramètre | Décision |
| --- | --- |
| Taille d’entrée **d’entraînement / inférence** | **512 × 288** (ratio 16:9, divisible par 32) |
| Letterbox 512 × 512 existant | Conservé comme artefact historique sous `data/processed/` ; **non** utilisé comme tenseur d’entrée par défaut |
| Masques originaux (`frames/`) | **Jamais modifiés** |
| Encodage cible pour le modèle | **0 = fond, 1 = anomalie** |
| Fichiers prepared actuels (0 / 255) | Peuvent rester ; le **loader** remappe `>0 → 1` |
| Interpolation image | Bilinéaire (ou équivalent continu) |
| Interpolation masque | **Nearest-neighbour** uniquement |
| Normalisation images | `[0, 1]` pour le CNN de zéro ; moyenne/écart-type **ImageNet** pour MobileNetV2 (documenter laquelle dans chaque run) |
| Même pipeline | Train, val, test, Streamlit : **une seule** fonction d’inférence partagée |

### 6.2 Pipeline déterministe (spécification)

Pour chaque couple `(image.webp, labels_semantic.png)` :

1. Charger l’image RGB et le masque sémantique.
2. Vérifier `shape` identique (1280×720).
3. Remapper le masque : `2 → 1`, tout autre non-zéro invalide → erreur ; `0` reste `0`.
4. Redimensionner à **512 × 288** (pas de padding).
5. Normaliser l’image selon la config de l’expérience.
6. Ne pas appliquer d’augmentation hors split `train`.

Alternative d’ablation autorisée plus tard (Phase 5) : entraîner aussi sur le letterbox 512×512 pour mesurer l’effet du padding — **une** comparaison, pas une recherche en grille.

### 6.3 Augmentation (train uniquement)

Transformations **synchronisées** image + masque :

| Transformation | Autorisée | Notes |
| --- | --- | --- |
| Flip horizontal | Oui | Sémantiquement acceptable pour une route |
| Flip vertical | Non (par défaut) | Invente une route à l’envers |
| Petite rotation (±5 à 10°) | Oui, limitée | Interpolation nearest sur le masque |
| Translation / zoom léger | Oui, limitée | Éviter de faire disparaître un petit objet |
| Luminosité / contraste | Oui | Photométrique ; masque inchangé |
| Cutout agressif / mélange d’images | Non | Trop risqué avec n = 42 |

Probabilités et amplitudes exactes seront fixées dans `configs/default.json` et figées au début de la Phase 3.

### 6.4 Opérations destructives interdites

- Ne pas appeler `shutil.rmtree` sur `data/processed` depuis le code d’entraînement.
- Ne pas régénérer le split sans nouvelle approbation écrite.
- Ne pas écraser le notebook historique `01_exploration_and_preprocessing.ipynb`.

---

## 7. Protocole d’évaluation

### 7.1 Définitions (pixels)

Sur le masque binaire seuillé :

- **TP** : anomalie prédite et vraie  
- **FP** : anomalie prédite, fond vrai  
- **FN** : fond prédit, anomalie vraie  
- **TN** : fond prédit et vrai  

### 7.2 Métriques obligatoires

Soit \(P = TP/(TP+FP)\), \(R = TP/(TP+FN)\) :

| Métrique | Formule | Rôle |
| --- | --- | --- |
| Precision | \(P\) | Fausses alarmes |
| Recall | \(R\) | Obstacles manqués |
| F1 | \(2PR/(P+R)\) | Compromis |
| IoU | \(TP/(TP+FP+FN)\) | Chevauchement |
| Dice | \(2TP/(2TP+FP+FN)\) | Équivalent F1 pixel pour un masque binaire |
| Accuracy | \((TP+TN)/(TP+TN+FP+FN)\) | **Secondaire** ; montrer le piège avec E0 |

### 7.3 Dénominateurs nuls

| Cas | Convention |
| --- | --- |
| `TP+FP = 0` (aucune anomalie prédite) | Precision = 1.0 si `TP+FN = 0`, sinon 0.0 |
| `TP+FN = 0` (masque GT vide) | Ne doit pas arriver sur RoadAnomaly ; si arrivée : Recall = 1.0 si `TP+FP = 0`, sinon 0.0 ; logger un avertissement |
| Union IoU nulle | IoU = 1.0 si les deux masques sont vides, sinon 0.0 |

Ces conventions seront unit-testées (Phase 2).

### 7.4 Agrégation

Pour chaque split d’évaluation :

1. **Métriques globales pixel** : compter TP/FP/FN/TN sur toutes les images concaténées.
2. **Moyenne par image** : moyenne des IoU (etc.) des 9 (ou 42) images, avec min / médiane / max.

Les deux sont rapportées. On ne laisse pas le fond dominer la lecture.

### 7.5 Seuillage

- Sortie modèle : probabilités sigmoïde.
- Seuil par défaut de reporting : **0,5**.
- Sélection éventuelle d’un seuil optimal : **uniquement sur validation** (ex. maximiser le Dice moyen par image).
- Le seuil choisi est alors **figé** pour le test et pour Streamlit.
- Interdiction de retuner le seuil après avoir vu le test.

### 7.6 Sélection de modèle

Critère principal de checkpoint : **Dice moyen par image sur validation** (ou IoU moyen par image — un seul critère, fixé avant l’entraînement E2 : **Dice moyen par image val**).

Early stopping sur la même métrique. Patience et nombre d’epochs max : dans `configs/default.json`.

### 7.7 Évaluation qualitative

Pour chaque image évaluée (au minimum val, puis test final) :

1. Image originale  
2. Masque vérité terrain  
3. Masque prédit  
4. Superposition  
5. Carte d’erreurs (FP / FN) si utile  

Inclure réussites **et** échecs (petits objets, grands objets, faux positifs, faux négatifs).

### 7.8 Test final

Une seule passe test après :

1. choix d’architecture / hyperparamètres sur val ;  
2. choix du seuil sur val ;  
3. gel du checkpoint.

---

## 8. Matrice d’expériences

| ID | Nom | Entraînement | Objectif |
| --- | --- | --- | --- |
| E0 | Baseline tout-fond | Aucun | Montrer le piège de l’accuracy |
| E1 | CNN / FCN from scratch | Train 42 | RQ1 / H1 |
| E2a | U-Net MobileNetV2, encodeur gelé | Train 42 + ImageNet | RQ2 / H2 |
| E2b | U-Net MobileNetV2, fine-tuning partiel | Train 42 | RQ3 / H3 |
| E3 | Ablations limitées | Selon besoin | Loss BCE vs BCE+Dice ; 512×288 vs letterbox (si temps) |
| E4 | Autoencodeur | **Non implémenté** comme baseline concurrente | Voir §9 |

Toutes les expériences valides sont journalisées (voir §10). Les échecs d’entraînement sont documentés, pas effacés.

---

## 9. Autoencodeurs — décision de faisabilité

### 9.1 Distinction

| Approche | Idée | Données nécessaires |
| --- | --- | --- |
| Reconstruction (AE) | Erreur de reconstruction élevée ⇒ anomalie | Beaucoup de scènes **normales** sans obstacle |
| Segmentation supervisée | Apprendre le masque d’anomalie | Images **avec** masques |

### 9.2 Constat sur RoadAnomaly

- 60 / 60 images contiennent des pixels d’anomalie.  
- Aucun sous-ensemble « route vide » identifiable.  
- Entraîner un AE sur ces 60 images risque d’apprendre à reconstruire aussi les obstacles.  
- Le papier ICCV 2019 critique déjà les baselines autoencodeur / RBM pour ce type de tâche lorsqu’elles ne ciblent pas correctement l’inattendu.  
- Une comparaison IoU supervisé vs score de reconstruction n’est pas équitable sans protocole commun.

### 9.3 Décision (approuvée)

**Ne pas** fabriquer un jeu « normal » artificiel.  
**Ne pas** présenter un AE entraîné sur RoadAnomaly seul comme baseline concurrente valide.  
**Rédiger** dans le rapport et le notebook final une section :

**« Évaluation de faisabilité des autoencodeurs et justification scientifique »**

Une extension AE ne serait envisageable que si un corpus externe de routes normales (ex. patches Cityscapes) était explicitement ajouté au scope — **hors scope actuel**.

---

## 10. Suivi d’expériences et reproductibilité

### 10.1 Fichier de configuration

Source des hyperparamètres par défaut : `configs/default.json`.

### 10.2 Métadonnées obligatoires par run

Chaque run écrit un JSON dans `experiments/metadata/` :

```text
experiment_id, model_name, architecture, input_height, input_width,
split_manifest_sha256, seed, optimizer, learning_rate, batch_size,
epochs_planned, epochs_run, loss, metrics_monitored,
augmentation_config, checkpoint_selection_criterion,
best_epoch, best_val_metric_name, best_val_metric_value,
threshold_selection, threshold_value,
hardware, software_versions, duration_seconds,
checkpoint_path, notes, status
```

Modèle de fichier : `experiments/metadata/_template.json`.

### 10.3 Graines et versions

- Fixer `seed` (NumPy, TensorFlow, Python `random`).  
- Enregistrer versions : Python, TensorFlow, Keras, NumPy, Pillow.  
- Entraînement prévu sur **Google Colab** (GPU). Machine locale : développement, tests légers, Streamlit, documentation.  
- Téléchargement des poids ImageNet MobileNetV2 : **autorisé**.

### 10.4 Artefacts

| Artefact | Emplacement prévu |
| --- | --- |
| Checkpoints Keras | `checkpoints/` |
| Courbes et figures d’expérience | `experiments/figures/` |
| Tables de résultats | `experiments/results/` |
| Données brutes | `frames/` (non versionnées en Git si trop lourdes ; présentes localement) |

### 10.5 Règles d’intégrité

- Ne jamais inventer de métriques, courbes ou captures.  
- Ne jamais affirmer qu’un entraînement a eu lieu sans artefact vérifiable.  
- Distinguer clairement : littérature / implémentation / observation / interprétation.

---

## 11. Stack technique (figée)

| Composant | Choix |
| --- | --- |
| Langage | Python |
| Deep learning | TensorFlow / Keras |
| Vision / tableaux | Pillow, NumPy, pandas |
| Visualisation | Matplotlib |
| Métriques | NumPy (+ scikit-learn si utile) |
| Démo | Streamlit |
| Entraînement | Google Colab |
| Documentation projet | **Français** |
| Identifiants code | Anglais |

Pas de bascule vers PyTorch. Pas d’ajout de dépendance sans nouvelle approbation (sauf ce qui est déjà listé dans le protocole Phase 2).

---

## 12. Structure cible (évolution progressive)

Le dépôt existant est **préservé**. Les ajouts se font **à côté** :

- conserver `src/road_anomaly.py`, `notebooks/01_...`, `reports/01_...`, `data/processed/` ;  
- ajouter modules d’entraînement, `configs/`, `experiments/`, `checkpoints/`, `app/`, `tests/` ;  
- notebook final français **nouveau** ; notebook actuel → matériel historique.

Le détail d’implémentation des loaders appartient à la **Phase 2**.

---

## 13. Décisions Phase 0 — registre figé

| # | Décision | Statut |
| --- | --- | --- |
| 1 | Préserver le split 42/9/9 | Approuvé |
| 2 | Entrée 512×288 sans padding | Approuvé |
| 3 | Remapper vers 0/1 dans le loader ; ne pas altérer les originaux | Approuvé |
| 4 | Autoriser les poids ImageNet MobileNetV2 | Approuvé |
| 5 | Autoencodeur : section de faisabilité seulement | Approuvé |
| 6 | Enchaîner Phase 1 | Approuvé |
| 7 | Initialiser Git + `.gitignore` | Approuvé |

---

## 14. Critères de passage à la Phase 2

La Phase 2 (pipeline de données) peut démarrer lorsque ce protocole est accepté, car il fixe :

- formulation et RQ/hypothèses ;  
- split ;  
- prétraitement 512×288 et masques 0/1 ;  
- évaluation et sélection ;  
- matrice d’expériences ;  
- exclusion motivée de l’AE ;  
- format de métadonnées.

**Phase 1 = documentation et fondations. Aucun modèle n’a été entraîné ici.**

---

## Références

1. Lis, K., Nakka, K., Fua, P., Salzmann, M. (2019). Detecting the Unexpected via Image Resynthesis. *ICCV*.  
2. Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.  
3. Dataset RoadAnomaly — notes locales `readme.md`, `credits.txt`.  
4. Audit Phase 0 — `docs/INITIAL_PROJECT_AUDIT.md`.
