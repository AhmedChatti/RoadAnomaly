# Contenu slide-par-slide (Canva) — 15–20 min

**Langue :** français  
**Durée cible :** ~17–18 min  
**Règle :** métriques **validation** uniquement ci-dessous ; cases test = « après Stage 5 » si absentes.

Sources figures : `reports/figures/`, `experiments/figures/`, Drive `…/phase4/*`.

---

## Slide 1 — Titre (1 min)
**Titre :** Détection et localisation des anomalies routières par segmentation sémantique supervisée  
**Sous-titre EN :** Road Anomaly Detection Through Supervised Semantic Segmentation  
**Visuel :** `reports/figures/05_image_mask_overlay_examples.png` ou `11_mask_correspondence_zoom.png`  
**Message :** prototype académique ; dataset RoadAnomaly (Lis et al., 2019).

## Slide 2 — Problème & motivation (1 min)
Obstacles hors vocabulaire Cityscapes-like ; besoin de **où** (masque), pas seulement d’un score image.  
Visuel : 3 scènes (animal / cône / pneu) via figs 05–06.

## Slide 3 — Objectifs & question de recherche (1 min)
BO : détecter, localiser, couvrir tailles/types, sortie vérifiable.  
RQ (synthèse) : dans quelle mesure transfer learning + fine-tuning contrôlé améliorent la généralisation sur un petit jeu ?  
Métriques : Dice/IoU/P/R/F1 ; accuracy secondaire.

## Slide 4 — Données (1 min 30)
60 images 1280×720 ; labels {0,2}→{0,1} ; ~9,85 % anomalies ; catégories filename.  
Figs : `01_category_counts.png`, `04_pixel_class_balance.png`.  
Limite : pas de sous-ensemble « normal only ».

## Slide 5 — Préparation (1 min 30)
Split **42/9/9** verrouillé (SHA `9a01c05c…`) ; entrée **512×288** ; aug. train only ; anti-fuite scènes répétées.  
Figs Phase 2 : `phase2_01`–`04`.

## Slide 6 — Méthodologie CRISP-DM (1 min)
Six phases ; sélection sur val ; test une fois ; pas de reproduction ICCV.

## Slide 7 — Modèles (2 min)
E0 fond ; E1 CNN scratch ; E2a U-Net+MobileNetV2 gelé ; E2b fine-tune dès `block_13_expand` (BN gelé).  
Loss : BCE pondérée (pos=15) + soft Dice.

## Slide 8 — Protocole expérimental (1 min)
Checkpoint = val mean Dice @ 0.5 ; seuil opérationnel grille val ; batch 4 (E2) / 2 (E1) ; seed 42 ; Colab T4.

## Slide 9 — Résultats validation (2 min)
Tableau (Dice mean/image) :

| | E0 | E1 | E2a | E2b |
|---|---:|---:|---:|---:|
| Dice | 0.00 | 0.57 | 0.70 | **0.78** |
| IoU | 0.00 | 0.45 | 0.58 | **0.69** |
| Seuil | 0.50 | 0.35 | 0.15 | 0.55 |

Source : `phase4_E0_E1_E2_val_comparison.csv`.  
Message : accuracy E0 trompeuse.

## Slide 10 — Courbes & qualitatif val (1 min 30)
Drive : `E2*_training_curves.png`, `E2*_val_qualitative.png` ; E1 curves locales.  
Mentionner figure E1 qualitative **régénérée**.

## Slide 11 — Résultats test (1 min)
**Uniquement si Stage 5 terminé** — sinon slide « protocole figé, exécution one-shot en cours / à coller ».  
Ne jamais inventer.

## Slide 12 — Analyse critique (1 min 30)
Petit n ; déséquilibre ; pas de lost-tire en test ; E2b meilleur en val mais incertitude élevée ; ≠ protocole ICCV.

## Slide 13 — Conclusion & perspectives (1 min)
Transfer learning utile en validation ; AE conventionnel non justifié ; démo Streamlit proposée non production.

## Slide 14 — Références
Lis et al., ICCV 2019 ; `credits.txt` ; Git SHA entraînement E2 `9e734a1…`.
