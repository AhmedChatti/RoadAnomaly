# Plan de présentation orale (15–20 minutes)

**Public :** jury académique technique (Deep Learning)  
**Langue :** français  
**Durée cible :** 18 minutes de parole + ~2 minutes de marge / transition  
**Support final :** à produire en Phase 9 (`reports/presentation/`)  
**Règle :** aucune métrique inventée ; slides de résultats en placeholder jusqu’aux expériences

Durée parlée totale planifiée : **≈ 17 min 30 s** (hors questions du jury).

---

## Vue d’ensemble du déroulé

| Bloc | Contenu | Temps |
| --- | --- | ---: |
| 1 | Titre et cadrage | 1 min |
| 2 | Problème, BO–DSO | 2 min 30 s |
| 3 | Données et annotations | 3 min |
| 4 | Préparation et anti-fuite | 2 min 30 s |
| 5 | Modèles et méthodologie | 3 min 30 s |
| 6 | Résultats et interprétation | 3 min |
| 7 | Limites, AE, ICCV, conclusion | 2 min |
| | **Total** | **≈ 17 min 30 s** |

---

## Slide 1 — Titre et contexte

| | |
| --- | --- |
| **Titre** | Détection et localisation des anomalies routières par segmentation sémantique supervisée |
| **Contenu** | Titre FR + EN ; noms ; module ; mention « prototype académique » ; dataset RoadAnomaly |
| **Visuel** | Une image + overlay GT (ex. Fig. 11 ou reindeer) |
| **Message clé** | Nous localisons des obstacles inattendus au pixel près, de façon supervisée. |
| **Temps** | 1 min |
| **Oral** | Présenter l’équipe en 10 s, lire le titre, montrer l’exemple : « le masque rouge suit l’objet, ce n’est pas une classe d’image entière ». |

---

## Slide 2 — Problème et motivation

| | |
| --- | --- |
| **Titre** | Pourquoi ce problème ? |
| **Contenu** | Objets hors classes Cityscapes-like ; enjeu sécurité ; besoin de *où* ; pas classification / pas boîtes seules |
| **Visuel** | Montage 3 scènes (animal, cône, pneu) sans jargon excessif |
| **Message clé** | L’anomalie routière est rare, hétérogène, et doit être localisée. |
| **Temps** | 1 min 15 s |
| **Oral** | Relier à un scénario ADAS : un drapeau image-level ne suffit pas pour freiner ou contourner. |

---

## Slide 3 — Objectifs métier et data science (BO–DSO)

| | |
| --- | --- |
| **Titre** | Objectifs métier et objectifs data science |
| **Contenu** | Tableau BO ↔ DSO ↔ métrique (Precision/Recall/IoU/Dice) ; accuracy explicitement secondaire |
| **Visuel** | Tableau 2 colonnes BO / DSO |
| **Message clé** | Chaque objectif métier impose une métrique ; l’accuracy seule est un piège. |
| **Temps** | 1 min 15 s |
| **Oral** | Insister : « 90 % d’accuracy en prédisant tout fond = échec total (IoU = 0) ». Citer Fig. 4 mentalement. |

---

## Slide 4 — Dataset RoadAnomaly

| | |
| --- | --- |
| **Titre** | Données : 60 scènes, masques pixel |
| **Contenu** | 1280×720 ; labels 0/2 ; 9,85 % anomalies ; catégories filename ; 46 instance masks |
| **Visuel** | `01_category_counts.png` + `04_pixel_class_balance.png` |
| **Message clé** | Petit jeu, fort déséquilibre pixel, aucune image « normale seule ». |
| **Temps** | 1 min 30 s |
| **Oral** | Dire clairement que les préfixes ne sont pas des classes du modèle. |

---

## Slide 5 — Annotations et correspondance

| | |
| --- | --- |
| **Titre** | Qualité et alignement image–masque |
| **Contenu** | Audit 60/60 ; zoom bord jaune ; limites sur parties fines |
| **Visuel** | `11_mask_correspondence_zoom.png` |
| **Message clé** | La supervision est dense et alignée ; ce n’est pas un dataset de classification. |
| **Temps** | 1 min 30 s |
| **Oral** | Montrer le pneu 0,37 % : « c’est le cas difficile que l’accuracy ignore ». |

---

## Slide 6 — Préparation et split

| | |
| --- | --- |
| **Titre** | Préparation : 512×288 et split 42/9/9 |
| **Contenu** | Resize 16:9 sans padding ; masques 0/1 ; split anti-fuite ; groupes répétés en train |
| **Visuel** | `07_split_composition.png` ; schéma 1280×720 → 512×288 |
| **Message clé** | On refuse le letterbox qui crée 44 % de noir artificiel comme entrée par défaut. |
| **Temps** | 1 min 30 s |
| **Oral** | Mentionner le hash de split et l’absence de pneus en test (honnêteté). |

---

## Slide 7 — Augmentation et fuite de données

| | |
| --- | --- |
| **Titre** | Augmentation (train) et prévention des fuites |
| **Contenu** | Flip H, photométrie limitée ; sync image/masque ; val/test sans aug. ; seuil hors test |
| **Visuel** | Petite liste « autorisé / interdit » |
| **Message clé** | Avec n=42, l’augmentation aide, mais une fuite ruinerait toute conclusion. |
| **Temps** | 1 min |
| **Oral** | Une phrase sur pourquoi le flip vertical est refusé. |

---

## Slide 8 — Méthodologie expérimentale

| | |
| --- | --- |
| **Titre** | Plan d’expériences |
| **Contenu** | E0 → E1 → E2a/E2b ; critère Dice val ; Colab ; TF/Keras |
| **Visuel** | Flèche méthodologique |
| **Message clé** | Progression contrôlée, pas de « big model first ». |
| **Temps** | 1 min |
| **Oral** | Annoncer que l’AE n’est pas un concurrent ici (slide 12). |

---

## Slide 9 — Baseline et CNN from scratch

| | |
| --- | --- |
| **Titre** | E0 et E1 : baseline et CNN |
| **Contenu** | E0 tout fond ; E1 FCN/CNN léger ; perte pondérée / Dice ; attente : faibles petits objets |
| **Visuel** | Schéma CNN simple ; `[Courbes à compléter après les expériences]` |
| **Message clé** | E0 ancre le discours métrique ; E1 montre la limite sans transfer. |
| **Temps** | 1 min 15 s |
| **Oral** | Ne pas inventer de scores ; si slides avant runs, dire « résultats en cours / à venir ». |

---

## Slide 10 — U-Net et transfer learning

| | |
| --- | --- |
| **Titre** | E2 : U-Net + MobileNetV2 |
| **Contenu** | Encodeur–décodeur, skips, ImageNet, gel vs fine-tune |
| **Visuel** | Schéma U-Net ; mention MobileNetV2 |
| **Message clé** | Avec 42 images, le pré-entraînement est la stratégie réaliste. |
| **Temps** | 1 min 30 s |
| **Oral** | Expliquer skip = récupérer les détails spatiaux perdus par le pooling. |

---

## Slide 11 — Résultats et comparaison

| | |
| --- | --- |
| **Titre** | Résultats quantitatifs |
| **Contenu** | Tableau val E0/E1/E2 ; ligne test du modèle final ; métriques protocole |
| **Visuel** | Tableau `[Résultats à compléter après les expériences]` + 1–2 overlays |
| **Message clé** | On compare sous le **même** protocole ; le test n’a servi qu’une fois. |
| **Temps** | 2 min |
| **Oral** | Lire Precision/Recall ensemble ; montrer un échec à côté d’un succès. |

---

## Slide 12 — Interprétation, AE et ICCV

| | |
| --- | --- |
| **Titre** | Interprétation, autoencodeur et référence ICCV |
| **Contenu** | Limites n=9 ; exclusion AE ; tableau 3 lignes ICCV vs nous (resynthèse vs segmentation supervisée) |
| **Visuel** | Mini-tableau comparaison |
| **Message clé** | Nous ne reproduisons pas ICCV 2019 ; l’AE n’a pas de routes normales ici. |
| **Temps** | 1 min 15 s |
| **Oral** | Phrase exacte : « RoadAnomaly est un benchmark de test dans le papier ; chez nous c’est la supervision. »

---

## Slide 13 — Déploiement et limites

| | |
| --- | --- |
| **Titre** | Démonstration et limites |
| **Contenu** | Streamlit ; même pipeline ; disclaimer sécurité ; pas de routes vides mesurées |
| **Visuel** | Capture app `[à compléter Phase 8]` ou mockup annoté « prototype » |
| **Message clé** | Démo pédagogique ≠ système de conduite. |
| **Temps** | 45 s |
| **Oral** | Une phrase sur l’usage humain-in-the-loop. |

---

## Slide 14 — Conclusion et perspectives

| | |
| --- | --- |
| **Titre** | Conclusion |
| **Contenu** | Réponses courtes RQ ; apports ; 3 perspectives réalistes (données, robustesse, eval OOD) |
| **Visuel** | 3 puces max |
| **Message clé** | Projet défendable, reproductible, honnête sur ses limites. |
| **Temps** | 1 min |
| **Oral** | Finir sur l’honnêteté scientifique, pas sur une promesse industrielle. Remercier. |

---

## Slide optionnelle (si avance) — Une erreur analysée

| | |
| --- | --- |
| **Titre** | Étude d’un cas d’échec |
| **Contenu** | FP/FN sur une image test ; hypothèse (taille, texture, domaine) |
| **Temps** | 45–60 s (pris sur la marge) |

---

## Consignes de cohérence orale

1. Toujours dire **segmentation sémantique binaire supervisée**, jamais « on classifie les images ».  
2. Toujours séparer **val** (choix) et **test** (jugement final).  
3. Ne jamais comparer verbalement votre IoU à l’AUROC du papier ICCV.  
4. Si une question porte sur un chiffre absent : « Ce run n’est pas encore figé ; le protocole est déjà verrouillé. »  
5. Vocabulaire aligné : 512×288, masque 0/1, Dice validation, split 42/9/9.

---

## Matériel à préparer avant la soutenance (Phase 9)

- [ ] Export PDF/PPT des slides  
- [ ] Figures haute résolution (pas de captures floues)  
- [ ] Tableau de résultats **réel**  
- [ ] Une démo Streamlit de secours (vidéo 20 s si le Wi-Fi échoue)  
- [ ] Version papier du guide de soutenance (`docs/06_...`)
