# Exigences d’évaluation du professeur — matrice de conformité

**Projet :** Détection et localisation des anomalies routières par segmentation sémantique supervisée  
**Nature :** workstream complémentaire d’évaluation (hors Phases 0–1)  
**Date :** 8 novembre 2026  
**Contrainte :** ce document **ne modifie pas** le protocole Phase 1 (`docs/02_protocole_scientifique_phase1.md`) ni `configs/default.json`.

Légende des statuts :

| Statut | Signification |
| --- | --- |
| **Disponible** | Artefact déjà présent et réutilisable |
| **Planifié** | Prévu par le protocole ; pas encore produit |
| **Placeholder** | Structure rédigée ; contenu chiffré à compléter après expériences |
| **Bloqué** | Impossible sans entraînement / évaluation réelle |

---

## 1. Inventaire rapide de l’existant

| Artefact | Chemin | Couvre surtout |
| --- | --- | --- |
| Audit scientifique | `docs/INITIAL_PROJECT_AUDIT.md` | Phase 0 |
| Protocole verrouillé | `docs/02_protocole_scientifique_phase1.md` | Décisions scientifiques |
| Config | `configs/default.json` | Hyperparamètres par défaut |
| Verrou de split | `experiments/metadata/split_lock.json` | 42/9/9 |
| Compréhension métier / données / prep | `reports/01_business_and_data_understanding.md` | CRISP-DM 1–3 (partiel) |
| Notebook exploration | `notebooks/01_exploration_and_preprocessing.ipynb` | Exploration + prep |
| Figures 01–11 | `reports/figures/` | Data Understanding / Preparation |
| Données préparées | `data/processed/` | Split letterbox historique |
| Code exploration | `src/road_anomaly.py` | Audit, split, figures |
| Modèles / métriques / checkpoints | — | **Absents** |
| Notebook final FR | — | **Absent** |
| Rapport technique final FR | — | **Absent** (structure : doc 04) |
| Streamlit | — | **Absent** |
| Présentation / soutenance | — | Ce workstream (docs 05–06) |

---

## 2. Matrice exigence professeur → livrable

### A. Rapport technique — structure CRISP-DM

| Exigence | Livrable | Statut | Quand le compléter |
| --- | --- | --- | --- |
| Business Understanding | Rapport §CRISP-1 ; déjà amorcé dans `reports/01_...md` | Disponible (ébauche) → à intégrer au rapport FR | Phase 9 |
| Data Understanding | Idem + figures 01–06, 10–11 | Disponible | Phase 9 |
| Data Preparation | Rapport + protocole §5–6 ; letterbox vs 512×288 | Partiel (letterbox documenté ; 512×288 décidé, pipeline Phase 2) | Phases 2 puis 9 |
| Modeling | Architectures E0–E2, hyperparamètres, justifications | Planifié / Placeholder | Phases 3–5, 9 |
| Evaluation | Métriques, comparaisons, échecs | Bloqué jusqu’aux runs | Phases 6, 9 |
| Deployment | Streamlit + limites prototype | Planifié | Phases 8–9 |
| Outline CRISP-DM détaillé | `docs/04_technical_report_structure.md` | **Disponible (ce workstream)** | Immédiat |

### B. Notebook / Colab

| Exigence | Livrable | Statut | Quand |
| --- | --- | --- | --- |
| Notebook final organisé (FR) | `notebooks/RoadAnomaly_Final.ipynb` (nom prévu) | Planifié | Phase 7 |
| Markdown avant code important | Règle du protocole + checklist | Planifié | Phase 7 |
| Architectures et configs | Aligné sur `configs/default.json` | Planifié | Phases 3–7 |
| Métriques et visualisations | Résultats réels uniquement | Bloqué | Phases 6–7 |
| Reproductibilité | Seeds, versions, chemins checkpoints | Planifié | Phases 3–7, 9 |
| Cohérence rapport ↔ notebook | `docs/07_deliverables_consistency_checklist.md` | **Disponible (ce workstream)** | Continu |

Le notebook historique `01_exploration_and_preprocessing.ipynb` est **conservé** comme matériel historique ; il ne remplace pas le notebook final.

### C. Présentation orale 15–20 min

| Exigence | Livrable | Statut |
| --- | --- | --- |
| Plan slide-par-slide | `docs/05_presentation_plan.md` | **Disponible (ce workstream)** |
| Diapositives finales (PPT/PDF) | `reports/presentation/` (à créer plus tard) | Placeholder — figures de résultats bloquées |
| Timing 15–20 min | Affectation dans doc 05 | Disponible |

### D. Soutenance technique individuelle

| Exigence | Livrable | Statut |
| --- | --- | --- |
| ≥ 25 questions / réponses | `docs/06_individual_defense_guide.md` | **Disponible (ce workstream)** |
| Follow-ups difficiles | Inclus dans doc 06 | Disponible |
| Réponses liées aux **résultats chiffrés** | Sections marquées placeholder | Bloqué jusqu’après expériences |

---

## 3. Cartographie CRISP-DM ↔ artefacts projet

| Phase CRISP-DM | Contenu attendu par le professeur | Déjà disponible | À produire |
| --- | --- | --- | --- |
| **1. Business Understanding** | Problème, BO, DSO, pertinence | `reports/01_...`, protocole §1 | Reformulation FR dans rapport final |
| **2. Data Understanding** | Structure, annotations, EDA, déséquilibre | Notebook 01, figures, inventaire CSV | Intégration rapport ; pas de nouvelles « fausses » stats |
| **3. Data Preparation** | Resize, masques, normalisation, aug., split, anti-fuite | Split 42/9/9 + letterbox historique ; décision 512×288 | Pipeline loader Phase 2 ; section rapport |
| **4. Modeling** | E0, E1, E2, ablations, hyperparamètres | Matrice dans protocole §8 | Code + entraînements Phases 3–5 |
| **5. Evaluation** | Precision/Recall/F1/IoU/Dice, analyses | Définitions protocole §7 | Mesures réelles Phase 6 |
| **6. Deployment** | Démo, usage, limites | Spécification dans audit / protocole | App Streamlit Phase 8 |

---

## 4. Alignement avec les décisions Phase 1 (non négociables)

Ce workstream **réaffirme** sans les modifier :

- Split **42 / 9 / 9** verrouillé (`experiments/metadata/split_lock.json`).
- Entrée **512 × 288**, sans letterbox comme tenseur d’entraînement par défaut.
- Masques modèle **0 / 1** via le loader ; originaux intacts.
- Critère de checkpoint : **Dice moyen par image (validation)**.
- Seuil choisi sur **val** uniquement ; test une fois.
- Autoencodeur : **section de faisabilité**, pas baseline concurrente.
- Pas de reproduction de Lis et al. (2019) ; tableau de comparaison protocolaire uniquement.
- Prototype académique ≠ système de sécurité routière certifié.

---

## 5. Ce qui ne peut pas encore être complété

| Élément | Raison |
| --- | --- |
| Tableaux de métriques test / val | Aucun modèle entraîné |
| Courbes d’apprentissage | Aucun run |
| Galeries de prédictions / échecs | Pas d’inférence |
| Choix final E2a vs E2b | Nécessite validation |
| Diapositives avec chiffres | Dépendent des résultats |
| Réponses de soutenance du type « notre IoU vaut X » | Placeholder obligatoire |
| Démo Streamlit branchée | Pas de checkpoint |

Toute mention de résultats dans les docs 04–07 utilise la mention :

`[Résultats à compléter après les expériences]`

---

## 6. Plan d’intégration sans casser le phasage

| Workstream évaluation | Phase technique associée |
| --- | --- |
| Docs 03–07 (maintenant) | Après Phase 1, **avant** Phase 2 |
| Remplir placeholders numériques | Après Phase 6 |
| Notebook final FR | Phase 7 |
| Streamlit | Phase 8 |
| Rapport / slides / model card finaux | Phase 9 |

**Ne pas démarrer la Phase 2 dans ce workstream.**
