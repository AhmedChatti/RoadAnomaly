# Checklist de cohérence des livrables

**But :** garantir que rapport, notebook, présentation, code, configs et résultats expérimentaux **racontent la même histoire scientifique**.  
**Référence normative :** `docs/02_protocole_scientifique_phase1.md` + `configs/default.json`  
**Usage :** cocher avant dépôt final (Phase 9–10). Les items marqués 🔒 sont déjà verrouillés.

Légende : ☐ à faire · 🔒 figé Phase 1 · ⏳ dépend des expériences · ✅ fait dans ce workstream doc

---

## 1. Terminologie commune (tous livrables)

| Terme autorisé | Terme interdit / trompeur | Statut |
| --- | --- | --- |
| Segmentation sémantique binaire supervisée | « Classification d’images RoadAnomaly » | ☐ |
| Anomalie / obstacle inattendu (classe positive unique) | Prédire animals vs cones comme classes du réseau | ☐ |
| Détection d’anomalies routières = problème applicatif | Confondre avec la tâche pixel | ☐ |
| Prototype académique | « Système ADAS certifié / production-ready » | ☐ |
| Différent de Lis et al. 2019 | « Nous reproduisons ICCV 2019 » | ☐ |
| Autoencodeur : faisabilité / exclusion | « Baseline AE entraînée avec succès sur les 60 images » | ☐ |

---

## 2. Paramètres scientifiques figés

| Paramètre | Valeur attendue partout | Vérifié dans |
| --- | --- | --- |
| Split | 42 / 9 / 9 | 🔒 protocole, `split_lock.json` |
| `assignment_sha256` | `9a01c05cfc7d33783c4f64b91b0ae423a4d3ad57439fd71ffc029a8f04585c78` | 🔒 |
| Input | **512 × 288**, pas de letterbox par défaut | 🔒 |
| Masque modèle | 0 / 1 (remap loader) | 🔒 |
| Masques `frames/` | non modifiés | 🔒 |
| Checkpoint metric | `val_mean_dice` | 🔒 |
| Seuil | choisi sur **val** ; test figé | 🔒 |
| Seed | 42 (sauf run documenté autrement) | 🔒 config |
| Stack | TensorFlow / Keras ; Streamlit | 🔒 |
| E4 AE concurrent | `false` | 🔒 |

**Règle :** si un notebook ou une slide contredit une ligne ci-dessus, c’est un bug de cohérence, pas une « amélioration silencieuse ».

---

## 3. Rapport technique ↔ protocole

| Contrôle | Statut |
| --- | --- |
| Structure principale = 6 phases CRISP-DM | ✅ prévu dans `docs/04_...` · ☐ rapport final |
| BO et DSO alignés protocole §1 | ☐ |
| Tableau ICCV vs nous sans comparaison de scores fallacieuse | ☐ |
| Section AE = faisabilité / exclusion | ☐ |
| Aucune métrique inventée | ☐ |
| Figures numérotées = fichiers réels du dépôt | ☐ |
| Résultats test présentés **une** fois | ⏳ |
| Disclaimer déploiement | ☐ |

---

## 4. Notebook final ↔ rapport ↔ config

| Contrôle | Statut |
| --- | --- |
| Sections notebook correspondent aux phases du rapport | ⏳ Phase 7 |
| Markdown avant chaque décision importante | ⏳ |
| Hyperparamètres lus depuis / identiques à `configs/default.json` | ⏳ |
| Chemins de checkpoints = ceux cités dans le rapport | ⏳ |
| Métriques notebook = tables rapport (tolérance d’arrondi documentée) | ⏳ |
| Notebook historique `01_...` non écrasé | 🔒 à préserver |
| Cellules « interprétation » séparées des cellules « calcul » | ⏳ |
| Environnement (versions TF/Keras) consignées | ⏳ |

---

## 5. Présentation ↔ rapport

| Contrôle | Statut |
| --- | --- |
| Plan 15–20 min suivi (`docs/05_...`) | ✅ plan · ☐ slides |
| Mêmes chiffres que le rapport (pas de slide « plus beau ») | ⏳ |
| Même message ICCV / AE / prototype | ☐ |
| Au moins un échec montré, pas seulement des succès | ⏳ |
| Timing respecté (±2 min) | ☐ répétition orale |

---

## 6. Code ↔ config ↔ expériences

| Contrôle | Statut |
| --- | --- |
| Loader 512×288 + remap 0/1 | ⏳ Phase 2 |
| Augmentation train only | ⏳ |
| E0 / E1 / E2 implémentés comme nommé dans le protocole | ⏳ Phases 3–4 |
| JSON metadata rempli par run (`experiments/metadata/`) | ⏳ |
| `split_manifest` hash vérifié en début de train | ⏳ |
| Pas d’appel `shutil.rmtree` sur `data/processed` depuis le training | ☐ à vérifier en revue code |
| Inférence Streamlit = même fonction que l’évaluation | ⏳ Phase 8 |
| Checkpoint chargé = best val Dice déclaré | ⏳ |

---

## 7. Résultats expérimentaux — intégrité

| Contrôle | Statut |
| --- | --- |
| Chaque chiffre du rapport a un artefact (log, JSON, figure) | ⏳ |
| Runs échoués documentés ou explicitement exclus avec raison | ⏳ |
| Placeholder `[Résultats à compléter après les expériences]` retiré seulement après mesure | ✅ règle actuelle |
| Accuracy jamais seule métrique principale | ☐ |
| Métriques globales pixel **et** moyennes par image | ⏳ |
| Liste des 9 images test avec scores individuels | ⏳ |

---

## 8. Livrables d’évaluation professeur — packing list

| Livrable | Emplacement prévu | Statut |
| --- | --- | --- |
| Rapport CRISP-DM FR | `reports/technical_report/` (Phase 9) | Structure ✅ (`docs/04`) · texte final ⏳ |
| Notebook / Colab final | `notebooks/RoadAnomaly_Final.ipynb` | ⏳ Phase 7 |
| Présentation 15–20 min | `reports/presentation/` | Plan ✅ (`docs/05`) · slides ⏳ |
| Guide soutenance ≥25 Q | `docs/06_individual_defense_guide.md` | ✅ (36 questions) |
| Matrice de conformité | `docs/03_assessment_requirements_and_compliance.md` | ✅ |
| Application Streamlit | `app/streamlit_app.py` | ⏳ Phase 8 |
| Code documenté | `src/` | Exploration ✅ · training ⏳ |
| Config / split lock | `configs/`, `experiments/metadata/` | ✅ |

---

## 9. Contrôles « ne pas casser Phase 1 »

Avant tout merge / dépôt :

- [ ] `docs/02_protocole_scientifique_phase1.md` inchangé (sauf correction typo approuvée explicitement)
- [ ] `configs/default.json` : décisions scientifiques non altérées sans révision de protocole
- [ ] `experiments/metadata/split_lock.json` présent et hash identique
- [ ] Aucun entraînement n’a modifié `frames/`
- [ ] Workstream évaluation n’a pas lancé la Phase 2

**Contrôle effectué le 8 nov. 2026 (ce workstream) :** Phase 1 non modifiée ; aucun training ; aucune installation ; aucun téléchargement de poids.

---

## 10. Signature de relecture (équipe)

| Membre | A relu rapport vs notebook | A relu slides vs tableaux | Date |
| --- | --- | --- | --- |
| | ☐ | ☐ | |
| | ☐ | ☐ | |
| | ☐ | ☐ | |
