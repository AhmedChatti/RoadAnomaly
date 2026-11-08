# Presentation materials index (for teammates)

**Purpose:** Map existing RoadAnomaly files to CRISP-DM slides.  
**Rule:** Use only verified content. Do **not** invent metrics. Distinguish **executed** experiments from **implemented-but-not-executed** ones.  
**Language of final oral slides:** French (plan in `docs/05_presentation_plan.md`).  
**Date of this index:** 2026-11-08  

**Experiment status snapshot**

| Experiment | Status | Use in slides? |
| --- | --- | --- |
| E0 all-background | **Executed** (val) | Yes — real numbers |
| E1 CNN from scratch (`E1_…_1510`) | **Executed** (val); frozen | Yes — real numbers |
| E1_1501 | Abandoned / non-canonical | Cite only as archive; **do not** present as main result |
| E2a U-Net + MobileNetV2 frozen encoder | **Implemented, not trained** | Architecture/config only; **no scores** |
| E2b selective fine-tune | **Implemented, not trained** | Architecture/config only; **no scores** |
| Final test evaluation | **Not started** (blocked) | Do **not** invent test scores |
| Streamlit / deployment app | **Not implemented** | Proposed only |

---

## Professor requirements (where to find them)

| Need | Exact path | What it contains |
| --- | --- | --- |
| Assessment / compliance mapping | `docs/03_assessment_requirements_and_compliance.md` | Professor deliverables mapped to artefacts; **note:** §1 inventory is partially stale (says models absent — E0/E1 now exist) |
| Live execution tracker | `docs/REQUIREMENTS_COMPLIANCE_MATRIX.md` | Current status per requirement (prefer this over stale parts of doc 03) |
| CRISP-DM report outline | `docs/04_technical_report_structure.md` | Required FR report sections |
| Oral presentation plan (15–20 min) | `docs/05_presentation_plan.md` | Slide-by-slide timing and messages |
| Individual defense Q&A | `docs/06_individual_defense_guide.md` | Technical oral-defense questions |
| Consistency checklist | `docs/07_deliverables_consistency_checklist.md` | Anti-contradiction rules (ICCV ≠ our protocol, no fake scores) |
| Frozen scientific protocol | `docs/02_protocole_scientifique_phase1.md` | Non-negotiable decisions (split, metrics, RQs, AE feasibility) |

**Do not invent:** final test metrics; E2a/E2b scores; claims that transfer learning “wins” before Stage 3–4 results exist; Streamlit screenshots if the app is not built.

---

## Reference papers and dataset acknowledgements

| Item | Exact path / citation | Notes for slides |
| --- | --- | --- |
| Dataset release paper | Lis, K., Nakka, K., Fua, P., & Salzmann, M. (2019). *Detecting the Unexpected via Image Resynthesis*. ICCV. BibTeX in `readme.md` | Motivates the **problem**; method is **unsupervised resynthesis** on ordinary scenes (e.g. Cityscapes), **not** our supervised split |
| Dataset readme | `readme.md` | Purpose, label values `{0,2}`, LabelGrab, contact |
| Image credits / licensing | `credits.txt` | Third-party images; annotations CC BY-SA 4.0 |
| Protocol comparison table (ICCV vs us) | `docs/02_protocole_scientifique_phase1.md` §4 | Mandatory slide content: different protocols → **no fair score comparison** with the paper |
| Early scientific audit | `docs/INITIAL_PROJECT_AUDIT.md` | Broader context; marks some paper claims as inferred if PDF not re-read locally |

**Do not invent:** numerical comparisons with ICCV paper scores; statements that we “reproduce ICCV 2019”.

---

## 1. Business Understanding

### Primary files

| Path | Contents |
| --- | --- |
| `reports/01_business_and_data_understanding.md` §1–2 | Problem situation, 4 business objectives, business success table, scope, learning task, data-science objectives, why accuracy fails |
| `docs/02_protocole_scientifique_phase1.md` §1–3 | Formal BO/DSO, research questions **RQ1–RQ4**, hypotheses, CRISP-DM mapping |
| `docs/04_technical_report_structure.md` §CRISP-1 | Report outline for BO↔DSO↔metrics |
| `docs/05_presentation_plan.md` slides 1–3 | Oral wording for title, motivation, BO–DSO |
| `docs/INITIAL_PROJECT_AUDIT.md` (problem sections) | Extra problem framing and stakeholder notes |

### Figures / tables to use

- Teaser overlays: `reports/figures/05_image_mask_overlay_examples.png`, `reports/figures/11_mask_correspondence_zoom.png`
- Accuracy trap foreshadowing: `reports/figures/04_pixel_class_balance.png`

### Completed vs missing

| Available | Missing — do not invent |
| --- | --- |
| Problem, BO, DSO, RQs, success criteria (protocol) | Production KPIs, fleet deployment claims, certified ADAS performance |

---

## 2. Data Understanding

### Primary files

| Path | Contents |
| --- | --- |
| `reports/01_business_and_data_understanding.md` §data sections | Origin, 60 images, categories, imbalance, repeated scenes, quality checks |
| `readme.md` + `frame_list.json` + `credits.txt` | Official dataset description and inventory |
| `docs/INITIAL_PROJECT_AUDIT.md` | Verified inventory notes (60 pairs, prefixes, instance masks) |
| `notebooks/01_exploration_and_preprocessing.ipynb` | Exploration notebook (historical / exploratory) |
| `reports/tables/frame_inventory.csv` | Per-frame inventory table |
| `src/road_anomaly.py` | Code that produced exploration figures |

### Figures / tables

| Path | Topic |
| --- | --- |
| `reports/figures/01_category_counts.png` | Category counts (animals/obstacles/cones/tires/vehicles) |
| `reports/figures/02_resolution_and_obstacle_size.png` / `02_image_resolutions.png` | Resolution / size |
| `reports/figures/03_anomaly_area_by_category.png` | Anomaly area by category |
| `reports/figures/04_pixel_class_balance.png` | ~9.85% anomaly vs ~90.15% background |
| `reports/figures/05_image_mask_overlay_examples.png` | Image–mask overlays |
| `reports/figures/06_small_and_large_anomalies.png` | Size extremes |
| `reports/figures/10_repeated_scenes.png` | Near-duplicate / repeated events |
| `reports/tables/frame_inventory.csv` | Tabular inventory |

### Completed vs missing

| Available | Missing — do not invent |
| --- | --- |
| 60/60 pairs, label values `{0,2}`, category distribution, pixel imbalance, repeated scenes | Extra unlabeled “normal-only” corpus; claims that instance masks exist for all 60 (readme: some instance files missing; project uses **semantic** masks) |

---

## 3. Data Preparation

### Primary files

| Path | Contents |
| --- | --- |
| `docs/02_protocole_scientifique_phase1.md` §5–6 | **Frozen** split 42/9/9, anti-leakage, resize **512×288**, mask 0/1 via loader, augmentation train-only |
| `docs/08_phase2_data_pipeline.md` | Implemented Phase 2 pipeline documentation |
| `configs/default.json` | Frozen default preprocessing / training config |
| `src/preprocessing.py`, `src/augmentation.py`, `src/dataset.py` | Executable preparation code |
| `experiments/metadata/split_lock.json` | Locked assignment hash `9a01c05c…04585c78` |
| `data/processed/split_manifest.csv` | Manifest used by loaders |
| `reports/01_business_and_data_understanding.md` preparation sections | Early prep narrative (**caution:** historical letterbox **512×512** text — superseded by frozen **512×288**) |
| `reports/tables/preparation_summary.json`, `data/processed/preparation_summary.json` | Prep summaries |

### Figures / tables

| Path | Topic |
| --- | --- |
| `reports/figures/07_split_composition.png` | Split composition |
| `reports/figures/08_preprocessing_letterbox.png` | **Historical** letterbox viz — label as historical if shown; training protocol is 512×288 without that letterbox path |
| `reports/figures/09_all_scenes_by_split.png` | Scenes by split |
| `experiments/figures/phase2_01_512x288_example.png` | Current 512×288 example |
| `experiments/figures/phase2_02_augmentation_sync.png` | Train augmentation sync image↔mask |
| `experiments/figures/phase2_03_split_examples.png` | Split examples |
| `experiments/figures/phase2_04_pipeline_summary.png` | Pipeline summary |

### Completed vs missing

| Available | Missing — do not invent |
| --- | --- |
| Frozen split + hash; Phase 2 loader; train-only aug; integrity tests | Any new split; test-set peeking; changing resolution without approval |

**Teammate warning:** Prefer `docs/02` + `docs/08` + Phase 2 figures over older “512×512 letterbox” wording in `reports/01`.

---

## 4. Modeling

### Primary files

| Path | Contents | Experiment status |
| --- | --- | --- |
| `docs/02_protocole_scientifique_phase1.md` §8–9 | Experiment matrix E0–E2b; AE feasibility decision | Protocol |
| `docs/09_phase3_baseline_and_cnn.md` | E0/E1 architectures, training, BN exclusion rationale | **Executed** |
| `docs/10_phase3_freeze_and_phase4_plan.md` | Freeze of E0/E1 + E2 plan | Mixed |
| `docs/12_phase4_implementation.md` | E2a/E2b implementation (no scores) | **Implemented only** |
| `docs/PHASE4_READINESS_REPORT.md` | Ready to train after sync; not trained | Ready / not executed |
| `configs/default.json` | E1-era defaults (batch, loss weights, etc.) | Frozen |
| `configs/phase4.json` | E2a/E2b LRs, batch 4, MobileNetV2, finetune `block_13_expand` | Declared |
| `src/models/baseline.py` | E0 | Executed |
| `src/models/cnn.py` | E1 | Executed |
| `src/models/unet_mobilenetv2.py` | E2a/E2b builder | Implemented |
| `src/losses.py`, `src/training.py` | Loss / training helpers | Shared |
| `src/phase4_training.py`, `scripts/run_phase4_experiments.py` | Gated E2 training entrypoints | Not executed for real training |

### Figures / configs to show (architecture, not scores)

- E0/E1 narrative + curves: see Evaluation section  
- Phase 4: show `configs/phase4.json` fields and architecture diagram from docs — **no Dice/IoU for E2**

### Completed vs missing

| Available | Missing — do not invent |
| --- | --- |
| E0/E1 training configs and code; E2 architecture + hyperparameters on disk | E2a/E2b trained weights, histories, selected thresholds, any claim E2 > E1 |

---

## 5. Evaluation

### Primary files (real results = validation only so far)

| Path | Contents | Status |
| --- | --- | --- |
| `experiments/metadata/E0_all_background.json` | E0 val metrics | **Executed** |
| `experiments/metadata/E1_cnn_scratch_20261108_1510.json` | Canonical E1 history, thr 0.5 train / 0.35 selected, val metrics | **Executed / frozen** |
| `experiments/results/phase3_E0_E1_val_comparison.csv` | Summary table | **Executed** |
| `experiments/results/E0_all_background_val_per_image.csv` | Per-image E0 | Executed |
| `experiments/results/E1_cnn_scratch_20261108_1510_val_per_image.csv` | Per-image E1 | Executed |
| `experiments/results/E1_cnn_scratch_20261108_1510_val_threshold_grid.csv` | Val threshold grid | Executed |
| `docs/09_phase3_baseline_and_cnn.md` | Written analysis of E0/E1 | Executed |
| `docs/10_phase3_freeze_and_phase4_plan.md` | Freeze table + limitations | Executed |
| `experiments/metadata/phase3_freeze_audit.json` | Integrity audit PASS | Verified |
| `src/metrics.py`, `src/evaluation.py` | Metric definitions; test split blocked | Code |

### Verified validation numbers (safe to present)

From `experiments/results/phase3_E0_E1_val_comparison.csv` (mean per image where applicable):

| Experiment | Dice | IoU | Recall | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| E0 | 0.000 | 0.000 | 0.000 | 0.862 |
| E1 | 0.569 | 0.445 | 0.850 | 0.867 |

Also state clearly:

- Training / checkpoint threshold: **0.5**  
- Operational threshold (val-selected): **0.35**  
- E1 best epoch: **66**  
- Checkpoint SHA-256: `567bde326ee5b776bc34132b23012543b35f195ec8d60229d01196eb1a0fb234`

### Figures

| Path | Topic | Note |
| --- | --- | --- |
| `experiments/figures/phase3_E0_metric_trap.png` | Accuracy trap (E0) | Prefer this for accuracy critique |
| `experiments/figures/phase3_E1_training_curves.png` | E1 loss / val Dice curves | Original |
| `experiments/figures/phase3_E1_val_qualitative.png` | Val qualitative panel | **Regenerated** 2026-11-08 — say so; provenance in `phase3_E1_val_qualitative.REGENERATED.json` |

### Completed vs missing

| Available | Missing — do not invent |
| --- | --- |
| E0/E1 **validation** metrics, curves, qualitative (regenerated), limitations in docs 09–10 | **Test-set** scores; E2a/E2b any metrics; final model ranking including E2; confusion matrices beyond what CSVs already imply |

---

## 6. Deployment

### Primary files (proposal only)

| Path | Contents |
| --- | --- |
| `docs/02_protocole_scientifique_phase1.md` §11 / CRISP mapping | Planned **Streamlit** demo; shared inference pipeline with eval |
| `docs/04_technical_report_structure.md` deployment section | Report outline for prototype limits |
| `docs/05_presentation_plan.md` / `docs/06_individual_defense_guide.md` | How to talk about demo honestly |
| `docs/07_deliverables_consistency_checklist.md` | Planned `app/streamlit_app.py` — **not created yet** |
| `docs/11_environment_synchronization_and_colab_workflow.md` | Colab/Drive workflow (execution infra, not end-user app) |
| `docs/GOOGLE_DRIVE_COLAB_CONNECTION_REPORT.md` | Verified Colab+Drive training path; Cursor has no Drive API |

### Figures / demos

- **None implemented.** No Streamlit app, no hosted demo screenshots.

### Completed vs missing

| Available | Missing — do not invent |
| --- | --- |
| Academic prototype **plan**; Colab training deployment path verified | Production system; real-time ADAS; app UI mock results; latency/throughput claims |

---

## Cross-cutting slides helpers

| Need | Path |
| --- | --- |
| 15–20 min slide order | `docs/05_presentation_plan.md` |
| Defense questions | `docs/06_individual_defense_guide.md` |
| Project status one-pager | `docs/PROJECT_STATUS.md` |
| Phase 3 gap honesty (regenerated figure, missing E1_1501 weights) | `docs/PHASE3_ARTIFACT_GAP_NOTES.md` |
| What not to contradict | `docs/07_deliverables_consistency_checklist.md` |

---

## Recommended slide source pack (minimum)

1. **Business:** `reports/01_…` §1 + protocole §1–2 + Fig. 5/11  
2. **Data:** Figs `01`, `04`, `06`, `10` + inventory CSV  
3. **Preparation:** protocole §5–6 + Phase 2 Figs `phase2_01`–`04` (not historical letterbox as “current”)  
4. **Modeling:** docs 09 (E0/E1) + `configs/phase4.json` (E2 **design only**)  
5. **Evaluation:** `phase3_E0_E1_val_comparison.csv` + E0 trap fig + E1 curves + regenerated qualitative  
6. **Deployment:** say “planned Streamlit prototype — not built yet”  
7. **ICCV:** protocole §4 comparison table — method difference, **no score duel**

---

## Absolute “do not invent” checklist for teammates

- [ ] No E2a/E2b Dice/IoU/Recall/Accuracy  
- [ ] No test-set metrics  
- [ ] No “we reproduce Lis et al. 2019”  
- [ ] No autoencoder results (feasibility exclusion only: protocole §9)  
- [ ] No Streamlit screenshots unless the app exists  
- [ ] No changing frozen split / resolution / E1 scores  
- [ ] If showing qualitative E1 panel, mention it was **regenerated** from the canonical checkpoint  

---

*This index only points to existing materials. It does not add scientific results. No commit was created as part of writing this file.*
