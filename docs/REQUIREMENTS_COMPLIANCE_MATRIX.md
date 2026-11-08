# RoadAnomaly — Requirements Compliance Matrix

**Date:** 2026-11-08 (Stage 0)  
**Rule:** status reflects verified evidence, not file existence alone.  
**Related older inventory:** `docs/03_assessment_requirements_and_compliance.md` (kept; this matrix is the execution tracker).

Status vocabulary:

| Status | Meaning |
| --- | --- |
| NOT STARTED | No substantive work |
| IN PROGRESS | Partial implementation or draft docs |
| IMPLEMENTED | Code/docs exist and were inspected |
| EXECUTED | Ran with real data / measured outputs |
| VERIFIED | Re-checked in Stage 0 (hash/test/audit) |

---

## A. CRISP-DM professor requirements

| ID | Exact requirement | Implementation / section | Evidence | Status | Remaining action | Limitation |
| --- | --- | --- | --- | --- | --- | --- |
| R-BU-01 | Business Understanding: road-safety problem, scope, research question, constraints | `reports/01_business_and_data_understanding.md`; protocole §1; outline `docs/04` | Draft report + protocol | IMPLEMENTED | Integrate into final FR report (Stage 8) | Final PDF not written |
| R-DU-01 | Data Understanding: source, counts, formats, labels, imbalance, categories, duplicates, quality, licensing | Notebook 01; reports/figures; `credits.txt` | Local 60 frames present; stats documented | IMPLEMENTED | Refresh figures in final notebook/report | Near-duplicates / no normal-only subset |
| R-DP-01 | Data Preparation: load, binary masks, resize 512×288, split 42/9/9, aug train-only, normalize, integrity, seeds | `src/preprocessing.py`, `dataset.py`, `augmentation.py`; `docs/08` | Tests VERIFIED; split lock VERIFIED | VERIFIED | Keep frozen; document in final report | Do not alter split |
| R-MO-01 | Modeling E0 all-background | `src/models/baseline.py`; E0 metadata | Val metrics frozen | EXECUTED / VERIFIED | Cite in comparison after E2 | Accuracy trap only |
| R-MO-02 | Modeling E1 CNN from scratch | `src/models/cnn.py`; E1_1510 | Checkpoint SHA VERIFIED; val Dice ≈0.569; freeze audit PASS | EXECUTED / VERIFIED | Keep frozen; cite regenerated qualitative with sidecar | Abandoned E1_1501 weights unavailable (documented) |
| R-MO-03 | Modeling E2a U-Net MobileNetV2 frozen encoder | `configs/phase4.json`, `unet_mobilenetv2.py`, `phase4_training.py` | Stage 1 readiness + 57 tests | IMPLEMENTED / READY | Stage 2 sync then authorized Colab training | No metrics yet |
| R-MO-04 | Modeling E2b selective fine-tune from `block_13_expand` | Same + `configure_e2b_finetuning` | Layer-trainability + readiness | IMPLEMENTED / READY | Authorized training after E2a | No metrics yet |
| R-EV-01 | Evaluation metrics: P/R/F1/IoU/Dice/Acc + limits of Acc | `src/metrics.py`, `evaluation.py`, Phase 3 docs | E0/E1 val tables | PARTIAL (E0/E1 EXECUTED; E2 pending) | E2a/E2b + final test | Test blocked until Stage 5 |
| R-EV-02 | Curves, qualitative predictions, model comparison, overfitting analysis | Phase 3 figures/CSVs; Phase 4 planned | E1 curves + **regenerated** qualitative; E2 pending | IN PROGRESS | Produce E2 artefacts after training | E2 metrics do not exist yet |
| R-DE-01 | Deployment / demo if applicable | `app.py`, `src/app_inference.py`, `docs/APP_USER_GUIDE.md` | Unit tests OK; live load needs Drive E2b `best.keras` | IMPLEMENTED (code); EXECUTE pending local ckpt | Copy E2b checkpoint from Drive | Academic prototype only; no ADAS claim |

---

## B. Additional deliverables

| ID | Exact requirement | Implementation | Evidence | Status | Remaining action | Limitation |
| --- | --- | --- | --- | --- | --- | --- |
| R-REP-01 | Technical report CRISP-DM (French) | Structure `docs/04`; drafts in `reports/` | Outline only | IN PROGRESS | Stage 8 after results | No invented metrics |
| R-NB-01 | Complete reproducible notebook | `notebooks/01_…` historical; Colab setup `00_…` | Setup notebook exists | IN PROGRESS | Final notebook Stage 6 | Final not created |
| R-PRES-01 | Presentation 15–20 min | `docs/05_presentation_plan.md` | Plan only | IN PROGRESS | Build slides Stage 9 | Needs real E2/test numbers |
| R-DEF-01 | Individual defense prep | `docs/06_individual_defense_guide.md` | Q/A draft | IN PROGRESS | Update with measured results | Placeholders remain |
| R-REF-01 | References + dataset acknowledgements | `credits.txt`; protocol refs | Present | IMPLEMENTED | Expand in final report | — |

---

## C. Scientific / reproducibility gates

| ID | Exact requirement | Implementation | Evidence | Status | Remaining action | Limitation |
| --- | --- | --- | --- | --- | --- | --- |
| R-SPL-01 | Frozen split 42/9/9 + SHA | `split_lock.json`, `verify_split_lock` | SHA match Stage 0 | VERIFIED | Never change | — |
| R-E1-01 | Canonical E1 checkpoint hash | `checkpoints/…/best.keras` | SHA match Stage 0 | VERIFIED | Keep on Drive backup | Local disk critically low |
| R-TST-01 | No test access before final eval | `evaluation` / Phase 4 hard blocks | Tests + freeze audit | VERIFIED | Unlock only Stage 5 w/ approval | — |
| R-AE-01 | Autoencoder feasibility only (no fake normal-only AE) | Protocol + Phase 4 plan docs | Documented scientific exclusion | IMPLEMENTED | Keep in report limitations | Dataset has anomalies in all 60 images |
| R-P4-AUTH | Dual auth before ImageNet/train | `phase4_training` + script flags | Unit tests | VERIFIED (gates) | User must approve Stage 3 | Flags not supplied |
| R-P4-GIT | Infrastructure commit ancestry | `assert_required_git_ancestry` | Ancestor OK | VERIFIED | Keep after future commits | — |
| R-P4-ISO | Phase 4 artefact isolation | `…/phase4/` dirs + denylist | Config + tests | VERIFIED | Use in training | — |
| R-DRV-01 | Honest Drive/Colab access status | Connection report | Stage 0 + Colab readiness JSON 2026-11-08T16:43:15Z | VERIFIED (Colab path); Cursor→Drive NOT AVAILABLE | Optional Drive-for-Desktop if Cursor sync desired | Cursor has no Drive API; Colab T4+Drive green |

---

## D. Acceptance checklist (Part 13) — Stage 0 snapshot

| Criterion | State |
| --- | --- |
| Compliance matrix exists | YES (this file) |
| Six CRISP-DM phases documented | PARTIAL (BU/DU/DP strong; MO/EV partial; DE absent) |
| Dataset/preprocessing documented | YES |
| Split verified | YES |
| E0 documented | YES |
| E1 preserved | YES (canonical) |
| E2a executed | NO |
| E2b executed | NO |
| Final test after approval | NO |
| Final FR report / notebook / slides | NO |
| Drive/Colab honestly documented | YES — not verified end-to-end |

---

## Notes on outdated statements in `docs/03_…`

Section 1 of `docs/03_assessment_requirements_and_compliance.md` still says models/checkpoints are absent. That is **historically stale**: E0/E1 and Phase 4 code now exist. Prefer **this matrix** for current status; do not silently rewrite frozen Phase 1 protocol text.

---

## E. Application / HITL (2026-11-08 acceptance audit)

| ID | Requirement | Evidence | Status | Limitation |
| --- | --- | --- | --- | --- |
| R-APP-01 | Official E2b@0.55 analyser | `app.py`, `app_inference.py`, tests | VERIFIED | Academic prototype |
| R-APP-02 | Anomaly Hunter train/val only | `pool.py`, acceptance tests | VERIFIED | Soft UI timer refresh PARTIAL |
| R-APP-03 | Discovery candidates + consent | store + UI | VERIFIED | Manual erasure only |
| R-APP-04 | Admin fail-closed auth | `admin_auth.py` | VERIFIED | Localhost demo assumed |
| R-APP-05 | E3 eligibility ≠ approval; no test/val train export | `eligibility.py`, `export.py` | VERIFIED | E3 training not implemented (intentional) |
| R-APP-06 | Freeze integrity pins | `test_acceptance_audit.py` | VERIFIED | — |
