# RoadAnomaly — Project Status

**Last verified:** 2026-11-08 (Stage 5 test complete; docs filled)  
**Branch:** `main`  
**E2 training commit:** `9e734a1d0af48342f92c084216b1dd1d4c254741`  
**Split lock:** `9a01c05cfc7d33783c4f64b91b0ae423a4d3ad57439fd71ffc029a8f04585c78`

## One-line status

**E2b** is the frozen primary model. Validation Dice ≈ **0.778** @ thr 0.55; **test** Dice mean/image ≈ **0.818** (global ≈ **0.871**). Stage 5 one-shot complete — no retuning.

## Stage board

| Stage | Status |
| --- | --- |
| 0 Audit | COMPLETED |
| 1 Readiness | COMPLETED |
| 2 Git sync Phase 4 | COMPLETED |
| 3 E2a/E2b training | COMPLETED |
| 4 Comparison (val) | COMPLETED |
| 5 Final test eval | **COMPLETED** |
| 6 FR CRISP-DM report | COMPLETED (val + test) |
| 7 Final notebook | COMPLETED |
| 8 Presentation materials | COMPLETED |
| 9 Defense guide | COMPLETED |
| Optional Streamlit | NOT STARTED (optional) |

## Validation leaderboard (mean/image)

| Exp | Thr | Dice | IoU |
| --- | ---: | ---: | ---: |
| E0 | 0.50 | 0.000 | 0.000 |
| E1 | 0.35 | 0.569 | 0.445 |
| E2a | 0.15 | 0.705 | 0.584 |
| E2b | **0.55** | **0.778** | **0.686** |

## Test leaderboard (mean/image, frozen thresholds)

| Exp | Thr | Dice | IoU |
| --- | ---: | ---: | ---: |
| E0 | 0.50 | 0.000 | 0.000 |
| E1 | 0.35 | 0.398 | 0.287 |
| E2a | 0.15 | 0.613 | 0.454 |
| E2b | **0.55** | **0.818** | **0.705** |

## Key paths

- Report: `docs/RAPPORT_TECHNIQUE_CRISP_DM.md`
- Val analysis: `docs/13_phase4_results_and_analysis.md`
- Test results: `docs/14_final_test_evaluation.md` + `experiments/results/phase4/phase4_final_test_comparison.csv`
- Slides: `docs/15_presentation_slide_content.md`
- Defense: `docs/16_oral_defense_guide_measured.md`
- Notebooks: `04_final_project_walkthrough.ipynb`, `03_final_test_evaluation_colab.ipynb`
