# RoadAnomaly — Project Status

**Last verified:** 2026-11-08 (Stages 4–9 deliverables; Stage 5 test pending Colab Run-all)  
**Branch:** `main`  
**HEAD:** `8c780d7dfc68b7e66ba45168608876f9210523f7`  
**E2 training commit:** `9e734a1d0af48342f92c084216b1dd1d4c254741`

## One-line status

Validation comparison and academic deliverables are ready. **E2b** is the frozen primary model (val Dice ≈ 0.778 @ thr 0.55). **Test metrics are not yet available** — Stage 5 requires one Colab Run-all of `notebooks/03_final_test_evaluation_colab.ipynb`.

## Stage board

| Stage | Status |
| --- | --- |
| 0 Audit | COMPLETED |
| 1 Readiness | COMPLETED |
| 2 Git sync Phase 4 | COMPLETED |
| 3 E2a/E2b training | COMPLETED |
| 4 Comparison (val) | COMPLETED |
| 5 Final test eval | **BLOCKED** — needs Colab one-shot Run-all |
| 6 FR CRISP-DM report | COMPLETED WITH LIMITATIONS (val only; test TBD) |
| 7 Final notebook | COMPLETED WITH LIMITATIONS |
| 8 Presentation materials | COMPLETED WITH LIMITATIONS |
| 9 Defense guide | COMPLETED WITH LIMITATIONS |
| Optional Streamlit | NOT STARTED (after mandatory) |

## Validation leaderboard (mean/image)

| Exp | Thr | Dice | IoU |
| --- | ---: | ---: | ---: |
| E0 | 0.50 | 0.000 | 0.000 |
| E1 | 0.35 | 0.569 | 0.445 |
| E2a | 0.15 | 0.705 | 0.584 |
| E2b | **0.55** | **0.778** | **0.686** |

## Key paths

- Report: `docs/RAPPORT_TECHNIQUE_CRISP_DM.md`
- Val analysis: `docs/13_phase4_results_and_analysis.md`
- Test protocol: `docs/14_final_test_evaluation.md` + freeze JSON
- Slides: `docs/15_presentation_slide_content.md`
- Defense: `docs/16_oral_defense_guide_measured.md`
- Notebooks: `04_final_project_walkthrough.ipynb`, `03_final_test_evaluation_colab.ipynb`

## Next human action (Stage 5)

Open  
https://colab.research.google.com/github/AhmedChatti/RoadAnomaly/blob/main/notebooks/03_final_test_evaluation_colab.ipynb  
→ T4 → Run all (AUTHORIZE_TEST_EVALUATION already True) → paste `phase4_final_test_comparison.csv` / `STAGE5_TEST_EVAL_DONE.json`.
