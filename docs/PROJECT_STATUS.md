# RoadAnomaly — Project Status

**Last verified:** 2026-11-08 (Streamlit app implemented; E2b weights must be copied from Drive)  
**Branch:** `main`  
**E2 training commit:** `9e734a1d0af48342f92c084216b1dd1d4c254741`  
**Split lock:** `9a01c05cfc7d33783c4f64b91b0ae423a4d3ad57439fd71ffc029a8f04585c78`

## One-line status

**E2b** primary model (val Dice ≈ 0.778 @ 0.55; test mean Dice ≈ 0.818). Streamlit demo code is ready; **local E2b `best.keras` is still required** from Google Drive for live inference.

## Stage board

| Stage | Status |
| --- | --- |
| 0–5 Science + test | COMPLETED |
| 6 FR CRISP-DM report | COMPLETED |
| 7 Final notebook | COMPLETED |
| 8 Presentation materials | COMPLETED |
| 9 Defense guide | COMPLETED |
| Streamlit app | **CODE COMPLETE** — blocked on local E2b checkpoint copy |

## Test leaderboard (mean/image)

| Exp | Thr | Dice |
| --- | ---: | ---: |
| E0 | 0.50 | 0.000 |
| E1 | 0.35 | 0.398 |
| E2a | 0.15 | 0.613 |
| E2b | **0.55** | **0.818** |

## Application

- Entry: `app.py` → `streamlit run app.py`
- Inference: `src/app_inference.py` (E2b only, thr 0.55, normalize `mobilenet_v2`)
- Docs: `docs/APP_USER_GUIDE.md`, `docs/APP_IMPLEMENTATION_PLAN.md`
- Checkpoint path: `checkpoints/phase4/E2b_unet_mobilenetv2_finetune_20261108_1721/best.keras`  
  SHA-256: `280170dc64347b0b4cc8ce3af9be40dec4ac3733b27d1abeb620cc0c8e88e432`

## Key science paths

- Report: `docs/RAPPORT_TECHNIQUE_CRISP_DM.md`
- Test: `docs/14_final_test_evaluation.md`
