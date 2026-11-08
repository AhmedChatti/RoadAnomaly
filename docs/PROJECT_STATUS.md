# RoadAnomaly — Project Status

**Last verified:** 2026-11-08 (Final acceptance audit — security, eligibility, freeze)  
**Branch:** `main`  
**E2 training commit:** `9e734a1d0af48342f92c084216b1dd1d4c254741`  
**Split lock:** `9a01c05cfc7d33783c4f64b91b0ae423a4d3ad57439fd71ffc029a8f04585c78`

## One-line status

**E2b** primary (val Dice ≈ 0.778 @ 0.55; test mean Dice ≈ 0.818). App + Anomaly Hunter accepted with PARTIAL UX limits; admin fail-closed; E3 export eligibility excludes test/val; freeze SHA intact.

## Stage board

| Stage | Status |
| --- | --- |
| 0–5 Science + test | COMPLETED |
| 6 FR CRISP-DM report | COMPLETED |
| 7 Final notebook | COMPLETED |
| 8 Presentation materials | COMPLETED |
| 9 Defense guide | COMPLETED |
| Streamlit app | **READY** — `streamlit run app.py` (E2b SHA verified locally) |
| Anomaly Hunter | **READY** — game + candidates + fail-closed admin + eligible export |
| Final acceptance audit | **PARTIAL** — see `docs/FINAL_ACCEPTANCE_AUDIT.md` |

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
- Game: `src/anomaly_hunter/` — training (train/val only) + discovery contributions
- Admin: `ROADANOMALY_ADMIN_TOKEN` required (≥16 chars; default banned)
- Docs: `docs/FINAL_ACCEPTANCE_AUDIT.md`, `docs/SECURITY_AND_PRIVACY_AUDIT.md`, `docs/DATA_LEAKAGE_PREVENTION.md`, hunter/pipeline/E3 guides
- Checkpoint path: `checkpoints/phase4/E2b_unet_mobilenetv2_finetune_20261108_1721/best.keras`  
  SHA-256: `280170dc64347b0b4cc8ce3af9be40dec4ac3733b27d1abeb620cc0c8e88e432`

## Key science paths

- Report: `docs/RAPPORT_TECHNIQUE_CRISP_DM.md`
- Test: `docs/14_final_test_evaluation.md`
