# Streamlit Full QA Audit — RoadAnomaly

**Date:** 2026-11-08  
**Scope:** Analyser · Jouer (Anomaly Hunter) · Revue  
**Method:** Code inventory + unit/AppTest automation + freeze hash verification. Full Playwright browser grid was not available in this pass.

## Architecture

| Layer | Path | Role |
| --- | --- | --- |
| Entry UI | `app.py` | Navigation (buttons), Analyser, welcome dialog, CSS theme |
| Inference | `src/app_inference.py` | Frozen E2b@0.55 load/predict/overlay/downloads |
| Game + admin | `src/anomaly_hunter/` | Tile picker, scoring, store, eligibility, export, admin UI |
| Storage | `data/annotations/` (gitignored) | Consented contributions + local leaderboard |
| Config | `.streamlit/config.toml` | Light theme, high-contrast text |

Navigation: **Analyser | Jouer | Revue** via plain `st.button` (Streamlit 1.67 radio/segmented_control session serde is unsafe).

## Feature inventory (actual behavior)

### Analyser
- Upload JPG/JPEG/PNG/WebP (≤8 Mo), example train image, preview, Analyse / Nouvelle image
- Official prediction only (threshold not shown); results tabs; downloads; FAQ; privacy caption
- Onboarding dialog once (query/session)

### Jouer
- **Tile grid 3×3** (reCAPTCHA-style) — **no brush/timer** (docs previously outdated; corrected)
- Training (train/val only) + Discovery upload
- Quit anytime; thank-you after contribute/quit
- Local gamification / leaderboard (device-local)

### Revue
- Fail-closed admin (`ROADANOMALY_ADMIN_TOKEN`, ≥16 chars; banned defaults)
- Filter / inspect / confirm before review & export
- Export = eligible only (approved ≠ eligible; no test/val)

## Defects found & disposition

| Defect | Status |
| --- | --- |
| Radio/segmented_control `TypeError` on Streamlit 1.67 | **Fixed** — button nav |
| Low-contrast dark-theme text on light CSS | **Fixed** — theme + CSS |
| Drawable-canvas `image_to_url` crash | **Obsolete** — tile UI replaces drawing |
| Prompt asked for timer/brush testing | **N/A** — current product uses tiles, no timer |
| Analyser uploader omitted WebP / size check | **Fixed** |
| Admin actions without confirmation | **Fixed** |
| Docs still describing brush/timer | **Updated** |

## Testing strategy

1. Unit: metrics, pool exclusions, store, eligibility, freeze hashes, AppTest smoke  
2. Manual still required: canvas tile clicks in real browser, mobile layout eyeball, real E2b visual overlay  
3. No Playwright run in this environment for full matrix

## Scientific freeze

- Threshold 0.55, 512×288, SHA `280170dc…` — verified in tests / local checkpoint when present.
