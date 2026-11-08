# RoadAnomaly — Final Acceptance Audit

**Audit date:** 2026-11-08  
**Scope:** Analyser + Anomaly Hunter + Revue + annotation export + freeze integrity  
**HEAD at audit close:** see Git SHA in the delivery report after push.

## Overall status

**PARTIAL acceptance** — core scientific freeze and application workflows are verified; admin security and E3 eligibility hardening are implemented; some UX items remain limited by Streamlit (timer refresh without optional autorefresh package; device-local leaderboard only).

| Area | Classification |
| --- | --- |
| E2b scientific freeze | **PASS** |
| Analyser official inference | **PASS** (unit + fake-model + real SHA load) |
| Anomaly Hunter training / discovery logic | **PASS** (unit/integration) |
| Revue + admin auth (fail-closed) | **PASS** |
| Privacy / consent storage rules | **PASS** (with documented retention limits) |
| Data leakage prevention (export eligibility) | **PASS** |
| Soft→hard countdown gameplay | **PARTIAL** (reject after expiry; UI refresh best-effort) |
| Global multi-user leaderboard | **BLOCKED** (not implemented — local only) |
| Automatic E3 training | **BLOCKED** (intentionally not implemented) |

## Features verified

1. **Analyser:** JPG/JPEG/PNG load, GIF reject, E2b@0.55, mask/overlay alignment, downloads encoded as PNG.
2. **Jouer:** train/val pool only; brush/rect/eraser path; scoring Dice/IoU; anti-replay points; discovery pending_review without validated Dice.
3. **Revue:** requires `ROADANOMALY_ADMIN_TOKEN`; banned default `roadanomaly-local-admin`; review + eligible export.
4. **Export:** versioned dirs, no overwrite, excludes test/val frames, requires consent/source/license/image retention.
5. **Freeze:** checkpoint SHA `280170dc…`; Stage 5 CSV + freeze JSON hashes pinned in tests.

## Features fixed during audit

- Removed predictable admin fallback; fail-closed if secret missing/banned/short.
- Distinguishes `approved` vs `eligible_for_e3_training`.
- Excludes **validation** (not only test) from E3 *training* exports.
- Hard reject late / empty / duplicate-score submissions.
- Near-duplicate fingerprint field + Hamming filter on export.
- Path-safe contribution IDs; export refuses traversal.

## Test results

Executed: `python -m unittest discover -s tests -v` → **95 tests, OK** (2026-11-08).

Acceptance-focused module: `tests/test_acceptance_audit.py`.

Post-audit E2b checkpoint SHA re-verified: `280170dc…` (unchanged).

## Remaining limitations

- Streamlit countdown UI updates only on interaction unless `streamlit-autorefresh` is installed (optional).
- Leaderboard is device-local JSON — must not be presented as global.
- Browser UI click-through of canvas drawing is not automated in CI (logic covered by unit tests).
- Near-duplicate detection is coarse (16×16 perceptual bits), not a full perceptual-hash library.
- No automatic deletion scheduler for old annotations (retention until admin deletes files on disk).

## Launch

```powershell
$env:ROADANOMALY_ADMIN_TOKEN = "<strong-secret-at-least-16-chars>"
pip install -r requirements-app.txt
streamlit run app.py
```

## E2b SHA (pre-audit)

`280170dc64347b0b4cc8ce3af9be40dec4ac3733b27d1abeb620cc0c8e88e432` — matched on disk and in code constants.
