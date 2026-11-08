# Streamlit Test Report

**Date:** 2026-11-08

## Automated

| Suite | Result | Notes |
| --- | --- | --- |
| `tests.test_acceptance_audit` | Executed in QA pass | Freeze hashes, admin auth, eligibility, analyser upload helpers |
| `tests.test_anomaly_hunter` | Executed in QA pass | Tiles, scoring, store, pool exclusions |
| `tests.test_app_inference` | Executed in QA pass | Threshold 0.55, preprocess, fake model |
| Full `unittest discover -s tests` | **98 OK, 2 skipped** | AppTest skipped unless `ROADANOMALY_RUN_APPTEST=1` |

## AppTest

`tests/test_streamlit_apptest.py` — opt-in (slow TF). Covers nav buttons + Jouer page markdown.

## Browser / devices

| Check | Status |
| --- | --- |
| Playwright full matrix | **Not executed** this pass |
| Desktop layout (CSS) | Implemented |
| Mobile CSS breakpoints | Present; **manual** verification still required |
| Tile click UX in live Streamlit | **Manual** after `streamlit run app.py` |

## Scientific freeze (this pass)

```
threshold=0.55  input=512×288
const_sha_match=True
checkpoint_on_disk=True
checkpoint_sha_match=True
SHA=280170dc64347b0b4cc8ce3af9be40dec4ac3733b27d1abeb620cc0c8e88e432
```

## Manual checklist remaining

1. Open Analyser — upload JPG/PNG/WebP, analyse, download.
2. Jouer — select tiles, validate, quit, see thank-you.
3. Revue — set `ROADANOMALY_ADMIN_TOKEN`, review with confirmation, export.
4. Private/incognito window if any residual Streamlit session errors.
