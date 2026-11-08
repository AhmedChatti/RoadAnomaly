# Streamlit Security & Privacy Audit

**Date:** 2026-11-08

## Admin auth

| Control | Status |
| --- | --- |
| `ROADANOMALY_ADMIN_TOKEN` env only | PASS |
| Banned `roadanomaly-local-admin` | PASS |
| Min length 16 | PASS |
| Fail closed if missing | PASS |
| Secret not shown in UI | PASS |
| Confirm before review/export | PASS |

## Uploads

| Control | Status |
| --- | --- |
| Extension allow-list | PASS (jpg/jpeg/png/webp) |
| Max 8 Mo | PASS (analyser + discovery) |
| Path basename only for display name | PASS |
| Contribution IDs hex-only | PASS |
| Contribution dirs under store root | PASS |

## Privacy

| Control | Status |
| --- | --- |
| Analyser: no permanent store | PASS |
| Discovery/train image retain only with consent | PASS |
| `data/annotations/` gitignored | PASS |
| No auto-promote to training | PASS |
| No claimed automatic deletion | Documented honestly |

## Data leakage

| Control | Status |
| --- | --- |
| Game pool excludes test | PASS |
| E3 export excludes test + val | PASS |
| approved ≠ eligible | PASS |
| Export never overwrites version | PASS |
