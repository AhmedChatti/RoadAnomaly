# Security and Privacy Audit

## Administrator authentication

| Rule | Status |
| --- | --- |
| Secret from `ROADANOMALY_ADMIN_TOKEN` only | Implemented |
| No predictable production fallback | Implemented — `roadanomaly-local-admin` **banned** |
| Fail closed if unset / short / banned | Implemented |
| Constant-time compare (SHA-256 digests + HMAC) | Implemented |
| Token never logged or shown in UI captions | Implemented |
| Review & export gated | Implemented (`enforce_admin` on export) |

**Local setup**

```powershell
# Generate a strong secret (≥16 characters). Do not commit it.
$env:ROADANOMALY_ADMIN_TOKEN = "replace-with-a-long-random-secret"
streamlit run app.py
```

Optional Streamlit secrets file is **not** required; environment variable is authoritative. Do not put the token in the repository.

## Upload and annotation safety

- Max upload size: 8 Mo (`MAX_UPLOAD_BYTES`).
- Formats: JPG/JPEG/PNG for analyser and discovery.
- Contribution IDs: 32 hex chars only; directory must be a direct child of `contributions/`.
- Masks must be binary `{0,1}` and non-empty on submit.
- No arbitrary file download endpoints; Streamlit serves only in-memory download buttons for analyser outputs.

## Privacy lifecycle

| Data | Stored? | Where | Duration |
| --- | --- | --- | --- |
| Analyser uploads | No permanent store | Memory / session only | Session |
| Training challenge image | Only if consent + retain | `data/annotations/contributions/<id>/image.png` | Until manually deleted |
| Discovery image | Requires consent | Same | Until manually deleted |
| Player mask / AI mask / probs | On submit (probs if consent) | Same folder | Until manually deleted |
| Leaderboard | Local JSON | `data/annotations/leaderboard.json` | Until manually deleted |
| Pseudo | Session + local leaderboard | Not an account system | Local |

**Consent** is stored on each contribution (`consent_to_contribute`). Without consent, training rounds may still score in-session but images are not retained.

**We do not claim automatic deletion.** Admins must remove files under `data/annotations/` if erasure is required. That directory is gitignored.

## Ordinary users vs admin

- Navigation **Revue** is visible but all mutating actions require a valid admin token.
- Without `ROADANOMALY_ADMIN_TOKEN` configured, the review UI shows an error and cannot approve/export.
- Private contribution files are not exposed through the Analyser/Jouer pages.

## Threat notes (accepted academic prototype limits)

- Streamlit apps are not hardened multi-tenant SaaS; bind to localhost for demos.
- Session state is not a security boundary against a hostile local user on the same machine.
- Do not deploy publicly with a weak or missing admin token.
