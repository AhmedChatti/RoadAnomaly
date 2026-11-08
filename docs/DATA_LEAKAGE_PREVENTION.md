# Data Leakage Prevention — Future E3

## Immutable scientific boundary

- Locked **test** split (Stage 5): never used in game challenges for scoring against GT in a way that feeds training; never exported for E3 training.
- Locked **validation** split: may appear in *game* training challenges (human practice with GT) but is **permanently excluded** from E3 *training* exports.
- E2b weights / threshold / Stage 5 metrics: immutable reference.

## Status vocabulary (not interchangeable)

| Status | Meaning |
| --- | --- |
| submitted | Contribution created (`annotation_lifecycle=submitted`) |
| reviewed | Admin touched the record |
| approved | Human accepted as a candidate annotation |
| eligible_for_e3_training | Passed `assess_contribution_eligibility` at export time |
| rejected / duplicate / requires_additional_review | Not eligible |

**Approval alone never implies E3 eligibility.**

## Eligibility gates (`anomaly_hunter.eligibility`)

A contribution is exportable for E3 training only if **all** hold:

1. `verification_status == approved`
2. Explicit consent
3. Image retained on disk
4. Non-trivial `image_source` and `license_info`
5. Not marked duplicate
6. `frame_id` not in locked test or val sets
7. Training-challenge origins other than `train` are rejected for E3 train export
8. Exact SHA-256 not already in the export batch
9. Near-duplicate fingerprint Hamming distance ≤ 5 vs items already kept → skip

If eligibility cannot be established, the item is listed under `excluded` with reasons.

## Versioned exports

- Path: `data/annotations/exports/v<UTC stamp>/`
- Never overwrites an existing version directory (suffix `_001`, …).
- Manifest lists every kept item (origin, license, scene_group, SHA) and every exclusion reason.
- Scene groups are recorded for **future** grouped splitting; automatic split execution is out of scope (see `FUTURE_E3_ROADMAP.md`).

## Game vs train leakage

| Activity | Allowed splits |
| --- | --- |
| Anomaly Hunter GT challenges | train + val only (test forbidden) |
| E3 training export | eligible discovery / consented train-origin only — **no test, no val** |
