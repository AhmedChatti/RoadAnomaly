# Phase 3 artefact gap notes (Stage 1)

**Date:** 2026-11-08  
**Purpose:** Document non-canonical / recovered artefacts without changing frozen scientific results.

## Canonical reproducibility (unchanged)

| Artefact | Status |
| --- | --- |
| `checkpoints/E1_cnn_scratch_20261108_1510/best.keras` | Present; SHA-256 `567bde32…0fb234` |
| E1 / E0 metadata and val CSVs | Untouched |
| Split lock | Untouched |
| Reloaded val mean Dice @ 0.35 | Matches frozen ≈ `0.568964` |

Canonical Phase 3 science does **not** depend on the abandoned E1_1501 weights file.

## Gap 1 — Validation qualitative figure

| Item | Detail |
| --- | --- |
| Missing path | `experiments/figures/phase3_E1_val_qualitative.png` |
| Cause | Local disk cleanup (ENOSPC), not a protocol change |
| Resolution | Regenerated via `scripts/regenerate_phase3_e1_val_qualitative.py` |
| Source | Canonical E1_1510 checkpoint + **validation split only** |
| Threshold | Operational val threshold from metadata (`0.35`) |
| Provenance | `experiments/figures/phase3_E1_val_qualitative.REGENERATED.json` |
| Label | Figure title includes `REGENERATED` |

Frozen metrics JSON was **not** rewritten by regeneration.

## Gap 2 — Abandoned E1_1501 checkpoint

| Item | Detail |
| --- | --- |
| Experiment | `E1_cnn_scratch_20261108_1501` (non-canonical / abandoned) |
| Metadata | Still present under `experiments/metadata/` |
| Checkpoint `best.keras` | **Unavailable** locally; no legitimate binary backup found |
| Policy | Do **not** fabricate weights; do **not** promote 1501 as canonical |
| Audit behaviour | Recorded as historical note / non-blocking (`abandoned_e1_checkpoint_status=unavailable`) |

## Local backup of canonical E1

`scripts/backup_phase3_checkpoint.py` restores/verifies  
`backups/phase3/E1_cnn_scratch_20261108_1510/best.keras` from the canonical source when missing.

Drive already holds a hash-verified copy (Colab readiness 2026-11-08T16:43:15Z).
