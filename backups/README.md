# Backups

Local and Drive mirrors of **model weights** and critical experiment snapshots live here.

- Do **not** commit `*.keras` or other weight files to GitHub (see `.gitignore`).
- Canonical Phase 3 checkpoint source of truth (local):  
  `checkpoints/E1_cnn_scratch_20261108_1510/best.keras`
- After running `python scripts/backup_phase3_checkpoint.py`, a verified copy appears under  
  `backups/phase3/E1_cnn_scratch_20261108_1510/`.
- Upload that folder to Google Drive:  
  `My Drive/RoadAnomaly/checkpoints/phase3/E1_cnn_scratch_20261108_1510/`  
  and keep an extra copy under  
  `My Drive/RoadAnomaly/backups/phase3/`.

Always verify SHA-256 after copy. Never overwrite the original local checkpoint.
