# Environment synchronization and Colab workflow

**Project:** RoadAnomaly  
**Date:** 8 November 2026  
**Scope:** Infrastructure only (no Phase 4 training)  
**GitHub:** https://github.com/AhmedChatti/RoadAnomaly.git (`main`)

---

## 1. Four-environment architecture

```
Cursor (dev) ──commit/push──► GitHub (private code) ──clone/pull──► Colab (GPU)
     │                                              ▲
     │ local checkpoints / docs                     │ mount
     ▼                                              │
Google Drive (MyDrive/RoadAnomaly) ◄── save artefacts / dataset ──┘
```

| Environment | Source of truth for | Not responsible for |
| --- | --- | --- |
| **Cursor** | Python code, tests, docs, notebooks, Git commits | Long GPU training |
| **GitHub** | Versioned source, configs, tests, notebooks, docs, lightweight metadata/CSV | Raw images, masks, `*.keras`, secrets |
| **Google Drive** | Original dataset copy, checkpoints, figures, Colab readiness reports, backups | Authoritative source code |
| **Google Colab** | GPU execution of **approved** Git commits; writes artefacts to Drive | Independent code fork |

Colab must always pull code from GitHub. Drive must always hold persistent experiment outputs.

---

## 2. Repository layout (code) vs Drive layout (data)

### 2.1 Local / GitHub project (Cursor)

Tracked (examples): `src/`, `tests/`, `configs/`, `docs/`, `notebooks/`, `scripts/`, `data/processed/split_manifest.csv`, `experiments/metadata/*.json`, `experiments/results/*.csv`.

Ignored (examples): `frames/`, `checkpoints/**/*.keras`, `experiments/figures/*`, `backups/*` weights, `.env`, tokens.

### 2.2 Official dataset on disk (must stay paired)

From the dataset `readme.md`:

```
RoadAnomaly/
  frame_list.json
  credits.txt
  readme.md
  frames/
    <frame>.webp
    <frame>.labels/
      labels_semantic.png    # 0=background, 2=anomaly
      labels_instance.png    # optional
      ...
```

Local discovery (this machine): **60** WebP images + **60** `.labels` directories under `frames/`.

Training code reads **original** frames via `split_manifest.csv` (on-the-fly 512×288). Historical letterboxed `data/processed/{train,val,test}/` images are **not** the Phase 2+ training tensors.

### 2.3 Google Drive target structure

```
My Drive/
└── RoadAnomaly/
    ├── README_DRIVE.txt
    ├── dataset/
    │   └── original/
    │       ├── frame_list.json      (optional copy)
    │       ├── credits.txt
    │       ├── readme.md
    │       └── frames/              ← FULL upload (webp + .labels)
    ├── checkpoints/
    │   ├── phase3/
    │   │   └── E1_cnn_scratch_20261108_1510/
    │   │       ├── best.keras
    │   │       └── SHA256.txt
    │   └── phase4/
    │       ├── E2a/
    │       └── E2b/
    ├── experiments/
    │   ├── metadata/
    │   ├── results/
    │   └── figures/
    └── backups/
        ├── phase3/
        └── phase4/
```

Create folders with (after Drive mount):

```bash
python scripts/init_drive_layout.py --root "/content/drive/MyDrive/RoadAnomaly"
```

Or run section 2 of `notebooks/00_colab_environment_setup.ipynb`.

---

## 3. GitHub synchronization workflow

### 3.1 Observed status (at documentation time)

| Item | Value |
| --- | --- |
| Remote `origin` | `https://github.com/AhmedChatti/RoadAnomaly.git` |
| Branch | `main` tracking `origin/main` |
| Last pushed commit | `1849009` — *Complete Phase 2 data pipeline* |
| Local uncommitted work | Phase 3 code fixes, docs 09/10, metadata JSON, audit script, plus this infrastructure work |

**No credentials or `frames/` files are tracked.** Report PNGs under `reports/figures/` are tracked (small exploration figures). Checkpoints are gitignored.

### 3.2 Safe workflow

1. Develop in Cursor.  
2. Review `git status` / `git diff`.  
3. Commit **only** source, docs, metadata, lightweight CSV (no tokens, no weights, no raw frames).  
4. Push to `main` (or a PR branch if preferred).  
5. In Colab, set `GITHUB_COMMIT_SHA` to that commit (recommended) and pull/checkout.  
6. Record `git_commit_sha` in experiment metadata.

**This infrastructure task does not auto-commit or auto-push.** Approve a commit separately.

### 3.3 Private repo authentication in Colab

1. GitHub → Settings → Developer settings → Personal access token (classic) with `repo` scope, **or** fine-grained read access to this private repo.  
2. Colab → Secrets → create `GITHUB_TOKEN`.  
3. Notebook reads the secret via `google.colab.userdata` — never hardcode the token in cells or files.

---

## 4. Manual Google Drive upload procedure

Cursor **does not** have direct access to your Google account. Perform these steps once:

### 4.1 Dataset (required for any Colab experiment)

1. On the PC, open `C:\Users\man\Downloads\RoadAnomaly\frames\`.  
2. Upload the **entire** `frames` folder to:  
   `My Drive/RoadAnomaly/dataset/original/frames/`  
3. Confirm after upload:
   - 60 × `*.webp`
   - 60 × `*.labels/` directories
   - each `.labels` contains `labels_semantic.png`
4. **Never** upload images without their `.labels` folders.

Optional: also upload `frame_list.json`, `credits.txt`, `readme.md` into `dataset/original/` (also present in the Git clone).

### 4.2 Phase 3 checkpoint (required before Phase 4)

1. Locally run: `python scripts/backup_phase3_checkpoint.py`  
2. Upload folder  
   `backups/phase3/E1_cnn_scratch_20261108_1510/`  
   to **both**:
   - `My Drive/RoadAnomaly/checkpoints/phase3/E1_cnn_scratch_20261108_1510/`
   - `My Drive/RoadAnomaly/backups/phase3/E1_cnn_scratch_20261108_1510/`
3. Expected SHA-256:  
   `567bde326ee5b776bc34132b23012543b35f195ec8d60229d01196eb1a0fb234`
4. Re-check hash on Colab (notebook section 8) or any SHA-256 tool.  
5. **Do not** delete or overwrite  
   `checkpoints/E1_cnn_scratch_20261108_1510/best.keras` on the PC.

### 4.3 Phase 3 lightweight artefacts (recommended)

Copy to Drive `experiments/`:

- `experiments/metadata/E0_*.json`, `E1_cnn_scratch_20261108_1510.json`, `phase3_freeze_audit.json`
- `experiments/results/*phase3*` and E1 CSV files
- `experiments/figures/phase3_*.png`

(Metadata/CSV can also live on GitHub after you approve a commit.)

---

## 5. Colab setup notebook

File: `notebooks/00_colab_environment_setup.ipynb`

It:

1. Mounts Drive.  
2. Defines configurable paths.  
3. Creates Drive folder skeleton.  
4. Verifies dataset pairing.  
5. Clones/pulls GitHub using Colab Secret `GITHUB_TOKEN`.  
6. Pins optional `GITHUB_COMMIT_SHA`.  
7. Sets `ROADANOMALY_*` environment variables for frames + artefact roots.  
8. Prints Python / TF / Keras / NumPy versions.  
9. Detects GPU.  
10. Imports project code and runs unit tests.  
11. Verifies split lock `9a01c05c…`.  
12. Checks writable Drive output dirs.  
13. Verifies Phase 3 checkpoint hash on Drive when present.  
14. Writes `experiments/metadata/colab_environment_readiness.json` on Drive.

It does **not** train, download ImageNet weights, reshuffle data, or evaluate the test set for selection.

Path overrides (implemented in `src/config.py`):

| Environment variable | Purpose |
| --- | --- |
| `ROADANOMALY_ROOT` | Project root (clone dir) |
| `ROADANOMALY_FRAMES_DIR` | Original `frames` on Drive |
| `ROADANOMALY_ARTIFACTS_ROOT` | Drive project root for artefacts |
| `ROADANOMALY_CHECKPOINTS_DIR` / `_METADATA_DIR` / `_RESULTS_DIR` / `_FIGURES_DIR` | Explicit artefact dirs |
| `ROADANOMALY_GIT_SHA` | Recorded commit for metadata |

---

## 6. Reproducibility convention

Every future experiment metadata JSON must include at least:

- `experiment_id`, `created_at_utc`
- `git_commit_sha`
- `dataset_fingerprint`
- `split_manifest_sha256` (must equal frozen lock)
- `input_height` / `input_width` (288 / 512)
- `model_name`, `architecture`
- `software_versions`, `gpu`
- `seed`, `hyperparameters`
- `history`
- `checkpoint_path`, `checkpoint_sha256`
- `validation_metrics`
- `threshold_selection` with **both** `default_during_training` (0.5) and `value` (val-selected)
- `artefact_locations` pointing at Drive paths
- `status`

Helper: `src/reproducibility.py` → `build_experiment_metadata_base()`.  
Template: `experiments/metadata/_template.json`.

**Crash safety:** write metadata + checkpoint to Drive on every best epoch (Phase 4 training code must do this). Colab runtime disks are ephemeral.

---

## 7. Phase 3 backup and recovery

| Location | Role |
| --- | --- |
| `checkpoints/E1_cnn_scratch_20261108_1510/best.keras` | Original local canonical weights |
| `backups/phase3/E1_cnn_scratch_20261108_1510/` | Verified local backup + `BACKUP_MANIFEST.json` |
| Drive `checkpoints/phase3/...` | Persistent copy for Colab |
| Drive `backups/phase3/...` | Second Drive copy |

Recovery: restore from Drive backup → verify SHA-256 → place under local `checkpoints/...` if needed. Never train over the canonical file in place; write new experiment IDs for new runs.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `Permission denied` cloning GitHub | Missing/invalid `GITHUB_TOKEN` secret | Recreate Colab Secret; ensure repo access |
| `Expected 60 webp` | Incomplete Drive upload | Re-upload full `frames/` including `.labels` |
| Split hash mismatch | Wrong `split_manifest.csv` / old clone | Checkout approved commit; do not reshuffle |
| No GPU in Colab | Runtime is CPU | Runtime → GPU (T4/L4/A100) |
| Native Windows “no GPU” in Cursor | TF≥2.11 limitation | Expected; use Colab for GPU training |
| Unit tests fail on Colab | `ROADANOMALY_FRAMES_DIR` unset / empty Drive | Fix env + dataset upload |
| Phase 3 hash mismatch on Drive | Corrupt/partial upload | Re-upload backup; compare `SHA256.txt` |
| Imports fail | `sys.path` / wrong clone dir | Re-run notebook sections 4–5 |

---

## 9. What is still manual

Until you complete them, Colab readiness will stay blocked:

1. Upload `frames/` (webp + labels) to Drive.  
2. Upload verified Phase 3 checkpoint to Drive.  
3. Create Colab Secret `GITHUB_TOKEN`.  
4. Review and approve a Git commit/push of pending Phase 3 + infrastructure files.  
5. Open the setup notebook on a **GPU** runtime and confirm `ready_for_phase4_training: true`.

---

## 10. Stopping rule

Infrastructure preparation ends here.  
**Do not start Phase 4 implementation or training** until you explicitly approve Phase 4 **and** the Colab readiness report is green.
