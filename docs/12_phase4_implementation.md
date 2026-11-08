# Phase 4 implementation — U-Net + MobileNetV2 (E2a / E2b)

**Status:** implementation + unit tests + audit corrections + **Stage 1 readiness**.  
**Training started:** NO  
**ImageNet weights downloaded:** NO  
**Test set accessed:** NO  
**Numerical Phase 4 results:** none yet  
**Readiness report:** `docs/PHASE4_READINESS_REPORT.md`

This document describes implemented code. It does **not** report validation scores. Scores exist only after an explicitly authorized Colab training run.

Stage 1 confirmed: config values are consumed by the builder/training helpers; artefact isolation and dual-auth gates hold; smoke CLI is the default and never calls `train_e2a`/`train_e2b`. Colab GPU/Drive infrastructure is green at commit `337cad0`, but **Phase 4 sources are still uncommitted** and must be synced (Stage 2) before Colab can execute E2a/E2b.

---

## Implemented and tested (no training required)

| Capability | Location |
| --- | --- |
| U-Net + MobileNetV2 builder (`weights=None`) | `src/models/unet_mobilenetv2.py` |
| E2a full encoder freeze + BN freeze | same |
| E2b exact `block_13_expand` start + BN freeze | `configure_e2b_finetuning` |
| Keras-compatible preprocess `uint8/127.5 - 1` | `normalize="mobilenet_v2"` in `src/preprocessing.py` |
| Dual auth gates | `phase4_training._require_training_authorization` |
| Git ancestry check vs `git_commit_required` | `assert_required_git_ancestry` |
| Val-only eval / threshold helpers | `src/phase4_evaluation.py` |
| Post-train threshold finalize + CSV | `finalize_phase4_threshold_selection` |
| Isolated artefact roots under `…/phase4/` | `phase4_artefact_dirs` |
| Smoke script default (`weights=None`) | `scripts/run_phase4_experiments.py` |

## Requires authorized training (not run in this phase)

Both flags together:

```bash
python scripts/run_phase4_experiments.py \
  --authorize-training \
  --authorize-imagenet-download
```

That path will:

1. Validate `337cad0…` is an **ancestor** of `HEAD` (exact equality not required).  
2. Download ImageNet MobileNetV2 weights.  
3. Train E2a then E2b (E2b loads best E2a checkpoint).  
4. Select checkpoints by **validation mean Dice @ threshold 0.5**.  
5. Run **validation-only** threshold grid; write CSV + metadata `threshold_selection.value`.  

## Future results (do not exist yet)

- E2a / E2b validation Dice, IoU, curves, qualitative panels  
- Comparison table vs E0 / E1 with real numbers  
- Any claim that transfer learning “works” on this dataset  

---

## Why MobileNetV2 + U-Net

| Choice | Rationale |
| --- | --- |
| **U-Net** | Pixel-level segmentation with skips recovers spatial detail. |
| **MobileNetV2** | Lightweight ImageNet encoder for Colab T4 + 42-image train set. |
| **Transfer learning** | Tests whether pretrained features beat E1 from-scratch CNN. |

| Experiment | Encoder | Encoder training | Purpose |
| --- | --- | --- | --- |
| E1 | CNN from scratch | Fully trained | Baseline |
| E2a | MobileNetV2 | Frozen | Transfer learning |
| E2b | MobileNetV2 | From `block_13_expand` onward (BN frozen) | Controlled fine-tuning |

---

## Architecture

- Input `(B, 288, 512, 3)`; output `(B, 288, 512, 1)` sigmoid.  
- Skips: `block_1/3/6/13_expand_relu`; bottleneck `out_relu`.  
- Decoder filters from `configs/phase4.json`; **`final_conv_filters` is consumed by the builder**.  
- No decoder BatchNorm.

---

## Preprocessing

| Mode | Transform | Used by |
| --- | --- | --- |
| `scale_0_1` | `/255` | E0 / E1 |
| `imagenet` | mean/std on `[0,1]` | not used for MobileNetV2 |
| **`mobilenet_v2`** | `uint8/127.5 - 1` → `[-1,1]` | **Phase 4** |

No in-model `preprocess_input` (avoids double normalization).

---

## Artefact isolation

| Kind | Path |
| --- | --- |
| Checkpoints | `checkpoints/phase4/<experiment_id>/` |
| Metadata | `experiments/metadata/phase4/` |
| Results | `experiments/results/phase4/` |
| Figures | `experiments/figures/phase4/` |

Phase 3 artefacts stay in their original locations. Writes into denylisted Phase 3 paths are refused.

---

## Threshold convention

| Role | Value | Where |
| --- | --- | --- |
| Training / checkpointing | **0.5** | `training_threshold` → `MeanDiceCallback` |
| Operational (post-hoc) | selected on **val** grid | `threshold_selection.value` + `*_val_threshold_grid.csv` |

Test split remains blocked until Phase 6.

---

## Safety gates

1. Both `--authorize-training` and `--authorize-imagenet-download` required together.  
2. Git ancestry of `git_commit_required` before training/ImageNet.  
3. Smoke mode (default, or `--smoke-only`) uses `weights=None`, no train, no download.  
4. `--smoke-only` cannot be combined with authorization flags.

---

## Limitations

1. No Phase 4 metrics until authorized Colab training.  
2. Fine-tune depth and skip choices are reasoned defaults, not ablated.  
3. Local Windows has no TF GPU; training is intended for Colab.
