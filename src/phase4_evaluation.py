"""Phase 4 evaluation helpers.

Hard-blocks test-set access for Phase 4 model selection / threshold tuning.
Reuses the shared metric definitions from `evaluation.py`.
"""

from __future__ import annotations

from typing import Any, Callable

from evaluation import evaluate_predictor_on_split, select_threshold_on_validation
from preprocessing import NormalizeMode

PHASE4_NORMALIZE: NormalizeMode = "mobilenet_v2"


def evaluate_phase4_on_split(
    predict_proba_fn: Callable,
    split: str,
    *,
    threshold: float = 0.5,
    batch_size: int = 4,
) -> dict[str, Any]:
    if split == "test":
        raise RuntimeError(
            "Phase 4 forbids test-split access. Test evaluation is reserved for Phase 6."
        )
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported Phase 4 split: {split}")
    return evaluate_predictor_on_split(
        predict_proba_fn,
        split,  # type: ignore[arg-type]
        threshold=threshold,
        normalize=PHASE4_NORMALIZE,
        batch_size=batch_size,
        allow_test=False,
    )


def select_phase4_threshold_on_validation(
    predict_proba_fn: Callable,
    *,
    batch_size: int = 4,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    return select_threshold_on_validation(
        predict_proba_fn,
        thresholds=thresholds,
        normalize=PHASE4_NORMALIZE,
        batch_size=batch_size,
    )
