"""Game scoring — clearly separate from scientific model metrics reporting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metrics import binary_segmentation_metrics


@dataclass(frozen=True)
class RoundScore:
    """Training-challenge scorecard. Game points ≠ scientific benchmark."""

    player_dice: float
    player_iou: float
    ai_dice: float
    ai_iou: float
    points_earned: int
    feedback_fr: str
    beat_ai: bool


def _points_from_dice(dice: float) -> int:
    """Map Dice to game points (0–100). Not a scientific claim."""
    d = float(np.clip(dice, 0.0, 1.0))
    if d >= 0.85:
        return 100
    if d >= 0.70:
        return 80
    if d >= 0.55:
        return 60
    if d >= 0.40:
        return 40
    if d >= 0.25:
        return 20
    if d > 0.05:
        return 10
    return 0


def _feedback(dice: float, beat_ai: bool) -> str:
    if dice >= 0.85:
        base = "Excellent contour ! Votre annotation est très proche de la référence."
    elif dice >= 0.70:
        base = "Très bon travail. Quelques contours restent à peaufiner."
    elif dice >= 0.55:
        base = "Pas mal. Essayez d’englober entièrement l’obstacle sans trop de fond."
    elif dice >= 0.35:
        base = "Résultat partiel. Concentrez-vous sur l’objet principal sur la chaussée."
    elif dice > 0.05:
        base = "Le chevauchement est faible. Observez bien la forme de l’obstacle."
    else:
        base = "Peu de correspondance avec la référence. Réessayez sur une nouvelle image."
    if beat_ai:
        base += " Bonus : vous avez dépassé la prédiction automatique sur cet essai (score de jeu)."
    return base


def score_training_round(
    *,
    ground_truth: np.ndarray,
    player_mask: np.ndarray,
    ai_mask: np.ndarray,
) -> RoundScore:
    """Compare player and AI masks to verified ground truth. Requires GT."""
    gt = (np.asarray(ground_truth) > 0).astype(np.uint8)
    player = (np.asarray(player_mask) > 0).astype(np.uint8)
    ai = (np.asarray(ai_mask) > 0).astype(np.uint8)
    if gt.shape != player.shape or gt.shape != ai.shape:
        raise ValueError(
            f"Shape mismatch GT{gt.shape} player{player.shape} ai{ai.shape}"
        )

    p_m = binary_segmentation_metrics(gt, player)
    a_m = binary_segmentation_metrics(gt, ai)
    points = _points_from_dice(p_m["dice"])
    beat = p_m["dice"] > a_m["dice"] + 1e-9
    if beat:
        points = min(100, points + 15)

    return RoundScore(
        player_dice=float(p_m["dice"]),
        player_iou=float(p_m["iou"]),
        ai_dice=float(a_m["dice"]),
        ai_iou=float(a_m["iou"]),
        points_earned=int(points),
        feedback_fr=_feedback(float(p_m["dice"]), beat),
        beat_ai=beat,
    )
