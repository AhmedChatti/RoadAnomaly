"""Anomaly Hunter — interactive annotation game (E2b remains frozen)."""

from anomaly_hunter.gamification import PlayerProfile, apply_training_score, default_profile
from anomaly_hunter.masks import (
    binary_mask_from_canvas_json,
    load_ground_truth_mask,
    remap_semantic_to_binary,
    validate_player_mask,
)
from anomaly_hunter.pool import Challenge, ChallengeType, load_training_challenges, pick_challenge
from anomaly_hunter.scoring import score_training_round
from anomaly_hunter.store import AnnotationStore, ContributionStatus

__all__ = [
    "AnnotationStore",
    "Challenge",
    "ChallengeType",
    "ContributionStatus",
    "PlayerProfile",
    "apply_training_score",
    "binary_mask_from_canvas_json",
    "default_profile",
    "load_ground_truth_mask",
    "load_training_challenges",
    "pick_challenge",
    "remap_semantic_to_binary",
    "score_training_round",
    "validate_player_mask",
]
