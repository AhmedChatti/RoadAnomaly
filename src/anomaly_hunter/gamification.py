"""Session gamification — game scores only, not scientific performance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ACHIEVEMENTS: dict[str, str] = {
    "first_hunt": "Première chasse",
    "sharp_eye": "Œil de lynx (Dice ≥ 0,85)",
    "beat_ai": "Plus précis que l’IA (essai)",
    "ten_rounds": "10 défis complétés",
    "discovery_contributor": "Contributeur découverte",
    "centurion": "100 points cumulés",
}


@dataclass
class PlayerProfile:
    session_id: str
    display_name: str = "Chasseur anonyme"
    points: int = 0
    personal_best: int = 0
    challenges_played: int = 0
    correctly_annotated: int = 0  # training rounds with Dice >= 0.55
    level: int = 1
    achievements: list[str] = field(default_factory=list)
    played_challenge_ids: list[str] = field(default_factory=list)
    scored_challenge_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlayerProfile":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        filtered.setdefault("scored_challenge_ids", [])
        filtered.setdefault("achievements", [])
        filtered.setdefault("played_challenge_ids", [])
        return cls(**filtered)


def default_profile(session_id: str) -> PlayerProfile:
    return PlayerProfile(session_id=session_id)


def level_for_points(points: int) -> int:
    # Soft curve: 0→1, 100→2, 250→3, 500→4, ...
    thresholds = [0, 100, 250, 500, 900, 1400, 2000]
    level = 1
    for i, thr in enumerate(thresholds, start=1):
        if points >= thr:
            level = i
    return level


def apply_training_score(
    profile: PlayerProfile,
    *,
    challenge_id: str,
    points_earned: int,
    player_dice: float,
    beat_ai: bool,
) -> PlayerProfile:
    # Anti-replay: each challenge_id awards points at most once per session profile.
    if challenge_id in profile.scored_challenge_ids:
        profile.challenges_played += 1
        if challenge_id not in profile.played_challenge_ids:
            profile.played_challenge_ids.append(challenge_id)
        return profile

    awarded = max(0, int(points_earned))
    profile.points += awarded
    profile.personal_best = max(profile.personal_best, awarded)
    profile.challenges_played += 1
    profile.scored_challenge_ids.append(challenge_id)
    if challenge_id not in profile.played_challenge_ids:
        profile.played_challenge_ids.append(challenge_id)
    if player_dice >= 0.55:
        profile.correctly_annotated += 1
    profile.level = level_for_points(profile.points)

    def unlock(key: str) -> None:
        if key not in profile.achievements:
            profile.achievements.append(key)

    if profile.challenges_played == 1:
        unlock("first_hunt")
    if player_dice >= 0.85:
        unlock("sharp_eye")
    if beat_ai:
        unlock("beat_ai")
    if profile.challenges_played >= 10:
        unlock("ten_rounds")
    if profile.points >= 100:
        unlock("centurion")
    return profile


def apply_discovery_submission(
    profile: PlayerProfile, *, challenge_id: str | None = None
) -> PlayerProfile:
    profile.challenges_played += 1
    # Participation points once per discovery challenge id
    if challenge_id and challenge_id in profile.scored_challenge_ids:
        return profile
    profile.points += 5  # participation only — not a correctness claim
    if challenge_id:
        profile.scored_challenge_ids.append(challenge_id)
    profile.level = level_for_points(profile.points)
    if "discovery_contributor" not in profile.achievements:
        profile.achievements.append("discovery_contributor")
    if "first_hunt" not in profile.achievements:
        profile.achievements.append("first_hunt")
    return profile


def leaderboard_row(profile: PlayerProfile) -> dict[str, Any]:
    return {
        "Joueur": profile.display_name,
        "Points (jeu)": profile.points,
        "Niveau": profile.level,
        "Défis": profile.challenges_played,
        "Annotations OK*": profile.correctly_annotated,
        "Record": profile.personal_best,
    }
