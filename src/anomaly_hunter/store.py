"""Human annotation contribution store (candidates only — never auto-promoted)."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from anomaly_hunter.masks import MAX_UPLOAD_BYTES, validate_player_mask
from anomaly_hunter.pool import content_fingerprint
from app_inference import E2B_CHECKPOINT_SHA256, E2B_EXPERIMENT_ID, E2B_THRESHOLD

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE_ROOT = ROOT / "data" / "annotations"

STATUSES = (
    "pending_review",
    "approved",
    "rejected",
    "duplicate",
    "requires_additional_review",
)


class ContributionStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"
    REQUIRES_ADDITIONAL_REVIEW = "requires_additional_review"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Contribution:
    contribution_id: str
    challenge_type: str
    player_session_id: str
    timestamp_utc: str
    verification_status: str
    consent_to_contribute: bool
    model_version: str
    model_threshold: float
    model_checkpoint_sha256: str
    image_source: str
    license_info: str
    frame_id: str | None = None
    image_retained: bool = False
    image_sha256: str | None = None
    duplicate_of: str | None = None
    review_comment: str = ""
    reviewer_id: str | None = None
    reviewed_at_utc: str | None = None
    game_score: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnnotationStore:
    """Filesystem-backed store under data/annotations/ (gitignored uploads)."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else Path(
            os.environ.get("ROADANOMALY_ANNOTATION_ROOT", str(DEFAULT_STORE_ROOT))
        )
        self.contributions_dir = self.root / "contributions"
        self.index_path = self.root / "index.jsonl"
        self.leaderboard_path = self.root / "leaderboard.json"
        self.contributions_dir.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.index_path.is_file():
            self.index_path.write_text("", encoding="utf-8")
        if not self.leaderboard_path.is_file():
            self.leaderboard_path.write_text("[]", encoding="utf-8")

    def _contrib_dir(self, contribution_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", contribution_id):
            raise ValueError("contribution_id invalide")
        return self.contributions_dir / contribution_id

    def find_duplicate_sha(self, image_sha256: str) -> str | None:
        if not image_sha256:
            return None
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("image_sha256") == image_sha256 and row.get("verification_status") != "rejected":
                return row.get("contribution_id")
        return None

    def submit(
        self,
        *,
        challenge_type: str,
        player_session_id: str,
        consent_to_contribute: bool,
        player_mask: np.ndarray,
        image_rgb: np.ndarray | None,
        image_bytes: bytes | None,
        ai_mask: np.ndarray | None,
        ai_probability: np.ndarray | None,
        image_source: str,
        license_info: str,
        frame_id: str | None = None,
        game_score: dict[str, Any] | None = None,
        retain_image: bool = False,
    ) -> Contribution:
        if image_bytes is not None and len(image_bytes) > MAX_UPLOAD_BYTES:
            raise ValueError("Image trop volumineuse (max 8 Mo).")

        h, w = int(player_mask.shape[0]), int(player_mask.shape[1])
        mask = validate_player_mask(player_mask, expected_hw=(h, w))

        image_sha = content_fingerprint(image_bytes) if image_bytes else None
        if image_rgb is not None and image_sha is None:
            # Fingerprint from array bytes for training challenges
            image_sha = content_fingerprint(
                np.ascontiguousarray(image_rgb).tobytes() + f"{h}x{w}".encode()
            )

        # Duplicate fingerprinting applies to discovery uploads; training frames may be replayed.
        dup = None
        if challenge_type == "discovery" and image_sha:
            dup = self.find_duplicate_sha(image_sha)
        status = (
            ContributionStatus.DUPLICATE.value
            if dup
            else ContributionStatus.PENDING_REVIEW.value
        )

        cid = _safe_id()
        cdir = self._contrib_dir(cid)
        cdir.mkdir(parents=True, exist_ok=False)

        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(cdir / "player_mask.png")

        if ai_mask is not None:
            ai_b = validate_player_mask(ai_mask, expected_hw=(h, w))
            Image.fromarray((ai_b * 255).astype(np.uint8), mode="L").save(cdir / "ai_mask.png")
        if ai_probability is not None:
            probs = np.asarray(ai_probability, dtype=np.float32)
            if probs.shape != (h, w):
                raise ValueError("Carte de probabilité mal formée.")
            np.savez_compressed(cdir / "ai_probability.npz", probability=probs)

        image_retained = False
        if consent_to_contribute and retain_image and image_rgb is not None:
            Image.fromarray(image_rgb.astype(np.uint8), mode="RGB").save(cdir / "image.png")
            image_retained = True
        elif consent_to_contribute and retain_image and image_bytes is not None:
            (cdir / "image_original.bin").write_bytes(image_bytes)
            image_retained = True

        meta = Contribution(
            contribution_id=cid,
            challenge_type=challenge_type,
            player_session_id=player_session_id,
            timestamp_utc=_utc_now(),
            verification_status=status,
            consent_to_contribute=bool(consent_to_contribute),
            model_version=E2B_EXPERIMENT_ID,
            model_threshold=float(E2B_THRESHOLD),
            model_checkpoint_sha256=E2B_CHECKPOINT_SHA256,
            image_source=image_source,
            license_info=license_info,
            frame_id=frame_id,
            image_retained=image_retained,
            image_sha256=image_sha,
            duplicate_of=dup,
            game_score=game_score or {},
            notes=(
                "Candidate annotation only — not ground truth. "
                "Not auto-promoted into training."
            ),
        )
        (cdir / "meta.json").write_text(
            json.dumps(meta.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        with self.index_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(meta.to_dict(), ensure_ascii=False) + "\n")
        return meta

    def list_contributions(
        self, *, status: str | None = None
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if status and row.get("verification_status") != status:
                continue
            rows.append(row)
        # Prefer freshest meta.json if present
        refreshed: list[dict[str, Any]] = []
        for row in rows:
            meta_path = self._contrib_dir(row["contribution_id"]) / "meta.json"
            if meta_path.is_file():
                refreshed.append(json.loads(meta_path.read_text(encoding="utf-8")))
            else:
                refreshed.append(row)
        refreshed.sort(key=lambda r: r.get("timestamp_utc") or "", reverse=True)
        return refreshed

    def load_contribution_assets(self, contribution_id: str) -> dict[str, Any]:
        cdir = self._contrib_dir(contribution_id)
        meta = json.loads((cdir / "meta.json").read_text(encoding="utf-8"))
        out: dict[str, Any] = {"meta": meta}
        for name, key in (
            ("image.png", "image"),
            ("player_mask.png", "player_mask"),
            ("ai_mask.png", "ai_mask"),
        ):
            path = cdir / name
            if path.is_file():
                out[key] = np.asarray(Image.open(path))
        prob_path = cdir / "ai_probability.npz"
        if prob_path.is_file():
            out["ai_probability"] = np.load(prob_path)["probability"]
        return out

    def review(
        self,
        contribution_id: str,
        *,
        status: str,
        comment: str = "",
        reviewer_id: str = "admin",
    ) -> dict[str, Any]:
        if status not in STATUSES:
            raise ValueError(f"Statut inconnu: {status}")
        cdir = self._contrib_dir(contribution_id)
        meta_path = cdir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["verification_status"] = status
        meta["review_comment"] = comment
        meta["reviewer_id"] = reviewer_id
        meta["reviewed_at_utc"] = _utc_now()
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        self._rewrite_index_row(meta)
        return meta

    def _rewrite_index_row(self, meta: dict[str, Any]) -> None:
        lines = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("contribution_id") == meta["contribution_id"]:
                lines.append(json.dumps(meta, ensure_ascii=False))
            else:
                lines.append(line)
        self.index_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def upsert_leaderboard(self, entry: dict[str, Any]) -> list[dict[str, Any]]:
        board = json.loads(self.leaderboard_path.read_text(encoding="utf-8") or "[]")
        sid = entry.get("session_id")
        board = [b for b in board if b.get("session_id") != sid]
        board.append(entry)
        board.sort(key=lambda r: int(r.get("points", 0)), reverse=True)
        board = board[:50]
        self.leaderboard_path.write_text(
            json.dumps(board, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return board
