"""Administrator authentication — fail-closed, no predictable production fallback."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from dataclasses import dataclass


ENV_VAR = "ROADANOMALY_ADMIN_TOKEN"
MIN_TOKEN_LENGTH = 16
# Explicitly banned even if somehow set — never acceptable in any deployment.
BANNED_TOKENS = frozenset(
    {
        "roadanomaly-local-admin",
        "admin",
        "password",
        "secret",
        "changeme",
        "123456",
        "token",
    }
)


@dataclass(frozen=True)
class AdminAuthResult:
    ok: bool
    reason: str


def configured_admin_token() -> str | None:
    """Return the configured token or None if missing/unsafe."""
    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        return None
    if raw.lower() in BANNED_TOKENS:
        return None
    if len(raw) < MIN_TOKEN_LENGTH:
        return None
    if re.search(r"[\x00-\x1f\x7f]", raw):
        return None
    return raw


def admin_auth_status() -> AdminAuthResult:
    token = configured_admin_token()
    if token is not None:
        return AdminAuthResult(True, "configured")
    env = os.environ.get(ENV_VAR, "").strip()
    if not env:
        return AdminAuthResult(
            False,
            f"Authentification admin non configurée. Définissez {ENV_VAR} "
            f"(≥{MIN_TOKEN_LENGTH} caractères, non prévisible).",
        )
    if env.lower() in BANNED_TOKENS:
        return AdminAuthResult(
            False,
            "Le jeton configuré est interdit (valeur prévisible / historique). "
            f"Choisissez un secret fort via {ENV_VAR}.",
        )
    if len(env) < MIN_TOKEN_LENGTH:
        return AdminAuthResult(
            False,
            f"Jeton trop court (minimum {MIN_TOKEN_LENGTH} caractères).",
        )
    return AdminAuthResult(False, "Jeton administrateur invalide.")


def verify_admin_token(provided: str | None) -> AdminAuthResult:
    """Constant-time compare against configured token. Fail closed."""
    status = admin_auth_status()
    if not status.ok:
        return status
    expected = configured_admin_token()
    assert expected is not None
    candidate = (provided or "").encode("utf-8")
    expected_b = expected.encode("utf-8")
    # Hash then compare so unequal lengths remain constant-time.
    a = hashlib.sha256(candidate).digest()
    b = hashlib.sha256(expected_b).digest()
    if hmac.compare_digest(a, b):
        return AdminAuthResult(True, "authenticated")
    return AdminAuthResult(False, "Jeton incorrect.")


def require_admin(provided: str | None) -> None:
    result = verify_admin_token(provided)
    if not result.ok:
        raise PermissionError(result.reason)
