"""Streamlit UI for Anomaly Hunter (French) + admin review.

Gameplay: reCAPTCHA-style tile selection (no freehand drawing, no timer).
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

from anomaly_hunter.export import export_approved_dataset
from anomaly_hunter.gamification import (
    ACHIEVEMENTS,
    apply_discovery_submission,
    apply_training_score,
    default_profile,
)
from anomaly_hunter.masks import (
    DEFAULT_TILE_GRID,
    MAX_UPLOAD_BYTES,
    difference_view,
    load_ground_truth_mask,
    mask_from_selected_tiles,
    overlay_mask_rgb,
    overlay_tile_grid_rgb,
    validate_player_mask,
)
from anomaly_hunter.pool import ChallengeType, load_training_challenges, pick_challenge
from anomaly_hunter.scoring import score_training_round
from anomaly_hunter.store import STATUSES, AnnotationStore
from app_inference import (
    E2B_THRESHOLD,
    colorize_mask,
    load_user_image,
    predict_e2b,
)

TILE_GRID = DEFAULT_TILE_GRID


def _ensure_profile() -> None:
    if "hunter_session_id" not in st.session_state:
        st.session_state["hunter_session_id"] = uuid.uuid4().hex
    if "hunter_profile" not in st.session_state:
        st.session_state["hunter_profile"] = default_profile(
            st.session_state["hunter_session_id"]
        ).to_dict()


def _profile():
    from anomaly_hunter.gamification import PlayerProfile

    _ensure_profile()
    return PlayerProfile.from_dict(st.session_state["hunter_profile"])


def _save_profile(profile) -> None:
    st.session_state["hunter_profile"] = profile.to_dict()
    store = AnnotationStore()
    store.upsert_leaderboard(
        {
            "session_id": profile.session_id,
            "display_name": profile.display_name,
            "points": profile.points,
            "level": profile.level,
            "challenges_played": profile.challenges_played,
            "correctly_annotated": profile.correctly_annotated,
            "personal_best": profile.personal_best,
        }
    )


@st.cache_resource(show_spinner=False)
def _get_model():
    from app_inference import load_e2b_model

    return load_e2b_model(verify_sha=True)


def _clear_game_state() -> None:
    for k in list(st.session_state.keys()):
        if k.startswith("game_") or k.startswith("tile_btn_"):
            del st.session_state[k]


def render_player_stats() -> None:
    profile = _profile()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Points (jeu)", profile.points)
    c2.metric("Niveau", profile.level)
    c3.metric("Défis joués", profile.challenges_played)
    c4.metric("Record", profile.personal_best)
    st.caption(
        "Ces scores sont ludiques. Ils ne mesurent pas la performance scientifique du modèle."
    )
    if profile.achievements:
        labels = [ACHIEVEMENTS.get(a, a) for a in profile.achievements]
        st.write("Succès : " + " · ".join(labels))


def _start_training_round() -> None:
    challenges = load_training_challenges()
    if not challenges:
        st.error(
            "Aucun défi d’entraînement disponible (images train/val introuvables). "
            "Vérifiez le dossier frames/ ou data/processed."
        )
        return
    profile = _profile()
    challenge = pick_challenge(
        challenges, exclude_ids=set(profile.played_challenge_ids)
    )
    if challenge is None:
        st.warning("Pool de défis vide.")
        return
    rgb = challenge.load_rgb()
    st.session_state["game_mode"] = "training"
    st.session_state["game_phase"] = "select"
    st.session_state["game_challenge"] = {
        "challenge_id": challenge.challenge_id,
        "frame_id": challenge.frame_id,
        "image_path": str(challenge.image_path),
        "mask_path": str(challenge.mask_path),
        "license_note": challenge.license_note,
        "source_note": challenge.source_note,
        "split": challenge.split,
    }
    st.session_state["game_image"] = rgb
    st.session_state["game_result"] = None
    st.session_state["game_submitted"] = False
    st.session_state["game_selected_tiles"] = set()
    st.session_state["game_round_id"] = uuid.uuid4().hex[:8]


def _start_discovery_round(image_rgb: np.ndarray, filename: str, meta: dict[str, Any]) -> None:
    st.session_state["game_mode"] = "discovery"
    st.session_state["game_phase"] = "select"
    st.session_state["game_challenge"] = {
        "challenge_id": f"discovery::{meta.get('content_sha256', '')[:12]}",
        "frame_id": None,
        "filename": filename,
        "license_note": meta.get("license_claim", ""),
        "source_note": meta.get("source_claim", ""),
        "content_sha256": meta.get("content_sha256"),
        "image_bytes_len": meta.get("byte_size"),
    }
    st.session_state["game_image"] = image_rgb
    st.session_state["game_image_bytes"] = meta.get("image_bytes")
    st.session_state["game_result"] = None
    st.session_state["game_submitted"] = False
    st.session_state["game_selected_tiles"] = set()
    st.session_state["game_round_id"] = uuid.uuid4().hex[:8]


def _selected_tiles() -> set[tuple[int, int]]:
    raw = st.session_state.get("game_selected_tiles") or set()
    out: set[tuple[int, int]] = set()
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.add((int(item[0]), int(item[1])))
    return out


def _toggle_tile(row: int, col: int) -> None:
    selected = _selected_tiles()
    key = (row, col)
    if key in selected:
        selected.remove(key)
    else:
        selected.add(key)
    st.session_state["game_selected_tiles"] = selected


def _render_tile_picker(image_rgb: np.ndarray) -> np.ndarray | None:
    """reCAPTCHA-style grid: click every tile that contains the obstacle."""
    h, w = image_rgb.shape[:2]
    selected = _selected_tiles()
    round_id = st.session_state.get("game_round_id", "r")

    st.markdown("### Sélectionnez les cases où se trouve l’obstacle")
    st.caption(
        "Comme sur Google : cliquez sur toutes les cases qui contiennent l’obstacle. "
        "Vous pouvez modifier votre choix librement, sans limite de temps."
    )

    preview = overlay_tile_grid_rgb(image_rgb, selected, grid=TILE_GRID)
    st.image(preview, use_container_width=True)

    st.write("Cases :")
    for row in range(TILE_GRID):
        cols = st.columns(TILE_GRID)
        for col in range(TILE_GRID):
            y0, y1, x0, x1 = (
                row * h // TILE_GRID,
                (row + 1) * h // TILE_GRID,
                col * w // TILE_GRID,
                (col + 1) * w // TILE_GRID,
            )
            tile = image_rgb[y0:y1, x0:x1]
            # Slightly enlarge tiny previews for buttons area
            with cols[col]:
                st.image(tile, use_container_width=True)
                is_on = (row, col) in selected
                label = "✓ Sélectionné" if is_on else "Choisir"
                if st.button(
                    label,
                    key=f"tile_btn_{round_id}_{row}_{col}",
                    use_container_width=True,
                    type="primary" if is_on else "secondary",
                ):
                    _toggle_tile(row, col)
                    st.rerun()

    c_clear, _ = st.columns([1, 3])
    with c_clear:
        if st.button("Tout désélectionner", use_container_width=True):
            st.session_state["game_selected_tiles"] = set()
            st.rerun()

    if not selected:
        return None
    return mask_from_selected_tiles((h, w), selected, grid=TILE_GRID)


def _assert_submission_allowed() -> None:
    if st.session_state.get("game_phase") != "select":
        raise ValueError("Aucune manche active.")
    if st.session_state.get("game_submitted"):
        raise ValueError("Cette manche a déjà été soumise.")


def _submit_training(player_mask: np.ndarray) -> None:
    _assert_submission_allowed()
    ch = st.session_state["game_challenge"]
    image_rgb = st.session_state["game_image"]
    h, w = image_rgb.shape[:2]
    player_mask = validate_player_mask(player_mask, expected_hw=(h, w), require_any_positive=True)
    gt = load_ground_truth_mask(ch["mask_path"], target_hw=(h, w))

    with st.spinner("Comparaison avec la référence et la prédiction automatique…"):
        model = _get_model()
        inf = predict_e2b(model, image_rgb)
        assert abs(float(inf.threshold) - E2B_THRESHOLD) < 1e-12
        ai_full = inf.binary_full
        score = score_training_round(
            ground_truth=gt, player_mask=player_mask, ai_mask=ai_full
        )

    profile = apply_training_score(
        _profile(),
        challenge_id=ch["challenge_id"],
        points_earned=score.points_earned,
        player_dice=score.player_dice,
        beat_ai=score.beat_ai,
    )
    _save_profile(profile)

    consent = st.session_state.get("hunter_consent", False)
    store = AnnotationStore()
    contrib = store.submit(
        challenge_type=ChallengeType.TRAINING.value,
        player_session_id=profile.session_id,
        consent_to_contribute=bool(consent),
        player_mask=player_mask,
        image_rgb=image_rgb if consent else None,
        image_bytes=None,
        ai_mask=ai_full,
        ai_probability=inf.probability_full if consent else None,
        image_source=ch["source_note"],
        license_info=ch["license_note"],
        frame_id=ch.get("frame_id"),
        game_score={
            "points": score.points_earned,
            "player_dice": score.player_dice,
            "player_iou": score.player_iou,
            "ai_dice": score.ai_dice,
            "ai_iou": score.ai_iou,
            "note": "game_metrics_only_tile_grid",
            "tile_grid": TILE_GRID,
        },
        retain_image=bool(consent),
    )

    st.session_state["game_submitted"] = True
    st.session_state["game_phase"] = "reveal"
    st.session_state["game_result"] = {
        "player_mask": player_mask,
        "gt": gt,
        "ai_mask": ai_full,
        "overlay_ai": inf.overlay_rgb,
        "score": score,
        "contribution_id": contrib.contribution_id,
        "status": contrib.verification_status,
    }


def _submit_discovery(player_mask: np.ndarray) -> None:
    _assert_submission_allowed()
    ch = st.session_state["game_challenge"]
    image_rgb = st.session_state["game_image"]
    h, w = image_rgb.shape[:2]
    player_mask = validate_player_mask(player_mask, expected_hw=(h, w), require_any_positive=True)
    consent = st.session_state.get("hunter_consent", False)
    if not consent:
        raise ValueError(
            "Pour une image découverte, cochez le consentement afin d’enregistrer "
            "votre contribution candidate."
        )

    with st.spinner("Prédiction automatique en cours…"):
        model = _get_model()
        inf = predict_e2b(model, image_rgb)
        assert abs(float(inf.threshold) - E2B_THRESHOLD) < 1e-12

    profile = apply_discovery_submission(
        _profile(), challenge_id=ch.get("challenge_id")
    )
    _save_profile(profile)

    image_bytes = st.session_state.get("game_image_bytes")
    store = AnnotationStore()
    contrib = store.submit(
        challenge_type=ChallengeType.DISCOVERY.value,
        player_session_id=profile.session_id,
        consent_to_contribute=True,
        player_mask=player_mask,
        image_rgb=image_rgb,
        image_bytes=image_bytes,
        ai_mask=inf.binary_full,
        ai_probability=inf.probability_full,
        image_source=ch.get("source_note") or ch.get("filename") or "upload",
        license_info=ch.get("license_note") or "non déclaré",
        frame_id=None,
        game_score={"points": 5, "note": "participation_only_no_gt", "tile_grid": TILE_GRID},
        retain_image=True,
    )

    st.session_state["game_submitted"] = True
    st.session_state["game_phase"] = "reveal"
    st.session_state["game_result"] = {
        "player_mask": player_mask,
        "gt": None,
        "ai_mask": inf.binary_full,
        "overlay_ai": inf.overlay_rgb,
        "score": None,
        "contribution_id": contrib.contribution_id,
        "status": contrib.verification_status,
    }


def _thanks_banner() -> None:
    st.success(
        "Merci pour votre contribution ! "
        "Votre aide nous permet d’améliorer la détection des anomalies routières."
    )


def _render_reveal() -> None:
    res = st.session_state.get("game_result") or {}
    image_rgb = st.session_state["game_image"]
    mode = st.session_state.get("game_mode")
    _thanks_banner()

    tabs = st.tabs(["Comparaison", "Détails", "Continuer"])
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Vos cases**")
            st.image(
                overlay_mask_rgb(image_rgb, res["player_mask"], color=(232, 93, 42)),
                use_container_width=True,
            )
        with c2:
            st.markdown("**Prédiction automatique**")
            st.image(res["overlay_ai"], use_container_width=True)
        with c3:
            if res.get("gt") is not None:
                st.markdown("**Référence vérifiée**")
                st.image(
                    overlay_mask_rgb(image_rgb, res["gt"], color=(40, 160, 90)),
                    use_container_width=True,
                )
            else:
                st.markdown("**Différence (vous vs IA)**")
                st.image(
                    difference_view(res["player_mask"], res["ai_mask"]),
                    use_container_width=True,
                )
                st.caption(
                    "Vert = accord · Rouge = vous seul · Bleu = IA seule. "
                    "Sans vérité terrain, aucun score n’est validé."
                )

    with tabs[1]:
        if mode == "training" and res.get("score") is not None:
            s = res["score"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Points gagnés (jeu)", s.points_earned)
            m2.metric("Votre Dice (vs référence)", f"{s.player_dice:.3f}")
            m3.metric("Dice IA (vs référence)", f"{s.ai_dice:.3f}")
            st.write(s.feedback_fr)
            st.caption(
                f"IoU joueur={s.player_iou:.3f} · IoU IA={s.ai_iou:.3f}. "
                "Ces chiffres servent au jeu ; le protocole scientifique Stage 5 reste figé."
            )
        else:
            st.info(
                "Défi découverte : aucune métrique Dice/IoU validée (pas de masque de référence). "
                "Votre annotation est une **contribution candidate** en attente de revue humaine."
            )
        st.write(
            f"ID contribution : `{res.get('contribution_id')}` — statut : `{res.get('status')}`"
        )

    with tabs[2]:
        a, b = st.columns(2)
        with a:
            if st.button("Nouvelle manche", type="primary", use_container_width=True):
                _clear_game_state()
                st.rerun()
        with b:
            if st.button("Quitter le jeu", use_container_width=True):
                _clear_game_state()
                st.session_state["hunter_lobby_thanks"] = True
                st.rerun()


def render_anomaly_hunter() -> None:
    _ensure_profile()
    st.markdown("## Anomaly Hunter")
    st.write(
        "Repérez les obstacles présents sur la route et aidez-nous à améliorer "
        "la détection des anomalies routières."
    )
    if st.session_state.pop("hunter_lobby_thanks", False):
        _thanks_banner()

    render_player_stats()

    name = st.text_input(
        "Pseudo (optionnel, classements locaux)",
        value=_profile().display_name,
        max_chars=32,
    )
    if name and name != _profile().display_name:
        p = _profile()
        p.display_name = name.strip() or "Chasseur anonyme"
        _save_profile(p)

    st.checkbox(
        "J’accepte que mon annotation (et l’image si découverte) soit conservée "
        "comme contribution candidate pour revue humaine — pas comme vérité terrain automatique.",
        key="hunter_consent",
    )

    mode_options = ["Entraînement (avec référence)", "Découverte (nouvelle image)"]
    # Plain buttons only (Streamlit 1.67 radio/segmented_control session serde is unsafe here).
    for dead in ("hunter_mode", "hunter_mode_seg"):
        if dead in st.session_state:
            del st.session_state[dead]
    if st.session_state.get("hunter_mode_btn") not in mode_options:
        st.session_state["hunter_mode_btn"] = mode_options[0]
    st.caption("Type de défi")
    mcols = st.columns(2)
    for i, opt in enumerate(mode_options):
        with mcols[i]:
            selected = st.session_state["hunter_mode_btn"] == opt
            if st.button(
                opt,
                key=f"hunter_mode_btn_{i}",
                use_container_width=True,
                type="primary" if selected else "secondary",
            ):
                st.session_state["hunter_mode_btn"] = opt
                st.rerun()
    mode = st.session_state["hunter_mode_btn"]

    if st.session_state.get("game_phase") == "reveal":
        _render_reveal()
        return

    if st.session_state.get("game_phase") == "select" and "game_image" in st.session_state:
        mask = _render_tile_picker(st.session_state["game_image"])
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            submit = st.button(
                "Valider ma sélection",
                type="primary",
                use_container_width=True,
                disabled=bool(st.session_state.get("game_submitted")),
            )
        with col_b:
            if st.button("Quitter", use_container_width=True):
                _clear_game_state()
                st.session_state["hunter_lobby_thanks"] = True
                st.rerun()
        with col_c:
            st.caption("Prenez votre temps — aucun chronomètre.")

        if submit:
            try:
                if mask is None or int((mask > 0).sum()) == 0:
                    st.error("Sélectionnez au moins une case contenant l’obstacle.")
                elif st.session_state.get("game_mode") == "training":
                    _submit_training(mask)
                    st.rerun()
                else:
                    _submit_discovery(mask)
                    st.rerun()
            except Exception as exc:
                st.error(str(exc))
        return

    # Lobby
    if mode.startswith("Entraînement"):
        st.info(
            "Choisissez les cases de la photo qui contiennent l’obstacle. "
            "Images autorisées : splits train/val uniquement (jamais le test verrouillé)."
        )
        if st.button("Lancer un défi d’entraînement", type="primary"):
            _start_training_round()
            st.rerun()
    else:
        st.warning(
            "Sans masque de référence : votre sélection est une proposition à revoir. "
            "Différer de l’IA ne signifie pas que vous avez raison."
        )
        uploaded = st.file_uploader(
            "Importer une image (JPG/JPEG/PNG, max 8 Mo)",
            type=["jpg", "jpeg", "png"],
            key="discovery_upload",
        )
        if uploaded is not None:
            raw = uploaded.getvalue()
            if len(raw) > MAX_UPLOAD_BYTES:
                st.error("Fichier trop volumineux (max 8 Mo).")
            elif st.button("Commencer le défi découverte", type="primary"):
                try:
                    rgb = load_user_image(raw, filename=uploaded.name)
                    from anomaly_hunter.pool import discovery_meta_from_upload

                    meta = discovery_meta_from_upload(
                        filename=uploaded.name,
                        image_bytes=raw,
                        license_claim="non déclaré",
                        source_claim=f"import joueur ({uploaded.name})",
                    )
                    meta["image_bytes"] = raw
                    store = AnnotationStore()
                    if store.find_duplicate_sha(meta["content_sha256"]):
                        st.warning(
                            "Une contribution similaire existe déjà (empreinte image). "
                            "Vous pouvez tout de même jouer, mais l’envoi pourra être marqué doublon."
                        )
                    _start_discovery_round(rgb, uploaded.name, meta)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    with st.expander("Classement local (cet appareil uniquement)"):
        st.warning(
            "Classement **local au navigateur/appareil** — pas un classement communautaire mondial. "
            "Aucun compte multi-utilisateur n’est implémenté."
        )
        board = AnnotationStore().leaderboard_path
        import json

        rows = json.loads(board.read_text(encoding="utf-8") or "[]")
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("Pas encore de scores enregistrés sur cet appareil.")

    with st.expander("Confidentialité — jeu"):
        st.markdown(
            """
- Aucun chronomètre : vous quittez quand vous voulez.
- Contributions conservées uniquement avec consentement.
- Stockage local sous `data/annotations/` pour revue admin.
- Aucune promotion automatique vers l’entraînement.
"""
        )


def render_admin_review() -> None:
    from anomaly_hunter.admin_auth import ENV_VAR, admin_auth_status, verify_admin_token
    from anomaly_hunter.eligibility import (
        assess_contribution_eligibility,
        detect_annotation_conflicts,
    )

    st.markdown("## Revue des annotations")
    st.caption(
        "Interface administrateur — validation humaine avant tout export. "
        "Approuvé ≠ éligible E3."
    )

    cfg = admin_auth_status()
    if not cfg.ok:
        st.error(cfg.reason)
        st.info(
            f"Configurez un secret fort dans la variable d’environnement `{ENV_VAR}` "
            "(≥16 caractères). Aucun jeton par défaut n’est accepté."
        )
        return

    token = st.text_input("Jeton administrateur", type="password")
    auth = verify_admin_token(token)
    if not auth.ok:
        st.warning("Accès refusé tant que le jeton correct n’est pas fourni.")
        return

    store = AnnotationStore()
    filter_opts = ["pending_review", "all", *STATUSES]
    # Deduplicate while preserving order
    seen = set()
    filter_opts = [x for x in filter_opts if not (x in seen or seen.add(x))]
    if st.session_state.get("admin_status_filter") not in filter_opts:
        st.session_state["admin_status_filter"] = "pending_review"
    fcols = st.columns(len(filter_opts))
    for i, opt in enumerate(filter_opts):
        with fcols[i]:
            on = st.session_state["admin_status_filter"] == opt
            if st.button(opt, key=f"admin_filt_{i}", use_container_width=True, type="primary" if on else "secondary"):
                st.session_state["admin_status_filter"] = opt
                st.rerun()
    status_filter = st.session_state["admin_status_filter"]
    rows = (
        store.list_contributions()
        if status_filter == "all"
        else store.list_contributions(status=status_filter)
    )
    st.write(f"{len(rows)} contribution(s)")

    conflicts = detect_annotation_conflicts(
        store.list_contributions(),
        store_load_mask=store.load_contribution_assets,
    )
    if conflicts:
        st.warning(f"{len(conflicts)} conflit(s) d’annotation détecté(s) (IoU bas / formes).")
        st.json(conflicts[:20])

    if not rows:
        st.info("Aucune contribution pour ce filtre.")
        return

    ids = [r["contribution_id"] for r in rows]
    if st.session_state.get("admin_selected_contrib") not in ids:
        st.session_state["admin_selected_contrib"] = ids[0]
    st.caption("Contribution")
    # Button list avoids selectbox serde issues on long sessions
    pick_cols = st.columns(min(3, len(ids)))
    for i, cid in enumerate(ids[:12]):
        with pick_cols[i % len(pick_cols)]:
            on = st.session_state["admin_selected_contrib"] == cid
            label = cid[:8] + "…"
            if st.button(label, key=f"pick_c_{cid}", use_container_width=True, type="primary" if on else "secondary"):
                st.session_state["admin_selected_contrib"] = cid
                st.rerun()
    selected = st.session_state["admin_selected_contrib"]
    assets = store.load_contribution_assets(selected)
    meta = assets["meta"]
    decision = assess_contribution_eligibility(meta)
    st.json({"meta": meta, "e3_eligibility": decision.to_dict()})

    cols = st.columns(3)
    with cols[0]:
        if "image" in assets:
            st.image(assets["image"], caption="Image", use_container_width=True)
        else:
            st.caption("Image non conservée (pas de consentement ou défi train sans rétention).")
    with cols[1]:
        if "player_mask" in assets:
            st.image(
                colorize_mask((assets["player_mask"] > 0).astype(np.uint8)),
                caption="Masque joueur",
                use_container_width=True,
            )
    with cols[2]:
        if "ai_mask" in assets:
            st.image(
                colorize_mask((assets["ai_mask"] > 0).astype(np.uint8)),
                caption="Masque IA",
                use_container_width=True,
            )

    if "player_mask" in assets and "ai_mask" in assets:
        st.image(
            difference_view(assets["player_mask"], assets["ai_mask"]),
            caption="Différences joueur vs IA",
            use_container_width=True,
        )

    st.markdown("### Décision de revue")
    status_labels = {
        "pending_review": "En attente",
        "approved": "Approuver",
        "rejected": "Rejeter",
        "duplicate": "Marquer doublon",
        "requires_additional_review": "Demander un avis supplémentaire",
    }
    if st.session_state.get("review_status_choice") not in STATUSES:
        st.session_state["review_status_choice"] = "approved"
    scols = st.columns(len(STATUSES))
    for i, status in enumerate(STATUSES):
        with scols[i]:
            on = st.session_state["review_status_choice"] == status
            if st.button(
                status_labels.get(status, status),
                key=f"rev_status_{i}",
                use_container_width=True,
                type="primary" if on else "secondary",
            ):
                st.session_state["review_status_choice"] = status
                st.rerun()
    new_status = st.session_state["review_status_choice"]
    comment = st.text_area("Commentaire de revue")
    confirm_review = st.checkbox(
        "Je confirme cette décision de revue (action persistante).",
        key="confirm_review_action",
    )
    if st.button("Enregistrer la revue", type="primary"):
        if not verify_admin_token(token).ok:
            st.error("Session admin invalide.")
        elif not confirm_review:
            st.warning("Cochez la confirmation avant d’enregistrer.")
        else:
            updated = store.review(selected, status=new_status, comment=comment)
            st.session_state["confirm_review_action"] = False
            st.success(f"Statut → {updated['verification_status']}")

    st.markdown("### Export dataset versionné (éligibles E3 uniquement)")
    st.caption(
        "Exclut test/val figés, doublons, consentement manquant, licence/source faibles. "
        "N’écrase jamais une version existante. Approuvé ≠ automatiquement éligible."
    )
    confirm_export = st.checkbox(
        "Je confirme l’export des annotations éligibles.",
        key="confirm_export_action",
    )
    if st.button("Exporter les annotations éligibles"):
        if not confirm_export:
            st.warning("Cochez la confirmation avant d’exporter.")
        else:
            try:
                manifest = export_approved_dataset(
                    store, admin_token=token, enforce_admin=True
                )
                st.session_state["confirm_export_action"] = False
                st.success(
                    f"Export {manifest['dataset_version']} — "
                    f"{manifest['n_eligible_exported']} éligible(s), "
                    f"{manifest['n_excluded']} exclu(s)."
                )
                st.json(manifest)
            except PermissionError as exc:
                st.error(str(exc))
