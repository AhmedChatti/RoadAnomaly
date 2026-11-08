"""Streamlit UI for Anomaly Hunter (French) + admin review."""

from __future__ import annotations

import time
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
    MAX_UPLOAD_BYTES,
    binary_mask_from_canvas_json,
    difference_view,
    load_ground_truth_mask,
    overlay_mask_rgb,
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

COUNTDOWN_SECONDS = 15
CANVAS_DISPLAY_WIDTH = 720


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


def _canvas_available() -> bool:
    try:
        import streamlit_drawable_canvas  # noqa: F401

        return True
    except Exception:
        return False


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
    st.session_state["game_phase"] = "draw"
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
    st.session_state["game_deadline"] = time.time() + COUNTDOWN_SECONDS
    st.session_state["game_result"] = None
    st.session_state["game_submitted"] = False
    st.session_state["game_expired"] = False
    st.session_state["canvas_key"] = st.session_state.get("canvas_key", 0) + 1


def _start_discovery_round(image_rgb: np.ndarray, filename: str, meta: dict[str, Any]) -> None:
    st.session_state["game_mode"] = "discovery"
    st.session_state["game_phase"] = "draw"
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
    st.session_state["game_deadline"] = time.time() + COUNTDOWN_SECONDS
    st.session_state["game_result"] = None
    st.session_state["game_submitted"] = False
    st.session_state["game_expired"] = False
    st.session_state["canvas_key"] = st.session_state.get("canvas_key", 0) + 1


def _render_drawing_tools(image_rgb: np.ndarray) -> np.ndarray | None:
    h, w = image_rgb.shape[:2]
    scale = CANVAS_DISPLAY_WIDTH / max(w, 1)
    canvas_w = int(round(w * scale))
    canvas_h = int(round(h * scale))
    # Keep aspect; clamp height for mobile
    if canvas_h > 480:
        scale = 480 / h
        canvas_w = int(round(w * scale))
        canvas_h = int(round(h * scale))

    deadline = float(st.session_state.get("game_deadline", time.time()))
    remaining = max(0.0, deadline - time.time())
    remaining_i = int(remaining)
    st.progress(
        min(1.0, remaining / COUNTDOWN_SECONDS) if COUNTDOWN_SECONDS else 0.0,
        text=f"Temps restant : {remaining_i} s",
    )
    if remaining <= 0:
        st.session_state["game_expired"] = True
        st.error("Temps écoulé. La validation est refusée — lancez une nouvelle manche.")
    else:
        st.session_state["game_expired"] = False
    # Force a periodic refresh while drawing so the countdown updates.
    if remaining > 0 and st.session_state.get("game_phase") == "draw":
        try:
            from streamlit_autorefresh import st_autorefresh  # type: ignore

            st_autorefresh(interval=1000, key="hunter_timer_refresh")
        except Exception:
            st.caption("Astuce : interagissez avec la page pour actualiser le compte à rebours.")

    tool_options = ["Pinceau", "Rectangle", "Gomme"]
    if st.session_state.get("hunter_tool") not in tool_options:
        st.session_state.pop("hunter_tool", None)
    tool = st.radio(
        "Outil",
        tool_options,
        horizontal=True,
        key="hunter_tool",
    )
    stroke_width = st.slider("Épaisseur", 4, 48, 18, key="hunter_brush")

    drawing_mode = {"Pinceau": "freedraw", "Rectangle": "rect", "Gomme": "freedraw"}[tool]
    stroke_color = "#ffffff" if tool == "Gomme" else "#e85d2a"

    mask: np.ndarray | None = None
    if _canvas_available():
        from streamlit_drawable_canvas import st_canvas

        bg = Image.fromarray(image_rgb).resize((canvas_w, canvas_h), Image.Resampling.BILINEAR)
        canvas = st_canvas(
            fill_color="rgba(232, 93, 42, 0.35)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=bg,
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode=drawing_mode,
            key=f"hunter_canvas_{st.session_state.get('canvas_key', 0)}",
            display_toolbar=True,
        )
        if canvas.json_data is not None:
            payload = {
                "objects": canvas.json_data.get("objects", []),
                "imageData": getattr(canvas, "image_data", None),
            }
            if canvas.image_data is not None:
                payload["imageData"] = canvas.image_data
            mask = binary_mask_from_canvas_json(
                payload,
                image_hw=(h, w),
                canvas_width=canvas_w,
                canvas_height=canvas_h,
            )
    else:
        st.info(
            "Module de dessin avancé indisponible. Utilisez le mode polygone simplifié "
            "(points séparés par des virgules : x,y)."
        )
        raw = st.text_area(
            "Sommets du polygone (x,y par ligne)",
            placeholder="120,80\n200,90\n180,160",
            key="poly_fallback",
        )
        if raw.strip():
            from PIL import ImageDraw

            pts = []
            for line in raw.strip().splitlines():
                parts = [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]
                if len(parts) >= 2:
                    pts.append((float(parts[0]), float(parts[1])))
            img = Image.new("L", (w, h), 0)
            if len(pts) >= 3:
                ImageDraw.Draw(img).polygon(pts, fill=255)
            mask = (np.asarray(img) > 0).astype(np.uint8)

    return mask


def _assert_submission_allowed() -> None:
    if st.session_state.get("game_phase") != "draw":
        raise ValueError("Aucune manche active.")
    if st.session_state.get("game_submitted"):
        raise ValueError("Cette manche a déjà été soumise.")
    deadline = float(st.session_state.get("game_deadline", 0))
    if time.time() > deadline + 0.25:  # small clock skew tolerance
        st.session_state["game_expired"] = True
        raise ValueError("Temps écoulé — soumission refusée. Aucun point attribué.")


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
        # Official threshold is frozen inside predict_e2b
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
            "note": "game_metrics_only",
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
            "Pour une image découverte, le consentement est requis afin d’enregistrer "
            "votre annotation candidate (sans statut de vérité terrain)."
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
        license_info=ch.get("license_note") or "déclaré par le joueur",
        frame_id=None,
        game_score={"points": 5, "note": "participation_only_no_gt"},
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


def _render_reveal() -> None:
    res = st.session_state.get("game_result") or {}
    image_rgb = st.session_state["game_image"]
    mode = st.session_state.get("game_mode")
    st.success("Manche terminée !")

    tabs = st.tabs(["Comparaison", "Détails", "Nouvelle manche"])
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Votre annotation**")
            st.image(overlay_mask_rgb(image_rgb, res["player_mask"], color=(232, 93, 42)), use_container_width=True)
        with c2:
            st.markdown("**Prédiction automatique**")
            st.image(res["overlay_ai"], use_container_width=True)
        with c3:
            if res.get("gt") is not None:
                st.markdown("**Référence vérifiée**")
                st.image(overlay_mask_rgb(image_rgb, res["gt"], color=(40, 160, 90)), use_container_width=True)
            else:
                st.markdown("**Différence (vous vs IA)**")
                st.image(difference_view(res["player_mask"], res["ai_mask"]), use_container_width=True)
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
        st.write(f"ID contribution : `{res.get('contribution_id')}` — statut : `{res.get('status')}`")

    with tabs[2]:
        if st.button("Nouvelle manche", type="primary"):
            for k in list(st.session_state.keys()):
                if k.startswith("game_"):
                    del st.session_state[k]
            st.rerun()


def render_anomaly_hunter() -> None:
    _ensure_profile()
    st.markdown("## Anomaly Hunter")
    st.write(
        "Repérez les obstacles présents sur la route et aidez-nous à améliorer "
        "la détection des anomalies routières."
    )
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
    if st.session_state.get("hunter_mode") not in mode_options:
        st.session_state.pop("hunter_mode", None)
    mode = st.radio(
        "Type de défi",
        mode_options,
        horizontal=True,
        key="hunter_mode",
    )

    if st.session_state.get("game_phase") == "reveal":
        _render_reveal()
        return

    if st.session_state.get("game_phase") == "draw" and "game_image" in st.session_state:
        st.markdown("### Annotez l’obstacle")
        st.caption("Dessinez autour de l’obstacle. Vous pouvez gommer et corriger avant de valider.")
        mask = _render_drawing_tools(st.session_state["game_image"])
        col_a, col_b = st.columns(2)
        expired = bool(st.session_state.get("game_expired")) or (
            time.time() > float(st.session_state.get("game_deadline", 0))
        )
        with col_a:
            submit = st.button(
                "Valider mon annotation",
                type="primary",
                use_container_width=True,
                disabled=expired or bool(st.session_state.get("game_submitted")),
            )
        with col_b:
            if st.button("Abandonner", use_container_width=True):
                for k in list(st.session_state.keys()):
                    if k.startswith("game_"):
                        del st.session_state[k]
                st.rerun()
        if submit:
            try:
                if expired:
                    raise ValueError("Temps écoulé — soumission refusée.")
                if mask is None or int((mask > 0).sum()) == 0:
                    st.error("Annotation vide refusée. Dessinez une zone avant de valider.")
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
            "Images autorisées uniquement (splits train/val). Le jeu **n’utilise jamais** "
            "le jeu de test verrouillé."
        )
        if st.button("Lancer un défi d’entraînement", type="primary"):
            _start_training_round()
            st.rerun()
    else:
        st.warning(
            "Sans masque de référence : votre dessin est une proposition à revoir. "
            "Différer de l’IA ne signifie pas que vous avez raison."
        )
        uploaded = st.file_uploader(
            "Importer une image (JPG/JPEG/PNG, max 8 Mo)",
            type=["jpg", "jpeg", "png"],
            key="discovery_upload",
        )
        license_claim = st.text_input(
            "Licence / droit d’usage déclaré",
            placeholder="Ex. photo personnelle, CC BY…",
        )
        source_claim = st.text_input("Source de l’image", placeholder="Ex. prise le… / URL…")
        if uploaded is not None:
            raw = uploaded.getvalue()
            if len(raw) > MAX_UPLOAD_BYTES:
                st.error("Fichier trop volumineux (max 8 Mo).")
            elif st.button("Commencer le défi découverte", type="primary"):
                try:
                    if not license_claim.strip() or not source_claim.strip():
                        raise ValueError("Indiquez la licence et la source avant de continuer.")
                    rgb = load_user_image(raw, filename=uploaded.name)
                    from anomaly_hunter.pool import discovery_meta_from_upload

                    meta = discovery_meta_from_upload(
                        filename=uploaded.name,
                        image_bytes=raw,
                        license_claim=license_claim,
                        source_claim=source_claim,
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
        st.caption("*Annotations OK = essais entraînement avec Dice ≥ 0,55 (métrique de jeu).")

    with st.expander("Confidentialité — jeu"):
        st.markdown(
            """
- Analyse : prédiction automatique en mémoire via le modèle de référence figé.
- Contributions : conservées **uniquement** si vous cochez le consentement.
- Images découverte : stockées localement sous `data/annotations/` pour revue admin.
- Aucune promotion automatique vers l’entraînement. Pas d’envoi vers un cloud tiers par cette app.
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
    status_filter = st.selectbox("Filtrer par statut", ["pending_review", "all", *STATUSES])
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
        st.markdown("### Export dataset versionné (éligibles E3 uniquement)")
        if st.button("Exporter les annotations éligibles"):
            try:
                manifest = export_approved_dataset(
                    store, admin_token=token, enforce_admin=True
                )
                st.success(
                    f"Export {manifest['dataset_version']} — "
                    f"{manifest['n_eligible_exported']} éligible(s), "
                    f"{manifest['n_excluded']} exclu(s)."
                )
                st.json(manifest)
            except PermissionError as exc:
                st.error(str(exc))
        return

    ids = [r["contribution_id"] for r in rows]
    selected = st.selectbox("Contribution", ids)
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

    new_status = st.selectbox("Nouveau statut", list(STATUSES), index=0)
    comment = st.text_area("Commentaire de revue")
    if st.button("Enregistrer la revue", type="primary"):
        # Re-check auth on mutating action
        if not verify_admin_token(token).ok:
            st.error("Session admin invalide.")
        else:
            updated = store.review(selected, status=new_status, comment=comment)
            st.success(f"Statut → {updated['verification_status']}")

    st.markdown("### Export dataset versionné (éligibles E3 uniquement)")
    st.caption(
        "Exclut test/val figés, doublons, consentement manquant, licence/source faibles. "
        "N’écrase jamais une version existante."
    )
    if st.button("Exporter les annotations éligibles"):
        try:
            manifest = export_approved_dataset(
                store, admin_token=token, enforce_admin=True
            )
            st.success(
                f"Export {manifest['dataset_version']} — "
                f"{manifest['n_eligible_exported']} éligible(s), "
                f"{manifest['n_excluded']} exclu(s)."
            )
            st.json(manifest)
        except PermissionError as exc:
            st.error(str(exc))
