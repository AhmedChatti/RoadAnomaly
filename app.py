"""
RoadAnomaly — application Streamlit grand public (interface française).

Inférence scientifique inchangée : prédiction officielle E2b au seuil figé 0,55
(implémentée dans src/app_inference.py — non exposée à l'utilisateur).

Lancement : streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app_inference import (  # noqa: E402
    E2B_THRESHOLD,
    colorize_mask,
    encode_png_mask,
    encode_png_probability,
    encode_png_rgb,
    load_e2b_model,
    load_user_image,
    predict_e2b,
    probability_heatmap_rgb,
)

st.set_page_config(
    page_title="RoadAnomaly",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Source+Sans+3:wght@400;500;600&display=swap');

:root {
  --ink: #15202b;
  --muted: #3d4b5c;
  --paper: #f6f8fa;
  --card: #ffffff;
  --line: #d5dde7;
  --brand: #0f6e56;
  --brand-deep: #0b5542;
  --accent: #e8f5f0;
  --warn: #8a5a00;
  /* Force readable light theme even if the browser/OS prefers dark */
  color-scheme: light;
}

html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
.stMarkdown, .stText, p, span, label, li {
  font-family: "Source Sans 3", "Segoe UI", sans-serif !important;
  color: var(--ink) !important;
}

.stApp {
  background:
    radial-gradient(900px 420px at 0% -10%, #dff3ea 0%, transparent 55%),
    radial-gradient(800px 380px at 100% 0%, #e8eef6 0%, transparent 50%),
    linear-gradient(180deg, #f7faf9 0%, #eef2f5 100%) !important;
}
.block-container {
  padding-top: 1rem;
  padding-bottom: 2.5rem;
  max-width: 1080px;
}
h1, h2, h3, h4, .brand {
  font-family: "Manrope", "Segoe UI", sans-serif !important;
  letter-spacing: -0.03em;
  color: var(--ink) !important;
}
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

/* Metrics, captions, radio/checkbox labels */
[data-testid="stMetricLabel"], [data-testid="stMetricValue"],
[data-testid="stMetricDelta"], [data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span,
div[role="radiogroup"] label, div[role="radiogroup"] p,
[data-testid="stCheckbox"] label p, [data-testid="stCheckbox"] span {
  color: var(--ink) !important;
  opacity: 1 !important;
}
[data-testid="stMetricValue"] { font-weight: 800 !important; color: var(--brand-deep) !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-weight: 700 !important; }

/* Inputs: dark text on white field */
.stTextInput input, .stTextArea textarea, .stNumberInput input,
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea,
[data-baseweb="select"] > div {
  background-color: #ffffff !important;
  color: var(--ink) !important;
  border-color: var(--line) !important;
  caret-color: var(--ink) !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
  color: #6b7a8c !important;
  opacity: 1 !important;
}

/* Info / warning boxes: keep readable */
[data-testid="stAlert"] p, [data-testid="stAlert"] span {
  color: var(--ink) !important;
}

.brand {
  font-size: 1.35rem; font-weight: 800; color: var(--brand-deep) !important;
}
.hero {
  background: linear-gradient(135deg, #0f6e56 0%, #148567 45%, #1a6f8a 100%);
  color: #f4fffb;
  border-radius: 22px;
  padding: 1.8rem 1.7rem 1.55rem;
  margin-bottom: 1.25rem;
  box-shadow: 0 18px 40px rgba(15, 110, 86, 0.22);
}
.hero h1 {
  color: #ffffff !important;
  font-size: clamp(1.7rem, 4vw, 2.35rem);
  margin: 0 0 0.55rem 0;
  font-weight: 800;
  max-width: 16ch;
  line-height: 1.15;
}
.hero p {
  margin: 0;
  max-width: 36rem;
  color: #e7fff6 !important;
  font-size: 1.05rem;
  line-height: 1.5;
}
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 1.15rem 1.2rem;
  box-shadow: 0 10px 28px rgba(21, 32, 43, 0.05);
}
.card h3 { margin: 0 0 0.35rem 0; font-size: 1.15rem; color: var(--ink) !important; }
.card p, .muted { color: var(--muted) !important; line-height: 1.5; opacity: 1 !important; }
.steps {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.75rem;
  margin: 1rem 0 0.2rem;
}
.step {
  background: var(--accent);
  border-radius: 14px;
  padding: 0.85rem 0.9rem;
}
.step .n {
  font-family: Manrope, sans-serif;
  font-weight: 800;
  color: var(--brand) !important;
  font-size: 0.85rem;
}
.step .t { font-weight: 700; margin: 0.2rem 0; color: var(--ink) !important; }
.step .d { color: var(--muted) !important; font-size: 0.9rem; margin: 0; }
.metric-lite {
  background: #f4faf7;
  border: 1px solid #d7ebe3;
  border-radius: 14px;
  padding: 0.9rem 1rem;
}
.metric-lite .lab {
  font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted) !important; font-weight: 700;
}
.metric-lite .val {
  font-family: Manrope, sans-serif; font-size: 1.6rem; font-weight: 800;
  color: var(--brand-deep) !important; margin-top: 0.15rem;
}
.disclaimer {
  background: #fff8e8;
  border-left: 4px solid #d4a017;
  border-radius: 0 12px 12px 0;
  padding: 0.85rem 1rem;
  color: #5c4a1d !important;
  line-height: 1.45;
  margin-top: 0.9rem;
}
.panel-title {
  font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.07em;
  color: var(--brand) !important; font-weight: 800; margin-bottom: 0.25rem;
}
div.stButton > button {
  color: #ffffff !important;
  font-weight: 700;
  border-radius: 12px;
}
div.stButton > button[kind="primary"] {
  background: linear-gradient(180deg, #14936f, #0f6e56) !important;
  border: none;
}
div.stButton > button[kind="secondary"] {
  background: #243040 !important;
  border: none;
  color: #ffffff !important;
}
@media (max-width: 800px) {
  .steps { grid-template-columns: 1fr; }
  .hero h1 { max-width: none; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_cached_model():
    return load_e2b_model(verify_sha=True)


def _is_onboarded() -> bool:
    if st.session_state.get("onboarded"):
        return True
    try:
        return st.query_params.get("onboarded") == "1"
    except Exception:
        return False


def _mark_onboarded() -> None:
    st.session_state["onboarded"] = True
    try:
        st.query_params["onboarded"] = "1"
    except Exception:
        pass


@st.dialog("Bienvenue sur RoadAnomaly !")
def welcome_dialog() -> None:
    st.markdown(
        "Analysez une image de route et identifiez automatiquement les zones "
        "potentiellement anormales."
    )
    st.markdown(
        """
<div class="steps">
  <div class="step"><div class="n">Étape 1</div><div class="t">Importez une image</div>
  <p class="d">Choisissez une photo depuis votre appareil.</p></div>
  <div class="step"><div class="n">Étape 2</div><div class="t">Lancez l'analyse</div>
  <p class="d">Notre système examine automatiquement l'image.</p></div>
  <div class="step"><div class="n">Étape 3</div><div class="t">Consultez les résultats</div>
  <p class="d">Visualisez les zones détectées et téléchargez votre résultat.</p></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.info(
        "Astuce : préférez une photo nette, bien éclairée, prise face à la route. "
        "Évitez les images très floues ou trop sombres."
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Commencer", type="primary", use_container_width=True):
            _mark_onboarded()
            st.rerun()
    with c2:
        if st.button("Fermer", use_container_width=True):
            _mark_onboarded()
            st.rerun()


def _purge_legacy_choice_keys() -> None:
    """Drop old radio widget keys that crash Streamlit 1.67 deserialize."""
    legacy_keys = (
        "nav_page",
        "nav_main",
        "hunter_mode",
        "hunter_tool",
    )
    for key in legacy_keys:
        if key in st.session_state:
            del st.session_state[key]


def _choice_bar(label: str, options: list[str], *, key: str, default: str | None = None) -> str:
    """Horizontal choice control without st.radio (avoids RadioSerde session bugs)."""
    default = default or options[0]
    current = st.session_state.get(key)
    if current not in options:
        st.session_state[key] = default
    try:
        value = st.segmented_control(label, options=options, key=key, label_visibility="collapsed")
    except Exception:
        value = st.selectbox(label, options=options, key=key, label_visibility="collapsed")
    if value not in options:
        return default
    return str(value)


def render_topbar() -> str:
    left, nav, right = st.columns([1.4, 3.2, 1])
    with left:
        st.markdown('<div class="brand">RoadAnomaly</div>', unsafe_allow_html=True)
    with nav:
        page = _choice_bar(
            "Navigation",
            ["Analyser", "Jouer", "Revue"],
            key="nav_seg_v2",
            default="Analyser",
        )
    with right:
        if st.button("Aide", use_container_width=True):
            st.session_state["force_help"] = True
    return page


def render_hero() -> None:
    st.markdown(
        """
<div class="hero">
  <h1>Rendez vos routes plus sûres.</h1>
  <p>Une analyse intelligente des images routières pour identifier les obstacles
  et anomalies visibles.</p>
</div>
""",
        unsafe_allow_html=True,
    )


def load_example_train_image() -> tuple[np.ndarray, str] | None:
    """Permitted non-test example from the training split only."""
    try:
        import pandas as pd

        manifest = pd.read_csv(ROOT / "data/processed/split_manifest.csv")
        train = manifest.loc[manifest["split"] == "train", "frame_id"].tolist()
        preferred = [
            "animals17_E6_near_Kvaenangen",
            "obstacles03_fallen_tree",
            "cones01",
        ]
        ordered = [f for f in preferred if f in train] + [f for f in train if f not in preferred]
        for frame_id in ordered[:8]:
            path = ROOT / "frames" / f"{frame_id}.webp"
            if path.is_file():
                # Convert webp bytes to PNG-compatible RGB via loader path API
                arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
                return arr, f"exemple_{frame_id}.png"
    except Exception:
        return None
    return None


def friendly_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "format" in msg or "supporté" in msg or "gif" in msg:
        return "Ce format n’est pas pris en charge. Utilisez une image JPG, JPEG ou PNG."
    if "trop petite" in msg:
        return "L’image est trop petite. Choisissez une photo plus grande."
    if "checkpoint" in msg or "no such file" in msg or "introuvable" in msg:
        return (
            "Le service d’analyse n’est pas prêt sur cet appareil "
            "(fichier modèle manquant). Réessayez plus tard ou contactez l’équipe projet."
        )
    return "L’analyse n’a pas pu être terminée. Vérifiez votre image et réessayez."


def render_faq() -> None:
    with st.expander("Questions fréquentes", expanded=False):
        st.markdown(
            """
**Que fait l’application ?**  
Elle examine une photo de route et met en évidence des zones potentiellement inhabituelles
(obstacles, objets inattendus), sous forme d’aide visuelle.

**Que signifie une zone colorée ?**  
Ce sont des régions que le système estime *potentiellement anormales*. Ce n’est pas une
preuve qu’un danger est confirmé, ni une classification du type d’objet.

**Quelles images fonctionnent le mieux ?**  
Photos nettes, de jour, face à la chaussée, sans trop de flou ni de contre-jour.

**Pourquoi une partie d’un obstacle peut manquer ?**  
Le système est expérimental et entraîné sur un jeu limité. Certains contours ou objets
peu courants peuvent être partiellement détectés.

**Mes images sont-elles conservées ?**  
Non. L’analyse se fait en mémoire pendant votre session. Aucun compte n’est requis et
aucune galerie serveur n’est créée par cette application.
"""
        )


def render_privacy() -> None:
    st.caption(
        "Confidentialité : les images restent en mémoire le temps de l’analyse dans cette "
        "session et ne sont pas enregistrées de façon permanente par l’application."
    )


def main() -> None:
    _purge_legacy_choice_keys()
    page = render_topbar()

    if st.session_state.get("force_help"):
        st.session_state["force_help"] = False
        welcome_dialog()
    elif not _is_onboarded() and page == "Analyser":
        welcome_dialog()

    if page == "Jouer":
        from anomaly_hunter.ui import render_anomaly_hunter

        render_anomaly_hunter()
        return

    if page == "Revue":
        from anomaly_hunter.ui import render_admin_review

        render_admin_review()
        return

    render_hero()

    st.markdown(
        """
<div class="card">
  <h3>Analysez une image de route</h3>
  <p class="muted">Importez une photo (JPG, JPEG ou PNG), lancez l’analyse, puis consultez
  les zones signalées. Simple, rapide, sans jargon technique.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")

    if "upload_nonce" not in st.session_state:
        st.session_state["upload_nonce"] = 0

    # --- Upload / example ---
    up_col, tip_col = st.columns([2.2, 1])
    with up_col:
        uploaded = st.file_uploader(
            "Glissez-déposez une image ici, ou cliquez pour la sélectionner",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            label_visibility="visible",
            key=f"uploader_{st.session_state['upload_nonce']}",
        )
    with tip_col:
        st.markdown(
            """
<div class="card">
  <div class="panel-title">Conseil</div>
  <p class="muted" style="margin:0">Photo claire, route bien visible.
  Formats : JPG, JPEG, PNG.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Essayer un exemple", use_container_width=True):
            example = load_example_train_image()
            if example is None:
                st.warning("Aucun exemple disponible sur cet appareil.")
            else:
                arr, name = example
                st.session_state["pending_image"] = arr
                st.session_state["pending_name"] = name
                st.session_state.pop("result", None)
                st.session_state.pop("busy", None)
                st.rerun()

    image = None
    image_name = "image.png"
    if "pending_image" in st.session_state and uploaded is None:
        image = st.session_state["pending_image"]
        image_name = st.session_state.get("pending_name", "exemple.png")
    elif uploaded is not None:
        try:
            image = load_user_image(uploaded.getvalue(), filename=uploaded.name)
            image_name = uploaded.name
            st.session_state.pop("pending_image", None)
        except Exception as exc:
            st.error(friendly_error(exc))
            render_faq()
            render_privacy()
            return

    if image is None:
        render_faq()
        render_privacy()
        return

    st.markdown("#### Aperçu")
    preview_c, action_c = st.columns([2.2, 1])
    with preview_c:
        st.image(image, use_container_width=True, caption=image_name)
    with action_c:
        st.markdown(
            f"""
<div class="metric-lite">
  <div class="lab">Taille</div>
  <div class="val" style="font-size:1.25rem">{image.shape[1]}×{image.shape[0]}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.write("")
        analyze = st.button(
            "Analyser l'image",
            type="primary",
            use_container_width=True,
            disabled=bool(st.session_state.get("busy")),
        )
        if st.button("Nouvelle image", use_container_width=True):
            st.session_state["upload_nonce"] = int(st.session_state.get("upload_nonce", 0)) + 1
            for key in ("result", "result_key", "pending_image", "pending_name", "busy", "mask_vis", "analyze_key"):
                st.session_state.pop(key, None)
            st.rerun()

    file_key = f"{image_name}:{image.shape}:{int(image.mean()*1000)}"
    if analyze:
        st.session_state["busy"] = True
        st.session_state["analyze_key"] = file_key

    need_run = (
        st.session_state.get("busy")
        and st.session_state.get("analyze_key") == file_key
        and (
            "result" not in st.session_state
            or st.session_state.get("result_key") != file_key
        )
    )

    if need_run:
        progress = st.empty()
        progress.info("Analyse de votre image en cours...")
        try:
            model = get_cached_model()
            # Official prediction only — threshold fixed inside predict_e2b().
            result = predict_e2b(model, image)
            assert abs(float(result.threshold) - E2B_THRESHOLD) < 1e-12
            st.session_state["result"] = result
            st.session_state["result_key"] = file_key
            st.session_state["mask_vis"] = colorize_mask(result.binary_full)
            st.session_state["busy"] = False
            progress.success("Analyse terminée. Voici votre résultat.")
        except Exception as exc:
            st.session_state["busy"] = False
            progress.empty()
            st.error(friendly_error(exc))
            render_faq()
            render_privacy()
            return

    if "result" not in st.session_state or st.session_state.get("result_key") != file_key:
        render_faq()
        render_privacy()
        return

    result = st.session_state["result"]
    mask_vis = st.session_state["mask_vis"]

    st.markdown("### Résultat de votre analyse")
    st.markdown(
        '<p class="muted">Les zones colorées représentent les régions que le système '
        "identifie comme potentiellement anormales.</p>",
        unsafe_allow_html=True,
    )

    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            f"""
<div class="metric-lite">
  <div class="lab">Surface signalée</div>
  <div class="val">{result.anomaly_pixel_pct:.1f}%</div>
  <p class="muted" style="margin:0.35rem 0 0 0;font-size:0.88rem">
  Estimation de la part de l’image mise en évidence — pas le pourcentage de route
  dangereuse, ni une mesure de risque certifiée.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """
<div class="metric-lite">
  <div class="lab">Statut</div>
  <div class="val" style="font-size:1.25rem">Analyse prête</div>
  <p class="muted" style="margin:0.35rem 0 0 0;font-size:0.88rem">
  Vous pouvez comparer les vues et télécharger vos fichiers.</p>
</div>
""",
            unsafe_allow_html=True,
        )

    tab_res, tab_cmp, tab_dl, tab_adv = st.tabs(
        ["Résultats", "Comparaison", "Téléchargements", "Options avancées"]
    )

    with tab_res:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="panel-title">Image originale</div>', unsafe_allow_html=True)
            st.image(result.original_rgb, use_container_width=True)
        with c2:
            st.markdown('<div class="panel-title">Zones détectées</div>', unsafe_allow_html=True)
            st.image(mask_vis, use_container_width=True)
        with c3:
            st.markdown('<div class="panel-title">Visualisation superposée</div>', unsafe_allow_html=True)
            st.image(result.overlay_rgb, use_container_width=True)

        with st.expander("Voir en grand — image annotée"):
            st.image(result.overlay_rgb, use_container_width=True)

    with tab_cmp:
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="panel-title">Avant</div>', unsafe_allow_html=True)
            st.image(result.original_rgb, use_container_width=True)
        with right:
            st.markdown('<div class="panel-title">Après</div>', unsafe_allow_html=True)
            st.image(result.overlay_rgb, use_container_width=True)

    with tab_dl:
        st.write("Téléchargez vos fichiers pour les conserver ou les partager.")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                "Télécharger le masque",
                data=encode_png_mask(result.binary_full),
                file_name="roadanomaly_zones.png",
                mime="image/png",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "Télécharger l'image annotée",
                data=encode_png_rgb(result.overlay_rgb),
                file_name="roadanomaly_image_annotee.png",
                mime="image/png",
                use_container_width=True,
            )
        with d3:
            with st.expander("Fichier avancé"):
                st.download_button(
                    "Télécharger la carte de probabilité",
                    data=encode_png_probability(result.probability_full),
                    file_name="roadanomaly_intensite.png",
                    mime="image/png",
                    use_container_width=True,
                )

    with tab_adv:
        st.caption(
            "Section réservée à une exploration visuelle optionnelle. "
            "Le résultat principal ci-dessus n’est pas modifié."
        )
        # Exploration kept but not prominent; import lazily to avoid cluttering main path.
        from app_inference import (  # noqa: WPS433
            EXPLORATION_LABEL,
            EXPLORATION_THRESHOLD_MAX,
            EXPLORATION_THRESHOLD_MIN,
            derive_exploratory_view,
        )

        with st.expander("Mode exploratoire (optionnel)", expanded=False):
            st.warning(EXPLORATION_LABEL)
            thr = st.slider(
                "Sensibilité d’affichage",
                min_value=float(EXPLORATION_THRESHOLD_MIN),
                max_value=float(EXPLORATION_THRESHOLD_MAX),
                value=float(E2B_THRESHOLD),
                step=0.05,
            )
            explore = derive_exploratory_view(result, thr)
            # Official result must remain untouched.
            assert abs(float(result.threshold) - E2B_THRESHOLD) < 1e-12
            a, b = st.columns(2)
            with a:
                st.caption("Résultat principal (inchangé)")
                st.image(mask_vis, use_container_width=True)
            with b:
                st.caption("Vue exploratoire")
                st.image(colorize_mask(explore.binary_full), use_container_width=True)

    st.markdown(
        """
<div class="disclaimer">
Cette analyse constitue une aide visuelle expérimentale. Elle ne remplace pas une
inspection humaine ni un système certifié de sécurité routière.
</div>
""",
        unsafe_allow_html=True,
    )
    render_faq()
    render_privacy()


if __name__ == "__main__":
    main()
