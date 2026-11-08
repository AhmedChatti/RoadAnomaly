"""
RoadAnomaly — application Streamlit d'inférence (modèle E2b figé).

Lancement :
  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app_inference import (  # noqa: E402
    E2B_CHECKPOINT_SHA256,
    E2B_EXPERIMENT_ID,
    E2B_INPUT_HEIGHT,
    E2B_INPUT_WIDTH,
    E2B_THRESHOLD,
    EXPLORATION_LABEL,
    EXPLORATION_THRESHOLD_MAX,
    EXPLORATION_THRESHOLD_MIN,
    DEFAULT_RELATIVE_CHECKPOINT,
    colorize_mask,
    derive_exploratory_view,
    encode_png_mask,
    encode_png_probability,
    encode_png_rgb,
    load_e2b_model,
    load_user_image,
    predict_e2b,
    probability_heatmap_rgb,
    resolve_e2b_checkpoint,
)

st.set_page_config(
    page_title="RoadAnomaly — Segmentation",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

:root {
  --ra-ink: #1c2430;
  --ra-muted: #5c6675;
  --ra-paper: #f3efe6;
  --ra-card: #fffcf7;
  --ra-line: #ddd4c6;
  --ra-signal: #e85d2a;
  --ra-signal-deep: #b8441a;
  --ra-asphalt: #243040;
}

html, body, [class*="css"]  {
  font-family: "Source Sans 3", "Segoe UI", sans-serif;
  color: var(--ra-ink);
}
.stApp {
  background:
    radial-gradient(1200px 500px at 10% -10%, #ffe8d6 0%, transparent 55%),
    radial-gradient(900px 400px at 100% 0%, #e7eef5 0%, transparent 50%),
    linear-gradient(180deg, #f7f3eb 0%, #efe9de 100%);
}
.block-container {
  padding-top: 1.2rem;
  padding-bottom: 2.5rem;
  max-width: 1180px;
}
h1, h2, h3, .ra-title {
  font-family: "Fraunces", Georgia, serif;
  letter-spacing: -0.02em;
  color: var(--ra-ink) !important;
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #1f2a38 0%, #16202b 100%);
}
[data-testid="stSidebar"] * {
  color: #e8eef5 !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption {
  color: #c5d0dc !important;
}

.ra-hero {
  position: relative;
  overflow: hidden;
  background:
    linear-gradient(120deg, rgba(28,36,48,0.92), rgba(36,48,64,0.88)),
    url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
  color: #f7f3eb;
  padding: 1.75rem 1.9rem 1.55rem;
  border-radius: 18px;
  margin-bottom: 1.1rem;
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 18px 40px rgba(28, 36, 48, 0.18);
}
.ra-hero h1 {
  color: #fff8f0 !important;
  margin: 0 0 0.45rem 0;
  font-size: 2.15rem;
  font-weight: 700;
}
.ra-hero p {
  margin: 0;
  color: #d9e0e8;
  max-width: 46rem;
  line-height: 1.55;
  font-size: 1.02rem;
}
.ra-badge-row { margin-top: 1rem; display: flex; flex-wrap: wrap; gap: 0.45rem; }
.ra-badge {
  display: inline-block;
  padding: 0.28rem 0.75rem;
  border-radius: 999px;
  background: var(--ra-signal);
  color: white;
  font-size: 0.76rem;
  font-weight: 700;
  letter-spacing: 0.02em;
}
.ra-badge.alt {
  background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.18);
}

.ra-section {
  margin: 1.35rem 0 0.55rem 0;
  font-size: 1.35rem;
}
.ra-lead {
  color: var(--ra-muted);
  margin-top: -0.35rem;
  margin-bottom: 0.9rem;
}

.ra-metric {
  background: var(--ra-card);
  border: 1px solid var(--ra-line);
  border-radius: 14px;
  padding: 1rem 1.05rem 0.95rem;
  height: 100%;
  box-shadow: 0 8px 22px rgba(28,36,48,0.05);
  position: relative;
  overflow: hidden;
}
.ra-metric::before {
  content: "";
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 4px;
  background: var(--ra-signal);
}
.ra-metric .label {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ra-muted);
  margin-bottom: 0.35rem;
  font-weight: 600;
}
.ra-metric .value {
  font-size: 1.75rem;
  font-weight: 700;
  color: var(--ra-ink);
  font-family: "Fraunces", Georgia, serif;
  line-height: 1.1;
}
.ra-metric .hint {
  font-size: 0.82rem;
  color: var(--ra-muted);
  margin-top: 0.35rem;
  line-height: 1.35;
}

.ra-panel {
  background: var(--ra-card);
  border: 1px solid var(--ra-line);
  border-radius: 14px;
  padding: 0.7rem 0.7rem 0.55rem;
  box-shadow: 0 10px 24px rgba(28,36,48,0.05);
  height: 100%;
}
.ra-panel .eyebrow {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ra-signal-deep);
  font-weight: 700;
  margin-bottom: 0.15rem;
}
.ra-panel h4 {
  margin: 0 0 0.55rem 0;
  font-size: 1.02rem;
  font-family: "Fraunces", Georgia, serif;
  color: var(--ra-ink);
}
.ra-panel .cap {
  margin-top: 0.45rem;
  font-size: 0.8rem;
  color: var(--ra-muted);
}

.ra-warn {
  background: #fff7ee;
  border: 1px solid #f0d7bf;
  border-left: 5px solid var(--ra-signal);
  padding: 1rem 1.1rem;
  border-radius: 0 12px 12px 0;
  color: #3f3a34;
  line-height: 1.5;
}
.ra-explore {
  background: #fffdf5;
  border: 1px dashed #c9a227;
  border-radius: 14px;
  padding: 1rem 1.1rem 1.15rem;
  margin-top: 0.35rem;
}
.ra-explore h4 {
  margin: 0 0 0.35rem 0;
  font-family: "Fraunces", Georgia, serif;
  color: #7a5b00;
}
.ra-explore p {
  margin: 0 0 0.75rem 0;
  color: #6b5a2a;
  font-size: 0.92rem;
  line-height: 1.45;
}
.ra-legend {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  margin: 0.35rem 0 0.9rem;
  color: var(--ra-muted);
  font-size: 0.86rem;
}
.ra-swatch {
  display: inline-block;
  width: 0.85rem;
  height: 0.85rem;
  border-radius: 3px;
  margin-right: 0.35rem;
  vertical-align: -1px;
}
div.stButton > button[kind="primary"] {
  background: linear-gradient(180deg, #f06a35, #d4511f);
  border: none;
  font-weight: 700;
}
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Chargement du modèle E2b…")
def get_cached_model():
    return load_e2b_model(verify_sha=True)


def render_home() -> None:
    st.markdown(
        """
<div class="ra-hero">
  <h1>RoadAnomaly</h1>
  <p>
    Prototype académique de <strong>segmentation sémantique binaire supervisée</strong>
    pour localiser des obstacles routiers inhabituels — animaux, cônes, débris —
    directement sur une image de route.
  </p>
  <div class="ra-badge-row">
    <span class="ra-badge">E2b figé</span>
    <span class="ra-badge alt">seuil 0,55</span>
    <span class="ra-badge alt">entrée 512×288</span>
    <span class="ra-badge alt">MobileNetV2</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
Le modèle **E2b** (U-Net + encodeur MobileNetV2, fine-tuning sélectif) a été choisi sur la
**validation**, puis évalué **une seule fois** sur le test. Le seuil **0,55** n’est pas
retuné sur vos uploads. Il détecte les anomalies **telles qu’annotées** dans RoadAnomaly —
ce n’est pas une détection non supervisée au sens de Lis et al. (ICCV 2019).
"""
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Protocole figé")
        st.write(f"**Expérience**  \n`{E2B_EXPERIMENT_ID}`")
        st.write(f"**Seuil**  \n`{E2B_THRESHOLD}` (non retuné)")
        st.write(f"**Entrée modèle**  \n`{E2B_INPUT_WIDTH}×{E2B_INPUT_HEIGHT}`")
        st.write("**Normalisation**  \n`mobilenet_v2`")
        st.caption(f"SHA-256 : `{E2B_CHECKPOINT_SHA256[:20]}…`")
        st.divider()
        st.markdown("### Checkpoint")
        try:
            path = resolve_e2b_checkpoint()
            st.success("E2b disponible")
            st.caption(str(path))
        except FileNotFoundError as exc:
            st.error("Checkpoint E2b manquant")
            st.code(str(exc), language=None)
            st.info(f"Attendu : `{DEFAULT_RELATIVE_CHECKPOINT}`")


def render_limitations() -> None:
    st.markdown('<h3 class="ra-section">Limites et interprétation</h3>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="ra-warn">
<strong>Prédiction.</strong> Probabilité d’anomalie par pixel, puis masque binaire au seuil 0,55.<br>
<strong>Pixel positif</strong> = classe anomalie <em>prédite</em>, pas une vérité terrain.<br>
<strong>Jeu petit</strong> (60 images) : échecs possibles hors distribution.<br>
<strong>Prototype académique</strong> — pas un système ADAS validé.
La « proba. moyenne » n’est <em>pas</em> une confiance calibrée.
</div>
""",
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, hint: str) -> None:
    st.markdown(
        f"""
<div class="ra-metric">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
  <div class="hint">{hint}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def panel(eyebrow: str, title: str, caption: str) -> None:
    st.markdown(
        f"""
<div class="ra-panel">
  <div class="eyebrow">{eyebrow}</div>
  <h4>{title}</h4>
</div>
""",
        unsafe_allow_html=True,
    )
    # caption rendered after image by caller optionally


def main() -> None:
    render_home()
    render_sidebar()

    st.markdown('<h3 class="ra-section">Analyse d’une image</h3>', unsafe_allow_html=True)
    st.markdown(
        '<p class="ra-lead">JPG / JPEG / PNG · RGB · le masque est recalculé à la résolution d’origine.</p>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Charger une image de route",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        render_limitations()
        st.caption("Astuce démo : préférez une image hors du jeu de test figé.")
        return

    try:
        image = load_user_image(uploaded.getvalue(), filename=uploaded.name)
    except Exception as exc:
        st.error(f"Fichier invalide : {exc}")
        return

    file_key = f"{uploaded.name}:{uploaded.size}:{image.shape}"
    c0, c1 = st.columns([2.2, 1])
    with c0:
        st.markdown(
            f"""
<div class="ra-panel">
  <div class="eyebrow">Entrée</div>
  <h4>{uploaded.name}</h4>
</div>
""",
            unsafe_allow_html=True,
        )
        st.image(image, use_container_width=True)
    with c1:
        metric_card("Dimensions", f"{image.shape[1]}×{image.shape[0]}", "Le réseau voit 512×288")
        st.write("")
        run = st.button("Lancer la prédiction E2b", type="primary", use_container_width=True)

    # Persist results across Streamlit reruns (downloads, widgets).
    if run or (
        st.session_state.get("ra_file_key") == file_key and "ra_result" in st.session_state
    ):
        need_infer = run or st.session_state.get("ra_file_key") != file_key
        if need_infer:
            try:
                model = get_cached_model()
            except FileNotFoundError as exc:
                st.error("Impossible de charger E2b — checkpoint manquant.")
                st.code(str(exc), language=None)
                return
            except Exception as exc:
                st.error(f"Échec du chargement : {type(exc).__name__}: {exc}")
                return
            with st.spinner("Inférence E2b en cours…"):
                try:
                    result = predict_e2b(model, image)
                except Exception as exc:
                    st.error(f"Échec de la prédiction : {type(exc).__name__}: {exc}")
                    return
            st.session_state["ra_file_key"] = file_key
            st.session_state["ra_result"] = result
            st.session_state["ra_mask_vis"] = colorize_mask(result.binary_full)
            st.session_state["ra_heat"] = probability_heatmap_rgb(result.probability_full)
        else:
            result = st.session_state["ra_result"]
    else:
        render_limitations()
        return

    # Official artefacts are frozen after inference; exploration never overwrites them.
    official = result
    mask_vis = st.session_state["ra_mask_vis"]
    heat = st.session_state["ra_heat"]
    assert abs(float(official.threshold) - E2B_THRESHOLD) < 1e-12

    st.markdown(
        '<h3 class="ra-section">Prédiction officielle (protocole scientifique)</h3>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="ra-lead">Modèle E2b · seuil figé <strong>{E2B_THRESHOLD:.2f}</strong> · '
        "non retuné sur cette image · résultats de test Stage 5 inchangés.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="ra-legend">
  <span><span class="ra-swatch" style="background:#e85d2a"></span>Anomalie prédite (officiel)</span>
  <span><span class="ra-swatch" style="background:#1c2430"></span>Fond</span>
  <span><span class="ra-swatch" style="background:#ffe8b3"></span>Contour du masque</span>
</div>
""",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card(
            "Pixels anomalie (officiel)",
            f"{official.anomaly_pixel_pct:.2f}%",
            f"Seuil figé {official.threshold:.2f}",
        )
    with m2:
        metric_card(
            "Proba. moyenne",
            f"{official.mean_anomaly_probability:.3f}",
            "Sortie modèle — non calibrée",
        )
    with m3:
        metric_card(
            "Proba. max",
            f"{float(official.probability_map.max()):.3f}",
            "Sur la grille 512×288",
        )
    with m4:
        metric_card("Seuil figé", f"{official.threshold:.2f}", "Choisi sur validation")

    st.write("")
    tab_cmp, tab_explore, tab_prob, tab_dl = st.tabs(
        [
            "Comparaison officielle",
            "Exploration visuelle",
            "Carte de probabilités",
            "Téléchargements",
        ]
    )

    with tab_cmp:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(
                '<div class="ra-panel"><div class="eyebrow">01</div><h4>Image originale</h4></div>',
                unsafe_allow_html=True,
            )
            st.image(official.original_rgb, use_container_width=True)
            st.caption("Résolution d’origine, non retouchée.")
        with col_b:
            st.markdown(
                '<div class="ra-panel"><div class="eyebrow">02</div>'
                "<h4>Masque officiel (0,55)</h4></div>",
                unsafe_allow_html=True,
            )
            st.image(mask_vis, use_container_width=True)
            st.caption("Orange = anomalie (≥ 0,55), fond asphalté.")
        with col_c:
            st.markdown(
                '<div class="ra-panel"><div class="eyebrow">03</div>'
                "<h4>Superposition officielle</h4></div>",
                unsafe_allow_html=True,
            )
            st.image(official.overlay_rgb, use_container_width=True)
            st.caption("Prédiction scientifique de référence pour cette image.")

    with tab_explore:
        st.markdown(
            f"""
<div class="ra-explore">
  <h4>{EXPLORATION_LABEL}</h4>
  <p>
    Ce curseur ne relance <strong>pas</strong> le modèle et ne modifie <strong>pas</strong>
    la prédiction officielle (seuil 0,55). Il re-binarise uniquement la carte de
    probabilités déjà calculée, pour l’interprétation visuelle.
    <strong>Ce n’est pas un résultat de test validé</strong> et ne peut pas servir
    à retuner le protocole.
  </p>
</div>
""",
            unsafe_allow_html=True,
        )
        explore_thr = st.slider(
            EXPLORATION_LABEL,
            min_value=float(EXPLORATION_THRESHOLD_MIN),
            max_value=float(EXPLORATION_THRESHOLD_MAX),
            value=float(E2B_THRESHOLD),
            step=0.05,
            help="Hors protocole scientifique — visualisation seulement.",
        )
        explore = derive_exploratory_view(official, explore_thr)
        # Prove UI path cannot mutate official session artefacts.
        assert np.array_equal(st.session_state["ra_result"].binary_full, official.binary_full)
        assert abs(float(st.session_state["ra_result"].threshold) - E2B_THRESHOLD) < 1e-12

        e0, e1, e2 = st.columns(3)
        with e0:
            metric_card(
                "Seuil exploratoire",
                f"{explore.threshold:.2f}",
                explore.label,
            )
        with e1:
            metric_card(
                "Pixels (exploratoire)",
                f"{explore.anomaly_pixel_pct:.2f}%",
                "Recalculé sans nouvelle inférence",
            )
        with e2:
            metric_card(
                "Pixels (officiel, inchangé)",
                f"{official.anomaly_pixel_pct:.2f}%",
                f"Toujours au seuil {E2B_THRESHOLD:.2f}",
            )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                '<div class="ra-panel"><div class="eyebrow">Officiel</div>'
                "<h4>Masque @ 0,55 (référence)</h4></div>",
                unsafe_allow_html=True,
            )
            st.image(mask_vis, use_container_width=True)
            st.caption("Ne change jamais avec le curseur.")
        with c2:
            st.markdown(
                f'<div class="ra-panel"><div class="eyebrow">Exploratoire</div>'
                f"<h4>Masque @ {explore.threshold:.2f}</h4></div>",
                unsafe_allow_html=True,
            )
            st.image(colorize_mask(explore.binary_full), use_container_width=True)
            st.caption(EXPLORATION_LABEL)

        st.markdown(
            f'<div class="ra-panel"><div class="eyebrow">Exploratoire</div>'
            f"<h4>Superposition @ {explore.threshold:.2f}</h4></div>",
            unsafe_allow_html=True,
        )
        st.image(explore.overlay_rgb, use_container_width=True)
        st.download_button(
            f"Télécharger masque exploratoire (@ {explore.threshold:.2f})",
            data=encode_png_mask(explore.binary_full),
            file_name=f"roadanomaly_mask_exploratory_{explore.threshold:.2f}.png",
            mime="image/png",
            use_container_width=True,
            key="dl_explore_mask",
        )

    with tab_prob:
        p1, p2 = st.columns(2)
        with p1:
            st.markdown(
                '<div class="ra-panel"><div class="eyebrow">Heatmap</div>'
                "<h4>Probabilités (taille d’origine)</h4></div>",
                unsafe_allow_html=True,
            )
            st.image(heat, use_container_width=True)
            st.caption("Carte partagée — l’exploration ne fait que re-seuiller.")
        with p2:
            st.markdown(
                '<div class="ra-panel"><div class="eyebrow">Modèle</div>'
                "<h4>Grille native 512×288</h4></div>",
                unsafe_allow_html=True,
            )
            st.image(probability_heatmap_rgb(official.probability_map), use_container_width=True)
            st.caption("Sortie sigmoid avant redimensionnement du masque.")

    with tab_dl:
        st.write("Exports **officiels** (seuil 0,55) pour le rapport ou la soutenance.")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                "Masque officiel",
                data=encode_png_mask(official.binary_full),
                file_name="roadanomaly_mask_official_0.55.png",
                mime="image/png",
                use_container_width=True,
                key="dl_official_mask",
            )
        with d2:
            st.download_button(
                "Superposition officielle",
                data=encode_png_rgb(official.overlay_rgb),
                file_name="roadanomaly_overlay_official_0.55.png",
                mime="image/png",
                use_container_width=True,
                key="dl_official_overlay",
            )
        with d3:
            st.download_button(
                "Probabilités",
                data=encode_png_probability(official.probability_full),
                file_name="roadanomaly_probability.png",
                mime="image/png",
                use_container_width=True,
                key="dl_official_prob",
            )
        st.caption(
            "Les exports exploratoires sont disponibles dans l’onglet "
            "« Exploration visuelle » et portent le libellé hors protocole."
        )

    render_limitations()


if __name__ == "__main__":
    main()
