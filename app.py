"""
RoadAnomaly — application Streamlit d'inférence (modèle E2b figé).

Lancement :
  streamlit run app.py

Checkpoint attendu (non versionné) :
  checkpoints/phase4/E2b_unet_mobilenetv2_finetune_20261108_1721/best.keras
ou variable d'environnement ROADANOMALY_E2B_CHECKPOINT.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    DEFAULT_RELATIVE_CHECKPOINT,
    encode_png_mask,
    encode_png_probability,
    encode_png_rgb,
    load_e2b_model,
    load_user_image,
    predict_e2b,
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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
}
.block-container {
  padding-top: 1.5rem;
  padding-bottom: 2rem;
  max-width: 1200px;
}
h1, h2, h3 {
  font-family: "DM Sans", "IBM Plex Sans", sans-serif;
  letter-spacing: -0.02em;
}
.ra-hero {
  background: linear-gradient(135deg, #1a2332 0%, #2c3e50 55%, #3d4f63 100%);
  color: #f4f1ea;
  padding: 1.6rem 1.8rem;
  border-radius: 12px;
  margin-bottom: 1.25rem;
  border: 1px solid rgba(255,255,255,0.08);
}
.ra-hero h1 {
  color: #f7f3eb !important;
  margin: 0 0 0.4rem 0;
  font-size: 1.85rem;
}
.ra-hero p {
  margin: 0;
  color: #d7dde5;
  max-width: 52rem;
  line-height: 1.5;
}
.ra-badge {
  display: inline-block;
  margin-top: 0.85rem;
  padding: 0.25rem 0.7rem;
  border-radius: 999px;
  background: #c45c26;
  color: white;
  font-size: 0.78rem;
  font-weight: 600;
}
.ra-metric {
  background: #f7f5f1;
  border: 1px solid #e4e0d8;
  border-radius: 10px;
  padding: 0.9rem 1rem;
  height: 100%;
}
.ra-metric .label {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #6b7280;
  margin-bottom: 0.25rem;
}
.ra-metric .value {
  font-size: 1.55rem;
  font-weight: 700;
  color: #1a2332;
  font-family: "DM Sans", sans-serif;
}
.ra-metric .hint {
  font-size: 0.8rem;
  color: #6b7280;
  margin-top: 0.2rem;
}
.ra-card {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 0.75rem;
}
.ra-card h4 {
  margin: 0 0 0.5rem 0;
  font-size: 0.95rem;
  color: #374151;
}
.ra-warn {
  background: #fff8ef;
  border-left: 4px solid #c45c26;
  padding: 0.85rem 1rem;
  border-radius: 0 8px 8px 0;
  color: #3f3a34;
  line-height: 1.45;
}
footer { visibility: hidden; }
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
    pour localiser des obstacles routiers inhabituels (animaux, cônes, débris, etc.)
    à partir d’une image de route.
  </p>
  <span class="ra-badge">Modèle figé E2b · seuil 0,55 · entrée 512×288</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
L’application charge le modèle **E2b** (U-Net + encodeur MobileNetV2 avec fine-tuning
sélectif), sélectionné sur le jeu de **validation** puis évalué **une seule fois** sur le
test. Le seuil opérationnel **0,55** est figé : il n’est pas retuné sur vos images.

Le modèle détecte les régions correspondant aux **anomalies représentées dans ses
labels d’entraînement** (masques pixel RoadAnomaly). Ce n’est pas une détection
d’anomalies non supervisée au sens de Lis et al. (ICCV 2019).
"""
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Paramètres scientifiques")
        st.write(f"**Expérience :** `{E2B_EXPERIMENT_ID}`")
        st.write(f"**Seuil :** `{E2B_THRESHOLD}` (figé)")
        st.write(f"**Entrée :** `{E2B_INPUT_WIDTH}×{E2B_INPUT_HEIGHT}`")
        st.write("**Normalisation :** `mobilenet_v2`")
        st.caption(f"SHA-256 attendu : `{E2B_CHECKPOINT_SHA256[:16]}…`")
        st.divider()
        st.markdown("### Checkpoint")
        try:
            path = resolve_e2b_checkpoint()
            st.success(f"Trouvé : `{path}`")
        except FileNotFoundError as exc:
            st.error("Checkpoint E2b manquant")
            st.code(str(exc), language=None)
            st.info(
                f"Copiez `best.keras` depuis Google Drive vers\n"
                f"`{DEFAULT_RELATIVE_CHECKPOINT}`"
            )


def render_limitations() -> None:
    st.markdown("### Limites et interprétation")
    st.markdown(
        """
<div class="ra-warn">
<strong>Ce que prédit le modèle.</strong> Pour chaque pixel, une probabilité d’appartenir
à la classe « anomalie » (obstacle inhabituel tel qu’annoté dans RoadAnomaly), puis un
masque binaire après seuil 0,55.<br><br>
<strong>Pixel positif</strong> = classe anomalie prédite (pas une vérité terrain).<br><br>
<strong>Différence prédiction / annotation :</strong> le masque affiché est une estimation
du modèle ; ce n’est pas un label humain.<br><br>
<strong>Jeu de données petit</strong> (60 images). Échecs possibles sur types d’anomalies,
éclairages ou environnements non vus.<br><br>
<strong>Prototype académique</strong> — <em>pas</em> un système de sécurité ADAS validé.
La « probabilité moyenne » n’est <em>pas</em> un score de confiance calibré.
</div>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    render_home()
    render_sidebar()

    uploaded = st.file_uploader(
        "Charger une image de route (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="L’image est convertie en RGB, redimensionnée en 512×288 pour le modèle, "
        "puis le masque est remis à la taille d’origine.",
    )

    if uploaded is None:
        render_limitations()
        st.caption(
            "Astuce : utilisez une image hors du jeu de test figé pour la démonstration."
        )
        return

    try:
        image = load_user_image(uploaded.getvalue(), filename=uploaded.name)
    except Exception as exc:
        st.error(f"Fichier invalide : {exc}")
        return

    st.markdown("### Image chargée")
    c0, c1 = st.columns([2, 1])
    with c0:
        st.image(image, caption=f"{uploaded.name} — {image.shape[1]}×{image.shape[0]}", use_container_width=True)
    with c1:
        st.markdown(
            f"""
<div class="ra-metric">
  <div class="label">Dimensions d’origine</div>
  <div class="value">{image.shape[1]}×{image.shape[0]}</div>
  <div class="hint">Le modèle voit 512×288</div>
</div>
""",
            unsafe_allow_html=True,
        )

    run = st.button("Lancer la prédiction E2b", type="primary", use_container_width=True)
    if not run:
        render_limitations()
        return

    try:
        model = get_cached_model()
    except FileNotFoundError as exc:
        st.error("Impossible de charger E2b — checkpoint manquant.")
        st.code(str(exc), language=None)
        return
    except Exception as exc:
        st.error(f"Échec du chargement du modèle : {type(exc).__name__}: {exc}")
        return

    with st.spinner("Inférence en cours…"):
        try:
            result = predict_e2b(model, image, threshold=E2B_THRESHOLD)
        except Exception as exc:
            st.error(f"Échec de la prédiction : {type(exc).__name__}: {exc}")
            return

    st.markdown("### Tableau de bord des résultats")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""
<div class="ra-metric">
  <div class="label">Pixels anomalie</div>
  <div class="value">{result.anomaly_pixel_pct:.2f}%</div>
  <div class="hint">Après seuil {result.threshold:.2f}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
<div class="ra-metric">
  <div class="label">Proba. moyenne (modèle)</div>
  <div class="value">{result.mean_anomaly_probability:.3f}</div>
  <div class="hint">Non calibrée — pas un score de confiance</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
<div class="ra-metric">
  <div class="label">Seuil figé</div>
  <div class="value">{result.threshold:.2f}</div>
  <div class="hint">Sélection validation, non retuné</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.write("")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown('<div class="ra-card"><h4>1. Image originale</h4></div>', unsafe_allow_html=True)
        st.image(result.original_rgb, use_container_width=True)
    with col_b:
        st.markdown('<div class="ra-card"><h4>2. Masque binaire</h4></div>', unsafe_allow_html=True)
        st.image(result.binary_full * 255, use_container_width=True, clamp=True)
    with col_c:
        st.markdown('<div class="ra-card"><h4>3. Superposition</h4></div>', unsafe_allow_html=True)
        st.image(result.overlay_rgb, use_container_width=True)

    with st.expander("Carte de probabilités (résolution modèle 512×288)", expanded=False):
        st.image(result.probability_map, use_container_width=True, clamp=True)

    st.markdown("### Téléchargements")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Masque binaire (PNG)",
            data=encode_png_mask(result.binary_full),
            file_name="roadanomaly_mask.png",
            mime="image/png",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Superposition (PNG)",
            data=encode_png_rgb(result.overlay_rgb),
            file_name="roadanomaly_overlay.png",
            mime="image/png",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            "Carte de probabilités (PNG)",
            data=encode_png_probability(result.probability_full),
            file_name="roadanomaly_probability.png",
            mime="image/png",
            use_container_width=True,
        )

    render_limitations()


if __name__ == "__main__":
    main()
