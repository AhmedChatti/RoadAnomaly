# Anomaly Hunter — guide utilisateur

Jeu interactif intégré à l’application Streamlit RoadAnomaly (`Jouer` dans la navigation).

## Objectif

Repérer les obstacles sur une photo de route en **sélectionnant des cases** (comme un captcha Google), puis comparer votre choix à une référence (défi entraînement) ou à la prédiction automatique (défi découverte).

## Lancer

```powershell
pip install -r requirements-app.txt
$env:ROADANOMALY_ADMIN_TOKEN = "<votre-secret>"
streamlit run app.py
```

Ouvrir l’onglet **Jouer**.

## Comment jouer

1. Lancez un défi (entraînement ou découverte).
2. Une grille 3×3 est superposée à la photo.
3. Cliquez sur **toutes les cases** qui contiennent l’obstacle.
4. Validez quand vous êtes prêt — **aucun chronomètre**.
5. Vous pouvez **Quitter** à tout moment.
6. Après validation : merci pour votre contribution, comparaison, puis nouvelle manche ou sortie.

## Défi entraînement

- Images train/val uniquement (jamais le test verrouillé).
- Comparaison avec la référence et la prédiction officielle.
- Points de jeu (distincts des résultats scientifiques Stage 5).

## Défi découverte

- Importez une image JPG/JPEG/PNG ≤ 8 Mo.
- Consentement requis pour enregistrer la contribution.
- Pas de Dice/IoU validé sans vérité terrain.
- Différer de l’IA ne prouve pas que vous avez raison.

## Confidentialité

- Contributions conservées uniquement avec consentement.
- Stockage local `data/annotations/` ; pas de promotion automatique vers l’entraînement.
- E2b n’est jamais modifié.

## Revue administrateur

Onglet **Revue** — jeton `ROADANOMALY_ADMIN_TOKEN` uniquement. Voir `docs/SECURITY_AND_PRIVACY_AUDIT.md`.
