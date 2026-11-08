# Anomaly Hunter — guide utilisateur

Jeu interactif intégré à l’application Streamlit RoadAnomaly (`Jouer` dans la navigation).

## Objectif

Repérer des obstacles sur une photo de route en dessinant autour d’eux, comparer votre annotation à une référence (défi entraînement) ou à la prédiction automatique (défi découverte), et éventuellement contribuer des annotations **candidates** pour une revue humaine.

## Lancer

```powershell
pip install -r requirements-app.txt
streamlit run app.py
```

Ouvrir l’onglet **Jouer**.

## Défi entraînement

- Images issues uniquement des splits **train** et **val** (jamais le test verrouillé).
- Chronomètre de 15 secondes, outils pinceau / rectangle / gomme.
- Après validation : comparaison avec la référence vérifiée et la prédiction officielle du modèle de référence.
- Scores Dice / IoU et points **de jeu** (distincts des résultats scientifiques Stage 5).

## Défi découverte

- Vous importez une nouvelle image (JPG/JPEG/PNG, ≤ 8 Mo) avec licence et source déclarées.
- Consentement obligatoire pour enregistrer l’annotation.
- Pas de Dice/IoU validé (pas de vérité terrain).
- Différer de l’IA **ne prouve pas** que vous avez raison.

## Scores et succès

Points, niveaux, record personnel, succès et classement local sont **ludiques**. Ils ne remplacent pas l’évaluation scientifique figée.

## Confidentialité

- Sans consentement : pas de conservation permanente de l’image.
- Avec consentement : stockage local sous `data/annotations/` pour revue administrateur.
- Aucune promotion automatique vers l’entraînement. Le modèle E2b n’est jamais modifié par le jeu.

## Revue administrateur

Onglet **Revue** — jeton `ROADANOMALY_ADMIN_TOKEN` (défaut local documenté dans l’UI). Voir aussi `docs/HUMAN_ANNOTATION_PIPELINE.md`.
