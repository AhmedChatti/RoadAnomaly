# Anomaly Hunter — guide utilisateur

Jeu interactif intégré à l’application Streamlit RoadAnomaly (`Jouer` dans la navigation).

## Objectif

Repérer des obstacles sur une photo de route en dessinant autour d’eux, comparer votre annotation à une référence (défi entraînement) ou à la prédiction automatique (défi découverte), et éventuellement contribuer des annotations **candidates** pour une revue humaine.

## Lancer

```powershell
pip install -r requirements-app.txt
# Requis pour la page Revue (secret ≥16 caractères, non prévisible) :
$env:ROADANOMALY_ADMIN_TOKEN = "<votre-secret>"
streamlit run app.py
```

Ouvrir l’onglet **Jouer**.

## Défi entraînement

- Images issues uniquement des splits **train** et **val** (jamais le test verrouillé).
- Chronomètre de 15 secondes : après expiration, la validation est **refusée** (aucun point).
- Outils pinceau / rectangle / gomme (ou polygone texte si le module canvas est absent).
- Après validation : comparaison avec la référence vérifiée et la prédiction officielle.
- Scores Dice / IoU et points **de jeu** (distincts des résultats scientifiques Stage 5).
- Chaque défi ne rapporte des points **qu’une seule fois** par session.

## Défi découverte

- Image JPG/JPEG/PNG ≤ 8 Mo ; consentement obligatoire pour enregistrer la contribution.
- Pas de formulaire URL/licence côté joueur (valeurs par défaut « non déclaré »).
- Sans droits déclarés, la contribution reste **non éligible** à un futur export E3.
- Pas de Dice/IoU validé (pas de vérité terrain).
- Différer de l’IA **ne prouve pas** que vous avez raison.
- Statut initial : `pending_review` (ou `duplicate` si empreinte déjà vue).

## Scores et succès

Points, niveaux, record, succès et classement sont **ludiques** et **locaux à l’appareil**.  
Ce n’est **pas** un classement communautaire mondial. Aucun compte multi-utilisateur n’est implémenté.

## Confidentialité

- Sans consentement : pas de conservation permanente de l’image.
- Avec consentement : stockage local sous `data/annotations/` pour revue admin.
- Pas de suppression automatique garantie — effacement manuel des fichiers si besoin.
- Aucune promotion automatique vers l’entraînement. E2b n’est jamais modifié.

## Revue administrateur

Onglet **Revue** — jeton via `ROADANOMALY_ADMIN_TOKEN` uniquement (aucun défaut).  
Voir `docs/SECURITY_AND_PRIVACY_AUDIT.md` et `docs/HUMAN_ANNOTATION_PIPELINE.md`.
