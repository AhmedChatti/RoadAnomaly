# Roadmap future — modèle E3 (non implémenté)

Ce document décrit la **préparation** d’un éventuel modèle E3. **Aucun entraînement automatique** n’est activé.

## Implémenté aujourd’hui

- Collecte de contributions + revue admin authentifiée.
- Export versionné des annotations **éligibles** (pas seulement approuvées).
- Exclusion permanente test + val des exports E3 train.
- Manifeste d’origine / licence / scene_group / SHA.
- E2b immuable comme référence officielle de l’app.

## Référence immuable

- **E2b** (`…1721`, seuil **0,55**, SHA `280170dc…`)
- Artefacts Stage 5 et split lock inchangés.

## Pipeline cible (manuel / documenté — futur)

1. Exporter les annotations éligibles.
2. Dédupliquer (SHA + near-dup + revue visuelle).
3. Versionner (`exports/v…/manifest.json`).
4. Split groupé par `scene_group` (anti-fuite) — **nouveau** hold-out ; ne pas réutiliser le test Stage 5 comme train.
5. Entraîner E3 hors poids E2b, sous autorisation explicite.
6. Évaluer E3 vs E2b figé (val d’abord ; test one-shot après freeze de sélection).
7. Promouvoir E3 seulement après rapport écrit + SHA. Jusque-là l’app expose E2b.

## Absences volontaires

- Pas de fine-tuning automatique depuis le jeu.
- Pas d’écriture dans le checkpoint E2b.
- Pas de claim « ground truth » sans revue.
- Pas de consensus multi-annotateur automatique.
