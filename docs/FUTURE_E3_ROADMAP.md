# Roadmap future — modèle E3 (non implémenté)

Ce document décrit la **préparation** d’un éventuel modèle E3. Aucun entraînement automatique n’est activé à ce stade.

## Référence immuable

- **E2b** (`E2b_unet_mobilenetv2_finetune_20261108_1721`, seuil opérationnel **0,55**, SHA `280170dc…`) reste le modèle de référence et la prédiction officielle de l’application.
- Les artefacts Stage 5 et le split test verrouillé ne doivent pas être altérés ni réutilisés pour tuning.

## Pipeline cible (manuel / documenté)

1. **Exporter** les annotations `approved` (`anomaly_hunter.export.export_approved_dataset`).
2. **Dédupliquer** (empreinte image + revue visuelle).
3. **Versionner** le dataset candidat (`exports/v…/manifest.json`).
4. **Splitter** train/val/**nouveau-test** avec regroupement de scènes liées (même logique anti-fuite que Phase 1 — ne pas mélanger un groupe de scènes entre splits). **Ne pas** réutiliser l’ancien test Stage 5 comme set d’entraînement.
5. **Entraîner** un nouvel E3 (architecture et hyperparamètres à définir) sous autorisation explicite, hors poids E2b.
6. **Évaluer** E3 contre E2b figé sur un protocole documenté (validation d’abord ; test one-shot seulement après freeze de sélection).
7. **Promouvoir** E3 uniquement après validation écrite (rapport + SHA checkpoint + comparaison tabulaire). Jusque-là, l’app continue d’exposer E2b.

## Ce qui est déjà en place

- Collecte de contributions + statuts de revue.
- Export versionné des annotations approuvées.
- Séparation stricte jeu / science dans l’UI et la doc.

## Ce qui est volontairement absent

- Pas de fine-tuning automatique depuis le jeu.
- Pas d’écriture dans `checkpoints/phase4/E2b_…`.
- Pas d’accès jeu au split test verrouillé.
- Pas de claim « ground truth » sur une annotation joueur non revue.
