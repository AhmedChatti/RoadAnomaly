# Pipeline d’annotations humaines

## Principes

1. Les annotations joueurs sont des **candidates**, jamais une vérité terrain automatique.
2. **E2b** (poids, seuil 0,55, résultats Stage 5) reste immuable.
3. Split **test** : jamais dans le pool jeu GT ; jamais dans l’export E3 train.
4. Split **val** : autorisé en jeu GT ; **exclu** de l’export E3 train.
5. `approved` ≠ `eligible_for_e3_training` (voir `docs/DATA_LEAKAGE_PREVENTION.md`).
6. Aucune promotion automatique vers un dataset d’entraînement.

## Stockage local

Racine : `data/annotations/` (binaires utilisateur gitignorés).

```
data/annotations/
  index.jsonl
  leaderboard.json          # local appareil — pas un ranking mondial
  contributions/<id>/
    meta.json
    player_mask.png
    ai_mask.png
    ai_probability.npz      # si consentement
    image.png               # seulement avec consentement + rétention
  exports/vYYYYMMDDTHHMMSSZ/
    images/ masks/ meta/ manifest.json EXPORT_COMPLETE
```

## Métadonnées (`meta.json`)

- Identité : `contribution_id`, `challenge_type`, `player_session_id`, `timestamp_utc`
- Consentement / rétention : `consent_to_contribute`, `image_retained`, `image_sha256`, `near_duplicate_fp`
- Provenance : `image_source`, `license_info`, `frame_id`
- Modèle figé : `model_version`, `model_threshold` (0,55), `model_checkpoint_sha256`
- Revue : `verification_status`, `review_comment`, `reviewer_id`, `reviewed_at_utc`
- Cycle de vie : `annotation_lifecycle` ∈ {submitted, reviewed, approved, rejected, requires_additional_review}
- `game_score` : métriques de jeu uniquement

## Statuts de revue

| Statut | Signification |
| --- | --- |
| `pending_review` | En attente |
| `approved` | Accepté comme candidate (pas encore éligible E3 par défaut logique d’export) |
| `rejected` | Refusé |
| `duplicate` | Empreinte image déjà vue (découverte) |
| `requires_additional_review` | Conflit / second avis |

Les conflits multi-annotateurs (IoU bas sur même `image_sha256`) sont **signalés** dans l’UI admin ; aucun consensus automatique n’est affirmé.

## Contrôles qualité

- Upload ≤ 8 Mo ; masques binaires non vides alignés à l’image.
- IDs hex uniquement (anti path traversal).
- Détection doublons SHA-256 (découverte) + filtre near-dup à l’export.

## Export

```python
from anomaly_hunter.store import AnnotationStore
from anomaly_hunter.export import export_approved_dataset
# Requiert ROADANOMALY_ADMIN_TOKEN si enforce_admin=True
manifest = export_approved_dataset(AnnotationStore(), enforce_admin=True, admin_token="...")
```

N’écrase jamais une version existante. N’entraîne pas de modèle.

## Confidentialité

Pas de cloud tiers. Pas de claim de suppression automatique. Voir `docs/SECURITY_AND_PRIVACY_AUDIT.md`.
