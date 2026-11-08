# Pipeline d’annotations humaines

## Principes

1. Les annotations joueurs sont des **candidates**, jamais une vérité terrain automatique.
2. **E2b** (poids, seuil 0,55, artefacts Stage 5) reste immuable.
3. Le split **test** verrouillé n’est jamais utilisé pour le jeu ni pour l’export d’entraînement futur.
4. Aucune promotion automatique vers un dataset d’entraînement sans revue humaine.

## Stockage local

Racine : `data/annotations/` (gitignorée pour les binaires utilisateur).

```
data/annotations/
  index.jsonl
  leaderboard.json
  contributions/<contribution_id>/
    meta.json
    player_mask.png
    ai_mask.png            # si disponible
    ai_probability.npz     # si consentement
    image.png              # seulement avec consentement + rétention
  exports/vYYYYMMDDTHHMMSSZ/
    images/ masks/ meta/ manifest.json
```

## Métadonnées (`meta.json`)

- `contribution_id`, `challenge_type` (`training` | `discovery`)
- `player_session_id`, `timestamp_utc`
- `consent_to_contribute`, `image_retained`, `image_sha256`
- `image_source`, `license_info`, `frame_id` (si applicable)
- `model_version`, `model_threshold`, `model_checkpoint_sha256`
- `verification_status`, `review_comment`, `reviewer_id`, `reviewed_at_utc`
- `game_score` (métriques de jeu uniquement)

## Statuts

| Statut | Signification |
| --- | --- |
| `pending_review` | En attente |
| `approved` | Accepté pour export candidat |
| `rejected` | Refusé |
| `duplicate` | Empreinte image déjà vue (défi découverte) |
| `requires_additional_review` | Besoin d’un second avis / consensus |

## Contrôles qualité

- Taille max upload : 8 Mo.
- Masques binaires alignés aux dimensions de l’image.
- Détection de doublons par SHA-256 (découverte).
- Revue admin : image, masque joueur, masque IA, différences, commentaire.

## Export

Depuis l’UI Revue ou :

```python
from anomaly_hunter.store import AnnotationStore
from anomaly_hunter.export import export_approved_dataset
manifest = export_approved_dataset(AnnotationStore())
```

L’export déduplique par `image_sha256` et produit un dossier versionné. Il **ne lance pas** d’entraînement.

## Confidentialité

Préférer le traitement en mémoire. Ne conserver une image que si le joueur a consenti. Pas de stockage cloud tiers introduit par cette application.
