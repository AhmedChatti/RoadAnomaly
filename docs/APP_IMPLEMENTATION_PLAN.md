# Plan d’implémentation — application Streamlit RoadAnomaly

**Date :** 2026-11-08  
**Modèle figé :** `E2b_unet_mobilenetv2_finetune_20261108_1721`  
**Seuil opérationnel figé :** `0.55`  
**Normalisation :** `mobilenet_v2` (`uint8/127.5 - 1` → ≈[-1, 1])  
**Entrée modèle :** largeur **512** × hauteur **288**

## 1. Constats d’inspection

### Checkpoint E2b

| Champ | Valeur |
| --- | --- |
| Expérience | `E2b_unet_mobilenetv2_finetune_20261108_1721` |
| SHA-256 attendu | `280170dc64347b0b4cc8ce3af9be40dec4ac3733b27d1abeb620cc0c8e88e432` |
| Chemin Drive (Colab) | `/content/drive/MyDrive/RoadAnomaly/checkpoints/phase4/E2b_unet_mobilenetv2_finetune_20261108_1721/best.keras` |
| Présent localement dans le dépôt | **NON** |
| Autre checkpoint local | E1 seulement (`checkpoints/E1_cnn_scratch_20261108_1510/best.keras`) |

**Bloqueur dépendance :** le fichier `best.keras` E2b n’est pas disponible sur la machine de développement. L’application est écrite pour le charger depuis un chemin configurable ; **aucun autre modèle ne sera substitué**.

Chemin local attendu après copie manuelle depuis Drive :

`checkpoints/phase4/E2b_unet_mobilenetv2_finetune_20261108_1721/best.keras`

Variable d’environnement alternative : `ROADANOMALY_E2B_CHECKPOINT`.

### Architecture et chargement

- Constructeur : `src/models/unet_mobilenetv2.py` → `build_unet_mobilenetv2`
- Sortie : `anomaly_prob`, activation **sigmoid** (probabilité [0, 1] par pixel)
- Chargement robuste Keras : `src/checkpoint_io.py` (`load_keras_model_compat` + rebuild + `load_weights_from_keras_zip`) — nécessaire car E1/E2 ont été sauvés sous Keras 3.x avec configs d’initialiseurs parfois incompatibles Colab/local

### Prétraitement exact E2b

Réutilisé depuis `src/preprocessing.py` (pas de duplication sémantique) :

1. RGB uint8  
2. `resize_image_rgb(..., width=512, height=288)` — bilinéaire  
3. `normalize_image(..., mode="mobilenet_v2")`  
4. Batch `(1, 288, 512, 3)` → `model.predict`  
5. Seuil **0.55** → masque binaire  
6. Remise à la taille d’origine du masque : **nearest-neighbour**  
7. Carte de probabilités conservée séparément (redimensionnement bilinéaire pour affichage uniquement)

### Contraintes scientifiques

- Pas de réentraînement  
- Pas d’accès au split **test** pour développement / réglage  
- Seuil et modèle **non** retunés sur des uploads utilisateurs  
- Pas de statistiques de performance inventées dans l’UI

## 2. Architecture applicative prévue

```
app.py                      # point d’entrée Streamlit (FR)
src/app_inference.py        # chargement modèle + prédiction (sans UI)
requirements-app.txt        # deps app (streamlit + TF déjà projet)
tests/test_app_inference.py # preprocess, seuil, dimensions, overlay
docs/APP_USER_GUIDE.md
```

## 3. Fonctionnalités UI

- Accueil FR + upload JPG/JPEG/PNG  
- Dashboard : image / masque / overlay / % pixels / proba moyenne (libellée non calibrée)  
- Téléchargements : masque, overlay, proba  
- Section limites académiques  

## 4. Validation

- Tests unitaires sans poids (preprocess / seuil / overlay)  
- Test d’intégration **conditionnel** : skip si checkpoint absent ; sinon vérifie SHA-256 + prédiction sur image **synthétique** (pas le test set)  
- Lancement `streamlit run app.py` après copie du checkpoint

## 5. Statut

| Élément | Statut |
| --- | --- |
| Inspection | FAIT |
| Code application | FAIT (`app.py`, `src/app_inference.py`) |
| Tests unitaires | FAIT (6 OK ; intégration E2b skip sans poids) |
| Checkpoint E2b local | **MANQUANT — action utilisateur** |
| Substitution E0/E1/E2a | Interdite |
