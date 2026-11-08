# Guide utilisateur — application Streamlit RoadAnomaly

## Prérequis

1. Python 3.10+ recommandé  
2. Checkpoint E2b (non inclus dans GitHub) :

```text
checkpoints/phase4/E2b_unet_mobilenetv2_finetune_20261108_1721/best.keras
```

Copiez ce fichier depuis Google Drive  
`MyDrive/RoadAnomaly/checkpoints/phase4/E2b_unet_mobilenetv2_finetune_20261108_1721/best.keras`

SHA-256 attendu : `280170dc64347b0b4cc8ce3af9be40dec4ac3733b27d1abeb620cc0c8e88e432`

Alternative :

```powershell
$env:ROADANOMALY_E2B_CHECKPOINT = "D:\chemin\vers\best.keras"
```

## Installation

```powershell
cd C:\Users\man\Downloads\RoadAnomaly
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-app.txt
```

## Lancement

```powershell
streamlit run app.py
```

Puis ouvrez l’URL locale affichée (souvent `http://localhost:8501`).

## Utilisation

1. Chargez une image JPG/JPEG/PNG.  
2. Cliquez sur **Lancer la prédiction E2b**.  
3. Consultez d’abord l’onglet **Comparaison officielle** (seuil figé **0,55**).  
4. (Optionnel) Ouvrez **Exploration visuelle** pour un seuil d’affichage différent.  
5. Téléchargez les PNG **officiels** pour le rapport.

## Prédiction officielle vs exploration visuelle

| | Officiel | Exploratoire |
| --- | --- | --- |
| Seuil | **0,55** figé (validation) | Curseur 0,10–0,90 |
| Inférence modèle | Une fois | **Aucune** (re-seuillage seulement) |
| Statut scientifique | Référence protocole | **Hors protocole** |
| Peut retuner le projet ? | Non | Non |
| Libellé UI | Prédiction officielle | *Seuil exploratoire — hors protocole scientifique* |

- La prédiction **officielle** est toujours produite au seuil **0,55**.  
- Le curseur exploratoire **ne relance pas** E2b et **n’écrase jamais** le masque / les métriques officiels.  
- Les vues exploratoires **ne sont pas** des résultats de test validés (Stage 5).  
- Ne citez que les sorties **officielles** dans le rapport ou la soutenance.

## Limites

Prototype académique sur un petit jeu (60 images). Pas un système de conduite autonome validé. La probabilité moyenne n’est pas une confiance calibrée. Un masque exploratoire plus « complet » sur une image donnée n’autorise pas à changer le seuil scientifique.
