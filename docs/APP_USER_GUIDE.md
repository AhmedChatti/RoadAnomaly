# Guide utilisateur — RoadAnomaly

Application française d’analyse d’images de route. Interface grand public ; le moteur d’analyse reste le modèle scientifique figé du projet (non affiché à l’écran).

## Installation

```powershell
cd C:\Users\man\Downloads\RoadAnomaly
pip install -r requirements-app.txt
```

Le fichier modèle doit être présent localement (non versionné dans Git) :

```text
checkpoints/phase4/E2b_unet_mobilenetv2_finetune_20261108_1721/best.keras
```

## Lancement

```powershell
streamlit run app.py
```

Ouvrez l’URL indiquée (souvent `http://localhost:8501`).

## Utilisation

1. Lisez le guide d’accueil **Bienvenue sur RoadAnomaly !** puis cliquez **Commencer**.  
2. Importez une image JPG / JPEG / PNG (ou **Essayer un exemple**).  
3. Cliquez **Analyser l'image**.  
4. Consultez **Résultat de votre analyse** (originale, zones, superposition).  
5. Téléchargez le masque ou l’image annotée si besoin.  
6. **Nouvelle image** pour recommencer.  
7. Bouton **Aide** pour rouvrir le guide.

## Confidentialité

Les images sont traitées **en mémoire** pendant la session. L’application ne crée pas de compte et n’enregistre pas de galerie permanente.

## Avertissement

Aide visuelle expérimentale uniquement. Ne remplace pas une inspection humaine ni un système certifié de sécurité routière.

## Pour l’équipe projet (hors UI)

- Prédiction officielle : seuil interne figé **0,55** (`predict_e2b`).  
- Mode exploratoire : section **Options avancées** uniquement ; n’écrase jamais le résultat officiel.  
- Détails scientifiques : `docs/APP_IMPLEMENTATION_PLAN.md`, rapport CRISP-DM.
