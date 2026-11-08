 
# Road Anomaly dataset

![](frames/animals06_sheep_roads_lambs.webp)

This dataset contains images of unusual dangers which can be encountered by a vehicle on the road - animals, rocks, traffic cones and other obstacles.
Its purpose is testing autonomous driving perception algorithms in rare but safety-critical circumstances.

The dataset contains images with associated per-pixel labels.
The labeling has been performed with our [LabelGrab](github.com/cvlab-epfl/LabelGrab) tool. Most frames retain the editor's files and can be further edited; some are missing because the labeling was done at a different resolution and rescaled. 
If you need the editor files and instance labels for all frames, please [contact us](mailto:krzysztof.lis@epfl.ch).

For any additional information, please contact [Krzysztof Lis](mailto:krzysztof.lis@epfl.ch).

The images are provided for research purposes.
We do not own the images; the authors and sources are listed in `credits.txt`.

### Directory structure

`RoadAnomaly`
- `readme.md`
- `frame_list.json` - list of all image file names
- `credits.txt` - sources of the images
- `frames/`
  - `frame_name.webp` - image
  - `frame_name.labels/labels_semantic.png` - per pixel label, background =  `0` and anomaly = `2`
  - `frame_name.labels/labels_instance.png` - instance id of objects, some are missing at this time.

### References
If you find this dataset useful, please cite:

```
@inproceedings{DetectingTheUnexpected2019,
	title = {Detecting the Unexpected via Image Resynthesis},
	author = {Krzysztof Lis and Krishna Nakka and Pascal Fua and Mathieu Salzmann},
	booktitle = {International Conference on Computer Vision},
	year = 2019,
	url = {https://infoscience.epfl.ch/record/269093?ln=en}
}
```

---

## Application RoadAnomaly (Streamlit, français)

Interface grand public pour analyser une image de route et visualiser les zones signalées.

Navigation :

- **Analyser** — upload d’image et prédiction officielle (modèle de référence figé).
- **Jouer** — *Anomaly Hunter* : annotez des obstacles, comparez, contribuez (candidates).
- **Revue** — interface admin pour valider / exporter les contributions.

### Installer et lancer

```powershell
pip install -r requirements-app.txt
# Requis pour Revue (secret ≥16 caractères ; aucun défaut autorisé) :
$env:ROADANOMALY_ADMIN_TOKEN = "<strong-secret>"
streamlit run app.py
```

Guides :

- Utilisateur app : `docs/APP_USER_GUIDE.md`
- Anomaly Hunter : `docs/ANOMALY_HUNTER_GUIDE.md`
- Annotations / anti-fuite : `docs/HUMAN_ANNOTATION_PIPELINE.md`, `docs/DATA_LEAKAGE_PREVENTION.md`
- Sécurité / confidentialité : `docs/SECURITY_AND_PRIVACY_AUDIT.md`
- Audit d’acceptance : `docs/FINAL_ACCEPTANCE_AUDIT.md`
- Roadmap E3 (futur, pas d’entraînement auto) : `docs/FUTURE_E3_ROADMAP.md`

### Fichier modèle (requis, hors Git)

Copier depuis Google Drive vers :

`checkpoints/phase4/E2b_unet_mobilenetv2_finetune_20261108_1721/best.keras`

SHA-256 attendu : `280170dc64347b0b4cc8ce3af9be40dec4ac3733b27d1abeb620cc0c8e88e432`  
(ou variable `ROADANOMALY_E2B_CHECKPOINT`).

Rapport technique : `docs/RAPPORT_TECHNIQUE_CRISP_DM.md` / dossier `deliverable/`.
