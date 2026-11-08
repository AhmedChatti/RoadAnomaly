# RoadAnomaly — initial technical and academic audit

Investigation date: 8 November 2026.  
Scope: inspection only. No files in the project were modified except this report. No models were trained, no packages were installed, and no external data were downloaded.

Evidence labels used throughout:

- **VERIFIED** — observed in files, measured from the images, or read from course slides.
- **INFERRED** — strongly supported, but not written down as a formal statement.
- **PROPOSED** — a recommendation for the next implementation phase.
- **UNKNOWN** — could not be established from the material on disk.

---

## 1. Executive summary

**RoadAnomaly is a 60-image binary semantic-segmentation benchmark, not a pothole classifier.** Each photograph shows an unusual obstacle on or beside a road. A pixel mask marks that obstacle. The official encoding, stated in `readme.md`, is background = 0 and anomaly = 2.

The repository already completes business understanding, data understanding, and a reproducible train/validation/test preparation. It does not train a model, does not evaluate one, and does not contain an application. There is no git history, no checkpoint, and no web UI.

Independent measurement of all 60 frames confirms the existing report:

- Every image is RGB WebP, 1280×720, 8 bits per channel, aspect ratio 16:9.
- Every semantic mask matches that grid and contains only the values 0 and 2, with at least one anomaly pixel.
- Anomaly pixels are 9.85% of the dataset. The median image is 6.62% anomaly. The range is 0.37% (`lost_tires08`, 3,453 pixels) to 42.33% (`animals09_working_together`, 390,100 pixels).
- The prepared split is 42 / 9 / 9, written as 512×512 PNG pairs under `data/processed/`.

The learning problem that the current code prepares is **supervised localisation of anomalous pixels**. The ICCV 2019 paper that released the set used a different protocol: train on ordinary Cityscapes scenes, where these obstacles are absent, and test on all 60 images. Those two protocols must not be mixed when scores are reported.

The Deep Learning course materials that could be read are an introduction by milestones, a long chapter on hyperparameters, regularisation and optimisation, a CNN chapter, a short autoencoder chapter, an image-segmentation deck (U-Net, FCN, Mask R-CNN), an RNN chapter, and a TensorFlow/Keras CIFAR-10 practical. CRISP-DM, classical feature engineering, object-detection training, and segmentation metrics (IoU, Dice, precision, recall) were **not** found in those slides. The assignment still asks for a CRISP-DM notebook, so the methodology comes from the project brief, not from a located lecture.

**PROPOSED direction:** keep the existing preparation, train a small segmentation model with TensorFlow/Keras (the course stack), use a trivial all-background baseline and a from-scratch convolutional segmenter before a transfer-learned U-Net, score with overlap metrics rather than accuracy, and wrap the trained model in a small local application. Do not train Mask R-CNN or an unsupervised autoencoder on these 60 images alone. There is no anomaly-free image in the set, and 42 training images cannot support a detection framework of that size. No NVIDIA GPU was visible on this machine (`nvidia-smi` is not installed); CPU training of a small U-Net on 42 images is feasible.

---

## 2. Existing repository audit

### 2.1 Folder structure

**VERIFIED.** The project root is `C:\Users\man\Downloads\RoadAnomaly`. It is not a git repository (no `.git` directory). A full file walk counted **821 files, 50,769,517 bytes**.

| Path | Role | Files |
| --- | --- | ---: |
| `readme.md` | Official dataset note: purpose, label encoding, citation | 1 |
| `credits.txt` | Image authors, licences, and source URLs. Annotations are CC BY-SA 4.0. The images are not owned by the dataset authors. | 1 |
| `frame_list.json` | Ordered list of the 60 `.webp` filenames | 1 |
| `requirements.txt` | `numpy>=1.26`, `pillow>=10`, `matplotlib>=3.8`, `pandas>=2.1` | 1 |
| `frames/` | Original images and LabelGrab annotation folders | 674 |
| `src/road_anomaly.py` | Exploration, audit, split, letterbox, and figure generation | 1 (+ `src/__pycache__/`) |
| `scripts/run_exploration_and_split.py` | Calls `road_anomaly.run` and rewrites processed data and figures | 1 |
| `scripts/_execute_notebook.py` | Executes the notebook with `nbclient` | 1 |
| `notebooks/01_exploration_and_preprocessing.ipynb` | 14-cell notebook covering business understanding through preparation. It has saved outputs. | 1 |
| `reports/01_business_and_data_understanding.md` | Written report for the same stage | 1 |
| `reports/figures/` | Eleven current figures, plus one extra file not written by the current script | 12 |
| `reports/tables/` | `frame_inventory.csv`, `split_assignment.csv`, `preparation_summary.json` | 3 |
| `data/processed/` | Letterboxed 512×512 images and masks, manifest, summary | 122 |
| `docs/` | Created for this audit only | this file |

There is no `app/`, `models/`, `tests/`, configuration file, experiment log, or checkpoint directory.

Extension counts across the whole tree: `.png` 499, `.npy` 201, `.webp` 60, `.json` 49, `.py` 3, `.csv` 3, `.md` 2, `.txt` 2, `.ipynb` 1, `.pyc` 1.

### 2.2 What the code actually does

**VERIFIED** by reading `src/road_anomaly.py` (887 lines) and `scripts/run_exploration_and_split.py`.

`discover_frames` pairs every name in `frame_list.json` with `frames/<stem>.labels/labels_semantic.png`. It rejects a missing file, a size mismatch, or any mask value other than 0 and 2. Where `labels_instance.png` exists, it checks that anomaly pixels have a non-zero instance id and background pixels have id 0.

`assign_splits` forces three filename groups into training, then stratifies the remaining frames by filename prefix and by anomaly area. The quotas are hardcoded:

- animals: 5 validation, 5 test
- obstacles: 1 and 1
- cones: 2 and 2
- vehicle: 1 and 1
- lost tires: none held out, because all three are in the forced training group

`letterbox` scales the long side to 512 (scale 0.4, content becomes 512×288), pads top and bottom by 112 black pixels, uses bilinear resampling for the photograph and nearest-neighbour for the mask, and remaps label 2 to 255.

`run` deletes `data/processed` with `shutil.rmtree` and rewrites it, the CSV tables, and the figures. Re-running the script is deterministic. It is also destructive of anything later placed inside `data/processed`.

The notebook does not reimplement this. It imports the module, displays the figures, and checks the manifest. Cell 0 states that it is the working version of `reports/01_business_and_data_understanding.md`.

### 2.3 What does not exist

**VERIFIED absent:**

- No trained weights (no `.pt`, `.pth`, `.h5`, `.keras`, `.onnx`, `.ckpt`).
- No training loop, loss function, or metric implementation.
- No application, API, or user interface.
- No tests.
- No `src/__init__.py`. The scripts add `src` to `sys.path` by hand.
- No dependency pin file and no environment lock.
- `requirements.txt` does not list `nbformat` or `nbclient`, which `_execute_notebook.py` imports, or `scipy`, which the figure code imports optionally for connected components.

### 2.4 Technical debt already visible

**VERIFIED**

- `reports/figures/02_image_resolutions.png` is not in the `paths` dictionary written by `save_figures`. It is a leftover. The current figure with that number is `02_resolution_and_obstacle_size.png`.
- `preparation_summary.json` stores absolute Windows paths (`C:\Users\man\Downloads\RoadAnomaly\...`). The manifest CSV does the same for image and mask paths.
- Duplicate-scene membership is a hand-written dictionary, not a measured clustering result. See section 4 for a correction to the lost-tire group.
- A comment at `src/road_anomaly.py` around the showcase frames says the median lost-tire frame “marks a person in the lane”. The three tire frames were inspected. None shows a person as the labelled object. `lost_tires03`’s colour mask marks two tires. The comment does not match the frames that are on disk.
- Filename prefixes are used as stratification keys. They are not model targets. One `vehicle` image also contains cones, and the mask marks both.

**INFERRED**

- The preparation stage is in good shape for a coursework hand-in of “understand and prepare the data”. The modelling stage has not started.
- The report’s narrative is clearer than a typical first notebook. The next notebook should continue from it rather than replace it.

---

## 3. Dataset technical audit

### 3.1 Source and purpose

**VERIFIED** from `readme.md` and `credits.txt`.

The set was released with Lis, Nakka, Fua, and Salzmann, “Detecting the Unexpected via Image Resynthesis”, ICCV 2019. The readme states the purpose in one sentence: images of unusual dangers a vehicle can meet (animals, rocks, traffic cones, and other obstacles), for testing autonomous-driving perception in rare, safety-critical cases.

Images are third-party photographs (Wikimedia Commons, Flickr, news pages, blogs, one YouTube clip, an Etsy listing). The dataset authors do not claim ownership. Annotations are CC BY-SA 4.0. Labelling was done with their LabelGrab tool. The readme says some editor files are missing because labelling was done at another resolution and then rescaled, and points to Krzysztof Lis (EPFL) for the missing instance files.

The paper PDF is not in the workspace. Claims about the paper’s training protocol below are taken from the project’s own report, which cites the paper, and are marked **INFERRED** from that secondary write-up plus the readme’s “testing” wording. They were not re-read from the ICCV PDF.

### 3.2 Inventory

**VERIFIED** by a file walk and by opening every listed image and semantic mask one at a time (no full-dataset array was kept in memory).

| Item | Count |
| --- | ---: |
| Files in the repository | 821 |
| Photographs listed in `frame_list.json` | 60 |
| `.webp` files directly under `frames/` | 60 |
| `*.labels` directories | 60 |
| `labels_semantic.png` | 60 |
| `labels_semantic_color.png` | 60 |
| `labels_instance.png` | 46 |
| `index.json` | 46 |
| `instance_*_gc_mask.png` | 201 |
| `instance_*_gc_state.npy` | 201 |
| Videos | 0 |
| Prepared images / masks | 42 + 9 + 9 each |

Filename categories, derived by the same rule as `category_of` in `src/road_anomaly.py` (prefix before the first underscore, digits stripped, with `lost_tires` kept intact):

| Prefix | Images | Median anomaly area | Mean anomaly area |
| --- | ---: | ---: | ---: |
| animals | 31 | 8.20% | 12.67% |
| obstacles | 15 | 6.31% | 9.50% |
| cones | 8 | 1.72% | 2.88% |
| lost_tires | 3 | 0.70% | 0.69% |
| vehicle | 3 | 9.97% | 10.20% |

These prefixes describe the scene. They are not stored as a class column inside the mask. Every instance record that exists has `"cls": 2` only.

`index.json` lists instances with `id`, `cls`, `crop_rect`, and `roi_rect`. It does not name “cow” or “cone”. The semantic mask is the supervision that exists for every frame.

The `.npy` files are GrabCut editor state, not labels. One file opened at random, `frames/animals01_Guiguinto_railway_station_Calves.labels/instance_001_gc_state.npy`, is `float64` with shape `(2, 65)`. All 201 npy files are 1,168 bytes.

### 3.3 Image characteristics

**VERIFIED** on all 60 WebP files with Pillow 12.

| Property | Result |
| --- | --- |
| Format | WebP, 60 / 60. Not JPEG, PNG, or BMP. |
| Mode | RGB, 60 / 60. No grayscale, no RGBA. |
| Animation | None (`n_frames` is 1). |
| Width | 1280 on every file. Minimum = maximum = mean = median. |
| Height | 720 on every file. |
| Aspect ratio | 1.777… (16:9) on every file. |
| Pixels per image | 921,600. Dataset total: 55,296,000. |
| Colour depth | 8 bits per channel (Pillow mode RGB). |
| Unreadable files | 0 |
| Exact duplicate files (MD5 of the WebP bytes) | 0 groups |
| Size mismatches between image and semantic mask | 0 |

File size of the photographs:

| | Bytes |
| --- | ---: |
| Minimum | 45,708 |
| Median | 196,946 |
| Mean | 199,591 |
| Maximum | 446,946 |
| Sum | 11,975,468 |

Semantic masks are single-channel PNG (`L`), 1280×720. Across all 60 masks the only values present are 0 and 2, and both values appear in every mask. Minimum anomaly count is 3,453 pixels. No empty mask.

Anomaly area, as a fraction of each 1280×720 frame:

| | Fraction | Percent |
| --- | ---: | ---: |
| Minimum | 0.003747 | 0.37% (`lost_tires08`) |
| Median | 0.066229 | 6.62% |
| Mean, also the share of all pixels | 0.098496 | 9.85% |
| Maximum | 0.423286 | 42.33% (`animals09_working_together`) |

Because every frame has the same number of pixels, the mean image and the global pixel share are the same number. Background is 90.15% of pixels.

A resolution histogram is unnecessary: the distribution is a single spike at 1280×720. **Resizing is not required to repair inconsistent dimensions.** It is required only if a chosen network wants another spatial size. The existing 512×512 letterbox is one choice, not a data repair. Section 10 argues that 512×288, which is already divisible by 32, avoids the black bars.

Near-duplicates were screened with an 8×8 average hash, Hamming distance ≤ 6 (out of 64). Six pairs crossed that loose threshold. The only pair that visual inspection confirmed as the same event a moment apart is `lost_tires08` and `lost_tires09`. The other pairs are different scenes that share a horizon or a road (for example a pheasant frame with the reindeer frame). **No hidden exact copies were found.**

### 3.4 Labels and the learning problem

**VERIFIED**

The task supported by the files is **binary semantic segmentation**:

| Mask value | Meaning |
| ---: | --- |
| 0 | Not the unexpected obstacle (road, sky, ordinary vehicles, people, buildings, vegetation) |
| 2 | Unexpected obstacle, one class, regardless of whether it is an animal, a rock, a cone, a tire, or an unusual vehicle |

There is no class 1. There are no bounding boxes, no polygons, no captions, and no anomaly-free images. Instance ids exist for 46 frames and agree with the semantic mask on all 46 (zero disagreements in the audit already stored in `preparation_summary.json`, and the code path that computes that check was read). Fourteen frames have a semantic mask and no instance mask. They are listed in `data/processed/preparation_summary.json` under `frames_without_instance_mask`.

Instance counts on the 46 indexed frames: 18 frames have 1 instance, then 2, 3, 4, … up to 31 (`cones01_Beeston_MMB_A6_Middle_Street`). The category is unbalanced at image level (animals are 31 of 60) and severely unbalanced at pixel level.

This is not:

- image classification of “pothole vs clean road” (no such labels, no clean roads);
- multi-class classification of animal vs cone vs rock (the mask does not encode the prefix);
- object detection with boxes (no boxes are stored; rectangles in `index.json` are editor crops, not a detection dataset);
- a video task (three frames were cut from a YouTube clip; the video is not in the repository).

Instance segmentation is possible only as a secondary experiment on 46 frames, and it would still be one class.

### 3.5 Partitions

**VERIFIED** by counting files and reading `split_manifest.csv`.

| Split | Images | Masks | Canvas | Image mode | Mask values |
| --- | ---: | ---: | --- | --- | --- |
| train | 42 | 42 | 512×512 | RGB | 0 and 255 only |
| val | 9 | 9 | 512×512 | RGB | 0 and 255 only |
| test | 9 | 9 | 512×512 | RGB | 0 and 255 only |

Every processed image has a mask of the same name. Scale is 0.4, content size 512×288, `pad_left` 0, `pad_top` 112, on every row. Smallest processed anomaly: 552 pixels (`lost_tires08`). No annotation was erased.

Original anomaly area by split (from the manifest, which copies the pre-resize ratio):

| Split | Min | Median | Max |
| --- | ---: | ---: | ---: |
| train | 0.37% | 6.48% | 36.85% |
| val | 1.37% | 9.96% | 42.33% |
| test | 0.80% | 4.32% | 31.18% |

After letterboxing, anomaly ratios fall because 43.75% of the canvas is black padding (224 / 512). Processed anomaly-area mean is 5.54%, median 3.72%, maximum 23.81%.

Category counts in the split:

| | animals | obstacles | cones | lost_tires | vehicle | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 21 | 13 | 4 | 3 | 1 | 42 |
| val | 5 | 1 | 2 | 0 | 1 | 9 |
| test | 5 | 1 | 2 | 0 | 1 | 9 |

The 11 frames forced into training are:

- `lost_tires03`, `lost_tires08`, `lost_tires09`
- `obstacles03`–`obstacles06` (cardboard Czech hedgehog)
- `obstacles12_rocks2`, `rocks4`, `rocks5`, `rocks6`

**INFERRED** from credits and from looking at the pictures: the cardboard group and the rock sequence are repeated views and should stay together. `lost_tires08` and `lost_tires09` are the same summer highway a moment apart (same watermark, same guardrail, same tire). `lost_tires03` is a different, snowy road with a different truck, credited to the same YouTube URL. Putting it in training avoids leakage from that source, but it is not a near-copy of frames 08 and 09. The test set therefore contains no loose tire and no snowy dashcam frame. That limitation is already stated in the existing report and should stay in any results table.

Validation is easier than test on median obstacle size (about 10% versus 4.3%). Both still span small and large objects. With 9 test images, a mean IoU will have a very wide uncertainty. Per-image scores are mandatory.

### 3.6 Annotation quality from a visual sample

Eleven photographs and six masks or report figures were opened. This is a sample, not a relabel of all 60.

| Frame | What is in the picture | What the mask does |
| --- | --- | --- |
| `lost_tires08`, `lost_tires09` | Dashcam, summer highway, one loose tire. Watermarks (“MIR”, frame numbers). | The report figure shows the tire outlined and the cars left as background. |
| `lost_tires03` | Different dashcam clip: snow, a dump truck, a rolling tire, and a tire against a car. | Colour mask marks both tires and not the truck or the cars. |
| `animals17_E6_near_Kvaenangen` | Forward view, overcast, one reindeer, motorcycles in the distance. | Figure 11 shows the reindeer outlined, including legs, and not the motorcycles. |
| `animals09_working_together` | Side view of oxen, a loaded cart, and a driver. Little road is visible. Largest mask (42%). | One region covers the animals and the load. Legs are thinner than the bodies. |
| `cones01_...` | Street works, many cones, workers, fencing. Test set. 31 instances. | Cones are marked. Workers and the site fence are not. A small stand among the cones is included. |
| `vehicle02_...` | Skid-steer loader on a street, three cones in front of it. | Loader and the three cones are marked. The building is not. |
| `obstacles02_rock` | Stylised stock photograph: a boulder on a double yellow line, heavy vignette. Darkest frame in the set (mean RGB about 58). | The boulder is cleanly filled. |
| `obstacles07_czech_hedgehog` | Close-up of a small metal obstacle on concrete. Credits point to an Etsy miniature. Validation set, 24.7% anomaly. | Not re-opened as a colour mask in this pass. The photograph itself is not a driving view. |
| `animals29_...` | Zebras crossing a pale desert road, side view, dark sky. Test set. | Not opened as a mask in this pass. The scene is a real road, but the camera is not a vehicle windshield. |

**INFERRED** annotation policy, consistent with the samples: mark the unexpected object, including several objects in one frame, and leave ordinary road users (cars, motorcycles, workers) as background. Boundaries on thin parts (legs, cone tips) are usable and slightly soft. The policy is coherent. The photographs are not.

### 3.7 Limitations and what they do to training

| Issue | Evidence | Consequence |
| --- | --- | --- |
| Pixel imbalance | 9.85% anomaly | Accuracy near 90% is achievable by predicting background. Use precision, recall, F1, IoU, Dice. Use a loss that can see the rare class. |
| Category imbalance | 31 animals, 3 tires, 3 vehicles | A single test IoU will be dominated by whichever animals landed in the 9-image test set. Report per-image and per-prefix scores. |
| No negative images | Every mask contains label 2 | False-alarm rate on a clean road cannot be measured. An autoencoder cannot learn “normal road” from this folder. |
| n = 60, test n = 9 | Counted | A deep network trained from scratch will memorise. Transfer learning and augmentation are the realistic path. Quote uncertainty, not a single heroic score. |
| Repeated events | 11 frames | Already kept in training. Do not reshuffle at random. |
| Domain mixture | Credits plus the visual sample | Wikimedia stills, stock art, a product photo, and three dashcam frames. A model can learn “web photograph of an animal” rather than “obstacle in the ego lane”. Say so in the limitations. |
| Watermarks | Tire frames | A dashcam demo can cheat by detecting the “MIR” logo if those frames were in test. They are in train, which is safer, but augmentation should not teach the logo as an obstacle. |
| Letterbox bars | 112 px top and bottom, label 0 | 43.75% of each training target is artificial black background. The model can learn that black means “safe”. |
| Val easier than test | Medians 10.0% vs 4.3% | Early stopping on validation can look better than the test table. Look at per-image errors. |
| Missing instance files | 14 frames | Do not drop them. Semantic masks are complete. |
| Licence | `credits.txt` | The app can be a local academic demo. Publishing the images as if the project owned them would contradict the readme. |

Data leakage of the random-split kind has been deliberately reduced. Leakage that remains is domain leakage: all images come from the same public-web collection, and the test images look like the training images. There is no external road-camera test set in the repository.

---

## 4. Visual data understanding

**VERIFIED** from the sample in section 3.6 and from `reports/figures/11_mask_correspondence_zoom.png`.

Typical case: a daylight road photograph, one or a few unexpected objects, mask covering those objects and not the rest of the scene. The reindeer frame and the cone street are good teaching examples. Obstacle area is usually under 10%.

Difficult cases:

- `lost_tires08`: 0.37% of the frame. After letterbox, 552 pixels. Easy to miss, and easy for a Dice score to collapse.
- `animals09`: the obstacle is most of the picture, the viewpoint is a close side view, and the “road scene” prior is weak.
- `cones01`: many small instances. A model can hit the large foreground cones and miss the distant ones, and still look plausible.
- `obstacles02_rock`: synthetic-looking lighting. A network can treat style as a feature.
- `obstacles07`: close-up, not a road corridor.
- Dashcam tires: motion blur, compression, on-screen graphics.

Camera perspectives present in the sample: forward dashcam, forward roadside, side-on wildlife, close-up object, and stylised composite. Weather present in the sample: snow, bright sun, overcast, and a heavily graded “dusk” stock look. There is no rain or night-driving series that was identified.

Visual biases: animals and livestock dominate; geography follows whatever Wikimedia happened to host; many frames are not from a vehicle; a few frames are illustrations. A smart-city or road-maintenance story is a poor fit. The credible story is a driver-assistance perception demo on unusual obstacles, with a clear statement that the photos are not a fleet dataset.

Existing figures worth keeping in the final notebook, all under `reports/figures/`:

1. `01_category_counts.png`
2. `02_resolution_and_obstacle_size.png`
3. `03_anomaly_area_by_category.png`
4. `04_pixel_class_balance.png`
5. `05_image_mask_overlay_examples.png`
6. `06_small_and_large_anomalies.png`
7. `07_split_composition.png`
8. `08_preprocessing_letterbox.png`
9. `09_all_scenes_by_split.png`
10. `10_repeated_scenes.png`
11. `11_mask_correspondence_zoom.png`

---

## 5. Deep Learning course synthesis

### 5.1 What was read, and what was not

**VERIFIED** files, all outside the repository, under `C:\Users\man\Downloads\`:

| File | What it is | How it was read |
| --- | --- | --- |
| `Ch1_Introduction.pdf` | 19-page PDF, created 2026-01-23. Slide bodies are raster images, so body text could not be extracted. The outline titles were extracted. | Outline only. Bodies not readable without a PDF renderer. |
| `V2 Chapter 3_HyperparametersTuning_Regularization_Optimization.pptx` | 67 slides. Core metadata: Ines Slimen / SMSB, 2024-07-05. | Slide text and speaker notes extracted from the PPTX XML. Equations drawn as pictures were not OCR’d. |
| `CNN V2.pptx` and `CNN V2 (1).pptx` | Identical files (same MD5). 52 slides. Title inside the deck: Chapter 4, CNNs. | Slide text and notes. Many architecture diagrams are images. |
| `Ch5_autoencoders.pptx` | 12 slides. Ines Slimen / SMSB, 2026-02-22. | Slide text. Short, and several slides are mostly diagrams. |
| `Segmentation.pptx` | 18 slides. Same author metadata, 2026-02-22. Unit: Deep Learning. No chapter number on the title slide. | Slide text. U-Net and Mask R-CNN diagrams are images. |
| `Ch10_Sequence Models.pptx` | 26 slides. Same author, 2026-04-07. Title: Chapter 10, RNN. | Slide text. BPTT, LSTM, and GRU formulas are images; the surrounding words were extracted. |
| `TP5-CNN.pdf` | French practical: CNN classification of CIFAR-10 with TensorFlow and Keras. | Text extracted. |

**Searched and excluded.** Desktop, Documents, Downloads, and OneDrive filenames matching chapter, CNN, segmentation, autoencoder, and CRISP patterns did not reveal chapters 2, 6, 7, 8, or 9. `C:\Users\man\Downloads\cours\` is an electronics course (diodes, transistors, Boolean algebra), not this unit. `TD2_segmentation_correction.docx` is IP subnetting (“masque” means subnet mask). `Chapitre 2 Classification-Naive bayes.pdf` and the `AN_CH2` / `Correction TD_AN` files were not part of the Deep Learning slide series and were not treated as course chapters. `Giovannetti2021_...Deep-MEG....pdf` is a research article, not a lecture.

**UNKNOWN:** the wording on the Ch1 slide bodies, and any chapter that lives only inside a platform the search did not index (a Moodle course, email, or USB stick).

### 5.2 Chapter-by-chapter synthesis

Only topics present in the extracted material are listed. If a topic the brief asked about is missing, it is marked as not found.

#### Chapter 1 — Introduction

1. Title: Introduction. Outline entries, not a printed chapter title on an extracted text slide.
2. Objectives: a historical map of AI, deep learning, generative models, and agents. No methodological objective was readable.
3. Concepts present as outline titles: AI evolution; ANN (1940s–1980s); CNN (1980); RNN (1986); LSTM (1997); bidirectional RNN (1997) and bidirectional LSTM (2000); VAE (2013); GANs (2014); Transformers (2017); diffusion models (2020); LLMs (since 2020); multimodal models (since 2021); AI agents (since 2018).
4. Mathematics: **not extractable**. Slide bodies are images.
5. Architectures: named in the timeline only.
6. Libraries: none in the outline.
7. Metrics: none in the outline.
8. Exercises: none found.
9. Applicable to RoadAnomaly: almost nothing operational. The timeline explains why a CNN and a U-Net exist. It does not say how to train one.
10. Enrichment: generative models are named. They are not taught as a method here.

**Not found in this file’s outline:** CRISP-DM, EDA, preprocessing recipes, loss functions, evaluation metrics.

#### Chapter 3 — Hyperparameters, regularisation, and optimisation

1. Title on slide 1: “Chapter 3: Hyperparameters Tuning, Regularization and Optimization”.
2. Outcomes on slide 2: initialisation, L2, dropout, batch normalisation, gradient checking; mini-batch gradient descent, momentum, RMSprop, Adam, and convergence checks.
3. Concepts actually developed in slide text: parameters `(w, b)` versus hyperparameters (learning rate, batch size, epochs, iterations, activation, depth, width); linear versus sigmoid, tanh, ReLU, Leaky ReLU; sigmoid and softmax at the output; overfitting and the bias–variance tradeoff; more data, architecture changes, cross-validation, early stopping, dropout, L1/L2; batch, stochastic, and mini-batch gradient descent; momentum; RMSprop; Adam; grid search, random search, coarse-to-fine, Bayesian search, genetic algorithms, manual search.
4. Mathematics named in words, with the algebra drawn on the slides and therefore not copied here: sigmoid range (0, 1); tanh range [−1, 1]; ReLU; Leaky ReLU small negative slope; L2 penalty on the weight norm added to the cost, called weight decay; L1 penalty on the absolute weights, called Lasso, encouraging exact zeros; cross-entropy used as the example cost; momentum update with a momentum coefficient; RMSprop as an exponential moving average of squared gradients; Adam with first and second moments, bias correction, and ε. The slides say λ = 0 means no regularisation and λ = 1 means “maximum” regularisation. That second claim is a simplification; λ is not bounded by 1 in the usual formula. Do not copy it into the project as a constraint.
5. Algorithms: the optimisers and search methods above.
6. Libraries: none. This chapter is theoretical.
7. Metrics: train and validation cost curves; bias and variance. No precision, recall, F1, IoU, or Dice.
8. Exercises: worked numerical stories (learning rates 0.8, 0.05, 0.2; a 5-observation SGD example; batch size 2). No code.
9. Applicable: Adam, early stopping on a validation loss, dropout, a small L2 penalty, learning-rate choice, and the train-versus-validation curve. With 42 images, random search over a short grid beats a genetic algorithm.
10. Enrichment: Bayesian optimisation and genetic search. They are not worth the compute here.
11. Named in the learning outcome and **not developed in the extracted slide text:** batch normalisation and gradient checking. Weight-initialisation schemes such as Xavier or He were not named.

#### Chapter 4 — Convolutional neural networks

1. Title: “Chapter 4: Convolutional Neural Networks (CNNs)”.
2. Outcomes: what a CNN is; 1D, 2D, and 3D convolution; max, min, and average pooling; LeNet-5, AlexNet, ZFNet, GoogLeNet, VGG, ResNet.
3. Concepts: RGB shape `(n, m, 3)`, grayscale `(n, m)`, binary pixels; hand-engineered features described as brittle; LeNet (Yann LeCun, the slide says 1990); convolution, ReLU after convolution, padding to preserve spatial size, stride, pooling, flatten, then dense classification. Slide 23 says typical CNN weights are randomly initialised and calls explainability an open research topic. It does not teach a method.
4. Mathematics: the convolution arithmetic is in drawings. Extracted text defines padding, stride, ReLU as “negative values become zero”, and pooling as max, min, or average. Output size is described qualitatively (larger stride, smaller map).
5. Architectures, with the numbers the slides state: LeNet-5, tanh then softmax, 0.8% test error on MNIST; AlexNet, ImageNet, 15.3% error, trained on GPUs; ZFNet; VGG (2014); GoogLeNet / Inception; ResNet (2015). A comparison slide exists as a figure.
6. Libraries: a Colab link is given for a 1D convolution regression. The practical that matches this chapter is `TP5-CNN.pdf`: `tensorflow`, `tensorflow.keras`, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, Adam, `sparse_categorical_crossentropy`, accuracy.
7. Metrics: accuracy on CIFAR-10. The practical text says a one-epoch run printed test accuracy about 0.707. That number is the handout’s illustration, not a result of this project.
8. Exercise: build the CIFAR-10 CNN (32 and 64 filters, 3×3 kernels, stride 1, 32×32×3 input, two dense layers, 10-way output), `model.summary()`, fit, plot accuracy.
9. Applicable: the layer vocabulary for a segmentation encoder. RGB in, ReLU, pooling, padding. Not the CIFAR classifier head.
10. Enrichment: reciting AlexNet’s ImageNet error. It does not move this project.

**Not found:** transfer learning as a procedure (that appears in the segmentation deck), detection heads, segmentation losses, data augmentation code.

#### Chapter 5 — Autoencoders

1. Title: “Chapter 5: Autoencoders”.
2. Outcomes are not listed as a formal box. The deck defines the model and its uses.
3. Concepts: unsupervised; lossy reconstruction; bottleneck code; encoder, code, decoder; code size, depth, and width as hyperparameters; loss named as MSE or binary cross-entropy; stacked, sparse, denoising, convolutional, and variational autoencoders; L1, L2, dropout; sparsity via an L1 term or KL divergence; denoising by corrupting the input and reconstructing the clean input; undercomplete bottleneck to avoid copying the input.
4. Mathematics: reconstruction is described as a non-linearity (sigmoid is the example) applied to a code. KL is named, not written out in extractable text.
5. Algorithms: the types above. VAE is a name on the types slide, not a derivation.
6. Libraries: none.
7. Metrics: the reconstruction loss. No anomaly threshold, no precision at the pixel level.
8. Exercises: none in the text.
9. Applicable only with a warning. The deck’s anomaly example is “points that do not resemble the rest of the data”. Every RoadAnomaly image contains an obstacle, so the set has no “normal” class to reconstruct. Training an autoencoder on these 60 images does not reproduce the ICCV protocol, which needs ordinary scenes. See section 10.
10. Enrichment: a convolutional denoising autoencoder on a separate normal-road corpus, if the student is allowed to add one. That corpus is not in the workspace.

#### Image segmentation (unnumbered deck)

1. Title: “Image Segmentation”. Unit: Deep Learning.
2. Outcomes: understand segmentation and its types; know U-Net, fully convolutional networks, and Mask R-CNN.
3. Concepts: partition an image by colour, texture, or edges; uses named include autonomous vehicles (lanes and pedestrians), medical imaging, augmented reality. Challenges: scarce labels, expressive appearance, synthetic ground truth. Remedies: augmentation (rotations, flips, scaling), transfer learning from classification, semi-supervised learning, attention, ensembles. U-Net: contracting encoder, expanding decoder, skip connections, symmetric, fully convolutional, variable input size. Source cited: Ronneberger et al., arXiv:1505.04597. FCN: arbitrary input size, transposed convolution, skip connections, dilated convolution mentioned as multi-scale context. Mask R-CNN: Faster R-CNN plus a mask branch, class and box and mask together; the slide says about 5 fps and “simple to train”.
4. Mathematics: none extracted. Skip connections and transposed convolution are described in words.
5. Architectures: U-Net, FCN, Mask R-CNN.
6. Libraries: none.
7. Metrics: **none**. IoU and Dice are not in the extracted text.
8. Exercises: none.
9. Applicable: this is the chapter the project should demonstrate. U-Net with an encoder initialised from classification is the method the slides recommend when labels are scarce. That description matches 60 masks.
10. Enrichment: Mask R-CNN and ensembles. Mask R-CNN needs instance masks (missing on 14 frames) and far more than 42 images. Attention and ensembles are optional appendix material.

#### Chapter 10 — Recurrent networks

1. Title: “Chapter 10: RNN”.
2. Outcomes: build RNNs, GRUs, and LSTMs; apply them to language and to audio, including speech recognition.
3. Concepts: sequences (prices, sensors, text, sound, video); many-to-many and many-to-one tasks drawn as a taxonomy that includes image captioning and video activity. CNN limitations stated for sequences: fixed length, no memory, order invariance. BPTT; vanishing and exploding gradients; gradient clipping; ReLU, initialisation, truncated BPTT, and gated cells as remedies. LSTM input, forget, and output gates with sigmoid and tanh. GRU (Cho et al., 2014) reset and update gates.
4. Mathematics: the recurrence and the gate equations are images. Extracted constraints: sigmoid outputs in (0, 1); tanh used to bound candidate memory; shared weights `W`, `V`, `U` across time; exploding-gradient rule described as clipping when the gradient exceeds a threshold.
5. Algorithms: vanilla RNN, LSTM, GRU, truncated BPTT.
6. Libraries: none.
7. Metrics: MSE is assumed in the BPTT derivation slide. No sequence-task metric is named.
8. Exercises: none extracted.
9. Applicable: **not essential**. RoadAnomaly is not a sequence. Three tire frames are not a sequence dataset.
10. Enrichment: a video model would require the source footage, which is not in the repository. Do not add an RNN to the grade’s critical path.

#### Practical TP5 — CIFAR-10 CNN

French handout. Stack: TensorFlow and Keras. Dataset: 60,000 colour images, 10 classes, 50,000 / 10,000 split, 32×32×3. The network is the small stack described under Chapter 4. Compile with Adam and sparse categorical cross-entropy. Metric: accuracy. The handout’s printed test accuracy is about 70% after the illustrated run.

This practical teaches classification, `model.summary()`, and a learning curve. It does not teach segmentation. It does fix the implementation stack: **TensorFlow / Keras, not PyTorch.** PyTorch is not mentioned in any extracted deck.

### 5.3 Topics the brief asked about that are absent

| Topic | Status in the inspected course files |
| --- | --- |
| CRISP-DM | Not found. |
| A data-science process chapter | Not found. |
| Exploratory data analysis as a method | Only the CIFAR practical’s “plot 25 images”. |
| Classical feature engineering | The CNN deck argues against hand-made features. No HOG, SIFT, or similar. |
| Traditional machine learning (trees, SVM, Naive Bayes) | Not in this unit. A separate Naive Bayes PDF exists and was not part of these decks. |
| Evaluation beyond accuracy and loss | Not found. IoU, Dice, precision, and recall are project needs, not lecture content. |
| Explainable AI methods | One sentence: CNNs are black boxes and interpretability is research. No Grad-CAM, no SHAP, no LIME. |
| Object-detection training (YOLO, SSD) | Not found. Mask R-CNN is named, not implemented. |
| Generative AI as a method | Timeline names only, plus VAE as an autoencoder type without a derivation. |

---

## 6. Course-to-project mapping

Priority is what a jury can see in the notebook and the application. Techniques that the course names but that do not fit 60 images are left out on purpose.

| Course concept | RoadAnomaly application | Proposed implementation | Priority |
| --- | --- | --- | --- |
| Segmentation, U-Net, skips, fully convolutional output | The mask is the target | U-Net, binary output, same spatial size as the input | Essential |
| Transfer learning when labels are scarce | 42 training images | Encoder weights from an image-classification network, decoder trained, encoder unfrozen only at a small learning rate | Essential |
| Data augmentation (flips, mild scale, colour) | Reduce memorisation | Horizontal flip, small zoom, brightness and contrast. Keep the mask in lockstep. Be careful with vertical flip: it is in the slides and is harmless for objects, but it invents an upside-down road. | Essential |
| CNN blocks: Conv, ReLU, pooling, padding | The from-scratch baseline | A small FCN, a few conv–pool stages and a 1×1 output, trained on this set | Essential |
| Train versus validation loss, early stopping | Chapter 3 | Plot both losses. Stop on validation Dice or IoU, not on accuracy. | Essential |
| Adam, learning rate, mini-batch | Chapter 3 and TP5 | Adam, batch size 2 or 4 on CPU, a short learning-rate comparison (for example 1e-3 and 1e-4) | Essential |
| Dropout and a light L2 penalty | Chapter 3 | Dropout in the decoder. L2 small enough that it does not erase the rare class. | High |
| Bias–variance, overfitting | Chapter 3 | With n = 42 this will happen. Show a model that fits training and fails on a small object in test. | Essential |
| Why accuracy is the CIFAR metric and the wrong metric here | TP5 versus the pixel counts | Report accuracy once, next to the all-background baseline, then stop using it as the headline | Essential |
| Loss: binary cross-entropy (autoencoder deck) and the idea of a cost | The pixel classifier | Weighted binary cross-entropy, or BCE combined with a soft Dice term. Dice itself is not in the slides; introduce it from the overlap definition and say so. | Essential |
| Softmax and multi-class cross-entropy | CIFAR head | Do not use a 10-way head. A two-logit softmax is optional and equivalent to a sigmoid if the implementation is careful. One sigmoid is enough. | Avoid misuse |
| Mask R-CNN | Named in the segmentation deck | Discuss in one paragraph why it is a poor fit: missing instance masks, 42 images, CPU. Do not train it. | Discussion only |
| Autoencoder anomaly detection | Chapter 5 and the ICCV paper | Do not train it on these 60 images. Explain the missing normal class. Optional only if a normal-road corpus is explicitly allowed. | Conditional |
| VAE, GAN, diffusion, LLM | Chapter 1 timeline | One limitations sentence: synthetic data could enlarge the set later. Do not build a generator for the grade. | Not essential |
| RNN, LSTM, GRU | Chapter 10 | Not used. The input is a still image. | Not applicable |
| Random / grid search over many hyperparameters | Chapter 3 | A manual or tiny grid: learning rate, freeze versus unfreeze, loss weight. Not a genetic algorithm. | High, kept small |
| Explainability as an open problem | CNN slide 23 | Show the predicted mask on the photo, plus the worst and best test images. That is the explanation a non-specialist can check. Grad-CAM on a segmentation output is optional and easy to misuse. | High as qualitative error analysis |
| CIFAR practical habits: `summary`, curves, held-out evaluation | Reproducible notebook | Same habits, different head and different metric | Essential |

---

## 7. Preliminary business understanding

### Documented facts

- **VERIFIED:** the dataset exists to test perception when a vehicle meets something outside the usual detector vocabulary.
- **VERIFIED:** the useful output in this coursework is a region, because the project’s own report and the masks agree that location matters. A single “anomaly / no anomaly” flag cannot tell a planner whether the object is in the lane.
- **VERIFIED:** a miss is a safety failure in the story the readme tells. The images themselves are public photographs, not an instrumented fleet.

### Hypotheses that need the student’s or the professor’s confirmation

- Stakeholders for a real system would be a driver-assistance team, a vehicle manufacturer, or a road operator. None of these is named in the repository.
- Financial consequences (braking events, insurance, road-repair dispatch) are not measured here and should not be invented as project results.
- Road-maintenance and smart-city inspection are weak fits. The masks mark unexpected objects, not pavement defects. There are no pothole labels.
- Business value of the student project is a demonstrator: upload a road photo, see the suspected obstacle, read a confidence cue, and compare it with the annotation when one exists. That is an academic proof, not an operational product.

### Proposed business objectives

1. Locate unexpected obstacles in a road photograph.
2. Prefer finding the obstacle over painting the whole road as obstacle (recall and precision both reported).
3. Remain usable from a small object (a tire) to a large one (a herd or a cart).
4. Show the region so a person can accept or reject it.
5. State, in the interface, that the tool was trained on 42 public photographs and is not a driving system.

### Proposed success criteria

These are **PROPOSED**, not given by a rubric in the repository.

| Kind | Criterion |
| --- | --- |
| Safety-shaped | Mean test recall and mean test IoU reported with the list of 9 images, not as a single unqualified percentage. |
| Usefulness | Precision reported beside recall. A model that marks the entire image is a failure even if recall is 1. |
| Honesty | The all-background baseline is in the same table. Accuracy is shown only to explain why it misleads. |
| Human check | The application draws the predicted region on the photo. |
| Scope | The interface does not claim autonomy, braking, or a municipal inventory. |

### Constraints

- 60 images, 42 for training, CPU unless the student has another machine. **VERIFIED** on this PC: `nvidia-smi` is not available. Whether a GPU exists and is simply missing its tools is **UNKNOWN**.
- TensorFlow 2.21 and Keras 3.15 are installed. PyTorch and OpenCV are not.
- Image licences restrict redistribution.
- The test protocol of the coursework (train on a split of RoadAnomaly) is not the protocol of the ICCV paper (train on Cityscapes, test on all 60). Scores must say which protocol was used.

---

## 8. Proposed data-science objectives

**PROPOSED**, aligned with the files.

| Item | Choice |
| --- | --- |
| Task | Binary semantic segmentation |
| Input | One RGB road photograph |
| Output | One probability per pixel, thresholded to a mask. Positive class = unexpected obstacle. |
| Granularity | Pixel. The filename prefix is a slice for error analysis, not a predicted class. |
| Formulation | Supervised, one sigmoid output, trained on the prepared masks. |
| Not the formulation | Multi-class classification; box detection; unsupervised anomaly detection on this folder alone. |

Candidate models, in the order they should appear:

1. **Baseline.** Predict background everywhere. Expected obstacle IoU = 0. Pixel accuracy ≈ 90%. This is the number that stops a bad metric choice.
2. **From-scratch FCN.** A few convolutional blocks, no pretrained weights. Expected to under-segment small objects. It exists so the jury can see what the CNN chapter does without transfer.
3. **Transfer U-Net.** Pretrained encoder, decoder with skips, weighted BCE and/or soft Dice, light augmentation, early stopping. This is the model the application should load.
4. **Optional ablation, only if time remains.** Freeze the encoder versus fine-tune it. Or 512×288 without black bars versus the current 512×512 letterbox. One ablation, not a leaderboard.
5. **Not recommended as a trained system.** Mask R-CNN. A category classifier. An RNN. A GAN. An autoencoder trained on these 60 images.

Technical success, **PROPOSED**: the transfer model beats the from-scratch model and the all-background baseline on test IoU and test Dice; the notebook shows at least one small-object failure in words and pictures; training is rerunnable from a documented command.

---

## 9. CRISP-DM implementation strategy

CRISP-DM was not found in the course. The brief requires it, and the existing report already covers the first three phases in substance. The notebook should name the phases explicitly so the marker can see them.

### A. Business understanding

Use section 7. Separate the documented perception problem from the hypothesised stakeholders. Write the success table before any model. State the protocol difference with Lis et al. (2019).

### B. Data understanding

Most of this is done. The final notebook should recompute, not paste, the counts: 60 pairs, 1280×720, labels {0, 2}, category counts, anomaly-area distribution, pixel balance, instance coverage 46/60, the three repeated groups, and the visual caveats (dashcam versus stock versus close-up). Keep figures 1–6 and 10–11. Add one short note that `lost_tires03` is not a near-duplicate of frames 08 and 09 even though the credits share a URL.

### C. Data preparation

Already implemented in `src/road_anomaly.py`. The notebook should explain:

- why label 2 becomes 255, and why training must treat any positive value as the obstacle;
- why nearest-neighbour is used on the mask and bilinear on the image;
- why the split is not a random row split;
- the black-bar side effect.

**PROPOSED change before training, not during this audit:** emit a 512×288 pair with no padding, or keep 512×512 and ignore the padded rows in the loss. 288 and 512 are both divisible by 32, so a five-level U-Net does not need the square canvas. The existing square files can stay as the audited artefact; the training loader can crop the central 288 rows.

Augmentation belongs here, applied on the fly to training only: horizontal flip, mild scale, colour jitter. No augmentation on test.

Normalisation: ImageNet mean and standard deviation if the encoder was trained that way; otherwise scale to [0, 1] and say which. Do not mix them.

### D. Modelling

Section 8 and section 10. One notebook section per model, each opened by a markdown cell that says why that model is next. Fixed seed. `model.summary()` as in TP5. Saved weights with the validation metric in the filename.

### E. Evaluation

Technical: per-image and mean precision, recall, F1, IoU, Dice on the 9 test images; the same numbers on validation for model selection; training curves; a panel of predictions. Threshold 0.5 is the default, with one sentence on how a lower threshold trades precision for recall. Do not tune the threshold on the test set.

Business: the same panel read as “would a person see the obstacle in time?”, plus an explicit statement that clean-road false alarms were not measured.

### F. Deployment

A local application, section 13. Not a cloud service. The model file is the one written by training, loaded at startup, with a clear error if the file is missing.

---

## 10. Modelling strategy and architecture recommendations

### Why this shape

The supervision is a dense binary mask. A classifier throws away the location the business objective asks for. A detector needs boxes the dataset does not provide. An autoencoder needs normal images the dataset does not contain. A U-Net is the architecture the segmentation lecture recommends for scarce pixel labels, and the CNN lecture supplies the encoder vocabulary.

### Progressive plan

**Model 0 — constant background.** No training. Establishes the metric trap.

**Model 1 — small FCN from scratch.** Four encoder stages are enough (for example 16, 32, 64, 128 filters, 3×3, ReLU, 2×2 max-pool), a 1×1 sigmoid at the lowest resolution or a short decoder without pretrained weights. Input 512×288. Loss: binary cross-entropy with a positive weight near the inverse of the class frequency (about 0.9 / 0.1, then adjusted if the weighted loss becomes unstable). Optimiser: Adam. Early stopping on validation IoU. This model will likely miss tires and small cones. That result is pedagogically useful.

**Model 2 — U-Net with a pretrained encoder.** Prefer an encoder that Keras can download once (for example a small ResNet or MobileNetV2 backbone exposed by a maintained segmentation package, or a short U-Net whose encoder matches a published Keras application). The exact backbone is a student decision constrained by CPU time and by whether a download of ImageNet weights is allowed on the lab network. Train the decoder first. Fine-tune the encoder at a lower learning rate only if validation IoU still improves. Augmentation as in section 9.C. Dropout in the decoder. Save the best validation checkpoint.

**Ablation, optional.** Crop away the 112-pixel bars versus training on the letterbox. Hypothesis: the bars waste capacity and create a black-background shortcut. One comparison table, not a third architecture family.

### What not to train

- **Mask R-CNN.** Course-aligned in name, misaligned with 14 missing instance masks and with CPU time. Discuss it.
- **Autoencoder on RoadAnomaly.** Chapter 5’s anomaly story requires a normal class. Here every image is positive. Reconstruction error would light up texture, not the obstacle. The ICCV method also resynthesises using a generator trained on ordinary street scenes (Cityscapes is named in the project report). Those images are not in this workspace. Adding them is a scope decision, a download, and a different scientific claim.
- **Category classifier** on the five prefixes. It answers the wrong question and leaks the fact that the prefix is in the filename.
- **RNN / sequence model.** No sequence.
- **Generative augmentation.** The course does not teach a usable GAN or diffusion recipe, and bad synthetic masks would dominate 42 real images.

### Loss and threshold

Sigmoid plus weighted binary cross-entropy matches the course losses (binary cross-entropy is named; class weights are the honest adaptation to a 9.85% positive class). A soft Dice term matches the metric the project must report. If both are used, say that Dice is introduced from the overlap definition because the lectures stop at accuracy and cross-entropy.

Threshold 0.5 for the reported table. A precision–recall curve over thresholds, computed on validation, is the Chapter-3 spirit of inspecting a hyperparameter without a huge search.

### Reproducibility

One Python module for the dataset, one for training, one for metrics. Seed NumPy and TensorFlow. Record the git-less reality by writing a `run_config` JSON next to each checkpoint: image size, backbone, loss weights, learning rate, epoch of the best validation score, and the split manifest hash. Do not call `shutil.rmtree` on `data/processed` from the training script.

---

## 11. Evaluation strategy

### Technical

| Metric | Why it is here | Course status |
| --- | --- | --- |
| Pixel accuracy | Show once. All-background scores about 90% and IoU 0. | Taught, via CIFAR accuracy. Misleading on this set. |
| Precision | Of the pixels marked obstacle, how many are. | Not in the slides. Required by the false-alarm side of the business table. |
| Recall | Of the true obstacle pixels, how many were found. | Not in the slides. Required by the safety side. |
| F1 | Harmonic mean. Punishes a model that buys recall with a huge mask. | Not in the slides. |
| IoU | Intersection over union of the two regions. Standard segmentation overlap. | Not named in the extracted slides. Define it in the notebook. |
| Dice | 2·intersection / (\|A\| + \|B\|). Equals pixel F1 for a binary mask. | Not in the slides. Report it because it is the usual twin of IoU and because a soft Dice loss needs the definition. |

Validation: early stopping and the learning-rate choice. Test: one final pass after the checkpoint is frozen. Also publish the nine per-image IoUs so a single herd cannot hide a missed tire. There is no tire in the test set; say that next to the table. Stratify the narrative by prefix and by small versus large area (threshold at the dataset median, 6.6%).

Error analysis: best image, worst image, a small object, a large object. Overlay prediction and ground truth in different colours. That is the explainability this course actually licenses.

### Business

A separate short section: does the mask sit on the obstacle a driver would care about, is the region small enough to be informative, and what was not tested (clean roads, night, rain, a real vehicle camera). No invented euro amounts.

### What would make a comparison invalid

- Training on all 60 images and then quoting a test score.
- Comparing a supervised IoU on 9 images with the ICCV paper’s unsupervised numbers on all 60.
- Choosing the decision threshold on the test images.
- Splitting `lost_tires08` and `lost_tires09` into different subsets.

---

## 12. Notebook specification

The existing `notebooks/01_exploration_and_preprocessing.ipynb` should stay. It already has the right tone: markdown before code, the metric warning, and the split rules. **Do not overwrite it.**

**PROPOSED** second notebook: `notebooks/02_modeling_and_evaluation.ipynb`. It may import the first stage’s manifest rather than redraw every figure. A single combined notebook is acceptable if the marker wants one file; in that case build a new file that imports the existing functions and do not edit the current notebook in place.

Required sections, in order:

1. Title, question, and protocol (supervised split versus the ICCV protocol).
2. Business understanding, short, with facts separated from hypotheses.
3. Data understanding: recompute the 60-frame audit, show three overlays, state imbalance.
4. Preparation: letterbox or the recommended crop, split table, leakage rules.
5. Metric definitions, including the formulas for IoU and Dice, and the all-background example.
6. Model 1, with `summary`, curves, and why a from-scratch CNN is expected to struggle.
7. Model 2, transfer U-Net, augmentation, early stopping, the checkpoint path.
8. Test table and per-image panel.
9. Error analysis and limitations.
10. How the application loads the same checkpoint.
11. Reproducibility: package versions, seed, command.

Markdown before every code cell that makes a decision. No cell whose only content is an uncommented training call. No simulated metrics.

---

## 13. Application specification

Nothing in the repository is an application. The scientific stack is Python, Pillow, NumPy, pandas, matplotlib, and, on this machine, TensorFlow/Keras. There is no Flask, Streamlit, or Gradio dependency yet.

**PROPOSED:** a local Streamlit or Gradio application, chosen for a short academic demo rather than for production. Streamlit fits a “one page, one picture, one result” brief. Gradio fits an upload widget with less layout work. Either is justified. A from-scratch Flask site is extra effort without course support. The student should pick one before implementation.

The page should:

- accept a PNG, JPEG, or WebP upload and reject other types with a readable message;
- apply the same resize and normalisation as training, and show that step in a caption (“the photo was scaled to the size the model saw”);
- run the saved Keras model, not a random mask and not a hardcoded example;
- draw the predicted region on the photo;
- show a confidence cue that is honest: for example the share of pixels above 0.5, and the mean probability inside the predicted region. Do not call the maximum logit a “percentage of safety”;
- warn when the predicted region is tiny or huge;
- state the training scope in one sentence a non-specialist can read;
- refuse to pretend the tool can brake a car.

If the upload is one of the 60 known frames, the page may also show the ground-truth mask, labelled as the annotation, not as the model. Unknown uploads show prediction only.

Error cases: missing weight file, corrupt image, image smaller than a few dozen pixels, model-input shape mismatch.

---

## 14. Proposed final repository architecture

Keep the current data and the current preparation code. Add training and the application beside them.

```
RoadAnomaly/
  readme.md
  credits.txt
  frame_list.json
  requirements.txt
  frames/                      # original data, untouched
  data/processed/              # current 512×512 split, still produced by the existing script
  src/
    road_anomaly.py            # keep; stop calling rmtree from new code
    data.py                    # loaders, optional 512×288 crop, augmentation
    models.py                  # FCN and U-Net builders
    metrics.py                 # precision, recall, F1, IoU, Dice
    train.py                   # fit, early stopping, checkpoint
    predict.py                 # one function used by the notebook and the app
  configs/
    default.json               # image size, loss weight, learning rate, seed
  notebooks/
    01_exploration_and_preprocessing.ipynb
    02_modeling_and_evaluation.ipynb
  models/                      # saved .keras files and the run JSON, created by training
  reports/                     # existing report plus later metric tables
  app/
    streamlit_app.py           # or gradio_app.py
  tests/
    test_metrics.py            # synthetic masks with a known IoU
    test_preprocess.py         # label 2 maps to positive; size is preserved
  docs/
    INITIAL_PROJECT_AUDIT.md
```

Separation: the notebook calls `train.py` and `predict.py`. It does not own a second copy of the resize logic. The application imports `predict.py` only.

Configuration: one JSON file. No scattered constants for image size.

Experiment tracking: a JSON sidecar per run is enough. MLflow is not justified for two models.

Tests: two small tests that do not load the 60 images. They protect the metric definitions and the mask remapping.

Documentation: extend `readme.md` later with how to regenerate the split, how to train, and how to launch the app. Do not rewrite the dataset authors’ note into a different dataset description.

Infrastructure: Python 3.13 is what this machine runs. TensorFlow 2.21 and Keras 3.15 are already importable. Pillow, NumPy, pandas, matplotlib, scikit-learn, and nbformat are installed. PyTorch and OpenCV are not. A new UI library and, if chosen, a segmentation helper library would be new installs. That step is outside this audit. ImageNet weights, if used, are a download at training time and should be cached.

---

## 15. Additional deliverables

| Deliverable | Academic value | Technical value | Effort | Recommendation |
| --- | --- | --- | --- | --- |
| Modeling notebook that continues the existing one | Highest | Highest | Already required | Do this. |
| Working application on the real checkpoint | Highest | Highest | Required | Do this. |
| Short academic report (8–15 pages) distilled from the notebook | High for a jury that will not read every cell | Medium | Medium | Do this if the marker expects a PDF. The current `reports/01_...md` is the first half. |
| Per-image error panel and a one-page model card (data, metric, limits, licence) | High | High | Low | Do this. It is the explainability the CNN lecture left open. |
| Reproducibility section: versions, seed, commands | High | High | Low | Do this inside the notebook and the readme. |
| Defence guide: 10 questions the jury will ask, with the answers in the student’s own words | High the week of the defence | Low | Low | Write it last, from the real results. |
| 8–10 slide jury presentation | High | Low | Low | After the numbers exist. |
| Architecture note beyond section 14 | Medium | Medium | Low | A page in `docs/` is enough. A second long document is not. |
| Experiment-tracking server | Low here | Low at this scale | Medium | Skip. JSON sidecars are enough. |
| Full dataset datasheet | Medium | Medium | Medium | The audit tables already cover it. Do not duplicate them unless the marker asks. |

---

## 16. Technical feasibility and risks

| Risk | Evidence | Alternative |
| --- | --- | --- |
| Overfitting | 42 images, a deep encoder | Frozen or lightly tuned encoder, augmentation, early stopping, the from-scratch model kept as a sanity check |
| Tiny test | 9 images, no tires, no snowy dashcam | Per-image table. Refuse a claim such as “the model detects tires” from the test number. |
| No GPU | `nvidia-smi` missing on this PC | 42 images at 512×288, batch size 2–4, a small backbone, on the order of minutes to a few hours per run on CPU. **PROPOSED** expectation, not a timed benchmark. Colab is the fallback if a chosen backbone is too slow. Do not start that fallback during a demo without having exported the notebook. |
| TensorFlow on Python 3.13 | TensorFlow 2.21 imported successfully here | Keep this interpreter. A fresh environment should pin versions after the first successful training, not before. |
| Black bars | 43.75% of each label is padding | Crop to 512×288 in the loader |
| Metric self-deception | 90% background | Lead with IoU and recall |
| Protocol confusion | Coursework split versus ICCV | Label every table with the protocol |
| Autoencoder temptation | Chapter 5 | Explain the missing normal class instead of forcing a fit |
| Mask R-CNN temptation | Segmentation deck | One discussion paragraph |
| Dependency drift | `requirements.txt` is four loose lines and omits packages the scripts already import | Extend it when training dependencies are chosen. Do not pretend it is complete today. |
| Licence | `credits.txt` | Local demo, credit line in the app, no claim of ownership |
| Re-running preparation | `shutil.rmtree` on `data/processed` | Training code must not call it |
| Reproducibility without git | No `.git` | A seed, a config file, and a saved manifest. A repository is a student decision. |
| Ch1 and several formulas unread as text | PDF bodies and PPTX equations are images | Do not invent slide equations. Define IoU and Dice in the notebook from first principles. |
| Domain shift | Mixed cameras and stock photos | Limitations section and the application disclaimer |

Memory: one 512×512×3 float32 batch is small. The risk is not RAM. The risk is training time and overfitting.

---

## 17. Outstanding questions

1. Does the marker confirm supervised segmentation on this split, or do they want the paper’s unsupervised Cityscapes protocol? The repository currently implements the first.
2. Is a second notebook acceptable, or is one notebook mandatory?
3. Is there a page limit, a required language (the practical is French; the dataset notes and the current report are English), and a required report PDF?
4. May ImageNet encoder weights be downloaded?
5. Is an external normal-road dataset in scope? If not, autoencoder anomaly detection should stay a discussion.
6. Which UI library is acceptable: Streamlit, Gradio, or something already used in another course?
7. Is a GPU or Colab available for the student, given that this PC has no `nvidia-smi`?
8. What numerical bar, if any, counts as success? None is written down.
9. Should filename categories be predicted, or only used to explain errors? The masks support only the second.
10. Should `lost_tires03` stay glued to frames 08 and 09? Visually it is a different scene from the same credited video page.
11. Is showing the 60 copyrighted photographs inside a local demo acceptable to the marker under the research-use note?
12. Are chapters 2 and 6–9 supposed to be in the student’s materials? They were not on Desktop, Documents, Downloads, or OneDrive.

---

## 18. Recommended implementation roadmap

1. Student answers section 17, at least questions 1, 4, 6, and 7.
2. Add `metrics.py` and unit-check IoU and Dice on a toy mask. No training yet.
3. Decide 512×288 crop versus letterbox, and implement it in one loader.
4. Train the constant baseline and the from-scratch FCN. Record curves and the 9 test rows.
5. Train one transfer U-Net. Keep the best validation checkpoint.
6. Write the modeling notebook around those real runs.
7. Build the application against that checkpoint and try a known test image plus a non-image upload.
8. Write the short report, the model card, and the defence slides from the numbers that actually occurred.
9. Only then consider one ablation (padding, or frozen encoder). Stop there.

---

## 19. Final recommendations

1. Treat the project as **supervised binary segmentation of unexpected obstacles**, and keep saying so.
2. Keep `src/road_anomaly.py`, the notebook, the report, and the prepared split. They are an asset.
3. Do not rebuild a classifier, a detector, or an autoencoder to decorate the course list.
4. Demonstrate the course with a from-scratch CNN segmenter, a transfer U-Net, Adam, early stopping, augmentation, dropout, and an honest metric section that explains why the CIFAR accuracy habit fails here.
5. Define IoU and Dice in the notebook, because the slides do not.
6. Put the trained model in a small local application that draws the region and states the limits.
7. Lead the defence with figure 11 (the mask follows the object), figure 4 (accuracy is a trap), and the per-image test table (nine images cannot support a universal claim).

---

## Appendix A. Direct answers to the twenty questions

1. **What is RoadAnomaly?** A 60-image benchmark of unusual road obstacles with pixel masks, released for ICCV 2019 perception testing. **VERIFIED**, `readme.md`.
2. **What does the repository already accomplish?** It audits all 60 pairs, documents the business and data story, builds a leakage-aware 42/9/9 split, letterboxes to 512×512, and draws eleven figures. It does not train or deploy. **VERIFIED**.
3. **Dataset source, structure, size, composition?** Lis et al. 2019; `frames/` plus LabelGrab folders; 821 project files; 60 RGB images; prefixes 31/15/8/3/3. **VERIFIED**.
4. **Dimensions, formats, channels, resolutions?** WebP, RGB, 8-bit, 1280×720 only, 16:9. File sizes 45,708–446,946 bytes. **VERIFIED**.
5. **Labels?** Semantic mask values 0 and 2 for every frame. Instance ids for 46 frames, class field 2 only. Editor GrabCut files are not labels. **VERIFIED**.
6. **Precise problem?** Supervised binary semantic segmentation. Not pothole classification. **VERIFIED** from the masks. The paper’s unsupervised protocol is a different problem and is only partly documented locally.
7. **Main quality issues?** Pixel imbalance, category imbalance, no negative images, mixed cameras and stock photos, watermarks, 14 missing instance masks, a 9-image test with no tires, letterbox padding, val/test difficulty shift. **VERIFIED** except the “stock / mixed camera” judgement, which is **INFERRED** from a visual sample and the credits.
8. **Business objectives?** Locate unexpected obstacles so a person can see where they are. Stakeholder and money claims are hypotheses. **PROPOSED** on top of the readme.
9. **Data-science objectives?** Pixel mask in, obstacle probability out, overlap metrics, transfer segmentation. **PROPOSED**.
10. **CRISP-DM activities?** Business and data understanding mostly done; preparation done with a recommended crop fix; modelling, evaluation, and deployment not started. **PROPOSED**. CRISP-DM itself was not in the course files.
11. **Course concepts to demonstrate?** U-Net, transfer learning, augmentation, a from-scratch CNN, Adam, early stopping, dropout, train/validation curves, and a written correction of accuracy. **PROPOSED** from chapters that were actually read.
12. **Algorithms?** Constant baseline, small FCN, transfer U-Net. Not Mask R-CNN, not an autoencoder on this folder, not an RNN. **PROPOSED**.
13. **Metrics?** Precision, recall, F1, IoU, Dice, plus accuracy only as a counterexample. Because the positive class is 9.85% of pixels. **PROPOSED**; IoU and Dice were not in the extracted lectures.
14. **Notebook contents?** Section 12. Keep the current notebook; add modelling and evaluation.
15. **Application contents?** Section 13. Upload, real model, overlay, honest confidence, errors, disclaimer.
16. **Stack?** Python, TensorFlow/Keras (already installed, and it is the practical’s stack), the existing NumPy/Pillow/pandas/matplotlib code, Streamlit or Gradio for the UI. Not PyTorch, which is not installed and not used in the course. **PROPOSED**.
17. **Extra deliverables worth doing?** Short report, model card, reproducibility notes, defence slides. Skip an experiment server.
18. **Main risks?** Overfitting, a tiny and biased test, CPU time, protocol confusion, black-bar shortcuts, licences. Section 16.
19. **Still missing?** Marker confirmation of the protocol, UI choice, permission to download encoder weights, chapters 2 and 6–9 if they exist elsewhere, Ch1 slide bodies, a success threshold, and any GPU the student may have on another machine.
20. **Decisions before implementation?** Questions 1, 4, 6, and 7 in section 17, plus whether to crop the letterbox.

---

## Appendix B. Environment observed on this machine

**VERIFIED** by import, not by a new install.

- Python 3.13.7
- Pillow 12.0.0, NumPy 2.3.4, pandas 2.3.3, matplotlib 3.10.7
- TensorFlow 2.21.0, Keras 3.15.1
- scikit-learn 1.8.0, nbformat 5.11.1
- Not importable: torch, cv2, fitz, pypdf, python-pptx
- `nvidia-smi` not on PATH

These versions describe the investigation machine. They are not a promise about the lab machines.
