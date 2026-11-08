# Guide de soutenance technique individuelle

**Objectif :** chaque membre doit pouvoir justifier **toutes** les décisions du projet, pas seulement sa partie.  
**Langue :** français (termes techniques EN conservés quand standards).  
**Alignement :** `docs/02_protocole_scientifique_phase1.md`  
**Chiffres d’expérience :** absents pour l’instant → réponses marquées `[Résultats à compléter après les expériences]` lorsque nécessaire.

Pour les questions importantes : réponse claire → justification projet → relance difficile → réponse à la relance.

---

## A. Business Understanding et formulation

### Q1. Quel est le problème traité ?

**Réponse.** Détecter et **localiser** des objets inattendus dans des scènes routières (animaux, rochers, cônes, pneus, véhicules atypiques) à partir d’une image.

**Pourquoi ici.** Le dataset RoadAnomaly fournit des masques pixel ; le besoin métier est spatial (où est l’obstacle).

**Relance.** Pourquoi pas se contenter d’un score image « anomalie / normal » ?  
**Réponse.** Un score global ne dit pas si l’objet est sur la voie, loin, ou hors route ; un planificateur ou un humain a besoin de la région.

### Q2. Quelle est la différence entre objectifs métier (BO) et objectifs data science (DSO) ?

**Réponse.** Les BO décrivent le besoin (localiser, être vérifiable, couvrir tailles variées). Les DSO traduisent cela en tâche ML (segmentation binaire, métriques d’overlap, protocole val/test).

**Pourquoi ici.** Le rapport CRISP-DM demande cette distinction explicitement.

### Q3. Pourquoi ce n’est pas de la classification d’images ?

**Réponse.** La classification assigne une étiquette à toute l’image. Ici chaque pixel reçoit fond ou anomalie ; plusieurs objets peuvent coexister ; la sortie est un masque.

### Q4. Pourquoi ce n’est pas de la détection d’objets (YOLO, etc.) ?

**Réponse.** Nous n’avons pas de boîtes englobantes comme cible principale ; la supervision officielle est un masque sémantique. La segmentation produit directement la région pixel.

**Relance.** Les `crop_rect` de `index.json` ne sont-ils pas des boxes ?  
**Réponse.** Ce sont des rectangles d’édition LabelGrab, pas un jeu de détection standard ; 14 images n’ont même pas d’instances. On ne construit pas YOLO dessus.

---

## B. Dataset et annotations

### Q5. Combien d’images et quelles dimensions ?

**Réponse.** 60 images RGB WebP, toutes **1280×720** (16:9).

### Q6. Que signifient les valeurs du masque sémantique ?

**Réponse.** Officiellement : `0` = fond / non-anomalie, `2` = anomalie. Il n’y a pas de classe `1` dans les fichiers. Pour le modèle, on remappera en `0/1` dans le loader.

### Q7. Les préfixes `animals`, `cones`… sont-ils des classes du modèle ?

**Réponse.** Non. Ce sont des descripteurs de fichier pour stratification et analyse d’erreurs. Le masque encode une seule classe positive « anomalie ».

**Relance.** Pourquoi alors les rapporter ?  
**Réponse.** Parce que le jeu est déséquilibré (31 animals / 60) et que le test peut être dominé par certaines scènes ; l’analyse par préfixe évite une lecture naïve d’une moyenne.

### Q8. Y a-t-il des images sans anomalie ?

**Réponse.** Non. Les 60 masques contiennent au moins un pixel d’anomalie. Donc on ne peut pas mesurer le taux de fausses alarmes sur « route vide » avec ce dossier seul.

### Q9. Que faire des 14 images sans masque d’instance ?

**Réponse.** Les garder : le masque sémantique est présent partout. Les supprimer pour utiliser les instances jetterait ~23 % d’un déjà petit jeu.

---

## C. Prétraitement et fuite de données

### Q10. Pourquoi redimensionner en 512×288 et non 512×512 letterbox ?

**Réponse.** 512×288 conserve le 16:9 et est divisible par 32 (U-Net). Le letterbox 512×512 ajoute ~43,75 % de bandes noires étiquetées fond, ce qui crée un raccourci d’apprentissage (« noir = sûr ») et dilue le signal.

**Relance.** Pourquoi alors des fichiers 512×512 existent déjà ?  
**Réponse.** Artefact historique de l’exploration Phase données ; le protocole Phase 1 fixe 512×288 pour l’entraînement. On ne détruit pas l’historique.

### Q11. Quelle interpolation pour l’image et pour le masque ?

**Réponse.** Image : interpolation continue (bilinéaire). Masque : **nearest-neighbour** pour ne pas créer de classes intermédiaires (ex. 0.5).

### Q12. Comment évitez-vous la fuite de données ?

**Réponse.** (1) Groupes répétés (pneus, sculpture, rochers) entièrement en train. (2) Augmentation train only. (3) Sélection modèle/seuil sur val. (4) Test une seule fois. (5) Split figé + hash SHA-256.

**Relance.** `lost_tires03` est-il vraiment un doublon de 08/09 ?  
**Réponse.** Même source créditée, mais scène visuelle différente (neige vs été). On le garde en train par prudence source ; 08/09 sont les vrais quasi-doublons temporels.

### Q13. Pourquoi pas de flip vertical ?

**Réponse.** Cela invente une géométrie de route irréaliste (ciel en bas). Le flip horizontal reste plausible.

---

## D. CNN et segmentation

### Q14. Qu’est-ce qu’une CNN apporte ici ?

**Réponse.** Extraction hiérarchique de motifs locaux (textures, contours) avec partage de poids ; adaptée aux images. En segmentation, on conserve (ou on restaure) la résolution spatiale pour prédire chaque pixel.

### Q15. Qu’est-ce qu’un modèle fully convolutional dans ce contexte ?

**Réponse.** Un réseau sans tête dense de classification globale : convolutions (et upsampling) jusqu’à une carte de la taille de sortie voulue, ici une probabilité par pixel.

### Q16. Pourquoi une activation sigmoïde en sortie ?

**Réponse.** Segmentation **binaire** : une probabilité d’anomalie par pixel dans \([0,1]\). Softmax multi-classe n’est pas nécessaire.

---

## E. U-Net, MobileNetV2, transfer learning

### Q17. Pourquoi U-Net ?

**Réponse.** Architecture encodeur–décodeur avec **skip connections**, conçue pour la segmentation avec peu de données annotées (origine biomédicale). Le cours de segmentation la met en avant ; elle matche nos masques denses et n=42.

### Q18. À quoi servent les skip connections ?

**Réponse.** Elles réinjectent les cartes haute résolution de l’encodeur dans le décodeur, pour récupérer les détails (bords de cônes, pattes) perdus par le pooling.

**Relance.** Sans skips, que se passe-t-il typiquement ?  
**Réponse.** Masques plus flous, petits objets moins bien délimités.

### Q19. Pourquoi MobileNetV2 comme encodeur ?

**Réponse.** Encodeur léger pré-entraîné ImageNet, adapté à Colab/CPU relatif, bon compromis précision/coût pour un prototype. Les poids ImageNet sont autorisés par décision Phase 1.

### Q20. Qu’est-ce que le transfer learning ici ?

**Réponse.** Réutiliser des filtres appris sur ImageNet comme extracteur de caractéristiques, puis entraîner le décodeur (et éventuellement fine-tuner l’encodeur) sur nos 42 masques routiers.

**Relance.** Les classes ImageNet ne contiennent pas « anomaly road ». Est-ce un problème ?  
**Réponse.** On ne transfère pas les classes finales, mais des représentations bas/moyen niveau (bords, textures). C’est précisément l’intérêt avec peu de données.

### Q21. Encodeur gelé vs fine-tuning : que choisissez-vous ?

**Réponse.** Les deux sont testés (E2a / E2b). On **choisit selon la validation** (Dice moyen). On ne suppose pas a priori que le fine-tuning est meilleur : avec n=42 il peut surapprendre.

`[Configuration gagnante à compléter après les expériences]`

---

## F. Pertes, hyperparamètres, surapprentissage

### Q22. Pourquoi ne pas utiliser uniquement la binary cross-entropy non pondérée ?

**Réponse.** Seulement ~9,85 % de pixels positifs : une BCE naïve peut sous-détecter. On prévoit une pondération de classe et/ou un terme Dice aligné sur la métrique.

### Q23. Quels hyperparamètres sont figés dans la config ?

**Réponse.** Entre autres : seed 42, Adam, tailles 288×512, batch 4, early stopping patience 12, critère `val_mean_dice` — voir `configs/default.json`.

### Q24. Comment détectez-vous le surapprentissage ?

**Réponse.** Écart train/val sur la loss et sur Dice/IoU ; early stopping ; comparaison E1 (from scratch) vs E2 ; analyse qualitative d’images val non mémorisées comme train.

**Relance.** Avec seulement 9 images val, l’early stopping n’est-il pas bruité ?  
**Réponse.** Oui : c’est une menace à la validité. D’où l’importance des courbes, du seed fixe, et de ne pas sur-interpréter un écart de 0,01 sur une moyenne de 9 images.

---

## G. Métriques et seuil

### Q25. Définir Precision et Recall sur des pixels.

**Réponse.** Precision = TP/(TP+FP) : parmi les pixels prédits anomalie, combien le sont. Recall = TP/(TP+FN) : parmi les vrais pixels anomalie, combien sont trouvés.

### Q26. Pourquoi IoU et Dice plutôt que l’accuracy ?

**Réponse.** L’accuracy est dominée par le fond (~90 %). IoU = TP/(TP+FP+FN) mesure le chevauchement des régions. Dice = 2TP/(2TP+FP+FN) est l’équivalent F1 pixel ; c’est aussi notre critère de checkpoint.

**Relance.** Dice et F1 pixel sont-ils identiques ?  
**Réponse.** Pour un masque binaire, Dice coïncide avec le F1 calculé sur les pixels de la classe positive.

### Q27. Comment choisissez-vous le seuil de binarisation ?

**Réponse.** Défaut 0,5 ; éventuelle optimisation du seuil **sur validation** (ex. max Dice moyen). Le seuil est ensuite figé pour le test et Streamlit. Jamais retuné après avoir vu le test.

### Q28. Validation vs test : quelle différence opérationnelle ?

**Réponse.** Val = décisions (architecture, early stopping, seuil). Test = estimation finale non retouchée. Utiliser le test pour choisir le modèle invaliderait l’évaluation.

---

## H. Autoencodeur et ICCV 2019

### Q29. Pourquoi un autoencodeur classique n’est pas une baseline valide ici ?

**Réponse.** Un AE d’anomalie par reconstruction suppose d’apprendre la distribution **normale**. RoadAnomaly n’a que des images déjà anormales. Entraîner sur ces 60 risque d’apprendre à reconstruire aussi les obstacles ; la comparaison à l’IoU supervisé serait protocolairement faible.

**Relance.** Le professeur autorise d’« explorer » les AE. N’avez-vous rien fait ?  
**Réponse.** Si : une **évaluation de faisabilité** documentée (protocole + rapport). Explorer ≠ forcer une expérience non valide. Sans corpus normal externe (hors scope), nous n’entraînons pas d’AE concurrent.

### Q30. En quoi votre projet diffère de Lis et al. (ICCV 2019) ?

**Réponse.** Le papier : segmenteur de classes connues → resynthèse (pix2pixHD) → réseau de discrépance ; entraînement sans voir les anomalies (Cityscapes + label swap) ; RoadAnomaly pour l’évaluation (AUROC). Nous : segmentation supervisée directement sur les masques RoadAnomaly ; métriques Precision/Recall/F1/IoU/Dice sur split 42/9/9.

**Relance.** Peut-on dire que vous améliorez leur AUROC ?  
**Réponse.** Non. Protocoles et métriques différents ; ce serait une comparaison fallacieuse.

---

## I. Déploiement et limites

### Q31. Que démontrera l’application Streamlit ?

**Réponse.** Upload d’image → même prétraitement que l’évaluation → checkpoint réel → masque + overlay. Pas de simulation de masque.

### Q32. Pourquoi le système n’est pas prêt pour la sécurité routière ?

**Réponse.** 42 images d’entraînement, photos web hétérogènes, pas de test night/rain systématique, pas de mesure sur routes vides, pas de redondance capteurs, pas de certification. C’est un **prototype académique**.

### Q33. Quelle est la principale menace à la validité statistique ?

**Réponse.** Test de 9 images (variance élevée), shift de difficulté val/test, absence de certaines catégories en test (pneus), risque de surapprentissage et de biais de domaine (Wikimedia ≠ caméra véhicule).

---

## J. Questions transverses « piège »

### Q34. Si votre accuracy test est 91 %, avez-vous réussi ?

**Réponse.** Pas nécessairement. La baseline tout-fond est déjà ~90 % accuracy avec IoU ≈ 0. Il faut lire IoU/Dice/Recall.

### Q35. Pourquoi TensorFlow/Keras et pas PyTorch ?

**Réponse.** Stack du module (TP CNN CIFAR) et décision de projet ; PyTorch n’est pas retenu pour éviter une bifurcation technique.

### Q36. Que signifie le fichier `split_lock.json` ?

**Réponse.** Empreinte SHA-256 du split 42/9/9 pour garantir qu’aucun run ne s’appuie sur une répartition silencieuse différente.

---

## Grille d’auto-évaluation (chaque membre)

| Compétence | Je peux l’expliquer sans notes | Besoin de revoir |
| --- | --- | --- |
| BO vs DSO | ☐ | ☐ |
| Masques 0/2 → 0/1 | ☐ | ☐ |
| 512×288 vs letterbox | ☐ | ☐ |
| Anti-fuite split | ☐ | ☐ |
| Skips U-Net | ☐ | ☐ |
| Transfer / gel / fine-tune | ☐ | ☐ |
| IoU vs accuracy | ☐ | ☐ |
| Val vs test / seuil | ☐ | ☐ |
| Exclusion AE | ☐ | ☐ |
| Différence ICCV 2019 | ☐ | ☐ |
| Limites déploiement | ☐ | ☐ |
| Lire nos tableaux de résultats | ☐ | ☐ `[après expériences]` |

---

## Phrase d’ouverture utile en cas de stress

> « Notre projet traite la détection d’anomalies routières comme une **segmentation sémantique binaire supervisée** sur RoadAnomaly. Nous ne reproduisons pas la resynthèse ICCV 2019 ; nous exploitons les masques disponibles avec un protocole val/test strict et des métriques d’overlap adaptées au déséquilibre. »
