# Guide de soutenance — réponses avec résultats mesurés

Complète `docs/06_individual_defense_guide.md`.  
Validation = sélection. Test = one-shot après gel (Stage 5 terminé).

## Timing oral suggéré (aligné slides docs/15)

| Bloc | Minutes |
| --- | ---: |
| Titre + problème + objectifs | 3 |
| Données + préparation | 3 |
| Modèles + protocole | 3,5 |
| Résultats val + figures | 4 |
| Limites + ICCV + conclusion | 3 |
| **Total** | **≈ 16,5** |

## Chiffres à connaître par cœur (mean/image)

| | Seuil | Dice val | Dice test | IoU val | IoU test |
| --- | ---: | ---: | ---: | ---: | ---: |
| E0 | 0,5 | 0 | 0 | 0 | 0 |
| E1 | 0,35 | 0,57 | 0,40 | 0,45 | 0,29 |
| E2a | 0,15 | 0,70 | 0,61 | 0,58 | 0,45 |
| E2b | **0,55** | **0,78** | **0,82** | **0,69** | **0,71** |

E2b Dice test **global** ≈ 0,87.  
E1 meilleure époque : 66. E2a : 10. E2b : 34.  
Split : 42/9/9. SHA split : `9a01c05c…`. E2b ckpt SHA : `280170dc…`.

## Questions difficiles — réponses courtes

**Pourquoi l’accuracy d’E0 est haute ?**  
Parce que ~90 % des pixels sont du fond ; prédire tout fond donne ~0,86 d’accuracy et 0 de Dice.

**Pourquoi deux seuils ?**  
0,5 fixe pour le checkpoint Dice pendant l’entraînement ; seuil opérationnel choisi sur **val uniquement**.

**Pourquoi E2b et pas E2a ?**  
Sur val, E2b a le meilleur Dice/IoU/F1. Sélection figée **avant** le test.

**Est-ce du few-shot ?**  
Non : 42 images d’entraînement, protocole de segmentation standard, pas d’épisode few-shot ni de méta-apprentissage.

**Pourquoi pas d’autoencodeur principal ?**  
Toutes les 60 images ont des anomalies ; pas de corpus « normal only » pour une reconstruction classique défendable.

**Différence avec Lis et al. 2019 ?**  
Ils font de la resynthèse / discordance ; nous faisons de la segmentation supervisée sur un split de RoadAnomaly. Scores non comparables directement.

**Surapprentissage ?**  
Risque élevé (n=42). Early stopping utilisé. Sur test, E1/E2a baissent ; E2b reste 1er (Dice ≈ 0,82) mais n_test=9 ⇒ ne pas sur-interpréter un écart val→test positif.

**BatchNorm absente d’E1 ?**  
Avec batch 2, BN a collapsé vers tout fond dans un run exploratoire ; décision figée : pas de BN dans E1.

## Checklist membre par membre

Chaque personne doit expliquer sans notes : split, loss, Dice vs accuracy, E2a vs E2b, seuil 0,5 vs opérationnel, pourquoi le test n’a pas servi à choisir le modèle.
