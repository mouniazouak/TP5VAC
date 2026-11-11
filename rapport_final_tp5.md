# Rapport final TP5 - Classification d'actions vidéo avec un CNN 3D

## 1. Objectif

L'objectif de ce TP est de reconnaître des actions humaines dans des vidéos du
dataset UCF101. Contrairement à une classification d'images, le modèle reçoit
un clip vidéo composé de plusieurs frames. Il peut donc apprendre à utiliser à
la fois l'apparence spatiale et le mouvement temporel.

Deux expériences ont été réalisées :

- un CNN 3D principal utilisant des noyaux temporels `3 x 3 x 3` ;
- une ablation 2D utilisant des noyaux `1 x 3 x 3`, afin de mesurer l'apport
  de l'information temporelle.

## 2. Dataset et séparation des données

Le dataset utilisé est UCF101 :

| Élément | Valeur |
|---|---:|
| Nombre de classes | 101 |
| Nombre total de vidéos | 13 320 |
| Vidéos de test officielles | 3 783 |
| Vidéos train + validation | 9 537 |
| Vidéos d'entraînement utilisées | 7 912 |
| Vidéos de validation utilisées | 1 625 |

Les fichiers `trainlist01.txt` et `testlist01.txt` ont été utilisés pour
conserver le split officiel de UCF101. La partie entraînement a ensuite été
séparée en train et validation par groupe de vidéos. Les vidéos provenant d'un
même groupe ne sont donc pas mélangées entre entraînement et validation.

Le manifeste final contient :

```text
test : 3783 vidéos
train : 7912 vidéos
val : 1625 vidéos
```

## 3. Préparation des vidéos

Le script `extract_frames.py` effectue les opérations suivantes :

1. lecture des fichiers `.avi` avec PyAV ;
2. échantillonnage régulier de 20 frames par vidéo ;
3. conversion des frames en RGB ;
4. redimensionnement à `80 x 80` pixels ;
5. sauvegarde au format NumPy `.npy` ;
6. création du fichier `manifest.csv`.

Chaque fichier extrait possède la forme :

```text
(20, 80, 80, 3)
```

Pendant l'entraînement, le DataLoader sélectionne 16 frames parmi les 20,
réalise un crop aléatoire `64 x 64` et applique parfois un retournement
horizontal. Les valeurs sont converties de `uint8` vers des nombres flottants
entre 0 et 1.

La forme envoyée au réseau est :

```text
(batch, channels, time, height, width)
(32, 3, 16, 64, 64)
```

Pour la validation et le test, un crop central est utilisé sans augmentation
aléatoire. Pour l'accuracy vidéo, cinq clips régulièrement espacés sont
évalués et leurs probabilités sont moyennées.

## 4. Architecture du CNN 3D

Le modèle principal est défini dans `model3d.py`. Il est entraîné depuis zéro,
sans poids pré-entraînés.

| Couche | Filtres | Noyau | Pooling |
|---|---:|---:|---|
| Conv1 + BatchNorm3D + ReLU | 64 | `3 x 7 x 7` | `1 x 2 x 2` |
| Conv2 + BatchNorm3D + ReLU | 128 | `3 x 3 x 3` | `2 x 2 x 2` |
| Conv3 + BatchNorm3D + ReLU | 256 | `3 x 3 x 3` | - |
| Conv4 + BatchNorm3D + ReLU | 256 | `3 x 3 x 3` | `2 x 2 x 2` |
| Conv5 + BatchNorm3D + ReLU | 512 | `3 x 3 x 3` | - |
| Conv6 + BatchNorm3D + ReLU | 512 | `3 x 3 x 3` | `2 x 2 x 2` |
| Global Average Pooling 3D | 512 | - | `1 x 1 x 1` |
| Dropout | - | `p = 0,5` | - |
| Couche finale | 101 | - | - |

Le réseau principal possède environ 13,6 millions de paramètres. La sortie est
constituée de logits vers les 101 classes. La fonction `CrossEntropyLoss`
applique elle-même la normalisation nécessaire ; aucun `Softmax` n'est donc
placé dans le modèle pendant l'entraînement.

## 5. Configuration d'entraînement

| Paramètre | Valeur |
|---|---:|
| Optimiseur | SGD |
| Learning rate initial | 0,025 |
| Momentum | 0,9 |
| Weight decay | 0,0005 |
| Batch size | 32 |
| Nombre d'époques | 60 |
| Label smoothing | 0,1 |
| Gradient clipping | 5,0 |
| Scheduler | CosineAnnealingLR |
| Device utilisé | Intel XPU |
| Graine demandée | 42 |

Le meilleur checkpoint est sauvegardé selon l'accuracy de validation dans :

```text
best_plain3d.pth
```

L'entraînement principal a donné :

| Mesure | Résultat |
|---|---:|
| Meilleure validation | 50,03 % |
| Validation finale | 48,74 % |
| Accuracy train finale | 99,96 % |

L'écart entre l'entraînement et la validation indique un surapprentissage,
mais la validation reste nettement supérieure au hasard, qui vaut environ
`1 / 101 = 0,99 %`.

## 6. Résultats du modèle 3D sur le test

| Niveau d'évaluation | Accuracy |
|---|---:|
| Clip-level | 51,65 % |
| Video-level | 52,10 % |

La métrique principale demandée est l'accuracy au niveau vidéo. Le résultat
final du modèle 3D est donc :

```text
Video-level accuracy = 52,10 %
```

L'agrégation des cinq clips améliore légèrement le résultat par rapport à un
seul clip :

```text
52,10 - 51,65 = +0,45 point
```

Le modèle dépasse le seuil de 50 % attendu pour le niveau excellent indiqué
dans le sujet.

## 7. Ablation 2D

L'ablation est définie dans `model3d1.py`. Elle conserve la même organisation
générale, mais remplace les noyaux temporels par des noyaux de profondeur 1 :

```text
Modèle 3D : (3, 3, 3)
Ablation 2D : (1, 3, 3)
```

Cette version ne peut donc pas apprendre directement les changements entre les
frames avec les convolutions.

Le checkpoint de l'ablation est :

```text
best_plain3d_2d.pth
```

Résultats obtenus :

| Modèle | Clip-level | Video-level |
|---|---:|---:|
| CNN 3D | 51,65 % | 52,10 % |
| Ablation 2D | 49,51 % | 49,72 % |

Gain du modèle 3D par rapport à l'ablation 2D :

```text
52,10 - 49,72 = +2,38 points
```

Ce résultat montre que l'information temporelle apporte un gain réel pour la
reconnaissance des actions vidéo.

## 8. Résultats par classe de l'ablation 2D

Les meilleures classes de l'ablation 2D sont :

| Classe | Accuracy |
|---:|---:|
| 8 | 100,00 % |
| 25 | 100,00 % |
| 63 | 100,00 % |
| 84 | 100,00 % |
| 11 | 97,50 % |
| 43 | 93,48 % |
| 95 | 92,86 % |
| 41 | 91,84 % |
| 18 | 89,29 % |
| 96 | 88,57 % |

Les classes les plus difficiles sont :

| Classe | Accuracy |
|---:|---:|
| 35 | 0,00 % |
| 37 | 0,00 % |
| 98 | 0,00 % |
| 23 | 2,04 % |
| 59 | 2,44 % |
| 46 | 5,26 % |
| 71 | 6,67 % |
| 19 | 8,33 % |
| 33 | 9,09 % |
| 57 | 9,09 % |

Ces difficultés peuvent venir de la ressemblance visuelle entre certaines
actions et de l'absence d'information temporelle dans l'ablation 2D.

## 9. Évaluation et fichiers produits

Le script `evaluate.py` produit :

- accuracy clip-level ;
- accuracy video-level ;
- accuracy par classe ;
- 16 exemples mal classés sous forme de bandes de frames ;
- `confusion_matrix.pt` ;
- `confusion_matrix_top10.pt` ;
- `learning_curves.png`.

Le script `evaluate1.py` produit les mêmes éléments pour l'ablation 2D :

- `best_plain3d_2d.pth` ;
- `learning_history_2d.json` ;
- `learning_curves_2d.png` ;
- `confusion_matrix_2d.pt`.

La matrice de confusion 2D peut être visualisée avec :

```powershell
python display_2d_results.py
```

## 10. Commandes de reproduction

Installation du support GPU Intel :

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
python -m pip install -r requirements.txt
```

Extraction des frames :

```powershell
python extract_frames.py C:\Users\DELL\mon_projet\TP5VAC\data\UCF101\UCF-101 --annotations C:\Users\DELL\mon_projet\TP5VAC\data\annotations --frames 20
```

Entraînement et évaluation du modèle 3D :

```powershell
python train.py --device xpu
python evaluate.py
```

Entraînement et évaluation de l'ablation 2D :

```powershell
python train1.py --device xpu
python evaluate1.py
```

## 11. Conclusion

Le projet répond aux objectifs principaux du TP. Le pipeline vidéo extrait les
frames, crée des clips, sépare les groupes, entraîne un CNN 3D depuis zéro et
évalue les résultats au niveau clip et vidéo.

Le CNN 3D obtient 52,10 % d'accuracy vidéo, tandis que l'ablation 2D obtient
49,72 %. Le gain de 2,38 points confirme que les convolutions 3D utilisent une
information de mouvement utile pour reconnaître les actions.

Le modèle présente un surapprentissage important avec une accuracy train de
99,96 %, mais la validation atteint 50,03 % et le test vidéo 52,10 %. Les
courbes, les matrices de confusion et les exemples mal classés permettent
d'analyser les limites du modèle et les classes difficiles.
