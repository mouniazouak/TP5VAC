# TP5 – Classification d'actions vidéo avec UCF101

Projet de classification de vidéos du jeu de données UCF101 avec PyTorch.

## Contenu

- `extract_frames.py` : extraction et redimensionnement des frames vidéo ;
- `data_loader.py` : chargement des clips et transformations ;
- `model3d.py` : modèle CNN 3D principal ;
- `model3d1.py` : ablation 2D (convolutions spatiales sans information temporelle) ;
- `train.py` : entraînement du modèle 3D ;
- `train1.py` : entraînement de l'ablation 2D ;
- `evaluate.py` : évaluation du modèle 3D ;
- `evaluate1.py` : évaluation de l'ablation 2D ;
- `rapport_final_tp5.md` : rapport du TP.

## Installation

Installer d'abord PyTorch correspondant à l'accélérateur disponible, puis :

```powershell
python -m pip install -r requirements.txt
```

Pour une machine Intel XPU, utiliser aussi les instructions de
`requirements-xpu.txt`.

## Données

Le jeu de données UCF101 n'est pas inclus dans ce dépôt. Télécharger UCF101
et les listes officielles train/test, puis placer les fichiers dans le dossier
`data/`. Les frames extraites et le manifeste sont générés localement.

## Exécution

```powershell
python extract_frames.py "data\UCF101\UCF-101" --annotations "data\annotations" --frames 20
python train.py --device xpu
python evaluate.py
python train1.py --device xpu
python evaluate1.py
```

## Résultats obtenus

- Modèle 3D : 51,65 % de précision au niveau clip et 52,10 % au niveau vidéo ;
- Ablation 2D : 49,51 % au niveau clip et 49,72 % au niveau vidéo.

Le modèle 3D obtient donc un gain de 2,38 points au niveau vidéo par rapport à
l'ablation 2D.
