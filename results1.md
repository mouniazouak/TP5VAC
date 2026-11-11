# Résultats TP5 — Ablation 2D

Cette expérience reprend exactement le pipeline du modèle CNN 3D principal.
La seule modification est le remplacement des noyaux temporels par des noyaux
de profondeur 1 : `(1, 3, 3)` et `(1, 7, 7)`.

| Modèle | Clip-level | Video-level | Validation |
|---|---:|---:|---:|
| CNN 3D principal | à mesurer | à mesurer | à mesurer |
| Ablation 2D | à mesurer | à mesurer | à mesurer |

Fichiers produits par l'ablation :

- `best_plain3d_2d.pth` ;
- `learning_history_2d.json` ;
- `learning_curves_2d.png` ;
- `confusion_matrix_2d.pt`.

Commande d'entraînement :

```powershell
python train1.py --device cpu
```

Commande d'évaluation :

```powershell
python evaluate1.py
```
