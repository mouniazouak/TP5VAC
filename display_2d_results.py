from pathlib import Path

import matplotlib.pyplot as plt
import torch

from model3d1 import Model


def display_confusion_matrix():
    matrix = torch.load(
        "confusion_matrix_2d.pt",
        map_location="cpu",
        weights_only=True,
    )

    plt.figure(figsize=(12, 10))
    plt.imshow(matrix.numpy(), cmap="Blues")
    plt.colorbar(label="Nombre de vidéos")
    plt.title("Matrice de confusion - Ablation 2D")
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.tight_layout()
    plt.savefig("confusion_matrix_2d.png", dpi=200)
    plt.show()

    print("Matrice de confusion affichée.")
    print("Image enregistrée dans confusion_matrix_2d.png")


def display_checkpoint():
    checkpoint_path = Path("best_plain3d_2d.pth")
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model = Model(num_classes=101)
    model.load_state_dict(state_dict)

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("\nModèle contenu dans best_plain3d_2d.pth :")
    print(model)
    print(f"Nombre total de paramètres : {number_of_parameters:,}")
    print(f"Nombre de tenseurs sauvegardés : {len(state_dict)}")


if __name__ == "__main__":
    display_confusion_matrix()
    display_checkpoint()
