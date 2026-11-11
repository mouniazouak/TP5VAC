import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from data_loader import get_test_loader
from model3d import Model


CHECKPOINT_PATH = "best_plain3d.pth"
MANIFEST_PATH = "manifest.csv"
NUM_CLASSES = 101
NUM_TEST_CLIPS = 5


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def save_misclassified_clip(data, prediction, target, index):
    output_directory = Path("misclassified")
    output_directory.mkdir(parents=True, exist_ok=True)

    # data possède la forme (3, temps, hauteur, largeur)
    clip = data.cpu()

    # Créer une bande horizontale avec les 16 frames du clip
    images = []
    for frame in clip.permute(1, 2, 3, 0):
        images.append(np.clip(frame.numpy(), 0, 1))
    image = np.concatenate(images, axis=1)

    output_path = output_directory / f"misclassified_examples{index}.png"

    plt.imshow(image)
    plt.title(f"Prediction: {prediction} - Cible: {target}")
    plt.axis("off")
    plt.savefig(output_path)
    plt.close()

    print(
        f"Clip {index}: Prediction: {prediction}, "
        f"Cible: {target}"
    )


def main():
    device = get_device()

    print(f"Device utilisé : {device}")

    model = Model(num_classes=NUM_CLASSES).to(device)

    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    # Résultats du clip central
    all_targets = []
    all_predictions = []

    # Résultats pour le niveau vidéo
    video_outputs = defaultdict(list)
    video_targets = {}
    misclassified_index = 0

    with torch.no_grad():
        for clip_index in range(NUM_TEST_CLIPS):
            test_loader = get_test_loader(
                manifest_path=MANIFEST_PATH,
                batch_size=32,
                num_workers=0,
                clip_index=clip_index,
            )

            for data, target, video_ids in test_loader:
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                probabilities = torch.softmax(output, dim=1)

                # Le premier clip sert à calculer l'accuracy clip
                if clip_index == 0:
                    predictions = output.argmax(dim=1)
                    all_targets.extend(target.cpu().tolist())
                    all_predictions.extend(predictions.cpu().tolist())

                    for sample_index, (prediction, label) in enumerate(
                        zip(predictions.cpu().tolist(), target.cpu().tolist())
                    ):
                        if prediction != label and misclassified_index < 16:
                            save_misclassified_clip(
                                data[sample_index],
                                prediction,
                                label,
                                misclassified_index,
                            )
                            misclassified_index += 1

                for video_id, label, probability in zip(
                    video_ids,
                    target.cpu().tolist(),
                    probabilities.cpu(),
                ):
                    video_outputs[video_id].append(probability)
                    video_targets[video_id] = label

    # Accuracy au niveau clip
    clip_correct = sum(
        prediction == target
        for prediction, target in zip(all_predictions, all_targets)
    )

    clip_accuracy = 100.0 * clip_correct / len(all_targets)

    print(f"Clip-level accuracy: {clip_accuracy:.2f}%")

    # Moyenne des cinq clips pour obtenir la prédiction vidéo
    video_targets_list = []
    video_predictions_list = []

    for video_id, outputs in video_outputs.items():
        mean_output = torch.stack(outputs).mean(dim=0)
        prediction = mean_output.argmax().item()

        video_targets_list.append(video_targets[video_id])
        video_predictions_list.append(prediction)

    video_correct = sum(
        prediction == target
        for prediction, target in zip(
            video_predictions_list,
            video_targets_list,
        )
    )

    video_accuracy = 100.0 * video_correct / len(video_targets_list)

    print(f"Video-level accuracy: {video_accuracy:.2f}%")

    # Afficher les erreurs au niveau vidéo
    error_index = 0

    for video_id, target, prediction in zip(
        video_outputs.keys(),
        video_targets_list,
        video_predictions_list,
    ):
        if prediction != target and error_index < 16:
            print(
                f"Video {video_id}: "
                f"Prediction: {prediction}, "
                f"Cible: {target}"
            )
            error_index += 1

    # Accuracy par classe
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    for target, prediction in zip(
        video_targets_list,
        video_predictions_list,
    ):
        class_total[target] += 1

        if prediction == target:
            class_correct[target] += 1

    class_results = []

    for class_index in range(NUM_CLASSES):
        if class_total[class_index] > 0:
            class_accuracy = (
                100.0
                * class_correct[class_index]
                / class_total[class_index]
            )

            class_results.append(
                (
                    class_index,
                    class_accuracy,
                    class_total[class_index],
                )
            )

    print("10 meilleures classes :")

    for class_index, accuracy, total in sorted(
        class_results,
        key=lambda item: item[1],
        reverse=True,
    )[:10]:
        print(
            f"Classe {class_index}: "
            f"{accuracy:.2f}% "
            f"({total} vidéos)"
        )

    print("10 classes les plus difficiles :")

    for class_index, accuracy, total in sorted(
        class_results,
        key=lambda item: item[1],
    )[:10]:
        print(
            f"Classe {class_index}: "
            f"{accuracy:.2f}% "
            f"({total} vidéos)"
        )

    # Construire la matrice de confusion vidéo
    confusion_matrix = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        dtype=torch.int64,
    )

    for target, prediction in zip(
        video_targets_list,
        video_predictions_list,
    ):
        confusion_matrix[target][prediction] += 1

    print("Matrice de confusion :")
    print(confusion_matrix)

    torch.save(
        confusion_matrix,
        "confusion_matrix.pt",
    )

    # Sauvegarder la matrice des 10 classes qui contiennent le plus d'erreurs
    class_errors = [
        (class_index, class_total[class_index] - class_correct[class_index])
        for class_index in range(NUM_CLASSES)
        if class_total[class_index] > 0
    ]
    top_error_classes = [
        class_index
        for class_index, _ in sorted(
            class_errors,
            key=lambda item: item[1],
            reverse=True,
        )[:10]
    ]
    top_error_indices = torch.tensor(top_error_classes, dtype=torch.long)
    top_confusion_matrix = confusion_matrix.index_select(0, top_error_indices).index_select(1, top_error_indices)
    torch.save(
        {
            "classes": top_error_classes,
            "matrix": top_confusion_matrix,
        },
        "confusion_matrix_top10.pt",
    )
    print("Matrice des 10 classes les plus confondues enregistrée dans confusion_matrix_top10.pt")

    # Ajouter les courbes d'apprentissage
    history_path = Path("learning_history.json")

    if history_path.exists():
        with history_path.open(
            encoding="utf-8"
        ) as history_file:
            history = json.load(history_file)

        epochs = range(1, len(history["train_loss"]) + 1)

        figure, (loss_axis, accuracy_axis) = plt.subplots(
            1,
            2,
            figsize=(12, 5),
        )

        loss_axis.plot(
            epochs,
            history["train_loss"],
            label="Entraînement",
        )
        loss_axis.plot(
            epochs,
            history["validation_loss"],
            label="Validation",
        )
        loss_axis.set(
            title="Perte",
            xlabel="Époque",
            ylabel="Cross-entropy",
        )
        loss_axis.legend()

        accuracy_axis.plot(
            epochs,
            history["train_accuracy"],
            label="Entraînement",
        )
        accuracy_axis.plot(
            epochs,
            history["validation_accuracy"],
            label="Validation",
        )
        accuracy_axis.set(
            title="Exactitude",
            xlabel="Époque",
            ylabel="Exactitude (%)",
        )
        accuracy_axis.legend()

        figure.tight_layout()
        figure.savefig("learning_curves.png")
        plt.close(figure)

        print(
            "Courbes d’apprentissage enregistrées dans "
            "learning_curves.png"
        )
    else:
        print(
            "Historique d’apprentissage introuvable. "
            "Lancez train.py avant evaluate.py."
        )


if __name__ == "__main__":
    main()
