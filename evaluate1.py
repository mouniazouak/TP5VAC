import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_loader import get_test_loader
from model3d1 import Model


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()
    model = Model(num_classes=101).to(device)
    model.load_state_dict(
        torch.load(
            "best_plain3d_2d.pth",
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    video_outputs = defaultdict(list)
    video_targets = {}
    clip_targets = []
    clip_predictions = []

    with torch.no_grad():
        for clip_index in range(5):
            test_loader = get_test_loader(
                manifest_path="manifest.csv",
                batch_size=32,
                num_workers=0,
                clip_index=clip_index,
            )

            for data, target, video_ids in test_loader:
                output = model(data.to(device))
                probabilities = torch.softmax(output, dim=1).cpu()

                if clip_index == 0:
                    clip_targets.extend(target.tolist())
                    clip_predictions.extend(
                        probabilities.argmax(dim=1).tolist()
                    )

                for video_id, label, probability in zip(
                    video_ids,
                    target.tolist(),
                    probabilities,
                ):
                    video_outputs[video_id].append(probability)
                    video_targets[video_id] = label

    video_targets_list = []
    video_predictions_list = []

    for video_id, outputs in video_outputs.items():
        mean_output = torch.stack(outputs).mean(dim=0)
        video_targets_list.append(video_targets[video_id])
        video_predictions_list.append(mean_output.argmax().item())

    clip_accuracy = 100.0 * np.mean(
        np.array(clip_targets) == np.array(clip_predictions)
    )
    video_accuracy = 100.0 * np.mean(
        np.array(video_targets_list) == np.array(video_predictions_list)
    )

    print(f"Ablation 2D - Clip-level accuracy: {clip_accuracy:.2f}%")
    print(f"Ablation 2D - Video-level accuracy: {video_accuracy:.2f}%")

    class_correct = [0] * 101
    class_total = [0] * 101

    for target, prediction in zip(
        video_targets_list,
        video_predictions_list,
    ):
        class_total[target] += 1
        if target == prediction:
            class_correct[target] += 1

    class_results = []
    for class_index in range(101):
        if class_total[class_index] > 0:
            accuracy = 100.0 * class_correct[class_index] / class_total[class_index]
            class_results.append((class_index, accuracy))

    print("10 meilleures classes :")
    print(sorted(class_results, key=lambda item: item[1], reverse=True)[:10])
    print("10 classes les plus difficiles :")
    print(sorted(class_results, key=lambda item: item[1])[:10])

    confusion_matrix = torch.zeros(101, 101, dtype=torch.int64)
    for target, prediction in zip(
        video_targets_list,
        video_predictions_list,
    ):
        confusion_matrix[target][prediction] += 1

    torch.save(confusion_matrix, "confusion_matrix_2d.pt")
    print("Matrice de confusion 2D enregistrée dans confusion_matrix_2d.pt")

    history_path = Path("learning_history_2d.json")
    if history_path.exists():
        with history_path.open(encoding="utf-8") as history_file:
            history = json.load(history_file)

        epochs = range(1, len(history["train_loss"]) + 1)
        figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 5))
        loss_axis.plot(epochs, history["train_loss"], label="Entraînement")
        loss_axis.plot(epochs, history["validation_loss"], label="Validation")
        loss_axis.set(title="Perte - Ablation 2D", xlabel="Époque", ylabel="Cross-entropy")
        loss_axis.legend()
        accuracy_axis.plot(epochs, history["train_accuracy"], label="Entraînement")
        accuracy_axis.plot(epochs, history["validation_accuracy"], label="Validation")
        accuracy_axis.set(title="Exactitude - Ablation 2D", xlabel="Époque", ylabel="Exactitude (%)")
        accuracy_axis.legend()
        figure.tight_layout()
        figure.savefig("learning_curves_2d.png")
        plt.close(figure)
        print("Courbes 2D enregistrées dans learning_curves_2d.png")


if __name__ == "__main__":
    main()
