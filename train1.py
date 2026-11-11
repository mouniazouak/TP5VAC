import argparse
import json

import torch
from torch import nn

from data_loader import train_loader, validation_loader
from model3d1 import Model


def get_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU indisponible")
        return torch.device("xpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA indisponible")
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Train the per-frame 2D ablation model"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "xpu", "cuda", "cpu"),
        default="auto",
    )
    args = parser.parse_args()
    device = get_device(args.device)

    model = Model(num_classes=101).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.025,
        momentum=0.9,
        weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=60,
    )
    num_epochs = 60
    best_accuracy = 0.0
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "learning_rate": [],
    }

    print(f"Device utilisé : {device}")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for batch in train_loader:
            data, target = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            train_correct += output.argmax(dim=1).eq(target).sum().item()

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100.0 * train_correct / len(train_loader.dataset)
        model.eval()
        validation_loss = 0.0
        validation_correct = 0

        with torch.no_grad():
            for batch in validation_loader:
                data, target = batch[0].to(device), batch[1].to(device)
                output = model(data)
                validation_loss += loss_fn(output, target).item() * data.size(0)
                validation_correct += output.argmax(dim=1).eq(target).sum().item()

        validation_loss /= len(validation_loader.dataset)
        validation_accuracy = 100.0 * validation_correct / len(validation_loader.dataset)
        learning_rate = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        history["learning_rate"].append(learning_rate)

        with open("learning_history_2d.json", "w", encoding="utf-8") as history_file:
            json.dump(history, history_file, indent=2)

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(model.state_dict(), "best_plain3d_2d.pth")

        scheduler.step()
        print(
            f"Epoch: {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Accuracy: {train_accuracy:.2f}%, "
            f"Validation Loss: {validation_loss:.4f}, "
            f"Validation Accuracy: {validation_accuracy:.2f}%"
        )

    print(f"Best validation accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
